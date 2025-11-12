#!/usr/bin/env python3
# semantic_bookmarks_app.py
"""
Semantic Search for Bookmarks — scalable & on‑disk (HNSW | IVF‑PQ/OPQ | IVF‑DISK)
---------------------------------------------------------------------------------
Single-file FastAPI app with:
- HNSW-Flat (cosine) for high-accuracy, mid-scale
- IVF-PQ/OPQ (cosine) for tens–hundreds of millions (in‑RAM inverted lists)
- IVF‑DISK (IVF-PQ/OPQ with on‑disk inverted lists) for 100M+ on NVMe (Option C)
- Memory-mapped loading of large FAISS indices
- Incremental upsert that skips already-processed entries by a stable key
- Streaming JSON ingestion; IVF-PQ trains on a sample for first build
- Minimal web UI; query knobs: Top‑K, efSearch (HNSW), nprobe (IVF)

Usage
-----
Install deps (Python >=3.9):
  pip install -U fastapi uvicorn[standard] sentence-transformers faiss-cpu numpy ujson tqdm python-multipart aiofiles

Build (first time) then reuse persisted index:
  # HNSW
  python semantic_bookmarks_app.py --data /path/to/bookmarks.json --persist ./index_hnsw --index-type hnsw

  # IVF-PQ with OPQ, tuned for 10M–100M (in-RAM inverted lists)
  python semantic_bookmarks_app.py \
    --data /path/to/bookmarks.json --persist ./index_ivfpq \
    --index-type ivfpq --nlist 32768 --m 64 --nbits 8 --nprobe 32 --opq-m 64

  # IVF‑DISK (IVF-PQ/OPQ with on-disk inverted lists) for 100M+
  python semantic_bookmarks_app.py \
    --data /path/to/bookmarks.json --persist ./index_ivfdisk \
    --index-type ivfdisk --nlist 32768 --m 64 --nbits 8 --nprobe 32 --opq-m 64

Incremental upsert via UI ("Upsert JSON") or API POST /api/upsert_json.

Environment overrides: SB_EMBED_MODEL, SB_BATCH, SB_HNSW_M, SB_EF_CON, SB_EF_SEARCH,
SB_MMAP (1 enables mmap read_index), SB_SAMPLE_MAX (IVF training cap), SB_IVF_ON_DISK (default for ivfpq only).
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable

import numpy as np
import ujson as json
from tqdm import tqdm

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

# -------------------------------
# Config defaults
# -------------------------------
MODEL_NAME = os.environ.get("SB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
EMB_DIM = 384
BATCH_SIZE = int(os.environ.get("SB_BATCH", 512))
HNSW_M = int(os.environ.get("SB_HNSW_M", 32))
EF_CONSTRUCTION = int(os.environ.get("SB_EF_CON", 200))
EF_SEARCH_DEFAULT = int(os.environ.get("SB_EF_SEARCH", 128))
TOPK_DEFAULT = 10
MMAP_ENABLED = bool(int(os.environ.get("SB_MMAP", "1")))
TRAIN_SAMPLE_MAX = int(os.environ.get("SB_SAMPLE_MAX", 1_000_000))
IVF_ON_DISK_DEFAULT = bool(int(os.environ.get("SB_IVF_ON_DISK", "0")))  # applies to ivfpq, ivfdisk forces on

# IVF-PQ defaults
IVF_NLIST_DEFAULT = 16384
PQ_M_DEFAULT = 64
PQ_NBITS_DEFAULT = 8
NPROBE_DEFAULT = 32
OPQ_M_DEFAULT = 0  # 0 disables OPQ

# -------------------------------
# Helpers
# -------------------------------

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def unique_key(rec: Dict[str, Any]) -> str:
    """Stable de-duplication key: prefer guid, else id, else url, else title+date."""
    for k in ("guid", "id", "url"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return f"{k}:{v.strip()}"
    title = (rec.get("title") or rec.get("name") or "").strip()
    date_added = (rec.get("date_added") or "").strip()
    return f"title:{title}|date:{date_added}"


# -------------------------------
# Index manager (HNSW | IVF-PQ/OPQ | IVF-DISK)
# -------------------------------
class VectorIndex:
    def __init__(self, persist_dir: Path, index_type: str = "hnsw", *,
                 nlist: int = IVF_NLIST_DEFAULT, pq_m: int = PQ_M_DEFAULT, nbits: int = PQ_NBITS_DEFAULT,
                 nprobe: int = NPROBE_DEFAULT, opq_m: int = OPQ_M_DEFAULT, ivf_on_disk: bool = False):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.meta: List[Dict[str, Any]] = []
        self.keys: set[str] = set()
        self.index_type = index_type.lower()
        # HNSW controls
        self.ef_search = EF_SEARCH_DEFAULT
        # IVF-PQ controls
        self.nlist = nlist
        self.pq_m = pq_m
        self.nbits = nbits
        self.nprobe = nprobe
        self.opq_m = opq_m
        self.ivf_on_disk = ivf_on_disk

    # Filenames
    @property
    def index_path(self) -> Path:
        return self.persist_dir / "faiss_index.bin"

    @property
    def meta_path(self) -> Path:
        return self.persist_dir / "meta.jsonl"

    @property
    def invlists_path(self) -> Path:
        return self.persist_dir / "ivf_lists.dat"  # FAISS OnDiskInvertedLists backing file

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    # ---- HNSW ----
    def _build_hnsw(self) -> faiss.Index:
        index = faiss.IndexHNSWFlat(EMB_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = EF_CONSTRUCTION
        index.hnsw.efSearch = self.ef_search
        return index

    def set_ef_search(self, ef: int):
        self.ef_search = max(8, int(ef))
        if self.index is not None and isinstance(self.index, faiss.IndexHNSW):
            self.index.hnsw.efSearch = self.ef_search

    # ---- IVF-PQ/OPQ ----
    def _build_ivfpq(self) -> faiss.Index:
        coarse = faiss.IndexFlatIP(EMB_DIM)
        ivfpq = faiss.IndexIVFPQ(coarse, EMB_DIM, self.nlist, self.pq_m, self.nbits, faiss.METRIC_INNER_PRODUCT)
        if self.opq_m and self.opq_m > 0:
            opq = faiss.OPQMatrix(EMB_DIM, self.opq_m)
            return faiss.IndexPreTransform(opq, ivfpq)
        return ivfpq

    def _unwrap_ivf(self, index: faiss.Index) -> faiss.IndexIVF:
        base = index
        if isinstance(base, faiss.IndexPreTransform):
            base = faiss.downcast_index(base.index)
        assert isinstance(base, faiss.IndexIVF)
        return base

    def set_nprobe(self, nprobe: int):
        self.nprobe = max(1, int(nprobe))
        base = self._unwrap_ivf(self.index) if self.index is not None else None
        if isinstance(base, faiss.IndexIVF):
            base.nprobe = self.nprobe

    # ---- Persistence ----
    def load(self):
        if not self.exists():
            raise FileNotFoundError("No persisted index found")
        flags = faiss.IO_FLAG_MMAP if MMAP_ENABLED else 0
        self.index = faiss.read_index(str(self.index_path), flags)
        # Load metadata & keys
        self.meta = [json.loads(line) for line in self.meta_path.read_text(encoding="utf-8").splitlines() if line]
        self.keys = {m.get("_key") for m in self.meta if m.get("_key")}
        # Post-load tuning
        if self.index_type == "hnsw":
            self.set_ef_search(self.ef_search)
        else:
            self.set_nprobe(self.nprobe)

    def _append_meta(self, metas: List[Dict[str, Any]]):
        with self.meta_path.open("a", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")
        self.meta.extend(metas)
        self.keys.update(m["_key"] for m in metas)

    def save_full_meta(self):
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m) + "\n")

    # ---- Build ----
    def build_hnsw(self, embeddings: np.ndarray, meta: List[Dict[str, Any]]):
        index = self._build_hnsw()
        index.add(embeddings.astype(np.float32))
        self.index = index
        self.meta = meta
        self.keys = {m["_key"] for m in meta}
        faiss.write_index(self.index, str(self.index_path))
        self.save_full_meta()

    def _prepare_on_disk_lists(self, index: faiss.Index):
        if not self.ivf_on_disk:
            return
        base = self._unwrap_ivf(index)
        code_size = None
        if isinstance(base, faiss.IndexIVFPQ):
            code_size = base.pq.code_size
        else:
            code_size = getattr(base, "code_size", None)
        if not code_size:
            raise RuntimeError("Cannot determine IVF code size for on-disk inverted lists")
        inv_path = str(self.invlists_path)
        on_disk = faiss.OnDiskInvertedLists(base.nlist, int(code_size), inv_path)
        base.replace_invlists(on_disk)
        base.own_invlists = False
        print(f"[ivf] On-disk inverted lists at {inv_path} (code_size={code_size} bytes)")

    def build_ivfpq(self, training: np.ndarray, all_vecs_iter: Iterable[np.ndarray], total: int, meta: List[Dict[str, Any]]):
        index = self._build_ivfpq()
        base_index = self._unwrap_ivf(index)
        base_index.cp.min_points_per_centroid = 39
        base_index.cp.max_points_per_centroid = 256
        print(f"[ivf] Training with sample {training.shape}")
        index.train(training.astype(np.float32))
        self._prepare_on_disk_lists(index)  # enable on-disk if configured
        added = 0
        for xb in all_vecs_iter:
            index.add(xb.astype(np.float32))
            added += xb.shape[0]
            if added % 100_000 == 0:
                print(f"[ivf] Added {added}/{total}")
        self.index = index
        self.set_nprobe(self.nprobe)
        self.meta = meta
        self.keys = {m["_key"] for m in meta}
        faiss.write_index(self.index, str(self.index_path))
        self.save_full_meta()

    # ---- Incremental add ----
    def add_vectors(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        if self.index is None:
            raise RuntimeError("Index not loaded")
        base = self.index if not isinstance(self.index, faiss.IndexPreTransform) else faiss.downcast_index(self.index.index)
        if isinstance(base, faiss.IndexIVF) and not self.index.is_trained:
            raise RuntimeError("IVF-PQ index is not trained; cannot add incrementally.")
        if isinstance(base, faiss.IndexIVF) and self.ivf_on_disk and not isinstance(base.invlists, faiss.OnDiskInvertedLists):
            self._prepare_on_disk_lists(self.index)
        self.index.add(vecs.astype(np.float32))
        faiss.write_index(self.index, str(self.index_path))
        self._append_meta(metas)

    def search(self, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index not loaded")
        k = max(1, int(k))
        D, I = self.index.search(qvec.astype(np.float32), k)
        return D, I


# -------------------------------
# Ingestor
# -------------------------------
class Ingestor:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size

    def _compose_text(self, rec: Dict[str, Any]) -> str:
        fields = [rec.get("title") or rec.get("name") or "",
                  rec.get("summary") or "",
                  rec.get("content") or "",
                  rec.get("url") or ""]
        text = "\n".join([s.strip() for s in fields if s and isinstance(s, str)])
        return text[:5000]

    def load_json_stream(self, path: Path) -> List[Dict[str, Any]]:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list of objects")
        return data

    def prepare_meta_and_texts(self, records: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for r in records:
            key = unique_key(r)
            text = self._compose_text(r)
            texts.append(text)
            metas.append({
                "_key": key,
                "id": r.get("id"),
                "guid": r.get("guid"),
                "title": r.get("title") or r.get("name") or "",
                "url": r.get("url") or "",
                "snippet": (r.get("summary") or r.get("content") or "")[:400],
                "date_added": r.get("date_added"),
            })
        return texts, metas

    def embed_batches(self, texts: List[str]) -> Iterable[np.ndarray]:
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding", ncols=80):
            batch = texts[i:i + self.batch_size]
            vecs = self.model.encode(batch, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
            yield vecs.astype(np.float32)


# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Semantic Bookmarks Search", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

vindex: Optional[VectorIndex] = None
ingestor: Optional[Ingestor] = None


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return UI_HTML


@app.post("/api/search")
async def api_search(payload: Dict[str, Any]):
    global vindex, ingestor
    if vindex is None or vindex.index is None or ingestor is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    q = (payload.get("q") or "").strip()
    k = int(payload.get("k") or TOPK_DEFAULT)
    ef = payload.get("ef_search")
    nprobe = payload.get("nprobe")
    if vindex.index_type == "hnsw" and ef is not None:
        vindex.set_ef_search(int(ef))
    if vindex.index_type != "hnsw" and nprobe is not None:
        vindex.set_nprobe(int(nprobe))
    if not q:
        return {"results": []}
    qvec = ingestor.model.encode([q], normalize_embeddings=True)
    D, I = vindex.search(qvec, k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        meta = vindex.meta[idx]
        results.append({
            "score": float(score),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "snippet": meta.get("snippet"),
            "id": meta.get("id"),
            "guid": meta.get("guid"),
            "date_added": meta.get("date_added"),
        })
    return {"results": results}


@app.post("/api/upload_json")
async def upload_json(file: UploadFile = File(...)):
    """Full rebuild from a JSON array (kept for parity)."""
    global vindex, ingestor
    if ingestor is None or vindex is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    content = await file.read()
    try:
        records = json.loads(content)
        if not isinstance(records, list):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="File must be a JSON array of objects")

    texts, meta = ingestor.prepare_meta_and_texts(records)

    if vindex.index_type == "hnsw":
        embs = np.vstack(list(ingestor.embed_batches(texts))).astype(np.float32) if texts else np.zeros((0, EMB_DIM), np.float32)
        vindex.build_hnsw(embs, meta)
    else:
        batches = list(ingestor.embed_batches(texts))
        if not batches:
            raise HTTPException(status_code=400, detail="No records to index")
        cum = 0
        train_list = []
        for b in batches:
            if cum >= TRAIN_SAMPLE_MAX:
                break
            take = min(TRAIN_SAMPLE_MAX - cum, b.shape[0])
            train_list.append(b[:take])
            cum += take
        train_mat = np.vstack(train_list)
        vindex.build_ivfpq(train_mat, iter(batches), total=sum(b.shape[0] for b in batches), meta=meta)

    return {"status": "ok", "count": len(meta)}


@app.post("/api/upsert_json")
async def upsert_json(file: UploadFile = File(...)):
    """Incrementally add new bookmarks; skip those with keys already indexed."""
    global vindex, ingestor
    if ingestor is None or vindex is None or vindex.index is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    content = await file.read()
    try:
        records = json.loads(content)
        if not isinstance(records, list):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="File must be a JSON array of objects")

    texts_all, metas_all = ingestor.prepare_meta_and_texts(records)
    to_add_texts: List[str] = []
    to_add_meta: List[Dict[str, Any]] = []
    skipped = 0
    for t, m in zip(texts_all, metas_all):
        if m["_key"] in vindex.keys:
            skipped += 1
            continue
        to_add_texts.append(t)
        to_add_meta.append(m)

    if not to_add_texts:
        return {"status": "ok", "added": 0, "skipped": skipped}

    base = vindex.index if not isinstance(vindex.index, faiss.IndexPreTransform) else faiss.downcast_index(vindex.index.index)
    if vindex.index_type != "hnsw" and isinstance(base, faiss.IndexIVF) and not vindex.index.is_trained:
        add_batches = list(ingestor.embed_batches(to_add_texts))
        train_mat = np.vstack(add_batches)
        vindex.build_ivfpq(train_mat, iter(add_batches), total=train_mat.shape[0], meta=to_add_meta)
        return {"status": "ok", "added": len(to_add_meta), "skipped": skipped, "trained_on_upsert": True}

    add_batches = list(ingestor.embed_batches(to_add_texts))
    if not add_batches:
        return {"status": "ok", "added": 0, "skipped": skipped}

    offset = 0
    for b in add_batches:
        batch_meta = to_add_meta[offset: offset + b.shape[0]]
        vindex.add_vectors(b, batch_meta)
        offset += b.shape[0]

    return {"status": "ok", "added": len(to_add_meta), "skipped": skipped}


# -------------------------------
# Minimal UI (Upsert control included)
# -------------------------------
UI_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Semantic Bookmarks Search</title>
  <style>
    :root {{ --bg:#0b0d10; --card:#12161b; --fg:#e6edf3; --muted:#91a0ad; --accent:#3aa6ff; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--fg); }}
    header {{ padding: 16px 20px; border-bottom: 1px solid #1f242c; position: sticky; top:0; background: rgba(11,13,16,0.9); backdrop-filter: blur(6px); }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
    .row {{ display: grid; grid-template-columns: 1fr auto auto; gap: 12px; }}
    input, button, label {{ padding: 12px 14px; border-radius: 12px; border: 1px solid #26313b; background: var(--card); color: var(--fg); outline: none; }}
    input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px rgba(58,166,255,0.2); }}
    button {{ cursor: pointer; }}
    .results {{ margin-top: 16px; display: grid; gap: 12px; }}
    .card {{ background: var(--card); border: 1px solid #1f242c; padding: 16px; border-radius: 16px; }}
    .title {{ font-weight: 600; font-size: 16px; margin: 0 0 6px; }}
    .url {{ color: var(--muted); font-size: 13px; word-break: break-all; }}
    .snippet {{ color: #b9c6d2; margin-top: 8px; line-height: 1.4; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .row2 {{ display:flex; gap:12px; margin-top: 12px; align-items:center; flex-wrap: wrap; }}
    .pill {{ display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; border:1px solid #26313b; }}
    .footer {{ text-align:center; padding: 20px; color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <div class=\"container\" style=\"display:flex; gap:12px; align-items:center; flex-wrap:wrap;\">
      <div style=\"font-weight:700;\">Semantic Bookmarks</div>
      <div class=\"pill\"><span>efSearch (HNSW)</span><input id=\"ef\" type=\"number\" value=\"{EF_SEARCH_DEFAULT}\" min=\"8\" max=\"1024\" style=\"width:90px;\"/></div>
      <div class=\"pill\"><span>nprobe (IVF)</span><input id=\"nprobe\" type=\"number\" value=\"{NPROBE_DEFAULT}\" min=\"1\" max=\"1024\" style=\"width:90px;\"/></div>
      <div class=\"pill\"><span>Top K</span><input id=\"k\" type=\"number\" value=\"10\" min=\"1\" max=\"100\" style=\"width:80px;\"/></div>
      <div class=\"pill\"><label for=\"file\">Rebuild JSON</label><input id=\"file\" type=\"file\" accept=\"application/json\"/></div>
      <div class=\"pill\"><label for=\"upsert\">Upsert JSON</label><input id=\"upsert\" type=\"file\" accept=\"application/json\"/></div>
      <div id=\"stats\" class=\"muted\" style=\"margin-left:auto;\"></div>
    </div>
  </header>
  <div class=\"container\">
    <div class=\"row\">
      <input id=\"q\" type=\"text\" placeholder=\"Ask anything about your bookmarks…\" autofocus />
      <button id=\"search\">Search</button>
      <button id=\"clear\">Clear</button>
    </div>
    <div class=\"muted\" style=\"margin-top:8px;\">Cosine ANN via FAISS HNSW or IVF‑PQ/OPQ · optional on‑disk inverted lists (ivfdisk) · all‑MiniLM‑L6‑v2 embeddings</div>
    <div id=\"results\" class=\"results\"></div>
  </div>
  <div class=\"footer\">Incremental upserts skip duplicates automatically.</div>
  <script>
    const $ = (id) => document.getElementById(id);
    const resultsEl = $("results");
    const qEl = $("q");
    const kEl = $("k");
    const efEl = $("ef");
    const nprobeEl = $("nprobe");
    const statsEl = $("stats");
    console.log('Script loaded, resultsEl:', resultsEl);
    
    async function search() {
      console.log('Search called, resultsEl:', resultsEl, 'data will be fetched');
      const q = qEl.value.trim();
      if (!q) { resultsEl.innerHTML = ""; return; }
      const k = parseInt(kEl.value || 10, 10);
      const ef = parseInt(efEl.value || """ + str(EF_SEARCH_DEFAULT) + """, 10);
      const nprobe = parseInt(nprobeEl.value || """ + str(NPROBE_DEFAULT) + """, 10);
      const t0 = performance.now();
      const r = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q, k, ef_search: ef, nprobe }) });
      console.log('Fetch response:', r);
      const data = await r.json();
      console.log('Data received:', data);
      const t1 = performance.now();
      statsEl.textContent = `Query ${(t1 - t0).toFixed(1)} ms · Results ${data.results.length}`;
      console.log('About to set innerHTML, resultsEl:', resultsEl, 'data.results:', data.results);
      resultsEl.innerHTML = data.results.map((it) => {
        const s = (it.snippet || '').replace(/</g, '&lt;');
        const url = it.url ? `<div class=\"url\">${it.url}</div>` : '';
        return `<div class=\"card\">
  <div class=\"title\"><a href=\"${it.url || '#'}\" target=\"_blank\" rel=\"noopener\" style=\"color:inherit; text-decoration:none;\">${it.title || '(untitled)'}</a></div>
  ${url}
  <div class=\"snippet\">${s}</div>
  <div class=\"muted\" style=\"margin-top:8px;\">score ${it.score.toFixed(4)} · id ${it.id || ''}</div>
</div>`;
      }).join('
');
    }

    $("search").addEventListener('click', search);
    qEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') search(); });
    $("clear").addEventListener('click', () => { qEl.value=''; resultsEl.innerHTML=''; statsEl.textContent=''; qEl.focus(); });

    async function postFile(inputId, url){
      const input = $(inputId);
      const file = input.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append('file', file);
      statsEl.textContent = (url.includes('upsert') ? 'Upserting (skipping duplicates)…' : 'Rebuilding index…');
      const r = await fetch(url, { method: 'POST', body: fd });
      const data = await r.json();
      statsEl.textContent = JSON.stringify(data);
      input.value = '';
    }

    $("file").addEventListener('change', () => postFile('file', '/api/upload_json'));
    $("upsert").addEventListener('change', () => postFile('upsert', '/api/upsert_json'));
  </script>
</body>
</html>
"""


# -------------------------------
# Bootstrap & CLI
# -------------------------------

def build_or_load(persist: Path, data_path: Optional[Path], *, index_type: str,
                  nlist: int, pq_m: int, nbits: int, nprobe: int, opq_m: int) -> Tuple[VectorIndex, "Ingestor"]:
    # ivfdisk enforces on-disk inverted lists; ivfpq uses env default; hnsw ignores
    ivf_on_disk = True if index_type == "ivfdisk" else (IVF_ON_DISK_DEFAULT if index_type == "ivfpq" else False)
    vi = VectorIndex(persist, index_type=index_type, nlist=nlist, pq_m=pq_m, nbits=nbits, nprobe=nprobe, opq_m=opq_m, ivf_on_disk=ivf_on_disk)
    ing = Ingestor(MODEL_NAME, BATCH_SIZE)
    if vi.exists():
        print(f"[load] Loading existing index from {persist} (mmap={MMAP_ENABLED})")
        vi.load()
    else:
        if not data_path or not data_path.exists():
            raise SystemExit("No persisted index found and --data not provided or invalid")
        print(f"[build] Ingesting {data_path}")
        records = ing.load_json_stream(data_path)
        print(f"[build] Records: {len(records)}")
        texts, meta = ing.prepare_meta_and_texts(records)
        if index_type == "hnsw":
            embs = np.vstack(list(ing.embed_batches(texts))).astype(np.float32) if texts else np.zeros((0, EMB_DIM), np.float32)
            vi.build_hnsw(embs, meta)
            print(f"[build] HNSW built → {persist}")
        else:
            batches = list(ing.embed_batches(texts))
            if not batches:
                raise SystemExit("No records to index")
            cum = 0
            train_list = []
            for b in batches:
                if cum >= TRAIN_SAMPLE_MAX:
                    break
                take = min(TRAIN_SAMPLE_MAX - cum, b.shape[0])
                train_list.append(b[:take])
                cum += take
            train_mat = np.vstack(train_list)
            total = sum(b.shape[0] for b in batches)
            vi.build_ivfpq(train_mat, iter(batches), total=total, meta=meta)
            print(f"[build] IVF ({'disk' if vi.ivf_on_disk else 'ram'}) built → {persist}")
    return vi, ing


def main():
    parser = argparse.ArgumentParser(description="Semantic Bookmarks Search Server")
    parser.add_argument("--data", type=str, default=None, help="Path to bookmarks JSON (array of objects)")
    parser.add_argument("--persist", type=str, default="./index_dir", help="Directory to store/load index")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index-type", type=str, default="hnsw", choices=["hnsw", "ivfpq", "ivfdisk"], help="ANN backend: hnsw | ivfpq | ivfdisk")
    parser.add_argument("--nlist", type=int, default=IVF_NLIST_DEFAULT, help="IVF: number of coarse centroids")
    parser.add_argument("--m", dest="pq_m", type=int, default=PQ_M_DEFAULT, help="IVF-PQ: number of subvectors (code size)")
    parser.add_argument("--nbits", type=int, default=PQ_NBITS_DEFAULT, help="IVF-PQ: bits per code (usually 8)")
    parser.add_argument("--nprobe", type=int, default=NPROBE_DEFAULT, help="IVF: number of lists to probe at search")
    parser.add_argument("--opq-m", type=int, default=OPQ_M_DEFAULT, help="Enable OPQ with given M (0 disables)")
    args = parser.parse_args()

    persist_dir = Path(args.persist)
    data_path = Path(args.data) if args.data else None

    global vindex, ingestor
    vindex, ingestor = build_or_load(
        persist_dir, data_path,
        index_type=args.index_type, nlist=args.nlist, pq_m=args.pq_m, nbits=args.nbits, nprobe=args.nprobe, opq_m=args.opq_m,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
