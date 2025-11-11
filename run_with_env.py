#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanceur multiplateforme d'un script avec variables d'environnement temporaires,
SANS séparateur '--' : la commande cible est une CHAÎNE unique en positionnelle.

Exemples :
  PS>  python .\run_with_env.py --env .env 'python .\app.py --mode test'
  PS>  python .\run_with_env.py --set API_KEY=abc 'python .\main.py'
  PS>  python .\run_with_env.py --exec 'python .\app.py --fast'
"""

from __future__ import annotations
import argparse
import os
import shlex
import sys
from typing import Dict, Iterable, Tuple, List

def parse_key_value(s: str) -> Tuple[str, str]:
    s = s.strip()
    if not s or s.startswith("#"):
        raise ValueError("Ligne vide ou commentaire")
    if s.lower().startswith("export "):
        s = s[7:].lstrip()
    if "=" not in s:
        raise ValueError(f"Format invalide (pas de '='): {s!r}")
    key, val = s.split("=", 1)
    key = key.strip()
    val = val.strip()
    if not key:
        raise ValueError(f"Clé vide dans: {s!r}")
    if (len(val) >= 2) and (val[0] == val[-1]) and val[0] in ("'", '"'):
        val = val[1:-1]
    val = val.replace(r"\n", "\n").replace(r"\t", "\t")
    return key, val

def load_dotenv(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                k, v = parse_key_value(line)
                env[k] = v
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return env

def merge_env(base: Dict[str, str], *layers: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    out = dict(base)
    for layer in layers:
        for k, v in layer:
            out[k] = v
    return out

def ensure_required(env: Dict[str, str], required: List[str]) -> None:
    missing = [k for k in required if k not in env or env[k] == ""]
    if missing:
        raise SystemExit("Variables d'environnement manquantes: " + ", ".join(missing))

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_with_env.py",
        description="Lance une commande avec des variables d'environnement temporaires (tous OS).",
    )
    p.add_argument("--env", dest="env_file", metavar="PATH",
                   help="Fichier .env optionnel (KEY=VALUE).")
    p.add_argument("--set", dest="sets", metavar="KEY=VALUE", action="append", default=[],
                   help="Définit/override une variable (répétable).")
    p.add_argument("--require", dest="required", metavar="KEY", action="append", default=[],
                   help="Exige qu'une variable soit définie (répétable).")
    p.add_argument("--exec", dest="use_exec", action="store_true",
                   help="Remplace le processus courant (execve).")
    p.add_argument("--python", dest="py", default=sys.executable,
                   help="Interpréteur Python à utiliser si la commande commence par un fichier .py.")
    # Positionnelle finale : CHAÎNE de commande complète, entre guillemets dans le shell
    p.add_argument("command", metavar="COMMAND", type=str,
                   help="Commande cible sous forme de chaîne, ex: 'python app.py --opt X'")
    return p

def main(argv: List[str]) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    base_env = dict(os.environ)

    file_layer: List[Tuple[str, str]] = []
    if args.env_file:
        if not os.path.exists(args.env_file):
            raise SystemExit(f"Fichier .env introuvable: {args.env_file}")
        file_layer = list(load_dotenv(args.env_file).items())

    cli_layer: List[Tuple[str, str]] = []
    for item in args.sets:
        try:
            k, v = parse_key_value(item)
            cli_layer.append((k, v))
        except ValueError as e:
            raise SystemExit(f"--set invalide: {e}") from e

    env_final = merge_env(base_env, file_layer, cli_layer)

    if args.required:
        ensure_required(env_final, args.required)

    # Transforme la CHAÎNE en liste d'arguments robustement (multiplateforme)
    # - shlex.split gère correctement les guillemets et espaces.
    cmd = shlex.split(args.command, posix=True)

    # Si la première "commande" semble être un fichier .py, préfixer par l'interpréteur demandé
    if cmd and cmd[0].lower().endswith(".py"):
        cmd = [args.py] + cmd

    if args.use_exec:
        import shutil
        program = cmd[0]
        try:
            os.execve(program, cmd, env_final)
        except FileNotFoundError:
            resolved = shutil.which(program, path=env_final.get("PATH"))
            if not resolved:
                raise
            os.execve(resolved, cmd, env_final)
        # unreachable
    else:
        import subprocess
        try:
            completed = subprocess.run(cmd, env=env_final)
            return int(completed.returncode or 0)
        except FileNotFoundError as e:
            sys.stderr.write(f"Commande introuvable: {cmd!r}\n{e}\n")
            return 127

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
