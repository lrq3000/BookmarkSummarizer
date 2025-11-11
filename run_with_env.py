#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-platform launcher to run a command with temporary environment variables,
without a '--' separator: the target command is a single positional STRING.

Improvements:
- Removes inline comments that are outside of quotes ("# ...") and trims leading/trailing spaces.
- Preserves previous behavior (--env, --set, --require, --exec).
"""

from __future__ import annotations
import argparse
import os
import shlex
import sys
from typing import Dict, Iterable, Tuple, List

# ---------- .env line sanitization (new) ----------

def _strip_inline_comment(line: str) -> str:
    """
    Remove inline comments not inside quotes:
    - Everything after an unquoted '#' is ignored.
    - '#' inside single or double quotes is kept.
    - Handles escaping of a quote with backslash *inside* the same quote type.
    """
    in_single = False
    in_double = False
    escaped = False
    out_chars: List[str] = []

    for ch in line:
        if in_single:
            if escaped:
                out_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                out_chars.append(ch)  # keep the backslash; it is part of the value
                continue
            if ch == "'":
                in_single = False
                out_chars.append(ch)
                continue
            out_chars.append(ch)
            continue

        if in_double:
            if escaped:
                out_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                out_chars.append(ch)
                continue
            if ch == '"':
                in_double = False
                out_chars.append(ch)
                continue
            out_chars.append(ch)
            continue

        # Context: outside quotes
        if ch == "#":
            break  # start of an inline comment
        if ch == "'":
            in_single = True
            out_chars.append(ch)
            continue
        if ch == '"':
            in_double = True
            out_chars.append(ch)
            continue
        out_chars.append(ch)

    return "".join(out_chars).strip()


# ---------- KEY=VALUE parsing ----------

def parse_key_value(s: str) -> Tuple[str, str]:
    """
    Robustly parse a KEY=VALUE pair.
    - Trims leading/trailing spaces, strips unquoted inline comments, ignores empty/comment lines.
    - Accepts "export KEY=VALUE".
    - Removes surrounding quotes from the value.
    - Interprets only \n and \t escapes in the value.
    """
    # 1) Global trim + inline comment removal
    s = _strip_inline_comment(s.strip())
    if not s:
        raise ValueError("Empty line after sanitization")

    # 2) Optional "export" prefix
    if s.lower().startswith("export "):
        s = s[7:].lstrip()

    # 3) Split KEY=VALUE
    if "=" not in s:
        raise ValueError(f"Invalid format (no '='): {s!r}")

    key, val = s.split("=", 1)
    key = key.strip()
    val = val.strip()

    if not key:
        raise ValueError(f"Empty key in: {s!r}")

    # 4) Remove strict surrounding quotes
    if (len(val) >= 2) and (val[0] == val[-1]) and val[0] in ("'", '"'):
        val = val[1:-1]

    # 5) Minimal escapes
    val = val.replace(r"\n", "\n").replace(r"\t", "\t")
    return key, val


def load_dotenv(path: str) -> Dict[str, str]:
    """
    Load a .env file.
    - Ignores empty lines and comments (including inline) after sanitization.
    - Raises ValueError on an unparsable line.
    """
    env: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            cleaned = _strip_inline_comment(raw.strip())
            if not cleaned:
                continue  # empty or pure comment
            try:
                k, v = parse_key_value(cleaned)
                env[k] = v
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return env


def merge_env(base: Dict[str, str], *layers: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """
    Merge assignment layers on top of a base environment.
    Later layers override earlier ones.
    """
    out = dict(base)
    for layer in layers:
        for k, v in layer:
            out[k] = v
    return out


def ensure_required(env: Dict[str, str], required: List[str]) -> None:
    """
    Enforce presence of required environment variables.
    Exits with an error if any is missing or empty.
    """
    missing = [k for k in required if k not in env or env[k] == ""]
    if missing:
        raise SystemExit("Missing required environment variables: " + ", ".join(missing))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_with_env.py",
        description="Run a command with temporary environment variables (all OS).",
    )
    p.add_argument("--env", dest="env_file", metavar="PATH",
                   help=".env file (optional, KEY=VALUE).")
    p.add_argument("--set", dest="sets", metavar="KEY=VALUE", action="append", default=[],
                   help="Define/override a variable (repeatable).")
    p.add_argument("--require", dest="required", metavar="KEY", action="append", default=[],
                   help="Require a variable to be present (repeatable).")
    p.add_argument("--exec", dest="use_exec", action="store_true",
                   help="Replace the current process (execve).")
    p.add_argument("--python", dest="py", default=sys.executable,
                   help="Python interpreter to use if the command begins with a .py file.")
    # Final positional: full TARGET COMMAND as a single string, quoted in the shell
    p.add_argument("command", metavar="COMMAND", type=str,
                   help="Target command as a single string, e.g., 'python app.py --opt X'")
    return p


def main(argv: List[str]) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    base_env = dict(os.environ)

    # Optional .env layer
    file_layer: List[Tuple[str, str]] = []
    if args.env_file:
        if not os.path.exists(args.env_file):
            raise SystemExit(f".env file not found: {args.env_file}")
        file_layer = list(load_dotenv(args.env_file).items())

    # --set overrides
    cli_layer: List[Tuple[str, str]] = []
    for item in args.sets:
        try:
            k, v = parse_key_value(item)
            cli_layer.append((k, v))
        except ValueError as e:
            raise SystemExit(f"Invalid --set: {e}") from e

    env_final = merge_env(base_env, file_layer, cli_layer)

    if args.required:
        ensure_required(env_final, args.required)

    # Robustly split the single-string command into argv (cross-platform)
    # - shlex.split handles quotes and spaces correctly.
    cmd = shlex.split(args.command, posix=True)

    # Heuristic: if the first token looks like a .py file, prefix with the chosen interpreter
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
    else:
        import subprocess
        try:
            completed = subprocess.run(cmd, env=env_final)
            return int(completed.returncode or 0)
        except FileNotFoundError as e:
            sys.stderr.write(f"Command not found: {cmd!r}\n{e}\n")
            return 127

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
