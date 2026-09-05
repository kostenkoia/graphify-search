#!/bin/bash
# Shared by install, unlock and lock: argument parsing, the dry-run switch, and the path lists.
set -euo pipefail

DRY_RUN=0
ROOT=""
REASON=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --root=*) ROOT="${arg#--root=}" ;;
    --root) ROOT="__next__" ;;
    *) if [ "$ROOT" = "__next__" ]; then ROOT="$arg"; else REASON="${REASON:+$REASON }$arg"; fi ;;
  esac
done
[ -z "$ROOT" ] && ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[ "$ROOT" = "__next__" ] && { echo "--root needs a path" >&2; exit 2; }
BENCH="$ROOT/benchmark"
[ -d "$BENCH" ] || { echo "no benchmark/ under $ROOT" >&2; exit 2; }
# why: the machine's own harness interpreter runs the seal verb once it exists; before install the
# repository's .venv is the preferred one, and a checkout with neither -- a CI runner -- still has
# the package importable from the interpreter on PATH
HARNESS_PY="$BENCH/envs/harness/bin/python"
[ -x "$HARNESS_PY" ] || HARNESS_PY="$ROOT/.venv/bin/python"
[ -x "$HARNESS_PY" ] || HARNESS_PY="$(command -v python3 || true)"
[ -x "$HARNESS_PY" ] || { echo "no harness interpreter: envs/harness, .venv or python3 on PATH" >&2; exit 2; }
OWNER="${SUDO_USER:-$(id -un)}"

run() {
  # inv: every state-changing command goes through here, so --dry-run prints exactly what a real run does
  if [ "$DRY_RUN" = 1 ]; then printf '%s\n' "$*"; else eval "$@"; fi
}

need_root() {
  if [ "$DRY_RUN" = 0 ] && [ "$(id -u)" != 0 ]; then echo "$1 must run as root: sudo $0" >&2; exit 1; fi
}

paths() { (cd "$ROOT" && "$HARNESS_PY" -m benchmark.harness seal --paths "$1"); }
launchers() { (cd "$ROOT" && "$HARNESS_PY" -m benchmark.harness seal --launchers); }

lock_instrument() {
  # inv: files 444, directories 555, uchg on both; the directory flag is what refuses a new file
  paths instrument | while read -r rel; do
    if [ -d "$ROOT/$rel" ]; then run chmod 555 "$rel"; else run chmod 444 "$rel"; fi
    run chown -h root:wheel "$rel"
  done
  paths instrument | while read -r rel; do run chflags uchg "$rel"; done
}

open_instrument() {
  paths instrument | while read -r rel; do run chflags nouchg "$rel"; done
  paths instrument | while read -r rel; do
    run chown -h "$OWNER" "$rel"
    if [ -d "$ROOT/$rel" ]; then run chmod 755 "$rel"; else run chmod 644 "$rel"; fi
  done
}
