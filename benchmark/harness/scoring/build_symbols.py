"""Build the symbol universe of a snapshot: one JSON line per addressable place."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from benchmark.harness import rules
from benchmark.harness.scoring.symbols_docs import extract as extract_docs
from benchmark.harness.scoring.symbols_docs import is_doc
from benchmark.harness.scoring.symbols_python import extract as extract_python
from benchmark.harness.scoring.symbols_shell import extract as extract_shell
from benchmark.harness.scoring.symbols_sql import extract as extract_sql
from benchmark.harness.scoring.symbols_typescript import extract as extract_typescript

EXTRACTORS = {
    ".py": extract_python,
    ".ts": extract_typescript,
    ".tsx": extract_typescript,
    ".sql": extract_sql,
    ".sh": extract_shell,
}

HASH_FILE = "symbols.sha256"


def write_hash(snapshot: Path) -> str:
    """Write `symbols.sha256` beside `symbols.jsonl` and return the digest.

    Parameters
    ----------
    snapshot : Path
        The snapshot directory holding `symbols.jsonl`.

    Returns
    -------
    str
        The sha256 hex digest of `symbols.jsonl`.
    """
    digest = rules.sha256_file(snapshot / "symbols.jsonl")
    # why: the `<digest>  <name>` line is what `shasum -c` reads, so the file checks without the harness
    (snapshot / HASH_FILE).write_text(f"{digest}  symbols.jsonl\n", encoding="utf-8")
    return digest


def load_universe(snapshot: Path) -> dict:
    """Read the snapshot's declaration of what belongs in the universe.

    Parameters
    ----------
    snapshot : Path
        Snapshot directory holding `meta.yaml`.

    Returns
    -------
    dict
        The `universe` section: `extensions`, `doc_roots`, `skip_dirs`.

    Raises
    ------
    SystemExit
        When the section is missing.
    """
    # why: what to index is a property of the snapshot, never a default of this script
    meta = yaml.safe_load((snapshot / "meta.yaml").read_text(encoding="utf-8"))
    universe = (meta or {}).get("universe")
    if not universe:
        raise SystemExit(f"{snapshot}/meta.yaml has no 'universe' section")
    return universe


def build(source: Path, universe: dict) -> tuple[list[dict], list[str], list[str]]:
    """Walk `source` and extract every addressable place the snapshot declares.

    Parameters
    ----------
    source : Path
        Root of the snapshot's frozen source tree.
    universe : dict
        The snapshot's `universe` section.

    Returns
    -------
    tuple of (list of dict, list of str, list of str)
        The records, sorted by path then start line and stamped with `id` and `sha256`;
        the files that could not be parsed; and the files whose declared extension has no
        extractor, one line each.
    """
    extensions = set(universe.get("extensions", []))
    doc_roots = list(universe.get("doc_roots", []))
    skip_dirs = set(universe.get("skip_dirs", []))
    records: list[dict] = []
    failed: list[str] = []
    holes: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(source)
        if skip_dirs & set(rel_path.parts):
            continue
        rel = str(rel_path)
        declared = path.suffix in extensions
        extractor = EXTRACTORS.get(path.suffix) if declared else None
        if extractor is None and is_doc(rel, doc_roots):
            extractor = extract_docs
        if extractor is None:
            # inv: a suffix the snapshot declares but no extractor covers leaves a hole in the
            # universe, and a hole nobody is told about is one no question can be scored against
            if declared:
                holes.append(f"no extractor for {path.suffix}: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
            found = extractor(rel, text)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as e:
            failed.append(f"{rel}: {type(e).__name__}: {e}")
            continue
        lines = text.splitlines()
        for rec in found:
            body = "\n".join(lines[rec["start"] - 1 : rec["end"]])
            rec["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
            records.append(rec)
    records.sort(key=lambda r: (r["path"], r["start"], r["fqname"]))
    for i, rec in enumerate(records):
        rec["id"] = f"sym_{i:05d}"
    return records, failed, holes


def main(argv: list[str] | None = None) -> int:
    """Write `symbols.jsonl` next to the snapshot's `source/` directory.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, plus `--hash-only` to hash an existing `symbols.jsonl`
        without rebuilding it, or None to use `sys.argv`.

    Returns
    -------
    int
        Zero.
    """
    ap = argparse.ArgumentParser(prog="benchmark.harness build-symbols")
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--hash-only", action="store_true",
                     help="hash an existing symbols.jsonl, build nothing")
    args = ap.parse_args(argv)
    if args.hash_only:
        digest = write_hash(args.snapshot)
        print(f"{digest}  symbols.jsonl")
        return 0
    universe = load_universe(args.snapshot)
    records, failed, holes = build(args.snapshot / "source", universe)
    target = args.snapshot / "symbols.jsonl"
    with target.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_hash(args.snapshot)
    print(f"{len(records)} symbols -> {target}")
    for line in failed:
        print(f"unparsed {line}", file=sys.stderr)
    if failed:
        print(f"{len(failed)} file(s) could not be parsed", file=sys.stderr)
    for line in holes:
        print(line, file=sys.stderr)
    if holes:
        print(f"{len(holes)} file(s) have no extractor", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
