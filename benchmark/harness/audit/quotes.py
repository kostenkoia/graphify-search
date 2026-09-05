"""Check a system's `quote:` scalars against its frozen vendor documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from benchmark.harness import config, rules
from benchmark.harness.audit._shared import _WS

if TYPE_CHECKING:
    from pathlib import Path


def fold(text: str) -> str:
    """Collapse runs of whitespace to one space and strip the ends."""
    return _WS.sub(" ", text).strip()


def _quotes_in(obj: object) -> tuple[list[str], list[str]]:
    quotes: list[str] = []
    problems: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "quote":
                if isinstance(v, str):
                    quotes.append(v)
                elif v is not None:
                    problems.append(f"quote is not a string: {v!r}")
            else:
                sub_q, sub_p = _quotes_in(v)
                quotes.extend(sub_q)
                problems.extend(sub_p)
    elif isinstance(obj, list):
        for v in obj:
            sub_q, sub_p = _quotes_in(v)
            quotes.extend(sub_q)
            problems.extend(sub_p)
    return quotes, problems


def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def check_quotes(benchmark: Path, system: str) -> list[str]:
    """Return one message per problem found in a system's quotes and their frozen docs.

    Parameters
    ----------
    benchmark : Path
        Root directory holding `systems/<system>/`.
    system : str
        System id under `systems/`.

    Returns
    -------
    list of str
        Empty when every doc listed exists, is not a symlink, resolves
        inside `docs/`, and matches its recorded hash, and every quote is a
        non-empty string found verbatim (whitespace-folded) in some doc.
    """
    h = config.load_harness(benchmark, system)
    docdir = benchmark / "systems" / system / "docs"
    problems: list[str] = []
    corpus: list[str] = []
    for name, meta in h.docs.items():
        path = docdir / name
        # inv: a symlink's recorded hash pins the link string, not the content the corpus reads
        if path.is_symlink():
            problems.append(f"doc is a symlink: {path}")
            continue
        # inv: a docs key must resolve inside docs/, or author-written text outside it could self-authorise a quote
        if not _within(docdir, path):
            problems.append(f"doc path escapes docs directory: {name}")
            continue
        if not path.is_file():
            problems.append(f"doc missing: {path}")
            continue
        expected = meta.get("sha256") if isinstance(meta, dict) else meta
        if rules.sha256_file(path) != expected:
            problems.append(f"doc sha256 differs: {path}")
        corpus.append(fold(path.read_text(encoding="utf-8")))
    sources: list[object] = [h.raw]
    manifest = benchmark / "systems" / system / "manifest.yaml"
    if manifest.is_file():
        sources.append(yaml.safe_load(manifest.read_text(encoding="utf-8")))
    quotes: set[str] = set()
    for src in sources:
        found, bad = _quotes_in(src)
        quotes.update(found)
        problems.extend(bad)
    for q in quotes:
        folded = fold(q)
        # inv: an empty folded quote is a substring of every corpus, so it must fail rather than vacuously match
        if not folded:
            problems.append(f"quote is empty: {q!r}")
        elif not any(folded in c for c in corpus):
            problems.append(f"quote not in docs: {q!r}")
    return problems
