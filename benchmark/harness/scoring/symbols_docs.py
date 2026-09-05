"""Treat a documentation file as one unit."""

from __future__ import annotations


def is_doc(rel: str, doc_roots: list[str]) -> bool:
    """Report whether a path is project documentation.

    Parameters
    ----------
    rel : str
        File path relative to the snapshot source root.
    doc_roots : list of str
        Directory prefixes the snapshot declares as documentation.

    Returns
    -------
    bool
        True for markdown files under one of the declared roots.
    """
    return rel.endswith(".md") and any(rel.startswith(root) for root in doc_roots)


def extract(rel: str, text: str) -> list[dict]:
    """Return the file as a single unit.

    Parameters
    ----------
    rel : str
        File path relative to the snapshot source root.
    text : str
        File content.

    Returns
    -------
    list of dict
        One record spanning the whole file, or nothing for an empty file.
    """
    lines = text.splitlines()
    if not lines:
        return []
    return [{
        "path": rel,
        "fqname": rel,
        "kind": "doc",
        "start": 1,
        "end": len(lines),
    }]
