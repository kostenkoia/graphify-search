"""Extract symbols from Python sources: functions, classes, methods, nested definitions."""

from __future__ import annotations

import ast

KINDS = ("function", "method", "class")


def extract(rel: str, text: str) -> list[dict]:
    """Return every definition in one Python file.

    Parameters
    ----------
    rel : str
        File path relative to the snapshot source root.
    text : str
        File content.

    Returns
    -------
    list of dict
        Records with `path`, `fqname`, `kind`, `start`, `end`.
    """
    tree = ast.parse(text)
    lines = text.splitlines()
    out: list[dict] = []
    _walk(tree, rel[:-3].replace("/", "."), rel, lines, out)
    return out


def _walk(node: ast.AST, prefix: str, rel: str, lines: list[str], out: list[dict]) -> None:
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        fqname = f"{prefix}.{child.name}" if prefix else child.name
        kind = "class" if isinstance(child, ast.ClassDef) else (
            "method" if isinstance(node, ast.ClassDef) else "function")
        out.append({
            "path": rel,
            "fqname": fqname,
            "kind": kind,
            "start": child.lineno,
            "end": child.end_lineno or child.lineno,
        })
        # why: nested definitions are addressable places a question can point at,
        # so the walk continues instead of stopping at module level.
        _walk(child, fqname, rel, lines, out)
