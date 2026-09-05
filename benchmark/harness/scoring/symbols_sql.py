"""Extract schema objects from SQL files: one unit per CREATE statement."""

from __future__ import annotations

import re

# why: DDL headers are regular enough to match directly, anchored at line start so a CREATE
# inside a string or a comment body is not picked up
# inv: the caret is redundant while this pattern is used with `match`, which anchors on its own;
# it is kept so the intent survives a change to `search`
_CREATE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:UNIQUE\s+)?"
    r"(TABLE|VIEW|MATERIALIZED\s+VIEW|INDEX|FUNCTION|PROCEDURE|TRIGGER|TYPE|SCHEMA|EXTENSION)"
    r"\s+(?:IF\s+NOT\s+EXISTS\s+)?((?:[\"`\[]?\w+[\"`\]]?\.?)+)",
    re.IGNORECASE,
)

# why: an identifier may arrive quoted and schema-qualified; the quoting is syntax
# and the schema is not the object a question points at, so both are stripped.
_QUOTES = str.maketrans("", "", '"`[]')


def extract(rel: str, text: str) -> list[dict]:
    """Return every CREATE statement in one SQL file.

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
    lines = text.splitlines()
    out: list[dict] = []
    module = rel.rsplit(".", 1)[0].replace("/", ".")
    for i, line in enumerate(lines):
        m = _CREATE.match(line)
        if not m:
            continue
        object_kind = " ".join(m.group(1).lower().split())
        name = m.group(2).translate(_QUOTES).rstrip(".").split(".")[-1]
        end = i + 1
        # why: a statement ends at its terminating semicolon; without one the unit
        # stops at the next CREATE so two objects never share a span.
        for j in range(i, len(lines)):
            if lines[j].rstrip().endswith(";"):
                end = j + 1
                break
            if j > i and _CREATE.match(lines[j]):
                end = j
                break
        else:
            end = len(lines)
        out.append({
            "path": rel,
            "fqname": f"{module}.{name}",
            "kind": f"sql_{object_kind.replace(' ', '_')}",
            "start": i + 1,
            "end": end,
        })
    return out
