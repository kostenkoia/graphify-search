"""Extract shell functions: one unit per function definition."""

from __future__ import annotations

import re

# why: both spellings a POSIX shell accepts -- `name() {` and `function name {`.
_FUNC = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w:.-]*)\s*\(\s*\)\s*\{|^\s*function\s+([A-Za-z_][\w:.-]*)\s*\{")


def extract(rel: str, text: str) -> list[dict]:
    """Return every function definition in one shell script.

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
    module = rel.rsplit(".", 1)[0].replace("/", ".")
    out: list[dict] = []
    for i, line in enumerate(lines):
        m = _FUNC.match(line)
        if not m:
            continue
        name = m.group(1) or m.group(2)
        # why: the body ends at the first closing brace in the same column as the
        # definition; nested blocks are indented deeper and are skipped by that test.
        indent = len(line) - len(line.lstrip())
        end = len(lines)
        for j in range(i + 1, len(lines)):
            stripped = lines[j].rstrip()
            if stripped == " " * indent + "}" or stripped == "}":
                end = j + 1
                break
        out.append({
            "path": rel,
            "fqname": f"{module}.{name}",
            "kind": "shell_function",
            "start": i + 1,
            "end": end,
        })
    return out
