"""Adapter for graphify-search's JSON output."""

from __future__ import annotations

import json

from benchmark.harness.scoring.adapters import normalize_paths

# inv: score.records holds a version step's output to this string only for a cell whose harness.yaml
# declares no `version.cli`; every shipped cell declares one, so none of them reaches this value
# why: it names the wheel this checkout builds, so a harness.yaml written without `version.cli` is
# scored against the version of the tree it lives in, never against the interpreter's installed
# package, which is the harness's own environment and not the measured cell's
VERSION = "0.6.0"


def parse(call: dict, text: str, *, search_modes: list[str] | None = None,
          path_prefix: str | None = None) -> list[dict]:
    """Turn one graphify-search `.out` into records.

    Parameters
    ----------
    call : dict
        The journal entry; `argv[1]` selects the subcommand.
    text : str
        The `.out` text.
    search_modes : list of str, optional
        The modes the configuration declares; an answer naming another one is refused.
    path_prefix : str, optional
        Not used; the tool prints corpus-relative paths.

    Returns
    -------
    list of dict
        One `place` per result; the whole text as one `unparsed` record when it is not the tool's
        JSON, when a result row lacks a field a place needs, or when `mode` is outside
        `search_modes`.
    """
    argv = list(call.get("argv") or [])
    if len(argv) < 2 or argv[1] != "query":
        return []
    try:
        doc = json.loads(text)
        # inv: a dense configuration whose answer fell back to bm25 measured a different
        # retrieval than the one on record, so its places are refused rather than scored
        if search_modes is not None and doc["mode"] not in search_modes:
            return [{"kind": "unparsed", "text": text}]
        # inv: a row missing rank, path or start names no place, and one bad row makes the whole
        # answer a shape this adapter was not written for, never a shorter list of places
        records = [{"kind": "place", "rank": int(r["rank"]), "path": r["path"],
                    "label": r.get("symbol") or r["path"], "symbol": r.get("symbol"),
                    "qualified_name": None, "start": int(r["start"]), "end": None}
                   for r in doc["results"]]
    except (ValueError, KeyError, TypeError):
        return [{"kind": "unparsed", "text": text}]
    return normalize_paths(records, path_prefix)
