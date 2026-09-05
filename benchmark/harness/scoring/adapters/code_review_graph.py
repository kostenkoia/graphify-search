"""Adapter for code-review-graph tool replies as written by the harness's MCP client."""

from __future__ import annotations

import json

from benchmark.harness.scoring.adapters import normalize_paths, symbol_of

# why: the vendor release whose output format this adapter parses; a `version.cli` in harness.yaml
# takes precedence, so this value scores only a cell that declares none
VERSION = "2.3.7"
_SILENT_TOOLS = {"get_minimal_context_tool", "list_graph_stats_tool", "list_tools"}
# inv: a status whitelist -- any status the vendor's query tool does not document is a shape this
# adapter was not written for
_KNOWN_STATUSES = frozenset({"ok", "ambiguous", "not_found"})


def _entry(kind: str, rank: int | None, item: dict) -> dict:
    name = str(item.get("name", ""))
    return {"kind": kind, "rank": rank, "path": item.get("file_path"), "label": name, "symbol": symbol_of(name),
            "qualified_name": item.get("qualified_name"), "start": item.get("line_start"), "end": item.get("line_end")}


def parse(call: dict, text: str, *, search_modes: list[str] | None = None,
          path_prefix: str | None = None) -> list[dict]:
    """Turn one reply object into records.

    Parameters
    ----------
    call : dict
        The journal entry; an `argv` call (`--version` etc.) produces nothing.
    text : str
        The reply body, either a JSON object or plain text for a system call.
    search_modes : list of str, optional
        The configuration's allowed `search_mode` values; a reply outside
        this set becomes `unparsed`.
    path_prefix : str, optional
        The index root this vendor's `file_path` values carry; stripped so
        records name corpus-relative paths, as the reference does.

    Returns
    -------
    list of dict
        Records for shapes this adapter recognizes; anything else becomes
        one `unparsed` record carrying the verbatim text.
    """
    if "argv" in call:
        return []
    tool = str(call.get("tool") or "")
    if tool in _SILENT_TOOLS:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [{"kind": "unparsed", "text": text}]
    if not isinstance(obj, dict):
        return [{"kind": "unparsed", "text": text}]
    if search_modes is not None and "search_mode" in obj and obj["search_mode"] not in search_modes:
        return [{"kind": "unparsed", "text": f"search_mode={obj['search_mode']!r} outside {search_modes}"}]
    status = obj.get("status")
    if status is not None and status not in _KNOWN_STATUSES:
        return [{"kind": "unparsed", "text": text}]
    records: list[dict] = []
    if status == "ambiguous":
        records.append({"kind": "no_results", "vendor_status": "ambiguous",
                        "vendor_message": str(obj.get("summary", ""))})
        records.extend(_entry("candidate", i, c) for i, c in enumerate(obj.get("candidates") or [], start=1))
        return normalize_paths(records, path_prefix)
    if status == "not_found":
        return [{"kind": "no_results", "vendor_status": "not_found", "vendor_message": str(obj.get("summary", ""))}]
    # inv: a result without line_start becomes kind: file, which score._is_hit never counts as a hit
    for i, r in enumerate(obj.get("results") or [], start=1):
        records.append(_entry("place" if r.get("line_start") is not None else "file", i, r))
    records.extend({**_entry("edge", None, e), "relation": e.get("relation")} for e in obj.get("edges") or [])
    return normalize_paths(records, path_prefix)
