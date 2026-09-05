"""Adapter for graphify's command-line output."""

from __future__ import annotations

import re

from benchmark.harness.scoring.adapters import normalize_paths, symbol_of

# why: the vendor release whose output format this adapter parses; a `version.cli` in harness.yaml
# takes precedence, so this value scores only a cell that declares none
VERSION = "0.9.27"

# inv: `label` is greedy and anchored at end-of-line, so it backtracks to the vendor's real
# trailing "[src=...]" group; a label that itself contains "[src=" cannot masquerade as it
_NODE = re.compile(
    r"^NODE (?P<label>.*) \[src=(?P<path>[^\s\]]*) loc=L?(?P<line>\d*) community=(?P<comm>[^\]]*)\]$")
_SOURCE = re.compile(r"^\s+Source:\s+(?P<path>\S+) L(?P<line>\d+)$")
_EDGE = re.compile(
    r"^\s+(?:<--|-->) (?P<label>.+?) \[(?P<rel>[^\]]+)\] \[(?P<conf>[^\]]+)\] (?P<path>\S+):L(?P<line>\d+)$")
# inv: a target label ending in "at=<path>:L<n>" is indistinguishable from a real location suffix,
# so a matched edge's path and start are a guess, never evidence
_EDGE_QUERY = re.compile(
    r"^EDGE (?P<src>.+) --(?P<rel>.+?) \[(?P<conf>[^\]]*)\]--> (?P<tgt>.+?)"
    r"(?: at=(?P<path>[^\s=]+):L(?P<line>\d+))?$")
# inv: a hop line carries no location, so the edge it yields claims none; the vendor prints the
# two labels and the relation and nothing else
# inv: the vendor prints a hop in whichever direction the edge runs; the label is the node it
# names last in both forms, as an explain connection line is read
_EDGE_PATH = re.compile(
    r"^\s+(?P<src>.+?) (?:--(?P<rel>.+?) \[(?P<conf>[^\]]*)\]-->"
    r"|<--(?P<back>.+?) \[(?P<bconf>[^\]]*)\]--) (?P<tgt>.+)$")
# inv: a grouped line names a file and a count, never a line, so the record it yields carries no
# start and can never score a hit; the direction and the count stay in the `.out`
_GROUPED = re.compile(r"^\s+(?:<--|-->) (?P<path>\S+): \d+ connections?$")
# inv: the vendor prints an empty Source for a node it cannot point at; that names no place, and
# the emptiness is matched exactly, so a Source carrying anything unexpected stays unparsed
_NO_SOURCE = re.compile(r"^\s+Source:\s*$")
_NO_NODE = re.compile(r"^No node matching '(?P<arg>.*)' found\.$", re.DOTALL)
_NO_PATH = re.compile(r"^No path found between '.*' and '.*'\.$", re.DOTALL)
_NO_MATCH = re.compile(r"^No matching nodes found\.$")
# inv: a warning names neither a place nor an edge, so it is no record; the vendor's own words
# stay in the `.out`, which the audit hashes, so nothing is lost from the evidence by skipping it
# inv: "... and N more" is the vendor saying it cut a list -- the connections, or the files it
# groups them by; it names neither a place nor an edge, so it is no record, and the cut count
# stays in the `.out`, which the audit hashes
_MORE = re.compile(r"^\s+\.\.\. and \d+ more(?: files)?$")
_SKIP = ("Traversal: ", "[!] TRUNCATED", "... (truncated", "  ID:", "  Type:", "  Community:", "  Degree:",
         "Connections (", "Shortest path (", "warning: ", "  Grouped by file:")
# inv: both messages quote a caller-supplied argument verbatim (cli.py); a line carrying this
# prefix outside a whole-text match is a shape this adapter was not written for, never a place
_MESSAGE_PREFIXES = ("No node matching '", "No path found between '")


def _place(rank: int, label: str, path: str, line: str) -> dict:
    return {"kind": "place", "rank": rank, "path": path or None, "label": label, "symbol": symbol_of(label),
            "qualified_name": None, "start": int(line) if line else None, "end": None}


def parse(call: dict, text: str, *, search_modes: list[str] | None = None,
          path_prefix: str | None = None) -> list[dict]:
    """Turn one graphify `.out` into records.

    Parameters
    ----------
    call : dict
        The journal entry; `argv[1]` selects the subcommand's shape.
    text : str
        The `.out` text.
    search_modes : list of str, optional
        Not used by this adapter.
    path_prefix : str, optional
        The index root this vendor's paths carry; this index stores them
        already corpus-relative, so `build.yaml` leaves it unset.

    Returns
    -------
    list of dict
        Records for shapes this adapter recognizes; unrecognized non-blank
        lines become `unparsed` records carrying the verbatim text.
    """
    del search_modes  # why: kept only so both adapters share one call signature
    argv = list(call.get("argv") or [])
    sub = argv[1] if len(argv) > 1 else ""
    if sub in ("--version", "--help"):
        return []
    # why: CRLF and LF must be judged identically, or the whole-text guard below and the
    # per-line pass disagree on where a line ends and a crafted "\r"-terminated message escapes it
    norm = text.replace("\r\n", "\n")
    whole = norm.rstrip("\n")
    if _NO_NODE.match(whole):
        return [{"kind": "no_results", "vendor_status": "no_node", "vendor_message": whole}]
    if _NO_PATH.match(whole):
        return [{"kind": "no_results", "vendor_status": "no_path", "vendor_message": whole}]
    if _NO_MATCH.match(whole):
        return [{"kind": "no_results", "vendor_status": "no_match", "vendor_message": whole}]
    if any(ln.startswith(_MESSAGE_PREFIXES) for ln in norm.split("\n")):
        return [{"kind": "unparsed", "text": text}]
    records: list[dict] = []
    rank = 0
    # inv: split on "\n" only; other separators stay inside the line and fail the anchored regexes
    lines = norm.split("\n")
    node_label = lines[0][len("Node: "):] if lines and lines[0].startswith("Node: ") else ""
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if sub == "explain" and i == 0 and line.startswith("Node: "):
            continue
        if sub == "query" and (m := _NODE.match(line)):
            rank += 1
            records.append(_place(rank, m["label"], m["path"], m["line"]))
            continue
        if sub == "query" and (m := _EDGE_QUERY.match(line)):
            records.append({"kind": "edge", "rank": None, "path": m["path"], "label": m["tgt"],
                            "symbol": symbol_of(m["tgt"]), "qualified_name": None,
                            "start": int(m["line"]) if m["line"] else None, "end": None, "relation": m["rel"]})
            continue
        if sub == "explain" and (m := _GROUPED.match(line)):
            records.append({"kind": "file", "rank": None, "path": m["path"], "label": m["path"],
                            "symbol": None, "qualified_name": None, "start": None, "end": None})
            continue
        if sub == "explain" and _MORE.match(line):
            continue
        if sub == "path" and (m := _EDGE_PATH.match(line)):
            records.append({"kind": "edge", "rank": None, "path": None, "label": m["tgt"],
                            "symbol": symbol_of(m["tgt"]), "qualified_name": None,
                            "start": None, "end": None, "relation": m["rel"] or m["back"]})
            continue
        if sub == "explain" and _NO_SOURCE.match(line):
            continue
        if sub == "explain" and (m := _SOURCE.match(line)):
            rank += 1
            records.append(_place(rank, node_label, m["path"], m["line"]))
            continue
        if sub == "explain" and (m := _EDGE.match(line)):
            records.append({"kind": "edge", "rank": None, "path": m["path"], "label": m["label"],
                            "symbol": symbol_of(m["label"]), "qualified_name": None,
                            "start": int(m["line"]), "end": None, "relation": m["rel"]})
            continue
        if line.startswith(_SKIP):
            continue
        records.append({"kind": "unparsed", "text": line})
    return normalize_paths(records, path_prefix)
