"""Read the vendor's graph.json: nodes, kinds, docstrings and one-hop edges."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from graphify_search.errors import InputError

# adr-0001: `contains`, `method` and `defines` are containment, which a result's own path and start
# already point at, and `rationale_for` reaches the reader as the docstring folded into the node's
# text, so all four stay out; `indirect_call` and `re_exports` are dropped to keep the five edge
# slots for the relations an agent follows, and `extends` joins `inherits` because both name what a
# declaration builds on
EDGE_RELATIONS = ("calls", "extends", "imports", "imports_from", "inherits", "uses", "references")
# NOT DERIVED: bounds the edge block of one result
EDGES_PER_RESULT = 5
_START = re.compile(r"^L(\d+)$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Node:
    """One graph node with the fields the index needs."""

    id: str
    label: str
    file_type: str
    path: str
    start: int | None
    community: str


@dataclass(frozen=True)
class Edge:
    """One edge as the answer prints it: relation, other label, and the edge site."""

    rel: str
    to: str
    file: str
    loc: str


class Graph:
    """The parsed graph.json."""

    def __init__(self, sha256: str, nodes: list[Node], docstrings: dict[str, list[str]],
                 edges: dict[str, list[Edge]]) -> None:
        self.sha256 = sha256
        self.nodes = nodes
        self._docstrings = docstrings
        self._edges = edges

    def eligible(self) -> list[Node]:
        """Return the nodes that may appear as results, in graph order."""
        # adr-0001: only code and document nodes with a start line are places; a docstring node
        # lies inside the symbol it describes and a concept names an idea, not a place
        return [n for n in self.nodes if n.file_type in ("code", "document") and n.start is not None]

    def docstrings(self, node_id: str) -> list[str]:
        """Return the docstring labels attached to `node_id` by rationale_for edges."""
        return list(self._docstrings.get(node_id, ()))

    def edges_of(self, node_id: str) -> list[Edge]:
        """Return the first EDGES_PER_RESULT filtered edges of `node_id`, in links order."""
        return list(self._edges.get(node_id, ())[:EDGES_PER_RESULT])


def sha256_of_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbol_of(label: str) -> str | None:
    """Return the label as a symbol, with a vendor's `.` prefix and `()` suffix removed."""
    core = label[:-2] if label.endswith("()") else label
    core = core[1:] if core.startswith(".") else core
    return core if _IDENT.match(core) else None


def kind_of(node: Node) -> str:
    """Return `document`, `file` or `symbol` for a node."""
    # adr-0001: the order matters; a file node's label ends with its basename and starts at line 1,
    # and a label at line 1 that spells no identifier names a whole file the vendor titled in prose,
    # since every declaration the vendor records carries its own identifier
    if node.file_type == "document":
        return "document"
    if node.start == 1 and (node.label.endswith(os.path.basename(node.path))
                            or symbol_of(node.label) is None):
        return "file"
    return "symbol"


def _start(loc: str | None) -> int | None:
    m = _START.match(loc or "")
    return int(m.group(1)) if m else None


def _community(n: dict[str, object]) -> str:
    # inv: community 0 is a real id, not an absent one; `or` would coerce it to the fallback
    name = n.get("community_name")
    if name is not None and name != "":
        return str(name)
    community = n.get("community")
    return str(community) if community is not None else ""


def _malformed(path: Path, what: str) -> InputError:
    return InputError(f"{path} is not a graphify graph: {what}", hint="run `graphify <path>` again")


def _node_of(path: Path, n: object) -> Node:
    if not isinstance(n, dict):
        raise _malformed(path, "a node is not an object")
    if "id" not in n:
        raise _malformed(path, "a node without id")
    loc = n.get("source_location")
    if loc is not None and not isinstance(loc, str):
        raise _malformed(path, f"node {n['id']}'s source_location is not a string")
    return Node(id=str(n["id"]), label=str(n.get("label") or ""), file_type=str(n.get("file_type") or ""),
                path=str(n.get("source_file") or ""), start=_start(loc), community=_community(n))


def load_graph(path: Path) -> Graph:
    """Parse graph.json into a Graph.

    Parameters
    ----------
    path : Path
        The vendor's graph.json.

    Returns
    -------
    Graph
        Nodes in file order, docstrings and filtered edges.

    Raises
    ------
    InputError
        When the file is missing, is not a node-link graph, holds a node or link of the wrong
        shape, or repeats a node id.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InputError(f"cannot read graph at {path}: {e}", hint="run `graphify <path>` first") from e
    if not isinstance(raw, dict) or "nodes" not in raw or "links" not in raw:
        raise InputError(f"{path} is not a graphify graph", hint="expected keys nodes and links")
    if not isinstance(raw["nodes"], list):
        raise _malformed(path, "nodes is not a list")
    if not isinstance(raw["links"], list):
        raise _malformed(path, "links is not a list")
    nodes = [_node_of(path, n) for n in raw["nodes"]]
    by_id: dict[str, Node] = {}
    for n in nodes:
        # inv: the index keys its bm25 corpus by node id, so two rows under one id would drop a
        # document the row list still carries and no rebuild could repair the disagreement
        if n.id in by_id:
            raise _malformed(path, f"duplicate node id {n.id}")
        by_id[n.id] = n
    docstrings: dict[str, list[str]] = defaultdict(list)
    edges: dict[str, list[Edge]] = defaultdict(list)
    for link in raw["links"]:
        if not isinstance(link, dict):
            raise _malformed(path, "a link is not an object")
        s, t, rel = str(link.get("source")), str(link.get("target")), str(link.get("relation") or "")
        if s not in by_id or t not in by_id:
            continue
        if rel == "rationale_for":
            # inv: the docstring node is whichever endpoint the vendor typed as rationale
            if by_id[s].file_type == "rationale":
                docstrings[t].append(by_id[s].label)
            elif by_id[t].file_type == "rationale":
                docstrings[s].append(by_id[t].label)
            continue
        if rel in EDGE_RELATIONS:
            site_file, site_loc = str(link.get("source_file") or ""), str(link.get("source_location") or "")
            edges[s].append(Edge(rel, by_id[t].label, site_file, site_loc))
            # inv: a self-loop's source and target are the same node; appending twice would
            # double-count the edge in edges_of
            if t != s:
                edges[t].append(Edge(rel, by_id[s].label, site_file, site_loc))
    return Graph(sha256_of_file(path), nodes, dict(docstrings), dict(edges))
