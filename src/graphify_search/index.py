"""Build, refresh and load the index beside graph.json."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from graphify_search import graph as g
from graphify_search import text as t
from graphify_search.bm25 import BM25
from graphify_search.errors import EndpointUnavailableError, InputError
from graphify_search.utils import atomic_write_bytes, atomic_write_text

if TYPE_CHECKING:
    from graphify_search.embed import Embedder

INDEX_DIR = ".graphify_search"
LOCK_NAME = "lock"
# NOT DERIVED: bounds the id list one record prints, which a moved repo would otherwise fill with
# every id in the graph
DROPPED_IN_RECORD = 20


def index_dir(graph_path: Path) -> Path:
    """Return the index directory beside a graph.json."""
    return Path(graph_path).parent / INDEX_DIR


def _package_version() -> str:
    try:
        return version("graphify-search")
    except PackageNotFoundError:
        return "0"


@dataclass
class Row:
    """One nodes.jsonl line."""

    id: str
    kind: str
    path: str
    symbol: str | None
    start: int
    community: str
    text_sha256: str
    snippet: str
    edges: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Manifest:
    """What the index was built from and with."""

    package_version: str
    graph_sha256: str
    model: str | None
    dims: int
    doc_prefix: str
    query_prefix: str
    endpoint: str | None
    files: dict[str, str]
    rows: int
    vectors: str

    def to_json(self) -> str:
        """Serialise the manifest."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Parse a manifest written by `to_json`."""
        return cls(**json.loads(text))


@dataclass
class Index:
    """The loaded index."""

    manifest: Manifest
    rows: list[Row]
    vectors: np.ndarray | None
    bm25: BM25


class _Source:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._resolved = self.root.resolve()
        self._lines: dict[str, list[str]] = {}
        self._sha: dict[str, str] = {}
        self._read: set[str] = set()

    def _inside(self, rel: str) -> Path | None:
        # inv: graph.json is read, not written, by this package, so a `source_file` that escapes
        # the source root reads a file the operator never offered and must yield nothing
        p = (self.root / rel).resolve()
        return p if p.is_relative_to(self._resolved) else None

    def lines(self, rel: str) -> list[str]:
        if rel not in self._lines:
            p = self._inside(rel)
            try:
                text = p.read_text(encoding="utf-8", errors="replace") if p else None
            except OSError:
                text = None
            # inv: a relative path resolves inside any root, so only a read that returns text proves
            # the file the graph names is the one this root holds
            if text is not None:
                self._read.add(rel)
            self._lines[rel] = text.splitlines() if text is not None else []
        return self._lines[rel]

    def read_count(self) -> int:
        return len(self._read)

    def is_empty(self, rel: str) -> bool:
        # inv: a read answers an empty string for a file of zero bytes alone, so a file that failed
        # to read is never called empty here
        return rel in self._read and not self._lines[rel]

    def sha(self, rel: str) -> str:
        if rel not in self._sha:
            p = self._inside(rel)
            try:
                self._sha[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p else ""
            except OSError:
                self._sha[rel] = ""
        return self._sha[rel]


def _edge_at(row_path: str, e: g.Edge) -> str:
    # why: the site is in the symbol's own file for most edges; the path is spelled only when it differs
    return e.loc if e.file == row_path else f"{e.file}:{e.loc}"


def _rows_and_texts(graph: g.Graph, src: _Source) -> tuple[list[Row], list[str]]:
    rows: list[Row] = []
    texts: list[str] = []
    for n in graph.eligible():
        start = n.start
        # inv: Graph.eligible() already filters to nodes with a start line
        if start is None:
            continue
        kind = g.kind_of(n)
        body = t.body_from(src.lines(n.path), start) if n.file_type == "code" else ""
        # inv: a file of zero bytes holds no snippet to read and no line to open, so its row would
        # rank an empty result above the code a reader asked for
        if kind == "file" and src.is_empty(n.path):
            continue
        docstrings = [d for d in graph.docstrings(n.id) if d.strip()]
        text = t.node_text(n.label, n.path, docstrings, body)
        # inv: a document's label is a heading, which is prose the reader must not read as the name
        # of a symbol even when it happens to spell one
        symbol = None if kind == "document" else g.symbol_of(n.label)
        rows.append(Row(
            id=n.id, kind=kind, path=n.path, symbol=symbol, start=start,
            community=n.community, text_sha256=t.text_sha256(text), snippet=t.snippet_of(body),
            edges=[{"rel": e.rel, "to": e.to, "at": _edge_at(n.path, e)} for e in graph.edges_of(n.id)],
        ))
        texts.append(text)
    return rows, texts


def _previous(graph_path: Path) -> tuple[Manifest, dict[str, tuple[Row, np.ndarray | None]]] | None:
    try:
        old = load_index(graph_path)
    except InputError:
        return None
    vec = old.vectors
    return old.manifest, {r.id: (r, vec[i] if vec is not None else None) for i, r in enumerate(old.rows)}


def _live_owner(lock: Path) -> int | None:
    try:
        pid = int(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        # inv: a lock that names no readable number names no builder to wait for, so it is stale
        return None
    # inv: pid 0 and the negative pids address process groups, which os.kill would signal, so a
    # lock naming one of them is stale rather than a question to ask the kernel
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except OSError:
        # inv: a refusal other than "no such process", such as a pid this user may not signal,
        # leaves the builder alive and the lock its own
        return pid
    return pid


def _held(lock: Path, pid: int | None) -> InputError:
    owner = f"pid {pid}" if pid is not None else "an unreadable pid"
    return InputError(f"another `graphify-search index` is running on this graph ({owner})",
                      hint=f"wait for it to finish, or delete {lock} if no such process exists")


def _open_lock(lock: Path) -> int:
    # why: O_EXCL is the create-or-fail both macOS and Linux answer for on a local filesystem,
    # and it leaves a name a person can inspect and delete after a crash
    return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)


def _acquire_lock(out: Path) -> Path:
    # inv: one builder at a time owns an index directory; a second one would read a `prev` the
    # first is still replacing and could unlink the vectors the first has just written
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK_NAME
    try:
        fd = _open_lock(lock)
    except OSError as first:
        owner = _live_owner(lock)
        if owner is not None:
            raise _held(lock, owner) from first
        # inv: a builder killed before its `finally` leaves a lock naming a process that is gone,
        # which is taken over here so no later build waits on a human to delete the file
        lock.unlink(missing_ok=True)
        try:
            fd = _open_lock(lock)
        except OSError as second:
            raise _held(lock, _live_owner(lock)) from second
    # inv: the lock names the process that holds it, so a later build can tell a running builder
    # from the file a killed one left behind
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return lock


def build_index(graph_path: Path, source_root: Path, client: Embedder | None, *,
                 full: bool = False, require_dense: bool = False) -> dict:
    """Write or refresh the index beside `graph_path`.

    Parameters
    ----------
    graph_path : Path
        The vendor's graph.json.
    source_root : Path
        Directory the graph's `source_file` paths are relative to.
    client : Embedder or None
        Endpoint to embed with; None writes the BM25 part only.
    full : bool
        Ignore the previous index and embed everything.
    require_dense : bool
        Fail instead of writing an index without vectors.

    Returns
    -------
    dict
        Counts of rows, source files read, embedded and reused rows, the first DROPPED_IN_RECORD
        dropped ids with `dropped_total`, the `vectors` state and seconds; and `embedding_error`,
        the endpoint's refusal, when the index was written without vectors after the endpoint was
        asked.

    Raises
    ------
    InputError
        When the graph holds no indexable node, when it names code files and none of them read
        under `source_root`, or when another build already holds the lock.
    EndpointUnavailableError
        When the endpoint does not answer and either `require_dense` is set or the previous index
        holds vectors.
    """
    t0 = time.monotonic()
    graph = g.load_graph(graph_path)
    src = _Source(source_root)
    rows, texts = _rows_and_texts(graph, src)
    if not rows:
        raise InputError("graph has no places to index",
                         hint="run `graphify <path>` on the source tree")
    files_read = src.read_count()
    # inv: a document is indexed by its label and path alone, so a graph of documents needs no
    # source file, while a graph naming code files none of which read names the wrong root
    if not files_read and any(n.file_type == "code" for n in graph.eligible()):
        raise InputError(f"no source file was found under {Path(source_root)}",
                         hint="pass --source pointing at the tree the graph's paths are relative to")
    out = index_dir(graph_path)
    lock = _acquire_lock(out)
    try:
        # inv: `prev` is read even under `full`, so a dropped id is still reported from the row the
        # previous index holds for it; only reuse of its vectors is gated on `full`
        prev = _previous(graph_path)
        # inv: load_index already refuses a manifest whose recorded width disagrees with its own
        # vectors, so `prev`, once returned, is always internally width-consistent
        # inv: two servers can answer one model name with different weights, so the endpoint is part
        # of what a cached vector must match before it may be reused
        same_backend = bool(not full and prev and client and prev[0].model == client.model
                           and prev[0].endpoint == client.endpoint
                           and prev[0].doc_prefix == client.doc_prefix
                           and prev[0].query_prefix == client.query_prefix and prev[0].vectors == "present")
        reuse = prev[1] if (prev and same_backend) else {}
        old_files = prev[0].files if prev else {}
        files = {p: src.sha(p) for p in sorted({r.path for r in rows})}
        vectors: list[np.ndarray | None] = []
        to_embed: list[int] = []
        for i, r in enumerate(rows):
            kept = reuse.get(r.id)
            # inv: a row is reused only when its text hash and its file hash both match the previous index
            if (kept and kept[1] is not None and kept[0].text_sha256 == r.text_sha256
                    and old_files.get(r.path) == files[r.path]):
                vectors.append(kept[1])
            else:
                vectors.append(None)
                to_embed.append(i)
        if client is not None and not to_embed and vectors and vectors[0] is not None:
            # inv: a refresh that would embed nothing still samples row 0's live width, so a server
            # answering a new width is caught here instead of failing the stack below on a real edit
            old_vector = vectors[0]
            try:
                probe = client.embed_documents([texts[0]])
                if probe.shape[1] != old_vector.shape[0]:
                    to_embed = list(range(len(rows)))
                    vectors = [None] * len(rows)
            except EndpointUnavailableError:
                # why: a probe that cannot reach the server proves nothing about the served width;
                # the cached vectors stand and the refresh proceeds as if nothing needed re-embedding
                pass
        state, dims = "absent", 0
        embedding_error: str | None = None
        if client is not None:
            try:
                fresh = client.embed_documents([texts[i] for i in to_embed])
                reused_widths = {v.shape[0] for v in vectors if v is not None}
                if fresh.shape[0] and reused_widths and fresh.shape[1] not in reused_widths:
                    # inv: a cached vector and a freshly embedded one must share one width to stack;
                    # a live server answering a new width forces the whole corpus through this call
                    to_embed = list(range(len(rows)))
                    vectors = [None] * len(rows)
                    fresh = client.embed_documents(texts)
                for k, i in enumerate(to_embed):
                    vectors[i] = fresh[k]
                state = "present"
            except EndpointUnavailableError as e:
                if require_dense:
                    raise
                # why: an index written without vectors discards work an outage cannot give back, so a
                # dense index that already exists outlives the outage and the refresh fails instead
                if prev is not None and prev[0].vectors == "present":
                    raise EndpointUnavailableError(str(e), hint="the previous index is kept; retry when the "
                                              "endpoint answers") from e
                # inv: an index written without vectors after the endpoint was asked names the
                # refusal in its record, so a silent bm25 index is never mistaken for a chosen one
                state, embedding_error = "absent", str(e)
        if state == "present":
            mat = np.stack([v for v in vectors if v is not None]).astype(np.float32)
            dims = int(mat.shape[1]) if mat.size else 0
            buf = io.BytesIO()
            np.save(buf, mat)
            atomic_write_bytes(out / "vectors.npy", buf.getvalue())
        else:
            (out / "vectors.npy").unlink(missing_ok=True)
        atomic_write_text(out / "nodes.jsonl",
                           "".join(json.dumps(asdict(r), ensure_ascii=False, separators=(",", ":")) + "\n"
                                   for r in rows))
        atomic_write_text(out / "bm25.json",
                           BM25.build({r.id: t.tokens(texts[i]) for i, r in enumerate(rows)}).to_json())
        manifest = Manifest(package_version=_package_version(), graph_sha256=graph.sha256,
                             model=client.model if client else None, dims=dims,
                             doc_prefix=client.doc_prefix if client else "",
                             query_prefix=client.query_prefix if client else "",
                             endpoint=client.endpoint if client else None, files=files, rows=len(rows), vectors=state)
        atomic_write_text(out / "manifest.json", manifest.to_json())
        # why: a dropped id is read off the previous index's own rows, not off `reuse`, since a model
        # change or `--full` empties `reuse` without meaning every previous row survived
        dropped = sorted(set(prev[1]) - {r.id for r in rows}) if prev else []
        embedded = len(to_embed) if state == "present" else 0
        record = {"graph_sha256": graph.sha256, "nodes": len(rows), "files_read": files_read,
                  "embedded": embedded,
                  "reused": len(rows) - len(to_embed) if state == "present" else 0,
                  "dropped": dropped[:DROPPED_IN_RECORD], "dropped_total": len(dropped),
                  "vectors": state, "seconds": round(time.monotonic() - t0, 3)}
        if embedding_error is not None:
            record["embedding_error"] = embedding_error
        return record
    finally:
        lock.unlink(missing_ok=True)


def load_index(graph_path: Path) -> Index:
    """Read the index beside `graph_path`.

    Raises
    ------
    InputError
        When the index is missing or its parts disagree.
    """
    d = index_dir(graph_path)
    try:
        manifest = Manifest.from_json((d / "manifest.json").read_text(encoding="utf-8"))
        rows = [Row(**json.loads(line))
                for line in (d / "nodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
        bm25 = BM25.from_json((d / "bm25.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise InputError(f"no usable index at {d}: {e}", hint="run `graphify-search index`") from e
    vectors = None
    if manifest.vectors == "present":
        try:
            vectors = np.load(d / "vectors.npy")
        except (OSError, ValueError) as e:
            raise InputError(f"vectors unreadable at {d}: {e}", hint="run `graphify-search index --full`") from e
    if (len(rows) != manifest.rows or len(bm25.ids) != len(rows)
            or (vectors is not None and vectors.shape[0] != len(rows))
            or (vectors is not None and (vectors.ndim != 2 or vectors.shape[1] != manifest.dims))):
        raise InputError(f"index parts disagree at {d}", hint="run `graphify-search index --full`")
    return Index(manifest, rows, vectors, bm25)
