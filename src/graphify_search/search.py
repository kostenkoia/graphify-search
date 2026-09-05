"""Rank the index against a question and assemble the answer."""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

from graphify_search import schema
from graphify_search import text as t
from graphify_search.errors import EndpointUnavailableError, InputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from graphify_search.embed import Embedder
    from graphify_search.index import Index, Row


_REBUILD = "run `graphify-search index --full`"


def _cosines(vectors: np.ndarray, text: str, client: Embedder) -> np.ndarray:
    q = client.embed_query(text)
    # inv: a width the index was not built at cannot be ranked; the refusal replaces the matmul's
    # own exception, which no caller of this command catches
    if q.shape[0] != vectors.shape[1]:
        raise InputError(f"index vectors are {vectors.shape[1]} wide, the endpoint answered {q.shape[0]}",
                         hint=_REBUILD)
    return vectors @ q


def _dense(vectors: np.ndarray, rows: list[Row], question: str, client: Embedder) -> list[tuple[int, float]]:
    scores = _cosines(vectors, question, client)
    # why: rounding here, not only in `_result`, keeps the sort order the caller sees consistent
    # with the order a caller re-sorting on the displayed `score` would recompute
    return [(i, round(float(scores[i]), 3)) for i in range(len(rows))]


def _bm25(idx: Index, question: str) -> list[tuple[int, float]]:
    pos = {r.id: i for i, r in enumerate(idx.rows)}
    words = t.tokens(question)
    # inv: bm25 matches words, so a question without one has nothing to match and is refused as
    # itself rather than answered with an empty array
    if not words:
        raise InputError("no word in the question to match (mode bm25)",
                         hint="ask with words; symbols and emoji alone match nothing")
    ranked = idx.bm25.rank(words)
    # inv: a question that has words and matches none of the index's is named as a vocabulary miss,
    # which an empty results array does not tell the caller apart from a question nothing answers
    if not ranked:
        raise InputError("no word of the question occurs in the indexed code (mode bm25)",
                         hint="start the embedding server and re-run `graphify-search index`, "
                              "or ask with the words the code uses")
    # why: a zero score means no term matched, so that row is not a match and never becomes a place
    return [(pos[i], round(s, 3)) for i, s in ranked if i in pos]


def _excluded(path: str, globs: Sequence[str]) -> bool:
    # inv: fnmatch's `*` crosses `/`, so `tests/*` names every path under `tests`, at any depth,
    # and not its top level alone
    return any(fnmatch.fnmatch(path, g) for g in globs)


def _result(rank: int, row: Row, score: float, snippets: bool, edges: bool) -> schema.Result:
    return schema.Result(rank=rank, kind=row.kind, path=row.path, symbol=row.symbol, start=row.start,
                          community=row.community, score=score,
                          snippet=row.snippet if snippets else None,
                          edges=[schema.EdgeOut(**e) for e in row.edges] if edges else None)


def query(idx: Index, question: str, client: Embedder | None, *, k: int, budget: int,
          require_dense: bool, snippets: bool, edges: bool, graph_sha256_now: str,
          exclude: Sequence[str] = ()) -> schema.Answer:
    """Answer one question from the loaded index.

    Parameters
    ----------
    idx : Index
        The loaded index.
    question : str
        The question as the user wrote it.
    client : Embedder or None
        Endpoint to embed the question with; None means BM25.
    k : int
        Places to return, clipped to the number of scored rows.
    budget : int
        Character budget for the results array.
    require_dense : bool
        Refuse to fall back to BM25.
    snippets, edges : bool
        Whether the optional fields are rendered.
    graph_sha256_now : str
        Hash of graph.json as it is on disk, for the staleness flag.
    exclude : sequence of str
        Globs matched against each result's path with `fnmatch`; a row matching any of them is
        dropped before `k` and before the budget ladder.

    Returns
    -------
    schema.Answer
        The answer after the budget ladder.

    Raises
    ------
    InputError
        When the question is blank, or the index and the client disagree on the model or on the
        vector width.
    EndpointUnavailableError
        Under `require_dense` when dense ranking is impossible.
    """
    if not question.strip():
        raise InputError("the question is empty", hint="pass the question in quotes")
    # inv: vectors of one model are meaningless against another model's question, and the
    # disagreement is silent whenever the two models share a width
    if client is not None and idx.manifest.model is not None and idx.manifest.model != client.model:
        raise InputError(f"index was built with {idx.manifest.model}, query asks for {client.model}", hint=_REBUILD)
    mode, vectors_state, model = "bm25", idx.manifest.vectors, None
    scored: list[tuple[int, float]] = []
    if idx.vectors is not None and client is not None:
        try:
            scored, mode, model = _dense(idx.vectors, idx.rows, question, client), "dense", client.model
        except EndpointUnavailableError:
            if require_dense:
                raise
            vectors_state = "unreachable"
    elif require_dense:
        if idx.vectors is None:
            raise EndpointUnavailableError("no vectors in the index",
                                      hint="run `graphify-search index` with a live endpoint")
        raise EndpointUnavailableError("no embedding endpoint configured for this query",
                                  hint="pass --endpoint or set GRAPHIFY_SEARCH_ENDPOINT")
    if mode == "bm25":
        scored = _bm25(idx, question)
    # inv: ties fall to the path, then the start line, so two runs print the same order
    scored.sort(key=lambda s: (-s[1], idx.rows[s[0]].path, idx.rows[s[0]].start))
    # inv: the exclusion runs before the `k` cut and before the ladder, so a path the caller ruled
    # out neither takes a place from a row that survives nor spends a character of the budget
    scored = [s for s in scored if not _excluded(idx.rows[s[0]].path, exclude)]
    top = scored[:max(0, k)]
    results = [_result(rank, idx.rows[i], s, snippets, edges) for rank, (i, s) in enumerate(top, 1)]
    # inv: the endpoint names where the question was sent, so it is reported whenever a client
    # exists, including a dense attempt that failed and answered from bm25 instead
    answer = schema.Answer(question=question, mode=mode, model=model,
                            endpoint=client.endpoint if client is not None else None,
                            index=schema.IndexState(idx.manifest.graph_sha256,
                                                     idx.manifest.graph_sha256 != graph_sha256_now,
                                                     vectors_state),
                            budget=schema.Budget(limit_chars=budget, used_chars=0), results=results)
    answer = schema.apply_budget(answer, budget)
    # inv: a field the caller switched off is reported the same way the ladder reports one it removed
    pre = [name for name, on in (("snippet", snippets), ("edges", edges)) if not on]
    # inv: `dropped` names only what the ladder removed or what a flag switched off, so a row kind
    # that carries no snippet of its own -- a `document` row -- leaves the list empty
    answer.budget.dropped = pre + [d for d in answer.budget.dropped if d not in pre]
    return answer
