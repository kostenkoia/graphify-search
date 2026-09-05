"""The answer's shape, its compact rendering and the budget ladder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

# NOT DERIVED: the smallest list the ladder shrinks to
K_FLOOR = 5
_OPTIONAL = ("snippet", "edges")


@dataclass
class EdgeOut:
    """One edge line of a result."""

    rel: str
    to: str
    at: str


@dataclass
class Result:
    """One place in the answer."""

    rank: int
    kind: str
    path: str
    symbol: str | None
    start: int
    community: str
    score: float
    snippet: str | None = None
    edges: list[EdgeOut] | None = None


@dataclass
class IndexState:
    """What the reader must know about the index that answered."""

    graph_sha256: str
    stale: bool
    vectors: str


@dataclass
class Budget:
    """The character budget and what the ladder removed to meet it."""

    limit_chars: int
    used_chars: int
    dropped: list[str] = field(default_factory=list)
    exceeded: bool = False


@dataclass
class Answer:
    """One query's whole output."""

    question: str
    mode: str
    model: str | None
    # inv: the field is keyword-only, so it renders next to `model` without displacing the
    # positional fields declared after it
    endpoint: str | None = field(default=None, kw_only=True)
    index: IndexState
    budget: Budget
    results: list[Result]


def _plain(obj: object) -> object:
    if isinstance(obj, list):
        return [_plain(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        # inv: fields are walked one by one, not through asdict, so a nested Result keeps the rule below
        d = {name: _plain(getattr(obj, name)) for name in obj.__dataclass_fields__}
        if isinstance(obj, Result):
            # inv: a field the ladder removed is absent, never null, so its cost is really gone
            d = {k: v for k, v in d.items() if not (k in _OPTIONAL and v is None)}
        return d
    return obj


def dumps(obj: object) -> str:
    """Render any answer part compactly, keys in dataclass order.

    Raises
    ------
    ValueError
        When a number in the answer is not finite.
    """
    # inv: `NaN` and `Infinity` are not JSON, so a non-finite number stops the render instead of
    # reaching a caller whose parser refuses the whole answer
    return json.dumps(_plain(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def results_chars(results: list[Result]) -> int:
    """Return the rendered length of the results array alone."""
    return len(dumps(results))


def render(answer: Answer) -> str:
    """Render the answer, writing `used_chars` into its budget, plus one newline."""
    # adr-0002: used_chars measures the results array, so the figure never depends on itself
    answer.budget.used_chars = results_chars(answer.results)
    return dumps(answer) + "\n"


def apply_budget(answer: Answer, limit: int) -> Answer:
    """Shrink the answer until its results fit `limit` characters.

    Parameters
    ----------
    answer : Answer
        The full answer.
    limit : int
        Character budget for the results array.

    Returns
    -------
    Answer
        A copy of the answer with `snippet`, then `edges`, then results beyond K_FLOOR removed, in
        that order, stopping at the first shape that fits; `exceeded` when none does.
    """
    out = replace(answer, budget=replace(answer.budget, limit_chars=limit, dropped=[], exceeded=False))
    out.results = [replace(r) for r in answer.results]

    def fits() -> bool:
        out.budget.used_chars = results_chars(out.results)
        return out.budget.used_chars <= limit

    if fits():
        return out
    # adr-0002: the ladder removes the most expensive optional field first and never reorders
    # inv: a rung is reported as dropped only when it actually removed something from a result
    if any(r.snippet is not None for r in out.results):
        out.budget.dropped.append("snippet")
    for r in out.results:
        r.snippet = None
    if fits():
        return out
    if any(r.edges is not None for r in out.results):
        out.budget.dropped.append("edges")
    for r in out.results:
        r.edges = None
    if fits():
        return out
    if len(out.results) > K_FLOOR:
        out.results = out.results[:K_FLOOR]
        out.budget.dropped.append("k")
        if fits():
            return out
    out.budget.exceeded = True
    return out
