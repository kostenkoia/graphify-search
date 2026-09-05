"""Check that the reference answer reached neither the prompt nor a tool definition."""

from __future__ import annotations

import json

# NOT DERIVED: a name shorter than this cannot be told apart from an ordinary word in prose,
# so a check built on it would report a leak on every question
MIN_SYMBOL = 3


class BlindError(Exception):
    """A reference cannot be checked against, so blindness cannot be shown either way."""


def _terms(place: dict) -> list[tuple[str, str]]:
    # inv: ordered most specific first, so the message names the longest thing that leaked
    # rather than a fragment of it
    terms: list[tuple[str, str]] = []
    if place.get("qualified_name"):
        terms.append(("qualified name", str(place["qualified_name"])))
    if place.get("path"):
        path = str(place["path"])
        terms.append(("path", path))
        terms.append(("file", path.rsplit("/", 1)[-1]))
    symbol = place.get("symbol")
    if symbol:
        if len(str(symbol)) < MIN_SYMBOL:
            raise BlindError(f"reference symbol {symbol!r} is too short to check blindness against")
        terms.append(("symbol", str(symbol)))
    if place.get("why"):
        terms.append(("reasoning", str(place["why"])))
    return terms


def violations(authored: str, tool_definitions: list[dict], reference: dict) -> list[str]:
    """Return one message per reference place the runner was told about before it started.

    Parameters
    ----------
    authored : str
        The part of the prompt above the harness heading, which the owner wrote.
    tool_definitions : list of dict
        Every tool the runner is offered, descriptions and schemas included.
    reference : dict
        The question's reference, whose `places` must reach the runner only as the
        output of a journaled action.

    Returns
    -------
    list of str
        Empty when nothing leaked; one message per place that did, naming where.

    Raises
    ------
    BlindError
        When a reference place cannot be checked against at all.
    """
    # inv: a tool is searched whole, schema included, because a leak in a property's own
    # description reaches the runner exactly as one in the summary does
    haystacks = [("the prompt", authored)] + [
        (f"the definition of {tool.get('name')}", json.dumps(tool, ensure_ascii=False))
        for tool in tool_definitions
    ]
    folded = [(where, text.lower()) for where, text in haystacks]
    found: list[str] = []
    for index, place in enumerate(reference.get("places") or [], start=1):
        for label, term in _terms(place):
            # why: folded, because a leak spelled in another case is the same leak
            where = next((name for name, text in folded if term.lower() in text), None)
            if where is not None:
                found.append(f"place {index}: {where} carries the reference {label} {term!r}")
                break
    return found
