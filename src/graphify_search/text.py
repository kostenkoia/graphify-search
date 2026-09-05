"""Assemble the text one node is indexed by."""

from __future__ import annotations

import hashlib
import re

# NOT DERIVED: the cap buys a body long enough to carry a symbol's own vocabulary without
# letting one long function dominate the index; the corpus it was chosen against is private, so
# the count behind it is not published and a reader tuning this has only their own corpus to go on
BODY_LINES = 30
# NOT DERIVED: what a reader needs to recognise a symbol
SNIPPET_LINES = 6
# inv: an identifier starts with a letter or an underscore in any script, so a question written
# outside the ASCII alphabet still yields tokens
_IDENT = re.compile(r"[^\W\d]\w*")
# inv: only ASCII case marks a camelCase boundary, so a run of letters in any other script
# stays one part instead of being cut at each character `[a-z]` cannot match
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[^\W\dA-Z_]+|[A-Z]+|\d+")


def split_identifiers(text: str) -> list[str]:
    """Split one identifier into lower-cased parts of at least two letters.

    Parameters
    ----------
    text : str
        A single identifier in camelCase, snake_case, or a mix of both.

    Returns
    -------
    list[str]
        The identifier's parts, lower-cased, each at least two characters
        and never purely digits.
    """
    # why: camelCase and snake_case both hide the words a question uses; a two-letter floor drops
    # loop counters and digits without touching real words
    return [p.lower() for p in _PARTS.findall(text) if len(p) >= 2 and not p.isdigit()]


def tokens(text: str) -> list[str]:
    """Return the split parts of every identifier in `text`, in order.

    Parameters
    ----------
    text : str
        Arbitrary text, such as a path or a body, to scan for identifiers.

    Returns
    -------
    list[str]
        The concatenated split parts of every identifier found, in the
        order they occur.
    """
    out: list[str] = []
    for ident in _IDENT.findall(text):
        out.extend(split_identifiers(ident))
    return out


def node_text(label: str, path: str, docstrings: list[str], body: str) -> str:
    """Join label, split path, docstrings and body into the text a node is embedded by.

    Parameters
    ----------
    label : str
        The node's display label.
    path : str
        The node's file path, split into tokens before joining.
    docstrings : list[str]
        The node's docstrings, joined with a single space.
    body : str
        The node's source body.

    Returns
    -------
    str
        The newline-joined text, skipping any part that is empty.
    """
    parts = [label, " ".join(tokens(path)), " ".join(docstrings), body]
    return "\n".join(p for p in parts if p)


def body_from(lines: list[str], start: int) -> str:
    """Return at most BODY_LINES lines from `start` (1-based) to the end of the file.

    Parameters
    ----------
    lines : list[str]
        The full file, split into lines.
    start : int
        The first line to include, 1-based.

    Returns
    -------
    str
        The newline-joined slice, or an empty string when `start` is out of
        range.
    """
    # why: the vendor records no end line for a symbol, and the next symbol's start is a wrong bound
    # for any class, whose first method starts a few lines in; a fixed window from the start is the
    # only cut that never hides a body
    if start < 1 or start > len(lines):
        return ""
    return "\n".join(lines[start - 1:start + BODY_LINES - 1])


def snippet_of(body: str) -> str:
    """Return the first SNIPPET_LINES lines of a body.

    Parameters
    ----------
    body : str
        The full body text.

    Returns
    -------
    str
        The newline-joined first SNIPPET_LINES lines.
    """
    return "\n".join(body.splitlines()[:SNIPPET_LINES])


def text_sha256(text: str) -> str:
    """Return the hex sha256 of a text's UTF-8 bytes.

    Parameters
    ----------
    text : str
        The text to hash.

    Returns
    -------
    str
        The hex-encoded sha256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
