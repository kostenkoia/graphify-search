"""Derive the expansion tokens of a question from its text and the graph vocabulary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

CODE_REVIEW_GRAPH = "code-review-graph"
GRAPHIFY = "graphify"

# why: graphify's step 0 ends the query when no vocabulary token matches, so for this system the
# harness journals the step that carries `<expansion>` as halted instead of running it on ""
HALTS_ON_EMPTY = frozenset({GRAPHIFY})

# NOT DERIVED: the ordinary service words of an English question; the list is fixed and the same
# for every question, so no question can be helped by a wording choice
STOP = frozenset({
    "how", "is", "are", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or",
    "what", "which", "where", "when", "who", "does", "do", "did", "was", "were", "be",
    "by", "with", "from", "that", "this", "it", "its", "at", "as", "can", "we", "you",
})

STEM = 4  # NOT DERIVED: the shared prefix length that makes two tokens count as one word

MIN_WORD = 3  # NOT DERIVED: the shortest question word that carries a subject, not a particle

# why: these two bounds belong to the pinned vocabulary script, and the two must agree, or a
# token accepted here would be missing from the vocabulary the run itself writes
MIN_TOKEN = 3
MAX_TOKEN = 30

_ASCII_WORD = re.compile(r"[a-z]+")
_LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)
_CASE_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


def content_words(text: str) -> list[str]:
    """Return the question words without the service ones, in the order of the question.

    Parameters
    ----------
    text : str
        The question as written.

    Returns
    -------
    list of str
        Lowercased words, each appearing once.
    """
    out: list[str] = []
    for word in _ASCII_WORD.findall(text.lower()):
        if len(word) >= MIN_WORD and word not in STOP and word not in out:
            out.append(word)
    return out


def vocabulary(graph_json: Path) -> set[str]:
    """Return the token vocabulary of a graph, by the procedure the pinned script follows.

    Parameters
    ----------
    graph_json : Path
        The `graph.json` of an index.

    Returns
    -------
    set of str
        Every lowercased case part of every node label, within the token length bounds.
    """
    data = json.loads(graph_json.read_text(encoding="utf-8"))
    vocab: set[str] = set()
    for node in data["nodes"]:
        for chunk in _LETTERS.findall(node.get("label", "") or ""):
            for part in _CASE_PART.findall(chunk) or [chunk]:
                token = part.lower()
                if MIN_TOKEN <= len(token) <= MAX_TOKEN:
                    vocab.add(token)
    return vocab


def _relatives(word: str, vocab: set[str]) -> list[str]:
    # why: the word itself answers best, then the tokens closest to it in length; the alphabet
    # settles the rest, so a set's iteration order can never reach the result
    candidates = [token for token in vocab if token[:STEM] == word[:STEM]]
    return sorted(candidates, key=lambda token: (token != word, abs(len(token) - len(word)), token))


def _from_vocabulary(words: list[str], vocab: set[str], max_tokens: int) -> list[str]:
    per_word = [_relatives(word, vocab) for word in words]
    # inv: the slots go round the question words, or the first word takes the whole cap
    picked: list[str] = []
    for rank in range(max((len(c) for c in per_word), default=0)):
        for candidates in per_word:
            if rank < len(candidates) and candidates[rank] not in picked:
                picked.append(candidates[rank])
    return picked[:max_tokens]


def expand(text: str, vocab: set[str], max_tokens: int) -> dict[str, list[str]]:
    """Return the expansion tokens of each system, by one rule that leaves no choice.

    Parameters
    ----------
    text : str
        The question as written.
    vocab : set of str
        The token vocabulary of the graph the question is asked against.
    max_tokens : int
        The cap the harness enforces on an expansion.

    Returns
    -------
    dict of str to list of str
        One token list per system name.
    """
    words = content_words(text)
    return {CODE_REVIEW_GRAPH: words, GRAPHIFY: _from_vocabulary(words, vocab, max_tokens)}


MECHANICAL = "mechanical"

# why: the generated block names its own origin, so a hand edit is visible in the diff
ORIGIN = "produced by the expand verb from the question text and the graph vocabulary"


def expansion_block(text: str, vocab: set[str], max_tokens: int) -> dict[str, dict[str, object]]:
    """Return the `expansion` mapping of a question, ready to be written into its file.

    Parameters
    ----------
    text : str
        The question as written.
    vocab : set of str
        The token vocabulary of the graph the question is asked against.
    max_tokens : int
        The cap the harness enforces on an expansion.

    Returns
    -------
    dict of str to dict
        One `{tokens, why}` mapping per system name.
    """
    return {system: {"tokens": tokens, "why": ORIGIN}
            for system, tokens in expand(text, vocab, max_tokens).items()}


def mismatches(question: dict, vocab: set[str], max_tokens: int) -> list[str]:
    """Return one message per system whose written tokens the rule does not reproduce.

    Parameters
    ----------
    question : dict
        A loaded question file.
    vocab : set of str
        The token vocabulary of the graph the question is asked against.
    max_tokens : int
        The cap the harness enforces on an expansion.

    Returns
    -------
    list of str
        Empty when the question does not claim the mechanical rule, or when every
        written token list is the one the rule produces.
    """
    if question.get("rule") != MECHANICAL:
        return []
    expected = expand(question["text"], vocab, max_tokens)
    found = []
    for system, block in (question.get("expansion") or {}).items():
        if system not in expected:
            found.append(f"{question['id']}: no mechanical rule for system {system}")
        elif list(block["tokens"]) != expected[system]:
            found.append(f"{question['id']}: {system} tokens are {block['tokens']}, "
                         f"the rule produces {expected[system]}")
    return found


def main(argv: list[str] | None = None) -> int:
    """Print the `expansion` block of one question text.

    Parameters
    ----------
    argv : list of str or None
        `--graph`, `--max-tokens` and the question text; None reads `sys.argv`.

    Returns
    -------
    int
        Zero.
    """
    ap = argparse.ArgumentParser(prog="benchmark.harness expand")
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("text")
    args = ap.parse_args(argv)
    block = expansion_block(args.text, vocabulary(args.graph), args.max_tokens)
    print(yaml.safe_dump({"rule": MECHANICAL, "expansion": block},
                         sort_keys=False, allow_unicode=True), end="")
    return 0
