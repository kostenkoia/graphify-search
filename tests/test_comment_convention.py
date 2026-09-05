"""Scan comments and docstrings under the source roots for the forbidden vocabulary.

String literals are never inspected, so a user-facing message carrying an id
does not trip the guard.
"""
import ast
import io
import pathlib
import re
import tokenize

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOTS = [
    REPO / "src" / "graphify_search",
    REPO / "benchmark" / "harness",
]

# why: banning every <letter><digit> token would false-positive on SHA256 and UTF8, so single-letter
# id-codes other than U, W, X and R are not matched by the pattern
# why: the narration words match without regard to case, since the ceremony reads the same either way
# why: the past-habitual phrase is anchored to a subject, so "the harness used to drain" is caught
# while the passive "a root, used to resolve a key" stays legal
_FORBIDDEN = re.compile(
    r"""(
        (?:19|20)\d{2}-\d{2}-\d{2}   # a date -- history belongs in the commit
      | spec\s§                       # spec citation -- we cite ADRs, not specs
      | §\d                           # bare spec-section marker
      | (?i:\btask\s\d)               # sprint task narration
      | \b[UWX]\d\b                   # campaign item ids (U5, W3, X2, ...)
      | \bR\d+\b                     # requirement ids (R7, R12, ...)
      | \bW\d-[A-Z]                   # wave-item ids (W4-A.10)
      | (?i:\bwave\s\d)               # campaign wave narration
      | (?i:\bcriterion\s\d)          # acceptance-criterion narration
      | (?i:\bcheckpoint\b)           # review-checkpoint narration
      | \bRULING\b                    # ruling narration
      | (?i:\bfinding\s\d)            # review-finding narration
      | -transition\b                 # bare removal-plan marker tags
      | (?i:\bpreviously\b)           # change-history narration
      | (?i:\b(?:it|this|that|these|those|they|we|which|there|the\s\w+)\sused\sto\b)
    )""",
    re.VERBOSE,
)


def _iter_py():
    return sorted(p for root in ROOTS for p in root.rglob("*.py"))


def _comment_hits(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")
    hits = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            m = _FORBIDDEN.search(tok.string)
            if m:
                hits.append((tok.start[0], "comment", m.group(0), tok.string.strip()[:80]))
    return hits


def _docstring_hits(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")
    hits = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ds = ast.get_docstring(node, clean=False)
            if ds:
                m = _FORBIDDEN.search(ds)
                if m:
                    line = getattr(node, "lineno", 0)
                    hits.append((line, "docstring", m.group(0), ds.strip().splitlines()[0][:80]))
    return hits


@pytest.mark.parametrize("path", _iter_py(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_forbidden_tokens_in_comments_or_docstrings(path):
    hits = _comment_hits(path) + _docstring_hits(path)
    if hits:
        rel = path.relative_to(REPO).as_posix()
        lines = "\n".join(
            f"  {rel}:{ln} ({where}) matched {tok!r} -- {snippet!r}"
            for ln, where, tok, snippet in sorted(hits)
        )
        pytest.fail(
            f"{len(hits)} comment/docstring convention violation(s) in {rel}.\n"
            "Comments state a present-tense invariant/why, signed inv:/why:/adr-NNNN; "
            "dates, campaign ids, spec-§ citations and change-history belong in the "
            "commit message, not the code. Docstrings stay NumPy-style (what/args/returns).\n"
            f"{lines}",
        )


def test_scan_actually_reads_something():
    """A guard on the guard: if a root's glob found no files, or the tokenizer
    found no comments anywhere, the parametrized test would pass vacuously."""
    total_comments = 0
    for root in ROOTS:
        files = sorted(root.rglob("*.py"))
        assert files, f"expected at least one scanned .py file under {root}"
        for path in files:
            src = path.read_text(encoding="utf-8")
            total_comments += sum(
                1 for tok in tokenize.generate_tokens(io.StringIO(src).readline)
                if tok.type == tokenize.COMMENT
            )
    assert total_comments > 0, "expected at least one comment across the scanned roots"


def test_blacklist_catches_the_vocabulary_the_convention_names():
    """A guard on the blacklist: every banned form is matched, in either case."""
    # inv: a scanned root holding none of these reads the same whether the pattern works or
    # not, so the pattern is exercised directly on the forms the convention names
    for text in ("previously this drained the pipe", "Previously the pipe was drained",
                 "this used to drain the pipe", "the harness used to drain the pipe",
                 "which used to be a pipe", "Task 12 cleanup", "task 4 follow-up",
                 "Wave 3 moved this", "wave 3 moved this", "Checkpoint after this",
                 "Criterion 2 says", "finding 3 says", "2026-08-25 was the day"):
        assert _FORBIDDEN.search(text), text


def test_blacklist_leaves_ordinary_description_alone():
    """A guard on the blacklist: the passive reading of a banned phrase is not history."""
    # why: an over-broad ban would be answered with a suppression, which this project treats
    # as a defect -- so the phrase that describes rather than narrates must stay legal
    for text in ("Label to root directory, used to resolve a changed key back to a file to hash.",
                 "the argv is used to name the stem",
                 "a listing is used to detect a change",
                 "the sha256 of every RECORD-listed file"):
        assert not _FORBIDDEN.search(text), text
