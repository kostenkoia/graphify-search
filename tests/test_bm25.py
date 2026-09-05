import os
import subprocess
import sys

from graphify_search.bm25 import BM25

DOCS = {"a": ["asset", "archived", "handler"], "b": ["asset", "created"], "c": ["dashboard"]}

_RANK_SCRIPT = (
    "from graphify_search.bm25 import BM25\n"
    "docs = {'a': ['asset', 'archived', 'handler'], 'b': ['asset', 'created'], 'c': ['dashboard']}\n"
    "scores = BM25.build(docs).rank(['archived', 'asset', 'handler'])\n"
    "print(repr(scores))\n"
)


def test_rank_orders_by_score_and_drops_zero():
    ranked = BM25.build(DOCS).rank(["archived", "asset"])
    assert [i for i, _ in ranked] == ["a", "b"]
    assert ranked[0][1] > ranked[1][1] > 0


def test_unknown_terms_yield_nothing():
    assert BM25.build(DOCS).rank(["zzz"]) == []


def test_json_round_trip_is_exact():
    idx = BM25.build(DOCS)
    again = BM25.from_json(idx.to_json())
    assert again.rank(["asset"]) == idx.rank(["asset"])
    assert again.ids == ["a", "b", "c"]


def test_ties_break_by_id():
    idx = BM25.build({"y": ["x"], "x": ["x"]})
    assert [i for i, _ in idx.rank(["x"])] == ["x", "y"]


def test_scores_are_deterministic_across_processes():
    runs = [
        subprocess.run(
            [sys.executable, "-c", _RANK_SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    ]
    assert runs[0] == runs[1]
