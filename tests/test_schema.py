import json

import pytest

from graphify_search import schema as s


def result(rank, snippet="x" * 50, edges=True):
    return s.Result(rank=rank, kind="symbol", path=f"p{rank}.py", symbol=f"f{rank}", start=rank,
                    community="c", score=0.5, snippet=snippet,
                    edges=[s.EdgeOut("calls", "g()", "L2")] if edges else None)


def answer(n, **kw):
    return s.Answer(question="q", mode="dense", model="m",
                    index=s.IndexState("abc", False, "present"),
                    budget=s.Budget(limit_chars=kw.get("limit", 6000), used_chars=0, dropped=[]),
                    results=[result(i) for i in range(1, n + 1)])


def test_render_is_compact_with_trailing_newline_and_fixed_key_order():
    text = s.render(answer(1))
    assert text.endswith("\n")
    assert ": " not in text
    assert ", " not in text
    doc = json.loads(text)
    assert list(doc) == ["question", "mode", "model", "endpoint", "index", "budget", "results"]
    expected_keys = ["rank", "kind", "path", "symbol", "start", "community", "score", "snippet", "edges"]
    assert list(doc["results"][0]) == expected_keys
    assert doc["budget"]["used_chars"] == s.results_chars(answer(1).results)


def test_absent_snippet_and_edges_are_omitted_not_null():
    a = answer(1)
    a.results[0].snippet = None
    a.results[0].edges = None
    doc = json.loads(s.render(a))
    assert "snippet" not in doc["results"][0]
    assert "edges" not in doc["results"][0]
    assert list(doc["results"][0]) == ["rank", "kind", "path", "symbol", "start", "community", "score"]


def test_ladder_drops_snippet_then_edges_then_k():
    a = answer(10, limit=1)
    out = s.apply_budget(a, limit=1)
    assert out.budget.dropped == ["snippet", "edges", "k"]
    assert len(out.results) == s.K_FLOOR
    assert out.budget.exceeded is True
    assert [r.rank for r in out.results] == [1, 2, 3, 4, 5]


def test_ladder_stops_at_the_first_step_that_fits():
    a = answer(2)
    without_snippets = s.results_chars([result(1, snippet=None), result(2, snippet=None)])
    out = s.apply_budget(a, limit=without_snippets)
    assert out.budget.dropped == ["snippet"]
    assert out.budget.exceeded is False
    assert out.budget.used_chars == without_snippets


def test_no_drop_when_it_fits():
    out = s.apply_budget(answer(2), limit=100000)
    assert out.budget.dropped == []
    assert out.results[0].snippet is not None


def test_apply_budget_leaves_the_input_untouched():
    a = answer(3)
    before = s.dumps(a)
    out = s.apply_budget(a, limit=1)
    assert s.dumps(a) == before
    assert out.results is not a.results


def test_empty_answer_reports_no_drops():
    out = s.apply_budget(answer(0), limit=0)
    assert out.budget.dropped == []
    assert out.budget.exceeded is True


def test_ladder_reports_a_k_rung_that_fits():
    bare = [result(i, snippet=None, edges=False) for i in range(1, 11)]
    limit = s.results_chars(bare[:s.K_FLOOR])
    assert s.results_chars(bare) > limit
    out = s.apply_budget(answer(10), limit=limit)
    assert out.budget.dropped == ["snippet", "edges", "k"]
    assert len(out.results) == s.K_FLOOR
    assert out.budget.exceeded is False
    assert out.budget.used_chars == limit


def test_a_non_finite_score_is_refused_instead_of_rendered_as_invalid_json():
    a = answer(1)
    a.results[0].score = float("nan")
    with pytest.raises(ValueError, match="Out of range float values"):
        s.render(a)
