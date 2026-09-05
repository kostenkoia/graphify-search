import re
import shutil
from pathlib import Path

import pytest

from graphify_search import index as ix
from graphify_search import search
from graphify_search import settings as st
from graphify_search.embed import EmbeddingClient
from graphify_search.errors import EndpointUnavailableError, InputError
from graphify_search.settings import DEFAULT_BUDGET
from tests.embed_stub import serve

FIX = Path(__file__).parent / "fixtures" / "graph_small"


@pytest.fixture
def built(tmp_path):
    shutil.copytree(FIX, tmp_path / "c")
    graph, src = tmp_path / "c" / "graph.json", tmp_path / "c" / "source"
    with serve() as (url, _):
        ix.build_index(graph, src, EmbeddingClient(url, "m", "search_document: ", "search_query: "))
    return graph


def run(graph, question="alpha doubles", client=None, **kw):
    idx = ix.load_index(graph)
    args = {"k": 10, "budget": 6000, "require_dense": False, "snippets": True, "edges": True,
            "graph_sha256_now": idx.manifest.graph_sha256}
    args.update(kw)
    return search.query(idx, question, client, **args)


def test_dense_answer_shape_and_ranks(built):
    with serve() as (url, _):
        a = run(built, client=EmbeddingClient(url, "m", "search_document: ", "search_query: "))
    assert a.mode == "dense"
    assert a.index.vectors == "present"
    assert a.index.stale is False
    assert [r.rank for r in a.results] == list(range(1, 11))
    assert all(r.score == round(r.score, 3) for r in a.results)
    assert sorted(a.results, key=lambda r: (-r.score, r.path, r.start)) == a.results
    alpha = next(r for r in a.results if r.symbol == "alpha")
    assert alpha.snippet.startswith("def alpha")
    assert alpha.edges[0].to == "beta()"


def test_bm25_when_endpoint_unreachable(built):
    dead = EmbeddingClient("http://127.0.0.1:9/v1", "m", "search_document: ", "search_query: ")
    a = run(built, client=dead)
    assert a.mode == "bm25"
    assert a.index.vectors == "unreachable"
    assert a.results[0].symbol == "alpha"
    with pytest.raises(EndpointUnavailableError):
        run(built, client=dead, require_dense=True)


def test_bm25_when_no_vectors_built(tmp_path):
    shutil.copytree(FIX, tmp_path / "c")
    graph = tmp_path / "c" / "graph.json"
    ix.build_index(graph, tmp_path / "c" / "source", None)
    a = run(graph)
    assert a.mode == "bm25"
    assert a.index.vectors == "absent"
    assert a.model is None
    with pytest.raises(EndpointUnavailableError):
        run(graph, require_dense=True)


def test_stale_flag_and_k_clip_in_dense_mode(built):
    with serve() as (url, _):
        client = EmbeddingClient(url, "m", "search_document: ", "search_query: ")
        a = run(built, graph_sha256_now="other", k=99, snippets=False, edges=False, client=client)
    assert a.index.stale is True
    assert len(a.results) == 10
    assert a.results[0].snippet is None
    assert a.results[0].edges is None
    assert a.budget.dropped == ["snippet", "edges"]


def test_bm25_k_clip_returns_only_matching_rows(built):
    a = run(built, k=99)
    assert all(r.score > 0 for r in a.results)
    # why: a body is BODY_LINES from its start, so in the 25-line main.py every symbol above
    # `run()` -- which calls alpha(3) -- carries the token "alpha"; `trailing()` and the other
    # files do not
    assert [r.symbol for r in a.results] == ["alpha", None, "run", "Gamma", "beta"]
    assert all(r.path == "app/main.py" for r in a.results)


def test_require_dense_without_vectors_names_missing_vectors(tmp_path):
    shutil.copytree(FIX, tmp_path / "c")
    graph = tmp_path / "c" / "graph.json"
    ix.build_index(graph, tmp_path / "c" / "source", None)
    with pytest.raises(EndpointUnavailableError, match="no vectors in the index"):
        run(graph, require_dense=True)


def test_require_dense_without_client_names_missing_endpoint(built):
    with pytest.raises(EndpointUnavailableError, match="no embedding endpoint configured for this query"):
        run(built, require_dense=True)


def test_k_zero_returns_a_well_formed_answer_with_no_results(built):
    a = run(built, k=0)
    assert a.results == []
    assert a.mode == "bm25"
    assert a.budget.used_chars == len("[]")


def test_dropped_lists_switched_off_field_once_alongside_ladder_removal(built):
    a = run(built, budget=1, edges=False)
    assert a.budget.dropped == ["edges", "snippet"]


def test_empty_question_is_refused(built):
    with pytest.raises(InputError):
        run(built, question="   ")


def test_a_model_the_index_was_not_built_with_is_refused(built):
    with serve() as (url, _), pytest.raises(InputError) as e:
        run(built, client=EmbeddingClient(url, "other", "search_document: ", "search_query: "))
    assert "other" in str(e.value)
    assert "--full" in e.value.hint


def test_a_query_vector_of_another_width_is_refused(built):
    with serve(dims=4) as (url, _), pytest.raises(InputError) as e:
        run(built, client=EmbeddingClient(url, "m", "search_document: ", "search_query: "))
    assert "--full" in e.value.hint


def test_dropped_stays_empty_when_no_result_carries_a_snippet(built):
    idx = ix.load_index(built)
    for row in idx.rows:
        row.snippet = ""
    a = search.query(idx, "alpha doubles", None, k=10, budget=6000, require_dense=False,
                     snippets=True, edges=True, graph_sha256_now=idx.manifest.graph_sha256)
    assert a.results
    assert all(not r.snippet for r in a.results)
    assert a.budget.dropped == []


def test_the_budget_ladder_names_each_rung_it_removes(built):
    def rungs(budget):
        with serve() as (url, _):
            client = EmbeddingClient(url, "m", "search_document: ", "search_query: ")
            return run(built, budget=budget, client=client).budget.dropped

    assert rungs(6000) == []
    assert rungs(2000) == ["snippet"]
    assert rungs(1200) == ["snippet", "edges"]
    assert rungs(700) == ["snippet", "edges", "k"]


def test_dropped_names_a_field_the_caller_switched_off(built):
    assert run(built, snippets=False).budget.dropped == ["snippet"]
    assert run(built, edges=False).budget.dropped == ["edges"]


def test_dropped_stays_silent_about_snippet_when_source_lines_were_read(built):
    a = run(built)
    assert any(r.snippet for r in a.results)
    assert "snippet" not in a.budget.dropped


def test_bm25_without_one_shared_word_is_refused(built):
    with pytest.raises(InputError) as e:
        run(built, question="как работает поиск")
    assert str(e.value) == "no word of the question occurs in the indexed code (mode bm25)"
    assert "graphify-search index" in e.value.hint


def test_bm25_with_a_shared_word_still_answers(built):
    a = run(built, question="alpha doubles")
    assert a.mode == "bm25"
    assert a.results


def test_bm25_refuses_a_question_without_a_single_word(built):
    with pytest.raises(InputError, match="no word"):
        run(built, question="🔥🔥🔥")


def test_the_budget_comment_states_the_size_a_ten_place_answer_renders(built):
    with serve() as (url, _):
        a = run(built, client=EmbeddingClient(url, "m", "search_document: ", "search_query: "),
                budget=DEFAULT_BUDGET)
    assert len(a.results) == 10
    assert a.budget.dropped == []
    # inv: the figure DEFAULT_BUDGET's comment names is this measurement, so a change to the
    # fixture or to the answer's shape has to restate it rather than leave it wrong
    source = Path(st.__file__).read_text(encoding="utf-8")
    claimed = re.search(r"render (\d+) characters on this\n#\s*package's own fixture", source)
    assert claimed, "DEFAULT_BUDGET's comment no longer names a rendered size"
    assert int(claimed.group(1)) == a.budget.used_chars
    assert a.budget.used_chars < DEFAULT_BUDGET


def test_exclude_drops_matching_paths_before_the_k_cut(built):
    def ask(**kw):
        with serve() as (url, _):
            client = EmbeddingClient(url, "m", "search_document: ", "search_query: ")
            return run(built, client=client, **kw)

    assert [r.path for r in ask(k=2).results] == ["web/useThing.ts", "app/main.py"]
    kept = ask(k=2, exclude=["app/*"])
    assert [r.path for r in kept.results] == ["web/useThing.ts", "docs/guide.md"]
    assert [r.rank for r in kept.results] == [1, 2]


def test_exclude_takes_several_globs_and_matches_at_any_depth():
    assert search._excluded("tests/client/test_redirects.py", ["tests/*"])
    assert search._excluded("tests/test_main.py", ["tests/*"])
    assert not search._excluded("httpx/_client.py", ["tests/*"])
    assert search._excluded("docs/api.md", ["tests/*", "docs/*"])
    assert not search._excluded("httpx/_client.py", [])


def test_excluding_everything_answers_with_no_results(built):
    a = run(built, exclude=["*"])
    assert a.results == []
    assert a.mode == "bm25"
    assert a.index.vectors == "present"
    assert a.budget.used_chars == len("[]")


def test_without_exclude_the_answer_is_unchanged(built):
    assert run(built).results == run(built, exclude=[]).results
