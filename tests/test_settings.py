import json
from pathlib import Path

import pytest

from graphify_search import settings as st
from graphify_search.errors import InputError


def test_resolve_graph_precedence(tmp_path):
    cwd = tmp_path
    assert st.resolve_graph(None, {}, cwd) == cwd / "graphify-out" / "graph.json"
    assert st.resolve_graph(None, {"GRAPHIFY_OUT": "out2"}, cwd) == cwd / "out2" / "graph.json"
    assert st.resolve_graph(None, {"GRAPHIFY_OUT": "/abs/out"}, cwd) == Path("/abs/out/graph.json")
    d = tmp_path / "g"
    d.mkdir()
    assert st.resolve_graph(str(d), {"GRAPHIFY_OUT": "ignored"}, cwd) == d / "graph.json"
    assert st.resolve_graph(str(d / "graph.json"), {}, cwd) == d / "graph.json"


def test_resolve_settings_precedence(tmp_path):
    graph = tmp_path / "graphify-out" / "graph.json"
    (tmp_path / "graphify-out" / ".graphify_search").mkdir(parents=True)
    (tmp_path / "graphify-out" / ".graphify_search" / "config.json").write_text(json.dumps({"model": "cfg", "k": 3}))
    s = st.resolve(graph, {}, {})
    assert (s.endpoint, s.model, s.k, s.budget) == (st.DEFAULT_ENDPOINT, "cfg", 3, st.DEFAULT_BUDGET)
    s = st.resolve(graph, {}, {"GRAPHIFY_SEARCH_MODEL": "env"})
    assert s.model == "env"
    s = st.resolve(graph, {"model": "flag", "k": 7}, {"GRAPHIFY_SEARCH_MODEL": "env"})
    assert (s.model, s.k) == ("flag", 7)


def test_a_non_integer_k_is_refused(tmp_path):
    (tmp_path / "graphify-out" / ".graphify_search").mkdir(parents=True)
    (tmp_path / "graphify-out" / ".graphify_search" / "config.json").write_text(json.dumps({"k": "many"}))
    with pytest.raises(InputError) as e:
        st.resolve(tmp_path / "graphify-out" / "graph.json", {}, {})
    assert "k must be an integer, got 'many'" in str(e.value)
    assert "config.json" in e.value.hint


def test_a_local_endpoint_defaults_to_no_prefixes(tmp_path):
    graph = tmp_path / "graphify-out" / "graph.json"
    s = st.resolve(graph, {}, {"GRAPHIFY_SEARCH_ENDPOINT": "local"})
    assert (s.endpoint, s.doc_prefix, s.query_prefix) == ("local", "", "")
    s = st.resolve(graph, {"endpoint": "local"}, {})
    assert (s.doc_prefix, s.query_prefix) == ("", "")


def test_a_local_endpoint_keeps_prefixes_the_config_file_sets(tmp_path):
    (tmp_path / "graphify-out" / ".graphify_search").mkdir(parents=True)
    (tmp_path / "graphify-out" / ".graphify_search" / "config.json").write_text(
        json.dumps({"doc_prefix": "passage: "}))
    s = st.resolve(tmp_path / "graphify-out" / "graph.json", {"endpoint": "local"}, {})
    assert (s.doc_prefix, s.query_prefix) == ("passage: ", "")


def test_an_http_endpoint_keeps_the_nomic_prefixes(tmp_path):
    s = st.resolve(tmp_path / "graphify-out" / "graph.json", {}, {})
    assert (s.doc_prefix, s.query_prefix) == (st.DEFAULT_DOC_PREFIX, st.DEFAULT_QUERY_PREFIX)


def _with_config(tmp_path, payload):
    d = tmp_path / "graphify-out" / ".graphify_search"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(payload))
    return tmp_path / "graphify-out" / "graph.json"


def test_a_remote_endpoint_in_the_config_file_is_refused(tmp_path):
    graph = _with_config(tmp_path, {"endpoint": "https://evil.example.com/v1"})
    with pytest.raises(InputError) as e:
        st.resolve(graph, {}, {})
    assert "https://evil.example.com/v1" in str(e.value)
    assert "config.json" in str(e.value)
    assert e.value.hint == ("remove `endpoint` from that file; a remote server goes in "
                            "GRAPHIFY_SEARCH_ENDPOINT or --endpoint")


@pytest.mark.parametrize("endpoint", ["http://localhost:1234/v1", "http://127.0.0.1:1234/v1",
                                      "http://127.5.5.5:1234/v1", "http://[::1]:1234/v1", "local"])
def test_a_loopback_endpoint_in_the_config_file_is_accepted(tmp_path, endpoint):
    assert st.resolve(_with_config(tmp_path, {"endpoint": endpoint}), {}, {}).endpoint == endpoint


def test_a_remote_endpoint_from_the_environment_or_the_flag_is_accepted(tmp_path):
    graph = tmp_path / "graphify-out" / "graph.json"
    assert st.resolve(graph, {}, {"GRAPHIFY_SEARCH_ENDPOINT": "https://api.example.com/v1"}).endpoint == \
        "https://api.example.com/v1"
    assert st.resolve(graph, {"endpoint": "https://api.example.com/v1"}, {}).endpoint == \
        "https://api.example.com/v1"


def test_a_remote_endpoint_in_the_config_file_is_refused_even_when_a_flag_overrides_it(tmp_path):
    graph = _with_config(tmp_path, {"endpoint": "https://evil.example.com/v1"})
    with pytest.raises(InputError):
        st.resolve(graph, {"endpoint": "local"}, {})


@pytest.mark.parametrize("body", ["[1, 2]", '"x"', "3", "null"])
def test_a_config_file_that_is_not_a_json_object_is_refused(tmp_path, body):
    d = tmp_path / "graphify-out" / ".graphify_search"
    d.mkdir(parents=True)
    (d / "config.json").write_text(body)
    with pytest.raises(InputError) as e:
        st.resolve(tmp_path / "graphify-out" / "graph.json", {}, {})
    assert "is not a JSON object" in str(e.value)
    assert "config.json" in str(e.value)
    assert e.value.hint == "fix or delete the file"


def test_a_config_file_naming_an_unparsable_url_is_refused(tmp_path):
    graph = _with_config(tmp_path, {"endpoint": "http://[::1/v1"})
    with pytest.raises(InputError) as e:
        st.resolve(graph, {}, {})
    assert "http://[::1/v1" in str(e.value)
    assert "config.json" in str(e.value)
    assert e.value.hint == "fix or delete the file"
