import json
from pathlib import Path

import pytest

from benchmark.harness.scoring.adapters import symbol_of as bench_symbol_of
from graphify_search import graph as g
from graphify_search.errors import InputError

FIX = Path(__file__).parent / "fixtures" / "graph_small"


def load():
    return g.load_graph(FIX / "graph.json")


def test_sha256_matches_the_file():
    assert load().sha256 == g.sha256_of_file(FIX / "graph.json")


def test_eligible_excludes_rationale_and_concept_and_unlocated():
    ids = [n.id for n in load().eligible()]
    assert "app_main_rationale_6" not in ids
    assert "concept_doubling" not in ids
    assert ids[:3] == ["app_init", "app_main", "app_main_alpha"]


def test_kind_order_document_file_symbol():
    nodes = {n.id: n for n in load().nodes}
    assert g.kind_of(nodes["docs_guide"]) == "document"
    # inv: a file node's label may carry a directory prefix and still end with the basename
    assert g.kind_of(nodes["app_init"]) == "file"
    assert g.kind_of(nodes["app_main"]) == "file"
    assert g.kind_of(nodes["app_main_alpha"]) == "symbol"


def test_docstrings_follow_rationale_for_edges():
    gr = load()
    assert gr.docstrings("app_main_alpha") == ["Compute alpha from x.      Doubles the input."]
    assert gr.docstrings("app_main_beta") == []


def test_edges_filtered_both_directions_in_links_order():
    gr = load()
    assert [(e.rel, e.to, e.file, e.loc) for e in gr.edges_of("app_main_alpha")] == [
        ("calls", "beta()", "app/main.py", "L10"),
        ("calls", ".run()", "app/main.py", "L21"),
        ("references", "useThing()", "web/useThing.ts", "L4"),
    ]


def test_edges_of_caps_at_edges_per_result_in_links_order(tmp_path):
    targets = [f"t{i}" for i in range(7)]
    nodes = [{"id": "src", "label": "src()", "file_type": "code",
              "source_file": "a.py", "source_location": "L1"}]
    nodes += [{"id": tid, "label": f"{tid}()", "file_type": "code",
               "source_file": "a.py", "source_location": f"L{i + 2}"} for i, tid in enumerate(targets)]
    links = [{"source": "src", "target": tid, "relation": "calls",
              "source_file": "a.py", "source_location": f"L{i + 2}"} for i, tid in enumerate(targets)]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": nodes, "links": links}))
    edges = g.load_graph(path).edges_of("src")
    assert len(edges) == g.EDGES_PER_RESULT == 5
    assert [e.to for e in edges] == [f"t{i}()" for i in range(5)]


def test_self_loop_edge_appears_once(tmp_path):
    nodes = [{"id": "n", "label": "n()", "file_type": "code", "source_file": "a.py", "source_location": "L1"}]
    links = [{"source": "n", "target": "n", "relation": "calls",
              "source_file": "a.py", "source_location": "L2"}]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": nodes, "links": links}))
    assert len(g.load_graph(path).edges_of("n")) == 1


def test_community_falls_back_to_numeric_zero(tmp_path):
    nodes = [{"id": "n", "label": "n()", "file_type": "code",
              "source_file": "a.py", "source_location": "L1", "community": 0}]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": nodes, "links": []}))
    assert g.load_graph(path).nodes[0].community == "0"


def test_load_graph_missing_file_raises_input_error(tmp_path):
    with pytest.raises(InputError) as exc:
        g.load_graph(tmp_path / "missing.json")
    assert "graphify" in exc.value.hint


def test_load_graph_without_nodes_and_links_raises_input_error(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"foo": 1}))
    with pytest.raises(InputError):
        g.load_graph(path)


def test_graph_exposes_no_end_hint_helpers():
    gr = load()
    assert not hasattr(g, "end_hint")
    assert not hasattr(gr, "starts_in")


def test_symbol_of_matches_the_benchmark_rule():
    assert g.symbol_of("alpha()") == "alpha"
    assert g.symbol_of(".run()") == "run"
    assert g.symbol_of("main.py") is None
    assert g.symbol_of("Compute alpha from x.") is None
    for label in ("alpha()", ".run()", "main.py", "Compute alpha from x.", "GET", "a.b()", ""):
        assert g.symbol_of(label) == bench_symbol_of(label)


def write_graph(tmp_path, raw):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(raw))
    return path


def test_load_graph_refuses_a_node_without_an_id(tmp_path):
    path = write_graph(tmp_path, {"nodes": [{"label": "alpha()", "file_type": "code"}], "links": []})
    with pytest.raises(InputError) as exc:
        g.load_graph(path)
    assert "node without id" in str(exc.value)
    assert "graphify" in exc.value.hint


def test_load_graph_refuses_nodes_that_are_not_a_list_of_objects(tmp_path):
    with pytest.raises(InputError) as exc:
        g.load_graph(write_graph(tmp_path, {"nodes": "oops", "links": []}))
    assert "nodes is not a list" in str(exc.value)
    with pytest.raises(InputError) as exc:
        g.load_graph(write_graph(tmp_path, {"nodes": {"a": {"id": "a"}}, "links": []}))
    assert "nodes is not a list" in str(exc.value)
    with pytest.raises(InputError) as exc:
        g.load_graph(write_graph(tmp_path, {"nodes": ["oops"], "links": []}))
    assert "node is not an object" in str(exc.value)


def test_load_graph_refuses_a_source_location_that_is_not_a_string(tmp_path):
    for bad in (5, ["L1"]):
        path = write_graph(tmp_path, {"nodes": [{"id": "n", "label": "n()", "file_type": "code",
                                                 "source_file": "a.py", "source_location": bad}],
                                      "links": []})
        with pytest.raises(InputError) as exc:
            g.load_graph(path)
        assert "source_location is not a string" in str(exc.value)


def test_load_graph_refuses_links_that_are_not_a_list_of_objects(tmp_path):
    node = {"id": "n", "label": "n()", "file_type": "code", "source_file": "a.py", "source_location": "L1"}
    with pytest.raises(InputError) as exc:
        g.load_graph(write_graph(tmp_path, {"nodes": [node], "links": "oops"}))
    assert "links is not a list" in str(exc.value)
    with pytest.raises(InputError) as exc:
        g.load_graph(write_graph(tmp_path, {"nodes": [node], "links": ["oops"]}))
    assert "link is not an object" in str(exc.value)


def test_load_graph_refuses_a_duplicate_node_id(tmp_path):
    node = {"id": "n", "label": "n()", "file_type": "code", "source_file": "a.py", "source_location": "L1"}
    path = write_graph(tmp_path, {"nodes": [node, {**node, "label": "other()"}], "links": []})
    with pytest.raises(InputError) as exc:
        g.load_graph(path)
    assert "duplicate node id n" in str(exc.value)


def test_extends_is_an_edge_relation(tmp_path):
    nodes = [{"id": "a", "label": "A", "file_type": "code", "source_file": "a.py", "source_location": "L1"},
             {"id": "b", "label": "B", "file_type": "code", "source_file": "a.py", "source_location": "L9"}]
    links = [{"source": "a", "target": "b", "relation": "extends",
              "source_file": "a.py", "source_location": "L1"}]
    gr = g.load_graph(write_graph(tmp_path, {"nodes": nodes, "links": links}))
    assert "extends" in g.EDGE_RELATIONS
    assert [(e.rel, e.to) for e in gr.edges_of("a")] == [("extends", "B")]


def test_kind_of_calls_a_code_node_whose_label_is_no_identifier_a_file():
    script = g.Node(id="s", label="build script", file_type="code", path="scripts/build",
                    start=1, community="0")
    assert g.symbol_of(script.label) is None
    assert g.kind_of(script) == "file"


def test_kind_of_keeps_a_declaration_a_symbol():
    for label, start in (("alpha()", 5), (".run()", 20), ("Gamma", 17)):
        node = g.Node(id="n", label=label, file_type="code", path="app/main.py",
                      start=start, community="0")
        assert g.symbol_of(node.label) is not None
        assert g.kind_of(node) == "symbol"


def test_every_code_label_a_symbol_kind_keeps_yields_an_identifier():
    for n in load().eligible():
        if g.kind_of(n) == "symbol":
            assert g.symbol_of(n.label) is not None
