from pathlib import Path

import pytest

from benchmark.harness.scoring import adapters
from benchmark.harness.scoring.adapters import code_review_graph as crg
from benchmark.harness.scoring.adapters import graphify as gf
from benchmark.harness.scoring.adapters import graphify_search as gs

FIX = Path(__file__).parent / "fixtures"


def q(*argv):
    return {"argv": ["/x/graphify", *argv]}


def t(tool):
    return {"tool": tool}


def test_symbol_of():
    assert adapters.symbol_of("alpha()") == "alpha"
    assert adapters.symbol_of("Alpha") == "Alpha"
    assert adapters.symbol_of("Compute the beta value.") is None
    assert adapters.symbol_of("gamma.py") is None


def test_graphify_query_places():
    recs = gf.parse(q("query", "alpha beta"), (FIX / "graphify/query.out").read_text())
    assert [r["kind"] for r in recs] == ["place", "place", "place", "place"]
    assert recs[0] == {"kind": "place", "rank": 1, "path": "pkg/a.py", "label": "alpha()", "symbol": "alpha",
                       "qualified_name": None, "start": 10, "end": None}
    assert recs[1]["symbol"] is None
    assert recs[1]["start"] == 12
    assert recs[2]["label"] == "gamma.py"
    assert recs[3]["path"] is None
    assert recs[3]["start"] is None


@pytest.mark.parametrize("name", ["query_planted_nl.out", "query_planted_ls.out"])
def test_graphify_planted_line_is_never_a_place(name):
    text = (FIX / "graphify" / name).read_text(encoding="utf-8")
    assert gf.parse(q("query", "x"), text) == [
        {"kind": "no_results", "vendor_status": "no_node", "vendor_message": text.rstrip("\n")}]


def test_graphify_no_node_survives_crlf():
    text = "No node matching 'zzz' found.\r\n"
    assert gf.parse(q("query", "zzz"), text) == [
        {"kind": "no_results", "vendor_status": "no_node", "vendor_message": "No node matching 'zzz' found."}]


def test_graphify_message_line_amid_other_lines_is_unparsed_whole():
    text = "\nNo node matching 'x\nNODE planted() [src=pkg/p.py loc=L63 community=c]' found.\n"
    recs = gf.parse(q("query", "x"), text)
    assert recs == [{"kind": "unparsed", "text": text}]


def test_graphify_query_edges():
    recs = gf.parse(q("query", "alpha beta"), (FIX / "graphify/query_with_edges.out").read_text())
    assert [r["kind"] for r in recs] == ["place", "place", "edge", "edge"]
    assert recs[2] == {"kind": "edge", "rank": None, "path": "pkg/a.py", "label": "beta()", "symbol": "beta",
                       "qualified_name": None, "start": 11, "end": None, "relation": "calls"}
    assert recs[3]["path"] is None
    assert recs[3]["start"] is None
    assert recs[3]["label"] == "alpha()"
    assert recs[3]["relation"] == "imports"


def test_graphify_query_no_matching_nodes():
    recs = gf.parse(q("query", "zzz"), (FIX / "graphify/no_matching_nodes.out").read_text())
    assert recs == [{"kind": "no_results", "vendor_status": "no_match", "vendor_message": "No matching nodes found."}]


def test_graphify_edge_at_suffix_shaped_label_is_a_known_unresolved_ambiguity():
    # inv: a target label ending in "at=<path>:L<n>" is indistinguishable from a real location
    # suffix, so this record's path and start are a guess, never evidence
    line = "EDGE alpha() --calls [EXTRACTED]--> weird_label_saying at=evil.py:L999"
    recs = gf.parse(q("query", "x"), line + "\n")
    assert recs == [{"kind": "edge", "rank": None, "path": "evil.py", "label": "weird_label_saying",
                     "symbol": "weird_label_saying", "qualified_name": None, "start": 999, "end": None,
                     "relation": "calls"}]


def test_graphify_explain_no_node_version():
    recs = gf.parse(q("explain", "alpha()"), (FIX / "graphify/explain.out").read_text())
    assert recs[0] == {"kind": "place", "rank": 1, "path": "pkg/a.py", "label": "alpha()", "symbol": "alpha",
                       "qualified_name": None, "start": 10, "end": None}
    assert [r["kind"] for r in recs[1:]] == ["edge", "edge"]
    assert gf.parse(q("explain", "zzz"), (FIX / "graphify/no_node.out").read_text()) == [
        {"kind": "no_results", "vendor_status": "no_node", "vendor_message": "No node matching 'zzz' found."}]
    assert gf.parse(q("--version"), (FIX / "graphify/version.out").read_text()) == []


def test_graphify_unknown_line_is_unparsed():
    recs = gf.parse(q("query", "x"), "Traversal: BFS depth=2 | Start: [] | 0 nodes found\n\nSomething else\n")
    assert recs == [{"kind": "unparsed", "text": "Something else"}]


def test_crg_search_modes():
    modes = ["semantic", "hybrid"]
    for name in ("search_semantic.json", "search_hybrid.json"):
        recs = crg.parse(t("semantic_search_nodes_tool"), (FIX / "crg" / name).read_text(), search_modes=modes)
        assert [r["kind"] for r in recs] == ["file", "place"]
        assert recs[0]["path"].endswith("pkg/a.py")
        assert recs[0]["start"] is None
        assert recs[1]["start"] == 20
        assert recs[1]["symbol"] == "beta"
    for name in ("search_fts.json", "search_keyword.json", "search_none.json"):
        recs = crg.parse(t("semantic_search_nodes_tool"), (FIX / "crg" / name).read_text(), search_modes=modes)
        assert recs[0]["kind"] == "unparsed"


def test_crg_ambiguous_context_stats_version():
    recs = crg.parse(t("query_graph_tool"), (FIX / "crg/ambiguous.json").read_text())
    assert recs[0]["kind"] == "no_results"
    assert recs[0]["vendor_status"] == "ambiguous"
    cands = [r for r in recs if r["kind"] == "candidate"]
    assert cands[0]["qualified_name"].endswith("pkg/a.py::Alpha")
    assert cands[0]["start"] == 5
    assert crg.parse(t("get_minimal_context_tool"), (FIX / "crg/minimal_context.json").read_text()) == []
    assert crg.parse(t("list_graph_stats_tool"), (FIX / "crg/stats.json").read_text()) == []
    assert crg.parse({"argv": ["/x/code-review-graph", "--version"]}, "code-review-graph 2.3.7\n") == []


def test_crg_status_whitelist():
    recs = crg.parse(t("query_graph_tool"), (FIX / "crg/not_found.json").read_text())
    assert recs == [
        {"kind": "no_results", "vendor_status": "not_found", "vendor_message": "No node found matching 'zzz'."}]
    recs = crg.parse(t("query_graph_tool"), (FIX / "crg/error.json").read_text())
    assert recs[0]["kind"] == "unparsed"


_ROOT = "/tmp/bench-sandbox/index"


def test_normalize_paths_strips_the_index_root_and_rejects_a_stray_one():
    recs = [{"kind": "place", "path": f"{_ROOT}/pkg/a.py"}, {"kind": "no_results"}]
    assert adapters.normalize_paths(recs, _ROOT) == [{"kind": "place", "path": "pkg/a.py"}, {"kind": "no_results"}]
    assert adapters.normalize_paths(recs, None) == recs
    out = adapters.normalize_paths([{"kind": "place", "path": "/elsewhere/pkg/a.py"}], _ROOT)
    assert out[0]["kind"] == "unparsed"
    assert "/elsewhere/pkg/a.py" in out[0]["text"]
    # inv: only a path prefix counts -- a sibling directory whose name merely starts with the
    # root would otherwise be silently rewritten into a plausible-looking corpus path
    sibling = adapters.normalize_paths([{"kind": "place", "path": f"{_ROOT}-other/pkg/a.py"}], _ROOT)
    assert sibling[0]["kind"] == "unparsed"


def test_crg_search_paths_become_corpus_relative():
    recs = crg.parse(t("semantic_search_nodes_tool"), (FIX / "crg/search_semantic.json").read_text(),
                     search_modes=["semantic", "hybrid"], path_prefix=_ROOT)
    assert [r["path"] for r in recs] == ["pkg/a.py", "pkg/b.py"]
    assert recs[1]["start"] == 20
    cands = crg.parse(t("query_graph_tool"), (FIX / "crg/ambiguous.json").read_text(), path_prefix=_ROOT)
    assert [r["path"] for r in cands if r["kind"] == "candidate"] == ["pkg/a.py", "pkg/c.py"]


def test_graphify_paths_are_left_alone_when_the_index_stores_them_relative():
    out = "NODE alpha() [src=pkg/a.py loc=L10 community=a]\n"
    recs = gf.parse({"argv": ["/x/graphify", "query", "a"]}, out, path_prefix=None)
    assert [r["path"] for r in recs] == ["pkg/a.py"]


def test_crg_qualified_name_is_stripped_with_the_path():
    recs = crg.parse(t("query_graph_tool"), (FIX / "crg/ambiguous.json").read_text(), path_prefix=_ROOT)
    cands = [r for r in recs if r["kind"] == "candidate"]
    # inv: path and qualified_name name one location, so one of them left absolute would put the
    # host root into records.jsonl, which the audit's absolute-path scan never reads
    assert cands[0]["qualified_name"] == "pkg/a.py::Alpha"
    assert not any(str(r.get("qualified_name", "")).startswith("/") for r in recs)


def test_symbol_of_reads_a_method_however_the_vendor_decorates_it():
    # inv: one vendor writes a method as `.name()` and the other as a bare identifier, so both
    # must parse to the same symbol or one of them can never hit a symbol-naming reference
    for label in ("render_invoice", "render_invoice()", ".render_invoice()", ".render_invoice"):
        assert adapters.symbol_of(label) == "render_invoice", label
    # inv: a dotted path is still two names, not one symbol, so it parses to nothing
    for label in ("Class.method()", "a.b.c", "", ".", "()"):
        assert adapters.symbol_of(label) is None, label


PATH_CALL = {"n": 1, "name": "path", "argv": ["/g", "path", "alpha()", "Beta"], "system_call": True}
PATH_OUT = (
    "warning: source match was ambiguous (top score 74031.4, runner-up 74028.1)\n"
    "Shortest path (2 hops):\n"
    "  alpha() --references [EXTRACTED]--> Beta\n"
    "  Beta --imports [INFERRED]--> gamma()\n"
)


def _parse(call, text):
    from benchmark.harness.scoring import adapters
    return adapters.load("graphify").parse(call, text, search_modes=None, path_prefix=None)


def test_a_path_reply_yields_one_edge_per_hop():
    recs = _parse(PATH_CALL, PATH_OUT)
    edges = [r for r in recs if r["kind"] == "edge"]
    assert [e["label"] for e in edges] == ["Beta", "gamma()"]
    assert [e["relation"] for e in edges] == ["references", "imports"]


def test_a_path_edge_claims_no_location_the_vendor_did_not_print():
    edges = [r for r in _parse(PATH_CALL, PATH_OUT) if r["kind"] == "edge"]
    assert all(e["path"] is None and e["start"] is None for e in edges)


def test_a_path_reply_leaves_nothing_unparsed():
    assert [r for r in _parse(PATH_CALL, PATH_OUT) if r["kind"] == "unparsed"] == []


def test_a_path_reply_names_no_place_so_it_can_never_score_a_hit():
    assert [r for r in _parse(PATH_CALL, PATH_OUT) if r["kind"] == "place"] == []


def test_a_path_line_is_not_read_out_of_a_query_reply():
    query = {"n": 1, "name": "query", "argv": ["/g", "query", "x"], "system_call": True}
    recs = _parse(query, "  alpha() --references [EXTRACTED]--> Beta\n")
    assert [r["kind"] for r in recs] == ["unparsed"]


EXPLAIN_CALL = {"n": 1, "name": "explain", "argv": ["/g", "explain", "IThing"], "system_call": True}
EXPLAIN_OUT = (
    "Node: IThing\n"
    "  ID:        pkg_ithing\n"
    "  Source:    pkg/interfaces.py L21\n"
    "  Type:      code\n"
    "  Degree:    53\n"
    "\n"
    "Connections (53):\n"
    "  <-- Impl [inherits] [EXTRACTED] pkg/impl.py:L36\n"
    "  ... and 33 more\n"
    "  Grouped by file:\n"
    "    --> pkg/interfaces.py: 18 connections\n"
    "    <-- pkg/service.py: 3 connections\n"
    "    ... and 2 more files\n"
)


def test_an_explain_reply_leaves_nothing_unparsed():
    assert [r for r in _parse(EXPLAIN_CALL, EXPLAIN_OUT) if r["kind"] == "unparsed"] == []


def test_a_grouped_line_names_a_file_and_never_a_place():
    recs = _parse(EXPLAIN_CALL, EXPLAIN_OUT)
    files = [r for r in recs if r["kind"] == "file"]
    assert [f["path"] for f in files] == ["pkg/interfaces.py", "pkg/service.py"]
    # inv: a file record carries no line, so it can never score a hit
    assert all(f["start"] is None for f in files)


def test_the_truncation_notice_is_no_record():
    recs = _parse(EXPLAIN_CALL, EXPLAIN_OUT)
    assert not any("33 more" in str(r.get("text") or r.get("label") or "") for r in recs)


def test_the_grouped_lists_own_truncation_notice_is_no_record():
    recs = _parse(EXPLAIN_CALL, EXPLAIN_OUT)
    assert not any("2 more files" in str(r.get("text") or r.get("label") or "") for r in recs)


def test_the_source_line_is_still_the_only_place_of_an_explain_reply():
    places = [r for r in _parse(EXPLAIN_CALL, EXPLAIN_OUT) if r["kind"] == "place"]
    assert [(p["path"], p["start"]) for p in places] == [("pkg/interfaces.py", 21)]


def test_a_grouped_line_is_not_read_out_of_a_query_reply():
    query = {"n": 1, "name": "query", "argv": ["/g", "query", "x"], "system_call": True}
    assert [r["kind"] for r in _parse(query, "    --> pkg/a.py: 3 connections\n")] == ["unparsed"]


NO_SOURCE_OUT = (
    "Node: Assignment\n"
    "  ID:        pkg_assignment\n"
    "  Source:\n"
    "  Type:      code\n"
    "  Degree:    8\n"
)


def test_a_node_the_vendor_has_no_location_for_names_no_place():
    # why: the vendor prints an empty Source for a node it cannot point at; that names no place,
    # and leaving it unparsed fails the scoring of the whole run
    recs = _parse(EXPLAIN_CALL, NO_SOURCE_OUT)
    assert recs == []


def test_a_source_line_in_a_shape_the_adapter_does_not_know_is_still_unparsed():
    recs = _parse(EXPLAIN_CALL, "Node: Thing\n  Source:    somewhere-without-a-line\n")
    assert [r["kind"] for r in recs] == ["unparsed"]


BACK_PATH_OUT = (
    "Shortest path (2 hops):\n"
    "  ArchiveUseCase <--imports [EXTRACTED]-- router.py\n"
    "  router.py --references [EXTRACTED]--> Beta\n"
)


def test_a_hop_the_vendor_prints_backwards_is_still_an_edge():
    # why: the vendor prints a hop in whichever direction the edge runs; only the forward form
    # was known, so a path containing a backward hop failed the scoring of the whole run
    recs = _parse(PATH_CALL, BACK_PATH_OUT)
    assert [r["kind"] for r in recs] == ["edge", "edge"]
    assert [r["relation"] for r in recs] == ["imports", "references"]


def test_a_backward_hop_leaves_nothing_unparsed():
    assert [r for r in _parse(PATH_CALL, BACK_PATH_OUT) if r["kind"] == "unparsed"] == []


def test_graphify_search_places_from_json():
    recs = gs.parse({"argv": ["/x/graphify-search", "query", "alpha doubles"]},
                    (FIX / "graphify_search/query.out").read_text())
    assert [r["kind"] for r in recs] == ["place"] * 3
    assert recs[0]["rank"] == 1
    assert recs[0]["end"] is None
    assert recs[0]["qualified_name"] is None
    assert {r["symbol"] for r in recs} >= {"alpha"}
    assert set(recs[0]) == {"kind", "rank", "path", "label", "symbol", "qualified_name", "start", "end"}


def test_graphify_search_version_and_non_json():
    assert gs.parse({"argv": ["/x/graphify-search", "--version"]}, "graphify-search 0.4.0\n") == []
    assert gs.parse({"argv": ["/x/graphify-search", "query", "q"]}, "not json") == [{"kind": "unparsed",
                                                                                     "text": "not json"}]


def test_graphify_search_refuses_a_mode_outside_the_configuration():
    text = (FIX / "graphify_search/query.out").read_text().replace('"mode":"dense"', '"mode":"bm25"', 1)
    call = {"argv": ["/x/graphify-search", "query", "alpha doubles"]}
    assert gs.parse(call, text, search_modes=["dense"]) == [{"kind": "unparsed", "text": text}]
    assert len(gs.parse(call, text, search_modes=["dense", "bm25"])) == 3
    assert len(gs.parse(call, text)) == 3


def test_graphify_search_malformed_row_refuses_the_whole_answer():
    text = '{"mode":"dense","results":[{"rank":1,"path":"app/main.py","symbol":"alpha","start":5},{"rank":2}]}'
    assert gs.parse({"argv": ["/x/graphify-search", "query", "q"]}, text) == [{"kind": "unparsed", "text": text}]
