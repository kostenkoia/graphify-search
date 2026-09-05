import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from graphify_search import index as ix
from graphify_search.embed import EmbeddingClient
from graphify_search.errors import EndpointUnavailableError, InputError
from tests.embed_stub import serve

FIX = Path(__file__).parent / "fixtures" / "graph_small"


@pytest.fixture
def corpus(tmp_path):
    shutil.copytree(FIX, tmp_path / "c")
    return tmp_path / "c" / "graph.json", tmp_path / "c" / "source"


def client(url):
    return EmbeddingClient(url, "m", "search_document: ", "search_query: ")


def test_build_writes_rows_vectors_bm25_and_manifest(corpus):
    graph, src = corpus
    with serve() as (url, calls):
        rec = ix.build_index(graph, src, client(url))
    d = ix.index_dir(graph)
    assert sorted(p.name for p in d.iterdir()) == ["bm25.json", "manifest.json", "nodes.jsonl", "vectors.npy"]
    rows = [json.loads(line) for line in (d / "nodes.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["app_init", "app_main", "app_main_alpha", "app_main_beta", "app_main_gamma",
                                       "app_main_gamma_run", "app_main_trailing", "web_usething",
                                       "web_usething_usething", "docs_guide"]
    alpha = rows[2]
    assert alpha["kind"] == "symbol"
    assert alpha["symbol"] == "alpha"
    assert alpha["start"] == 5
    assert "end_hint" not in alpha
    assert alpha["snippet"].startswith("def alpha(x):\n")
    assert alpha["edges"][0] == {"rel": "calls", "to": "beta()", "at": "L10"}
    assert alpha["edges"][2] == {"rel": "references", "to": "useThing()", "at": "web/useThing.ts:L4"}
    assert rows[9]["kind"] == "document"
    assert "end_hint" not in rows[9]
    assert rows[9]["snippet"] == ""
    assert np.load(d / "vectors.npy").shape == (10, 3)
    assert rec["nodes"] == 10
    assert rec["embedded"] == 10
    assert rec["reused"] == 0
    assert rec["vectors"] == "present"
    assert sum(len(c["input"]) for c in calls) == 10
    m = ix.Manifest.from_json((d / "manifest.json").read_text())
    assert m.rows == 10
    assert m.dims == 3
    assert m.doc_prefix == "search_document: "
    assert set(m.files) == {"app/__init__.py", "app/main.py", "web/useThing.ts", "docs/guide.md"}


def test_docstring_is_in_the_embedded_text(corpus):
    graph, src = corpus
    with serve() as (url, calls):
        ix.build_index(graph, src, client(url))
    alpha_text = [t for c in calls for t in c["input"] if t.startswith("search_document: alpha()")][0]
    assert "Compute alpha from x." in alpha_text
    assert "def alpha(x):" in alpha_text


def test_refresh_reembeds_only_changed_files(corpus):
    graph, src = corpus
    with serve() as (url, calls):
        ix.build_index(graph, src, client(url))
        (src / "app" / "main.py").write_text((src / "app" / "main.py").read_text().replace("y + 1", "y + 2"))
        rec = ix.build_index(graph, src, client(url))
    assert rec["embedded"] == 6        # six rows live in app/main.py
    assert rec["reused"] == 4
    assert sum(len(c["input"]) for c in calls) == 16


def test_model_change_forces_full_rebuild(corpus):
    graph, src = corpus
    with serve() as (url, calls):
        ix.build_index(graph, src, client(url))
        other = EmbeddingClient(url, "m2", "search_document: ", "search_query: ")
        rec = ix.build_index(graph, src, other)
    assert rec["embedded"] == 10
    assert rec["reused"] == 0


def test_endpoint_down_writes_bm25_only_unless_required(corpus):
    graph, src = corpus
    dead = EmbeddingClient("http://127.0.0.1:9/v1", "m", "search_document: ", "search_query: ")
    rec = ix.build_index(graph, src, dead)
    assert rec["vectors"] == "absent"
    assert not (ix.index_dir(graph) / "vectors.npy").exists()
    assert ix.load_index(graph).bm25.rank(["alpha"])
    with pytest.raises(EndpointUnavailableError):
        ix.build_index(graph, src, dead, require_dense=True)


def test_load_refuses_missing_or_inconsistent_index(corpus):
    graph, src = corpus
    with pytest.raises(InputError) as e:
        ix.load_index(graph)
    assert "graphify-search index" in e.value.hint
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    np.save(ix.index_dir(graph) / "vectors.npy", np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(InputError) as e:
        ix.load_index(graph)
    assert "--full" in e.value.hint


def test_load_refuses_index_whose_vectors_have_the_wrong_width(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    np.save(ix.index_dir(graph) / "vectors.npy", np.zeros((10, 5), dtype=np.float32))
    with pytest.raises(InputError) as e:
        ix.load_index(graph)
    assert "--full" in e.value.hint


def test_full_rebuild_on_dims_change(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    with serve(dims=4) as (url2, _):
        rec = ix.build_index(graph, src, client(url2))
    assert rec["embedded"] == 10
    assert rec["reused"] == 0
    assert np.load(ix.index_dir(graph) / "vectors.npy").shape == (10, 4)


def test_full_rebuild_reports_dropped_ids_from_previous_index(corpus):
    graph, src = corpus
    ix.build_index(graph, src, None)
    raw = json.loads(graph.read_text())
    raw["nodes"] = [n for n in raw["nodes"] if n["id"] != "docs_guide"]
    graph.write_text(json.dumps(raw))
    rec = ix.build_index(graph, src, None, full=True)
    assert rec["dropped"] == ["docs_guide"]


def test_noop_refresh_survives_a_dead_endpoint(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    # why: the endpoint is unchanged and only its server is gone, which is what an outage looks like
    dead = client(url)
    rec = ix.build_index(graph, src, dead)
    assert rec["vectors"] == "present"
    assert rec["reused"] == 10
    assert np.load(ix.index_dir(graph) / "vectors.npy").shape == (10, 3)
    rec2 = ix.build_index(graph, src, dead, require_dense=True)
    assert rec2["vectors"] == "present"


def test_load_refuses_one_dimensional_vectors(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    np.save(ix.index_dir(graph) / "vectors.npy", np.zeros(10, dtype=np.float32))
    with pytest.raises(InputError) as e:
        ix.load_index(graph)
    assert "--full" in e.value.hint


def test_source_file_outside_the_root_yields_no_body_and_no_sha(tmp_path):
    root = tmp_path / "a" / "b"
    root.mkdir(parents=True)
    (root / "inside.py").write_text("def gamma():\n    return 1\n")
    (tmp_path / "outside.txt").write_text("TOP-SECRET-LINE-1\nTOP-SECRET-LINE-2\n")
    (tmp_path / "graphify-out").mkdir()
    graph = tmp_path / "graphify-out" / "graph.json"
    graph.write_text(json.dumps({"nodes": [
        {"id": "up", "label": "alpha()", "file_type": "code", "source_file": "../../outside.txt",
         "source_location": "L1", "community": 0},
        {"id": "abs", "label": "beta()", "file_type": "code", "source_file": str(tmp_path / "outside.txt"),
         "source_location": "L1", "community": 0},
        # inv: one readable file keeps the build past the check for a source root that holds none
        {"id": "in", "label": "gamma()", "file_type": "code", "source_file": "inside.py",
         "source_location": "L1", "community": 0},
    ], "links": []}))
    ix.build_index(graph, root, None)
    loaded = ix.load_index(graph)
    assert [r.snippet for r in loaded.rows][:2] == ["", ""]
    assert "TOP-SECRET" not in (loaded.rows[0].snippet + loaded.rows[1].snippet)
    assert loaded.manifest.files["../../outside.txt"] == ""
    assert loaded.manifest.files[str(tmp_path / "outside.txt")] == ""


def test_refresh_outage_keeps_the_previous_dense_index(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    (src / "app" / "main.py").write_text((src / "app" / "main.py").read_text().replace("y + 1", "y + 2"))
    # why: the endpoint is unchanged and only its server is gone, which is what an outage looks like
    dead = client(url)
    with pytest.raises(EndpointUnavailableError) as e:
        ix.build_index(graph, src, dead)
    assert "previous index is kept" in e.value.hint
    assert np.load(ix.index_dir(graph) / "vectors.npy").shape == (10, 3)
    assert ix.load_index(graph).manifest.vectors == "present"


def test_load_refuses_a_bm25_corpus_that_disagrees_with_the_rows(corpus):
    graph, src = corpus
    ix.build_index(graph, src, None)
    (ix.index_dir(graph) / "bm25.json").write_text(
        json.dumps({"ids": ["a", "b"], "tf": [{"x": 1}, {"y": 1}], "df": {"x": 1, "y": 1}}))
    with pytest.raises(InputError) as e:
        ix.load_index(graph)
    assert "--full" in e.value.hint


def test_a_class_body_reaches_past_its_first_method(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    rows = {json.loads(line)["id"]: json.loads(line)
            for line in (ix.index_dir(graph) / "nodes.jsonl").read_text().splitlines()}
    gamma = rows["app_main_gamma"]
    assert gamma["start"] == 17
    # inv: the assertion lands on line 21, past the `def run(self):` on line 20 that a
    # four-line window from the class start already reaches
    assert "return alpha(3)" in gamma["snippet"]


def test_an_index_written_with_an_end_hint_column_is_refused_then_rebuilt(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    nodes = ix.index_dir(graph) / "nodes.jsonl"
    old = [json.loads(line) for line in nodes.read_text().splitlines()]
    nodes.write_text("".join(json.dumps({**r, "end_hint": 9}) + "\n" for r in old))
    with pytest.raises(InputError, match="no usable index"):
        ix.load_index(graph)
    with serve() as (url, _):
        rec = ix.build_index(graph, src, client(url))
    assert rec["embedded"] == 10
    assert "end_hint" not in json.loads(nodes.read_text().splitlines()[0])


def test_build_refuses_a_graph_with_duplicate_node_ids(corpus):
    graph, src = corpus
    raw = json.loads(graph.read_text())
    raw["nodes"].append(dict(raw["nodes"][2]))
    graph.write_text(json.dumps(raw))
    with pytest.raises(InputError) as e:
        ix.build_index(graph, src, None)
    assert "duplicate node id app_main_alpha" in str(e.value)
    assert not ix.index_dir(graph).exists()


def test_build_refuses_a_graph_with_no_places(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"nodes": [
        {"id": "c", "label": "Doubling", "file_type": "concept",
         "source_file": "docs/guide.md", "source_location": "L3"},
    ], "links": []}))
    with pytest.raises(InputError) as e:
        ix.build_index(graph, tmp_path, None)
    assert "no places to index" in str(e.value)
    assert not ix.index_dir(graph).exists()


def _lock_holding(graph, text):
    lock = ix.index_dir(graph) / "lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(text)
    return lock


def test_build_refuses_to_run_while_another_holds_the_lock(corpus):
    graph, src = corpus
    lock = _lock_holding(graph, f"{os.getpid()}\n")
    with pytest.raises(InputError) as e:
        ix.build_index(graph, src, None)
    assert "another `graphify-search index` is running" in str(e.value)
    assert f"pid {os.getpid()}" in str(e.value)
    assert str(lock) in e.value.hint
    lock.unlink()
    assert ix.build_index(graph, src, None)["nodes"] == 10


def test_a_lock_naming_a_dead_builder_is_taken_over(corpus):
    graph, src = corpus
    done = subprocess.Popen(["true"])
    done.wait()
    lock = _lock_holding(graph, f"{done.pid}\n")
    assert ix.build_index(graph, src, None)["nodes"] == 10
    assert not lock.exists()


@pytest.mark.parametrize("text", ["", "not-a-pid\n", "0\n", "-1\n"])
def test_a_lock_naming_no_live_process_is_taken_over(corpus, text):
    graph, src = corpus
    lock = _lock_holding(graph, text)
    assert ix.build_index(graph, src, None)["nodes"] == 10
    assert not lock.exists()


class _LockReader:
    endpoint = "http://127.0.0.1:9/v1"
    model = "m"
    doc_prefix = ""
    query_prefix = ""

    def __init__(self, lock):
        self.lock = lock
        self.seen = None

    def embed_documents(self, texts):
        self.seen = self.lock.read_text()
        return np.zeros((len(texts), 3), dtype=np.float32)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_the_lock_names_the_builder_that_holds_it(corpus):
    graph, src = corpus
    reader = _LockReader(ix.index_dir(graph) / "lock")
    assert ix.build_index(graph, src, reader)["nodes"] == 10
    assert int(reader.seen.strip()) == os.getpid()


def test_a_build_that_fails_leaves_no_lock_behind(corpus):
    graph, src = corpus
    dead = EmbeddingClient("http://127.0.0.1:9/v1", "m", "search_document: ", "search_query: ")
    with pytest.raises(EndpointUnavailableError):
        ix.build_index(graph, src, dead, require_dense=True)
    assert not (ix.index_dir(graph) / "lock").exists()
    with serve() as (url, _):
        assert ix.build_index(graph, src, client(url))["nodes"] == 10
    assert not (ix.index_dir(graph) / "lock").exists()


def test_endpoint_change_forces_a_full_reembed(corpus):
    graph, src = corpus
    with serve() as (url, _):
        ix.build_index(graph, src, client(url))
    with serve() as (url2, _):
        rec = ix.build_index(graph, src, client(url2))
    assert rec["embedded"] == 10
    assert rec["reused"] == 0


def test_a_fresh_build_names_why_it_holds_no_vectors(corpus):
    graph, src = corpus
    rec = ix.build_index(graph, src, client("http://127.0.0.1:9/v1"))
    assert rec["vectors"] == "absent"
    assert "127.0.0.1:9" in rec["embedding_error"]


def test_a_build_without_a_client_carries_no_embedding_error(corpus):
    graph, src = corpus
    rec = ix.build_index(graph, src, None)
    assert rec["vectors"] == "absent"
    assert "embedding_error" not in rec


def test_an_embedding_that_overflows_float32_leaves_no_nan_in_the_index(corpus):
    graph, src = corpus
    with serve(row=[1e40, 1.0, 1.0]) as (url, _):
        rec = ix.build_index(graph, src, client(url))
    assert rec["vectors"] == "absent"
    assert "non-finite embedding" in rec["embedding_error"]
    assert not (ix.index_dir(graph) / "vectors.npy").exists()


def _graph_with(tmp_path, nodes, links=None):
    graph = tmp_path / "graphify-out" / "graph.json"
    graph.parent.mkdir(parents=True, exist_ok=True)
    graph.write_text(json.dumps({"nodes": nodes, "links": links or []}))
    return graph


def test_build_refuses_a_source_root_that_holds_none_of_the_paths(tmp_path):
    graph = _graph_with(tmp_path, [
        {"id": "m", "label": "main.py", "file_type": "code", "source_file": "pkg/main.py",
         "source_location": "L1", "community": 0},
        {"id": "a", "label": "alpha()", "file_type": "code", "source_file": "pkg/main.py",
         "source_location": "L4", "community": 0},
    ])
    root = tmp_path / "elsewhere"
    root.mkdir()
    with pytest.raises(InputError) as e:
        ix.build_index(graph, root, None)
    assert "no source file was found under" in str(e.value)
    assert "--source" in e.value.hint
    assert not ix.index_dir(graph).exists()


def test_a_graph_of_documents_alone_builds_without_any_source_file(tmp_path):
    graph = _graph_with(tmp_path, [
        {"id": "d", "label": "guide.md", "file_type": "document", "source_file": "docs/guide.md",
         "source_location": "L1", "community": 0},
    ])
    root = tmp_path / "elsewhere"
    root.mkdir()
    rec = ix.build_index(graph, root, None)
    assert rec["nodes"] == 1
    assert rec["files_read"] == 0


def test_files_read_counts_the_distinct_code_paths_read(corpus):
    graph, src = corpus
    rec = ix.build_index(graph, src, None)
    # inv: the fixture's code nodes live in three files; docs/guide.md is a document, which is read
    # for no body
    assert rec["files_read"] == 3
    (src / "web" / "useThing.ts").unlink()
    assert ix.build_index(graph, src, None)["files_read"] == 2


def test_dropped_ids_are_capped_and_counted_in_the_record(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text("".join(f"def f{i}():\n    return {i}\n" for i in range(41)))
    keep = {"id": "keep", "label": "m.py", "file_type": "code", "source_file": "m.py",
            "source_location": "L1", "community": 0}
    gone = [{"id": f"n{i}", "label": f"f{i}()", "file_type": "code", "source_file": "m.py",
             "source_location": f"L{2 * i + 1}", "community": 0} for i in range(40)]
    graph = _graph_with(tmp_path, [keep, *gone])
    assert ix.build_index(graph, src, None)["nodes"] == 41
    graph.write_text(json.dumps({"nodes": [keep], "links": []}))
    rec = ix.build_index(graph, src, None)
    assert rec["dropped_total"] == 40
    assert len(rec["dropped"]) == 20
    assert rec["dropped"] == sorted(f"n{i}" for i in range(40))[:20]


def test_a_short_dropped_list_is_complete(corpus):
    graph, src = corpus
    ix.build_index(graph, src, None)
    raw = json.loads(graph.read_text())
    raw["nodes"] = [n for n in raw["nodes"] if n["id"] != "docs_guide"]
    graph.write_text(json.dumps(raw))
    rec = ix.build_index(graph, src, None, full=True)
    assert rec["dropped"] == ["docs_guide"]
    assert rec["dropped_total"] == 1


def test_a_document_row_carries_no_symbol(tmp_path):
    src = tmp_path / "src"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "changes.md").write_text("# Fixed\n\nA heading that reads as an identifier.\n")
    graph = _graph_with(tmp_path, [
        {"id": "d", "label": "Fixed", "file_type": "document", "source_file": "docs/changes.md",
         "source_location": "L1", "community": 0},
    ])
    ix.build_index(graph, src, None)
    row = ix.load_index(graph).rows[0]
    assert row.kind == "document"
    assert row.symbol is None


def test_a_row_for_an_empty_source_file_is_dropped(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "empty.py").write_text("")
    (src / "empty.md").write_text("")
    (src / "full.py").write_text("def gamma():\n    return 1\n")
    graph = _graph_with(tmp_path, [
        {"id": "empty", "label": "empty.py", "file_type": "code", "source_file": "empty.py",
         "source_location": "L1", "community": 0},
        {"id": "missing", "label": "gone.py", "file_type": "code", "source_file": "gone.py",
         "source_location": "L1", "community": 0},
        {"id": "full", "label": "full.py", "file_type": "code", "source_file": "full.py",
         "source_location": "L1", "community": 0},
        {"id": "doc", "label": "empty.md", "file_type": "document", "source_file": "empty.md",
         "source_location": "L1", "community": 0},
    ])
    rec = ix.build_index(graph, src, None)
    assert [r.id for r in ix.load_index(graph).rows] == ["missing", "full", "doc"]
    assert rec["nodes"] == 3
