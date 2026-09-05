import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from graphify_search import cli, settings
from graphify_search.embed import EmbeddingClient, LocalEmbeddingClient
from tests.embed_stub import serve

FIX = Path(__file__).parent / "fixtures" / "graph_small"


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copytree(FIX / "source", tmp_path / "src")
    (tmp_path / "src" / "graphify-out").mkdir()
    shutil.copy(FIX / "graph.json", tmp_path / "src" / "graphify-out" / "graph.json")
    monkeypatch.chdir(tmp_path / "src")
    monkeypatch.delenv("GRAPHIFY_OUT", raising=False)
    monkeypatch.delenv("GRAPHIFY_SEARCH_ENDPOINT", raising=False)
    return tmp_path / "src"


def out(capsys):
    o = capsys.readouterr()
    return o.out, o.err


def test_index_then_query_json_on_stdout(project, capsys):
    with serve() as (url, _):
        assert cli.main(["index", "--endpoint", url, "--model", "m"]) == 0
        rec = json.loads(out(capsys)[0])
        assert rec["nodes"] == 10
        assert rec["vectors"] == "present"
        assert cli.main(["query", "alpha doubles", "--endpoint", url, "--model", "m", "-k", "3"]) == 0
    text, err = out(capsys)
    assert err == ""
    assert text.endswith("\n")
    doc = json.loads(text)
    assert doc["mode"] == "dense"
    assert len(doc["results"]) == 3
    assert doc["index"]["stale"] is False


def test_query_without_index_refuses(project, capsys):
    assert cli.main(["query", "x"]) == 1
    text, err = out(capsys)
    assert text == ""
    assert "hint: run `graphify-search index`" in err


def test_query_bm25_when_endpoint_down_and_require_dense_refuses(project, capsys):
    assert cli.main(["index", "--endpoint", "http://127.0.0.1:9/v1"]) == 0
    assert json.loads(out(capsys)[0])["vectors"] == "absent"
    assert cli.main(["query", "alpha"]) == 0
    assert json.loads(out(capsys)[0])["mode"] == "bm25"
    assert cli.main(["query", "alpha", "--require-dense"]) == 1
    assert out(capsys)[0] == ""


def test_status_and_version(project, capsys):
    assert cli.main(["index", "--endpoint", "http://127.0.0.1:9/v1"]) == 0
    out(capsys)
    assert cli.main(["status"]) == 0
    doc = json.loads(out(capsys)[0])
    assert doc["vectors"] == "absent"
    assert doc["stale"] is False
    assert doc["rows"] == 10
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert out(capsys)[0].startswith("graphify-search ")


def test_query_never_writes(project, capsys):
    assert cli.main(["index", "--endpoint", "http://127.0.0.1:9/v1"]) == 0
    out(capsys)
    before = {p: p.stat().st_mtime_ns for p in (project / "graphify-out").rglob("*") if p.is_file()}
    assert cli.main(["query", "alpha"]) == 0
    after = {p: p.stat().st_mtime_ns for p in (project / "graphify-out").rglob("*") if p.is_file()}
    assert before == after


def test_a_missing_graph_is_reported_before_a_missing_index(project, capsys):
    (project / "graphify-out" / "graph.json").unlink()
    assert cli.main(["query", "alpha"]) == 1
    text, err = out(capsys)
    assert text == ""
    assert "no graph at" in err
    assert "graph.json" in err
    assert "no usable index" not in err


def test_a_stale_model_or_width_is_refused_on_stderr(project, capsys):
    with serve() as (url, _):
        assert cli.main(["index", "--endpoint", url, "--model", "m"]) == 0
        out(capsys)
        assert cli.main(["query", "alpha", "--endpoint", url, "--model", "other"]) == 1
    text, err = out(capsys)
    assert text == ""
    assert "index was built with m" in err
    assert "hint: run `graphify-search index --full`" in err
    with serve(dims=4) as (url2, _):
        assert cli.main(["query", "alpha", "--endpoint", url2, "--model", "m"]) == 1
    text, err = out(capsys)
    assert text == ""
    assert "hint: run `graphify-search index --full`" in err


def test_the_local_endpoint_selects_the_in_process_client(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda name, device=None: name
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    fields = {"model": "m", "doc_prefix": "", "query_prefix": "", "k": 10, "budget": 6000}
    assert isinstance(cli._client(settings.Settings(endpoint="local", **fields)), LocalEmbeddingClient)
    http = cli._client(settings.Settings(endpoint="http://127.0.0.1:1234/v1", **fields))
    assert isinstance(http, EmbeddingClient)
    assert http.endpoint == "http://127.0.0.1:1234/v1"


@pytest.mark.parametrize("value", ["Local", "LOCAL", " local "])
def test_the_endpoint_switch_reads_local_whatever_its_case(monkeypatch, value):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda name, device=None: name
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    fields = {"model": "m", "doc_prefix": "", "query_prefix": "", "k": 10, "budget": 6000}
    client = cli._client(settings.Settings(endpoint=value, **fields))
    assert isinstance(client, LocalEmbeddingClient)
    assert client.endpoint == "local"


def test_a_local_endpoint_without_the_extra_answers_from_bm25(project, capsys, monkeypatch):
    with serve() as (url, _):
        assert cli.main(["index", "--endpoint", url, "--model", "m"]) == 0
    assert json.loads(out(capsys)[0])["vectors"] == "present"
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert cli.main(["query", "alpha", "--endpoint", "local", "--model", "m"]) == 0
    doc = json.loads(out(capsys)[0])
    assert (doc["mode"], doc["index"]["vectors"]) == ("bm25", "unreachable")
    assert cli.main(["query", "alpha", "--endpoint", "local", "--model", "m", "--require-dense"]) == 1
    text, err = out(capsys)
    assert text == ""
    assert "sentence-transformers is not installed" in err


def test_a_local_endpoint_without_the_extra_still_writes_a_bm25_index(project, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert cli.main(["index", "--endpoint", "local", "--model", "m"]) == 0
    assert json.loads(out(capsys)[0])["vectors"] == "absent"
    assert cli.main(["index", "--endpoint", "local", "--model", "m", "--require-dense"]) == 1
    assert "sentence-transformers is not installed" in out(capsys)[1]


def test_the_query_json_names_the_endpoint_right_after_the_model(project, capsys):
    with serve() as (url, _):
        assert cli.main(["index", "--endpoint", url, "--model", "m"]) == 0
        out(capsys)
        assert cli.main(["query", "alpha", "--endpoint", url, "--model", "m"]) == 0
    doc = json.loads(out(capsys)[0])
    assert doc["endpoint"] == url
    assert list(doc)[:4] == ["question", "mode", "model", "endpoint"]


def test_the_index_record_names_the_endpoint(project, capsys):
    with serve() as (url, _):
        assert cli.main(["index", "--endpoint", url, "--model", "m"]) == 0
    assert json.loads(out(capsys)[0])["endpoint"] == url


def test_a_config_file_naming_a_remote_endpoint_refuses_before_any_request(project, capsys):
    d = project / "graphify-out" / ".graphify_search"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"endpoint": "https://evil.example.com/v1"}))
    assert cli.main(["index"]) == 1
    text, err = out(capsys)
    assert text == ""
    assert "https://evil.example.com/v1" in err
    assert ("hint: remove `endpoint` from that file; a remote server goes in "
            "GRAPHIFY_SEARCH_ENDPOINT or --endpoint") in err


def test_a_score_json_cannot_carry_is_refused_in_plain_text(project, capsys):
    with serve() as (url, _):
        assert cli.main(["index", "--endpoint", url, "--model", "m"]) == 0
        out(capsys)
        vectors = project / "graphify-out" / ".graphify_search" / "vectors.npy"
        poisoned = np.load(vectors)
        poisoned[0] = np.nan
        np.save(vectors, poisoned)
        assert cli.main(["query", "alpha", "--endpoint", url, "--model", "m"]) == 1
    text, err = out(capsys)
    assert text == ""
    assert "the answer cannot be rendered as JSON" in err
    assert "hint: run `graphify-search index --full`" in err


def test_the_index_record_spells_the_endpoint_as_the_manifest_does(project, capsys, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_SEARCH_ENDPOINT", "http://127.0.0.1:9/v1/")
    assert cli.main(["index"]) == 0
    rec = json.loads(out(capsys)[0])
    manifest = json.loads((project / "graphify-out" / ".graphify_search" / "manifest.json").read_text())
    assert rec["endpoint"] == "http://127.0.0.1:9/v1"
    assert rec["endpoint"] == manifest["endpoint"]


def test_query_exclude_is_repeatable_and_drops_matching_paths(project, capsys):
    with serve() as (url, _):
        assert cli.main(["index", "--endpoint", url, "--model", "m"]) == 0
        out(capsys)
        assert cli.main(["query", "alpha doubles", "--endpoint", url, "--model", "m"]) == 0
        plain = json.loads(out(capsys)[0])
        assert cli.main(["query", "alpha doubles", "--endpoint", url, "--model", "m",
                         "--exclude", "app/*", "--exclude", "docs/*"]) == 0
        kept = json.loads(out(capsys)[0])
        assert cli.main(["query", "alpha doubles", "--endpoint", url, "--model", "m",
                         "--exclude", "*"]) == 0
        nothing = json.loads(out(capsys)[0])
    assert any(r["path"].startswith(("app/", "docs/")) for r in plain["results"])
    assert [r["path"] for r in kept["results"]] == ["web/useThing.ts", "web/useThing.ts"]
    assert [r["rank"] for r in kept["results"]] == [1, 2]
    assert nothing["results"] == []
    assert nothing["mode"] == "dense"
    assert nothing["budget"]["used_chars"] == len("[]")
    assert nothing["index"]["vectors"] == "present"
