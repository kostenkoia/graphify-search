import sys
import types

import numpy as np
import pytest

from graphify_search import embed
from graphify_search.embed import (
    BATCH_SIZE,
    EmbeddingClient,
    LocalEmbeddingClient,
)
from graphify_search.errors import EndpointUnavailableError
from tests.embed_stub import serve, serve_redirect, vector


def client(url, model="m"):
    return EmbeddingClient(url, model, doc_prefix="search_document: ", query_prefix="search_query: ")


def test_documents_are_prefixed_batched_and_normalised():
    with serve() as (url, calls):
        vecs = client(url).embed_documents([f"t{i}" for i in range(BATCH_SIZE + 1)])
    assert vecs.shape == (BATCH_SIZE + 1, 3)
    assert vecs.dtype == np.float32
    assert [len(c["input"]) for c in calls] == [BATCH_SIZE, 1]
    assert calls[0]["input"][0] == "search_document: t0"
    assert calls[0]["model"] == "m"
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(vecs[0], np.array(vector("search_document: t0"), dtype=np.float32), atol=1e-6)


def test_query_uses_the_query_prefix():
    with serve() as (url, calls):
        v = client(url).embed_query("what is x")
    assert calls[0]["input"] == ["search_query: what is x"]
    assert v.shape == (3,)


def test_unreachable_endpoint_raises_endpoint_unavailable():
    with pytest.raises(EndpointUnavailableError) as e:
        client("http://127.0.0.1:9/v1").embed_query("x")
    assert "127.0.0.1:9" in str(e.value)
    assert e.value.hint


def test_http_error_raises_endpoint_unavailable():
    with serve() as (url, _), pytest.raises(EndpointUnavailableError):
        client(url, model="broken").embed_query("x")


def test_empty_input_needs_no_request():
    with serve() as (url, calls):
        assert client(url).embed_documents([]).shape == (0, 0)
    assert calls == []


def test_rows_follow_the_response_index_not_its_order():
    with serve(reverse=True) as (url, _):
        vecs = client(url).embed_documents(["t0", "t1"])
    np.testing.assert_allclose(vecs[0], np.array(vector("search_document: t0"), dtype=np.float32), atol=1e-6)


def test_short_response_raises_endpoint_unavailable():
    with serve(drop_last=True) as (url, _), pytest.raises(EndpointUnavailableError):
        client(url).embed_documents(["t0", "t1"])


def test_an_api_key_becomes_an_authorization_header():
    with serve() as (url, calls):
        EmbeddingClient(url, "m", "search_document: ", "search_query: ", api_key="k-1").embed_query("x")
    assert calls[0]["authorization"] == "Bearer k-1"
    with serve() as (url, calls):
        client(url).embed_query("x")
    assert calls[0]["authorization"] is None


class _FakeSentenceTransformer:
    seen = {}
    loads = 0

    def __init__(self, name, device=None):
        _FakeSentenceTransformer.seen = {"name": name, "device": device}
        _FakeSentenceTransformer.loads += 1
        self.name = name

    def encode(self, texts, **kwargs):
        _FakeSentenceTransformer.seen["kwargs"] = kwargs
        _FakeSentenceTransformer.seen["texts"] = list(texts)
        rows = [[1.0, 0.0, 0.0] for _ in texts]
        return np.asarray(rows, dtype=np.float64)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer
    _FakeSentenceTransformer.seen = {}
    _FakeSentenceTransformer.loads = 0
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return _FakeSentenceTransformer


def test_the_local_client_loads_the_named_model_on_the_cpu_at_the_first_encode(fake_sentence_transformers):
    c = LocalEmbeddingClient("a-model", doc_prefix="", query_prefix="")
    assert (c.endpoint, c.model, c.doc_prefix, c.query_prefix) == ("local", "a-model", "", "")
    assert fake_sentence_transformers.loads == 0
    c.embed_query("x")
    assert fake_sentence_transformers.seen["name"] == "a-model"
    assert fake_sentence_transformers.seen["device"] == "cpu"


def test_the_local_client_loads_the_model_once_for_many_encodes(fake_sentence_transformers):
    c = LocalEmbeddingClient("a-model", doc_prefix="", query_prefix="")
    c.embed_documents(["t0"])
    c.embed_query("q")
    assert fake_sentence_transformers.loads == 1


def test_the_local_client_encodes_with_prefixes_and_returns_float32(fake_sentence_transformers):
    c = LocalEmbeddingClient("a-model", doc_prefix="d: ", query_prefix="q: ")
    vecs = c.embed_documents(["t0", "t1"])
    assert vecs.shape == (2, 3)
    assert vecs.dtype == np.float32
    assert fake_sentence_transformers.seen["texts"] == ["d: t0", "d: t1"]
    kwargs = fake_sentence_transformers.seen["kwargs"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["batch_size"] == BATCH_SIZE
    assert kwargs["show_progress_bar"] is False
    assert c.embed_query("what is x").shape == (3,)
    assert fake_sentence_transformers.seen["texts"] == ["q: what is x"]


def test_the_local_client_answers_the_empty_shape_without_encoding(fake_sentence_transformers):
    c = LocalEmbeddingClient("a-model", doc_prefix="", query_prefix="")
    assert c.embed_documents([]).shape == (0, 0)
    assert "texts" not in fake_sentence_transformers.seen
    assert fake_sentence_transformers.loads == 0


def test_a_missing_sentence_transformers_is_an_unavailable_endpoint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    c = LocalEmbeddingClient("a-model", doc_prefix="", query_prefix="")
    with pytest.raises(EndpointUnavailableError) as e:
        c.embed_query("x")
    assert "sentence-transformers is not installed" in str(e.value)
    assert e.value.hint == "pip install 'graphify-search[local]'"


def test_a_model_that_cannot_be_loaded_is_an_unavailable_endpoint(monkeypatch):
    module = types.ModuleType("sentence_transformers")

    def boom(name, device=None):
        raise OSError(f"{name} is not here")

    module.SentenceTransformer = boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    c = LocalEmbeddingClient("a-model", doc_prefix="", query_prefix="")
    with pytest.raises(EndpointUnavailableError) as e:
        c.embed_documents(["t"])
    assert "no local model a-model" in str(e.value)
    assert e.value.hint


@pytest.mark.parametrize("endpoint", ["file:///etc", "ftp://host/v1", "localhost:1234/v1", ""])
def test_a_non_http_endpoint_is_refused_before_any_request(endpoint):
    with pytest.raises(EndpointUnavailableError, match="http or https"):
        EmbeddingClient(endpoint, "m", "", "")


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:9/v1", "https://api.example.invalid/v1/"])
def test_an_http_endpoint_is_admitted(endpoint):
    assert EmbeddingClient(endpoint, "m", "", "").endpoint == endpoint.rstrip("/")


def test_a_redirect_is_refused_and_the_second_host_is_never_called():
    with serve() as (target, target_calls):
        with serve_redirect(f"{target}/embeddings") as (url, hops), \
             pytest.raises(EndpointUnavailableError) as e:
            EmbeddingClient(url, "m", "", "", api_key="k-1").embed_query("x")
        assert target_calls == []
    assert len(hops) == 1
    assert f"{target}/embeddings" in str(e.value)
    assert "redirects are not followed" in str(e.value)
    assert e.value.hint == "point --endpoint at the final URL"


def test_a_response_larger_than_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(embed, "MAX_RESPONSE_BYTES", 10)
    with serve() as (url, _), pytest.raises(EndpointUnavailableError) as e:
        client(url).embed_query("x")
    assert "embedding response exceeds 10 bytes" in str(e.value)


def test_a_non_finite_embedding_is_refused():
    with serve(non_finite=True) as (url, _), pytest.raises(EndpointUnavailableError) as e:
        client(url).embed_query("x")
    assert "non-finite embedding" in str(e.value)


def test_an_embedding_that_overflows_float32_is_refused():
    # why: 1e40 is finite in the float64 the JSON body decodes to and infinite once cast, so it
    # is the value that tells a check made after the cast from one made before it
    with serve(row=[1e40, 1.0, 1.0]) as (url, _), pytest.raises(EndpointUnavailableError) as e:
        client(url).embed_documents(["t0", "t1"])
    assert "non-finite embedding" in str(e.value)


def test_an_all_zero_embedding_is_refused():
    with serve(row=[0.0, 0.0, 0.0]) as (url, _), pytest.raises(EndpointUnavailableError) as e:
        client(url).embed_query("x")
    assert "all-zero embedding" in str(e.value)


@pytest.mark.parametrize("endpoint", ["http://[::1/v1", "http://[oops]/v1"])
def test_an_endpoint_the_url_parser_refuses_is_an_unavailable_endpoint(endpoint):
    with pytest.raises(EndpointUnavailableError) as e:
        EmbeddingClient(endpoint, "m", "", "")
    assert endpoint in str(e.value)
    assert e.value.hint == "check --endpoint, GRAPHIFY_SEARCH_ENDPOINT or config.json"
