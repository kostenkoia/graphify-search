"""Clients that turn text into vectors."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

import numpy as np

from graphify_search.errors import EndpointUnavailableError

# NOT DERIVED: the number of texts one HTTP request carries; it buys fewer round trips than
# one request per text without a body large enough for a server to refuse
BATCH_SIZE = 64
# NOT DERIVED: the number of texts one in-process forward pass carries; it buys a padded batch
# small enough that the peak memory of a CPU pass stays flat across corpus sizes
FORWARD_BATCH = 64
# inv: the one endpoint value that names no URL; it selects the in-process backend instead
LOCAL_ENDPOINT = "local"
# NOT DERIVED: the largest embedding response this client reads; it buys a bound on the memory one
# answer can cost, well above the few megabytes a full batch of float lists occupies
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
# inv: every refusal of the configured endpoint value names the three places it can come from
_ENDPOINT_HINT = "check --endpoint, GRAPHIFY_SEARCH_ENDPOINT or config.json"


def _no_redirect_opener() -> urllib.request.OpenerDirector:
    """Return an opener that turns a redirect into an error instead of following it.

    Returns
    -------
    urllib.request.OpenerDirector
        An opener carrying no redirect handler, so a 3xx reaches the caller as an HTTPError.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (urllib.request.ProxyHandler(), urllib.request.HTTPHandler(),
                    urllib.request.HTTPSHandler(), urllib.request.HTTPDefaultErrorHandler(),
                    urllib.request.HTTPErrorProcessor()):
        opener.add_handler(handler)
    return opener


# inv: the Authorization header and the source text reach the host the operator named and no other,
# so the opener that carries them follows no redirect
_OPENER = _no_redirect_opener()


def is_local(endpoint: str) -> bool:
    """Return whether `endpoint` names the in-process backend rather than a URL.

    Parameters
    ----------
    endpoint : str
        The configured endpoint value.

    Returns
    -------
    bool
        True when the value is the local backend's name.
    """
    # inv: one rule decides this for the client choice and for the prefix defaults, so the two
    # can never disagree about what a given endpoint value means
    return endpoint.strip().casefold() == LOCAL_ENDPOINT


class Embedder(Protocol):
    """What the index and the search ask of an embedding backend."""

    endpoint: str
    model: str
    doc_prefix: str
    query_prefix: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents; one unit-length row per text, in order."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question; one unit-length row."""


class EmbeddingClient:
    """Embed texts through `<endpoint>/embeddings` with the configured prefixes."""

    def __init__(self, endpoint: str, model: str, doc_prefix: str, query_prefix: str,
                 timeout: float = 120.0, api_key: str | None = None) -> None:
        # inv: only http(s) reaches urlopen; config.json travels with a copied graph and may name
        # the endpoint, so a file:// or ftp:// value is refused here rather than opened
        try:
            scheme = urllib.parse.urlsplit(endpoint).scheme
        except ValueError as e:
            # inv: the split refuses a malformed host such as `http://[::1/v1`, which is an
            # endpoint this client cannot reach and not a fault of the process asking for it
            raise EndpointUnavailableError(f"endpoint is not a URL: {endpoint}: {e}",
                                           hint=_ENDPOINT_HINT) from e
        if scheme not in ("http", "https"):
            raise EndpointUnavailableError(f"endpoint scheme must be http or https: {endpoint}",
                                           hint=_ENDPOINT_HINT)
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # why: a local server needs no key and rejects none; a hosted one answers 401 without this
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        url = f"{self.endpoint}/embeddings"
        req = urllib.request.Request(  # noqa: S310 — __init__ admits only http(s), which is what this rule guards
            url, body, self._headers(),
        )
        try:
            with _OPENER.open(req, timeout=self.timeout) as r:
                # inv: the read stops one byte past the cap, so an endless body is refused on that
                # byte instead of being held whole in memory
                body_bytes = r.read(MAX_RESPONSE_BYTES + 1)
            if len(body_bytes) > MAX_RESPONSE_BYTES:
                raise EndpointUnavailableError(f"embedding response exceeds {MAX_RESPONSE_BYTES} bytes",
                                          hint="point --endpoint at an embeddings server")
            raw = json.loads(body_bytes)
            rows = sorted(raw["data"], key=lambda d: d["index"])
            embeddings = [list(map(float, d["embedding"])) for d in rows]
            # inv: one embedding per input, all the same width, so downstream shape checks hold
            if len(embeddings) != len(inputs):
                raise ValueError(f"{len(inputs)} inputs, {len(embeddings)} embeddings")
            if len({len(row) for row in embeddings}) > 1:
                raise ValueError("embeddings differ in length")
            return embeddings
        except urllib.error.HTTPError as e:
            # inv: a redirect is named as itself, since the header this request carries would
            # otherwise reach whatever host the Location points at
            location = e.headers.get("Location", "") if 300 <= e.code < 400 else ""
            e.close()
            if location:
                raise EndpointUnavailableError(
                    f"endpoint redirected to {location}; redirects are not followed",
                    hint="point --endpoint at the final URL") from e
            raise EndpointUnavailableError(f"no embeddings from {url}: {e}",
                                      hint="start the embeddings server or pass --endpoint") from e
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as e:
            raise EndpointUnavailableError(f"no embeddings from {url}: {e}",
                                      hint="start the embeddings server or pass --endpoint") from e

    @staticmethod
    def _normalise(rows: list[list[float]]) -> np.ndarray:
        # inv: every check below runs on the cast array, since a value finite in the float64 a JSON
        # body decodes to, such as 1e40, is an infinity in the float32 the index stores
        # why: the overflow of that cast is silenced because the refusal it leads to is this
        # command's whole report of it, and a warning on stderr would be the only other line there
        with np.errstate(over="ignore"):
            arr = np.asarray(rows, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        # inv: a NaN or an infinity poisons every cosine it is ranked against and renders as
        # invalid JSON, so it is refused here rather than stored in the index
        if not bool(np.isfinite(arr).all()):
            raise EndpointUnavailableError("endpoint returned a non-finite embedding",
                                      hint="point --endpoint at an embeddings server")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # inv: a row of length zero has no direction to rank and would divide to NaN, so it is
        # refused rather than scaled by an epsilon that hides it
        if not bool((norms > 0).all()):
            raise EndpointUnavailableError("endpoint returned an all-zero embedding",
                                      hint="point --endpoint at an embeddings server")
        # inv: rows are unit length, so a dot product is the cosine
        return arr / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents in batches; one row per text, in order."""
        rows: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            rows.extend(self._post([self.doc_prefix + t for t in texts[i:i + BATCH_SIZE]]))
        return self._normalise(rows)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question with the query prefix."""
        return self._normalise(self._post([self.query_prefix + text]))[0]


class LocalEmbeddingClient:
    """Embed texts with a sentence-transformers model loaded into this process."""

    def __init__(self, model: str, doc_prefix: str, query_prefix: str) -> None:
        self.endpoint = LOCAL_ENDPOINT
        self.model = model
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self._model: Any = None

    def _load(self) -> Any:  # noqa: ANN401 — the loader is an untyped third-party object
        # why: the import and the weights wait for the first encode, so a missing extra or a
        # missing model reaches index.py and search.py as an unreachable endpoint -- which they
        # degrade to bm25 -- instead of refusing before either of them is entered
        if self._model is not None:
            return self._model
        # why: the import is deferred so the package installs and every HTTP path runs without the
        # optional extra, which drags in torch
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EndpointUnavailableError("sentence-transformers is not installed",
                                      hint="pip install 'graphify-search[local]'") from e
        try:
            # why: CPU is chosen for byte-stable vectors across runs and hosts; code-review-graph
            # recorded that the same weights on MPS may answer differently from one host to the next
            self._model = SentenceTransformer(self.model, device="cpu")
        # why: the loader raises OSError, ValueError and its own library's errors for a name it
        # cannot resolve offline, so every failure to load is caught and named as one refusal
        except Exception as e:
            raise EndpointUnavailableError(f"no local model {self.model}: {e}",
                                      hint="download the weights, or unset HF_HUB_OFFLINE") from e
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        # inv: an empty corpus answers the (0, 0) shape the HTTP client answers, so index.py
        # reads one width rule for both backends
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # inv: the model normalises, so the rows are unit length and a dot product is the cosine
        rows = self._load().encode(texts, batch_size=FORWARD_BATCH, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(rows, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents with the document prefix; one row per text, in order."""
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question with the query prefix."""
        return self._encode([self.query_prefix + text])[0]
