"""Where the graph is and how the endpoint is reached: flag, environment, config file, defaults."""

from __future__ import annotations

import ipaddress
import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from graphify_search.embed import is_local
from graphify_search.errors import InputError
from graphify_search.index import index_dir

if TYPE_CHECKING:
    from collections.abc import Mapping

# NOT DERIVED: the owner's LM Studio address
DEFAULT_ENDPOINT = "http://localhost:1234/v1"
# NOT DERIVED: names one common local embedding model; nothing in this package requires it over another
DEFAULT_MODEL = "text-embedding-nomic-embed-text-v1.5"
# why: the nomic prefixes move the vector -- cosine to the unprefixed embedding measures 0.89-0.95
# for the document prefix and 0.95-0.99 for the query prefix on three texts, so a corpus and a
# question must be embedded with the pair the model documents or with neither
DEFAULT_DOC_PREFIX = "search_document: "
DEFAULT_QUERY_PREFIX = "search_query: "
# NOT DERIVED: the characters one answer's results may occupy; it buys room for the ten places
# `DEFAULT_K` asks for with their snippets and edges, which render 2529 characters on this
# package's own fixture, so the ladder drops nothing on an answer of that size
DEFAULT_BUDGET = 6000
# NOT DERIVED: the number of places one answer lists
DEFAULT_K = 10
_DEFAULTS = {"endpoint": DEFAULT_ENDPOINT, "model": DEFAULT_MODEL,
             "doc_prefix": DEFAULT_DOC_PREFIX, "query_prefix": DEFAULT_QUERY_PREFIX,
             "k": DEFAULT_K, "budget": DEFAULT_BUDGET}
# inv: the key is a secret, so the environment is its only source; config.json sits beside a graph
# that is copied and shared, so the endpoint that file may name is `local` or a loopback host and
# the operator's own environment or flag is the only way to send source text to a remote server
_ENV = {"endpoint": "GRAPHIFY_SEARCH_ENDPOINT", "model": "GRAPHIFY_SEARCH_MODEL",
        "api_key": "GRAPHIFY_SEARCH_API_KEY"}
_FIX_CONFIG = "fix or delete the file"


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one command."""

    endpoint: str
    model: str
    doc_prefix: str
    query_prefix: str
    k: int
    budget: int
    api_key: str | None = None


def resolve_graph(arg: str | None, env: Mapping[str, str], cwd: Path) -> Path:
    """Return the graph.json a command works on."""
    if arg:
        p = Path(arg)
        return p / "graph.json" if p.is_dir() or p.suffix != ".json" else p
    # inv: the vendor's own variable names the output directory, relative to the working directory
    out = Path(env.get("GRAPHIFY_OUT") or "graphify-out")
    return (out if out.is_absolute() else cwd / out) / "graph.json"


def _whole(values: Mapping[str, object], key: str, cfg: Path) -> int:
    try:
        return int(str(values[key]))
    except ValueError as e:
        raise InputError(f"{key} must be an integer, got {values[key]!r}",
                         hint=f"fix {cfg} or the flag") from e


def _is_loopback(endpoint: str) -> bool:
    """Return whether `endpoint` reaches this machine only.

    Parameters
    ----------
    endpoint : str
        An endpoint value, either a URL or the in-process backend's name.

    Returns
    -------
    bool
        True for the in-process backend and for a URL whose host is a loopback address.
    """
    if is_local(endpoint):
        return True
    host = urllib.parse.urlsplit(endpoint).hostname
    if host is None:
        return False
    # inv: the name `localhost` is loopback by definition, and every other host must parse as a
    # loopback address, so a resolvable name such as `evil.example.com` is not admitted
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve(graph_path: Path, args: Mapping[str, object], env: Mapping[str, str]) -> Settings:
    """Merge flag, environment, config file and defaults, in that precedence.

    Raises
    ------
    InputError
        When the config file is not a JSON object, when `k` or `budget` is not an integer, or
        when the config file names a remote or unparsable endpoint.
    """
    values: dict[str, object] = dict(_DEFAULTS)
    # inv: the keys some source other than _DEFAULTS sets, so a default can be re-chosen once
    # the endpoint is known without overriding what the operator asked for
    configured: set[str] = set()
    cfg = index_dir(graph_path) / "config.json"
    # inv: an unreadable or unparsable file leaves the defaults standing, since a graph copied
    # without its config is the ordinary case rather than an error
    try:
        raw: object = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    # inv: the file is a mapping of settings, so a JSON array, string or number is named as the
    # file it came in rather than reaching `.items()` as an attribute error
    if not isinstance(raw, dict):
        raise InputError(f"{cfg} is not a JSON object", hint=_FIX_CONFIG)
    chosen: dict[str, object] = {k: v for k, v in raw.items() if k in values}
    # inv: the check runs before any flag can override the value, so a graph carrying a remote
    # endpoint is refused as the file it came in rather than silently ignored
    if "endpoint" in chosen:
        try:
            loopback = _is_loopback(str(chosen["endpoint"]))
        except ValueError as e:
            # inv: a URL the parser cannot split is refused as the file's own value, since no
            # later source can repair what this file says
            raise InputError(f"{cfg} names an unusable endpoint: {chosen['endpoint']!r}: {e}",
                             hint=_FIX_CONFIG) from e
        if not loopback:
            raise InputError(f"{cfg} names a remote endpoint: {chosen['endpoint']!r}",
                             hint="remove `endpoint` from that file; a remote server goes in "
                                  "GRAPHIFY_SEARCH_ENDPOINT or --endpoint")
    values.update(chosen)
    configured.update(chosen)
    for key, var in _ENV.items():
        if env.get(var):
            values[key] = env[var]
            configured.add(key)
    given = {k: v for k, v in args.items() if k in values and v is not None}
    values.update(given)
    configured.update(given)
    # why: the nomic prefixes belong to the nomic model; a sentence-transformers model such as
    # all-MiniLM-L6-v2 was trained on raw text, and a prefix left over from the HTTP path would
    # move every vector it embeds
    if is_local(str(values["endpoint"])):
        values.update({k: "" for k in ("doc_prefix", "query_prefix") if k not in configured})
    return Settings(endpoint=str(values["endpoint"]), model=str(values["model"]),
                    doc_prefix=str(values["doc_prefix"]), query_prefix=str(values["query_prefix"]),
                    k=_whole(values, "k", cfg), budget=_whole(values, "budget", cfg),
                    api_key=str(values["api_key"]) if values.get("api_key") else None)
