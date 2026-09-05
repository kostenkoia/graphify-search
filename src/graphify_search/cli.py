"""Command line: the packaged skill plus index, query and status over a graphify graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from graphify_search import graph, index, schema, search, settings, skill
from graphify_search.embed import Embedder, EmbeddingClient, LocalEmbeddingClient, is_local
from graphify_search.errors import EndpointUnavailableError, InputError


def _version() -> str:
    try:
        return version("graphify-search")
    except PackageNotFoundError:
        return "0"


def _parser() -> argparse.ArgumentParser:
    """Build the parser for every subcommand.

    Returns
    -------
    argparse.ArgumentParser
        A parser naming no directory of its own.
    """
    ap = argparse.ArgumentParser(prog="graphify-search",
                                 description="Semantic search over a graphify graph, and the Claude Code skill for it.")
    ap.add_argument("--version", action="version", version=f"graphify-search {_version()}")
    sub = ap.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="write SKILL.md into the skills directory")
    install.add_argument("--root", type=Path, help="project root; defaults to the working directory")
    install.add_argument("--global", dest="global_install", action="store_true",
                         help="install under the home directory instead of the project")
    install.add_argument("--force", action="store_true", help="overwrite a file that is not this skill")

    uninstall = sub.add_parser("uninstall", help="remove the skill directory")
    uninstall.add_argument("--root", type=Path, help="project root; defaults to the working directory")
    uninstall.add_argument("--global", dest="global_install", action="store_true",
                           help="act under the home directory instead of the project")

    detect = sub.add_parser("detect", help="report where the skill is installed")
    detect.add_argument("--root", type=Path, help="project root; defaults to the working directory")

    def graph_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--graph", help="graph.json or its directory; defaults to $GRAPHIFY_OUT or graphify-out")
        p.add_argument("--endpoint",
                       help="OpenAI-compatible base URL, e.g. http://localhost:1234/v1, or `local`")
        p.add_argument("--model", help="embedding model name")
        p.add_argument("--require-dense", action="store_true", help="refuse to answer without vectors")

    idx = sub.add_parser("index", help="build or refresh the index beside graph.json")
    graph_flags(idx)
    idx.add_argument("--source",
                     help="source root the graph's paths are relative to; defaults to the graph directory's parent")
    idx.add_argument("--full", action="store_true", help="ignore the previous index")

    q = sub.add_parser("query", help="answer a question with JSON places")
    q.add_argument("question")
    graph_flags(q)
    q.add_argument("-k", type=int, help="places to return")
    q.add_argument("--budget", type=int, help="character budget for the results array")
    q.add_argument("--exclude", action="append", metavar="GLOB", default=[],
                   help="drop results whose path matches this glob; repeatable")
    q.add_argument("--no-snippets", action="store_true")
    q.add_argument("--no-edges", action="store_true")

    st = sub.add_parser("status", help="report the index state as JSON")
    st.add_argument("--graph", help="graph.json or its directory")
    return ap


def _client(s: settings.Settings) -> Embedder:
    # inv: `local` is the one endpoint that names no URL, so it is the switch between the
    # in-process model and the HTTP client rather than a flag of its own
    if is_local(s.endpoint):
        return LocalEmbeddingClient(s.model, s.doc_prefix, s.query_prefix)
    return EmbeddingClient(s.endpoint, s.model, s.doc_prefix, s.query_prefix, api_key=s.api_key)


def _run(args: argparse.Namespace, root: Path) -> object:
    if args.command == "install":
        return skill.install(root, global_install=args.global_install, force=args.force)
    if args.command == "uninstall":
        return skill.uninstall(root, global_install=args.global_install)
    if args.command == "detect":
        return skill.detect(root)
    graph_path = settings.resolve_graph(args.graph, os.environ, root)
    if args.command == "status":
        idx = index.load_index(graph_path)
        now = graph.sha256_of_file(graph_path) if graph_path.is_file() else ""
        return {"graph": str(graph_path), "graph_sha256": idx.manifest.graph_sha256,
                "stale": idx.manifest.graph_sha256 != now, "rows": idx.manifest.rows,
                "vectors": idx.manifest.vectors, "model": idx.manifest.model,
                "endpoint": idx.manifest.endpoint, "package_version": idx.manifest.package_version}
    s = settings.resolve(graph_path, vars(args), os.environ)
    if args.command == "index":
        source = Path(args.source) if args.source else graph_path.parent.parent
        client = _client(s)
        record = index.build_index(graph_path, source, client, full=args.full, require_dense=args.require_dense)
        # inv: the record names the host the source text was sent to, which the operator cannot
        # read off the counts alone, and it is the client's own spelling, so the record and the
        # manifest a later refresh compares against never differ by a trailing slash
        return {**record, "endpoint": client.endpoint}
    # inv: the graph is checked before the index, so a missing graph is named as itself rather than
    # as the missing index it also explains
    if not graph_path.is_file():
        raise InputError(f"no graph at {graph_path}", hint="run `graphify <path>` first")
    idx = index.load_index(graph_path)
    return search.query(idx, args.question, _client(s), k=s.k, budget=s.budget, require_dense=args.require_dense,
                        snippets=not args.no_snippets, edges=not args.no_edges,
                        graph_sha256_now=graph.sha256_of_file(graph_path), exclude=args.exclude)


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand and print its record as JSON on stdout.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        0 when the command succeeded, 1 when it was refused.
    """
    args = _parser().parse_args(argv)
    # inv: `_parser` names no directory, so the working directory is read here, when the command
    # runs, and never frozen into the parser
    root = getattr(args, "root", None) or Path.cwd()
    try:
        record = _run(args, root)
        # inv: the record is JSON a caller parses, so a non-finite number stops the render rather
        # than emitting `NaN`
        rendered = (schema.render(record) if isinstance(record, schema.Answer)
                    else json.dumps(record, indent=2, allow_nan=False) + "\n")
    except (InputError, EndpointUnavailableError) as exc:
        # inv: a refusal reaches the operator on stderr and leaves stdout empty, so a caller
        # parsing this command's JSON never reads an error message as a record
        print(str(exc), file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # inv: the budget ladder renders the answer too, so a number JSON cannot carry is caught
        # around the whole command and reaches the operator as plain text, never as a traceback
        print(f"the answer cannot be rendered as JSON: {exc}", file=sys.stderr)
        print("hint: run `graphify-search index --full`", file=sys.stderr)
        return 1
    # inv: nothing is written to stdout until the whole answer has rendered, so a refused render
    # leaves no half-written record for a caller to parse
    sys.stdout.write(rendered)
    return 0
