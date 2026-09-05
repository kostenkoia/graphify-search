---
name: graphify-search
description: >
  Answer a question about a codebase with ranked places (path, symbol, line) from the graphify
  graph, as JSON, in one call. Use when graphify-out/graph.json exists and the user asks where
  something happens, how something works, or which code does X.
---

# graphify-search

Semantic search over `graphify-out/graph.json`. One call, JSON out, no vocabulary expansion.

## Ask

```bash
graphify-search query "<the user's question, verbatim>"
```

Every answer is one JSON document on stdout; every failure is plain text on stderr with exit
code 1 (2 for a bad flag).

Read the answer before acting on it:

- `question` — the question as it was passed. `mode` — `dense` (embeddings) or `bm25` (no vectors
  indexed, or the endpoint did not answer this query); name the mode when you report the answer.
  `model` — the embedding model, `null` in `bm25`. `endpoint` — the endpoint configured for this
  call; in `bm25` with `vectors: absent` nothing was sent to it.
- `index` — `graph_sha256`, the graph the index was built from; `stale: true`, the graph changed
  after indexing, so run the refresh below first; `vectors`, one of `present`, `absent` (none were
  indexed) or `unreachable` (the index holds vectors, the endpoint did not answer this query).
- `budget` — `limit_chars` as asked, `used_chars` the rendered length of `results`, `dropped`, and
  `exceeded: true` when the results do not fit even after every removal. `dropped` names only what
  was removed to fit: `snippet`, then `edges`, then `k` (the list cut to five), each either removed
  by the budget or switched off with `--no-snippets` / `--no-edges`. Raise `--budget` (e.g. 9000)
  when the budget removed a field you did not switch off. A `document` row always has an empty
  `snippet` (`""`); that is not a `dropped` cause. An empty `snippet` on a code row means the index
  could not read that source — re-run `index` with the `--source` the graph's paths are relative to.
- `results[]` — places ranked best first: `rank`; `kind` (`symbol` | `file` | `document`); `path`;
  `symbol`, the bare name of a `symbol` row and `null` on a `file` or `document` row; `start`, the
  line to open (no end line is known); `community`; `score`. Optional: `snippet`, empty on a
  `document` row, and `edges`. One symbol can appear twice with the same `path` and `symbol` at
  different `start` lines — a sync and an async definition of the same name; open both.
- `community` — graphify's cluster label for the node, an opaque string. Two places carrying the
  same label were clustered together by graphify; it names no semantic category, so do not read
  meaning into its text.
- `score` — cosine similarity in `dense`, the BM25 score in `bm25`. Comparable within one answer,
  never across the two modes.
- `edges[]` — one hop out of this place: `{rel, to, at}`, the relation, the node it points at, and
  the site (`L12`, or `path:L12` when the site is in another file).

Options: `-k N` places (default 10), `--budget CHARS` (default 6000), `--no-snippets`,
`--no-edges`, `--require-dense` (refuse `bm25`), `--exclude GLOB` (repeatable; drop every result
whose `path` matches the glob before ranking is cut — `--exclude 'tests/*' --exclude 'docs/*'` keeps
implementation code, where a test or a doc whose name echoes the question would otherwise crowd the
top), `--graph PATH` (the `graph.json` or its directory; default `$GRAPHIFY_OUT`, then
`graphify-out`).

On an `index` refresh where nothing changed, `--require-dense` trusts the cached vectors even if
the endpoint is down.

In `bm25`, a question sharing no word with the indexed code is refused rather than answered with an
empty list; ask again with the words the code uses, or start the embedding server and re-index.

Put the question in the language the code and its comments are written in before you run `query`.
Keep names as they appear in the code, translate the rest. Word matching (`bm25`) finds nothing
across languages, and meaning-based ranking is weaker across them too. Report the answer in the
user's language.

## Keep the index current

After code changes:

```bash
graphify update . && graphify-search index
```

Every `query` answer reports whether the index is behind the graph: `index.stale` is `true` when
`graph.json` changed after the index was built. On a `stale: true` answer, run the two commands
above and ask again before reading the places; an answer from a stale index describes the code as
it was, not as it is. `stale` follows the graph, not the source: a file edited since the last
`graphify update .` shows as current until that command runs.

`index` reads `graph.json` and the source tree, embeds only what changed, and prints
`graph_sha256`, `nodes`, `embedded`, `reused`, `files_read` (distinct source files read; `0` under
a wrong `--source` is refused, not reported), `dropped` (up to 20 ids that left the graph) with
`dropped_total` (their full count), `vectors`, `seconds`, `endpoint` (the one configured for the
run, whether or not it was asked) and — only
when the endpoint was asked and refused, so the index holds no vectors — `embedding_error`, the
refusal's text. `status` prints `graph` (the file it read), `graph_sha256`, `stale`,
`rows`, `vectors` (`present` | `absent`), `model` and `endpoint` (the two the index was built
with), and `package_version`. `index --full` discards the previous index and rebuilds it — the fix
printed when index parts disagree or vectors are unreadable.
`index` holds `graphify-out/.graphify_search/lock` while it runs, so a second `index` on the same
graph is refused with the holder's PID; a lock left behind by a killed process is taken over by the
next `index` on its own.
If the endpoint is down during a refresh, the previous index is kept and the command fails; retry
when it answers. `--source PATH` names the source root when the graph's paths are not relative to
the graph directory's parent.

## Configuration

Embeddings come from an OpenAI-compatible endpoint: `--endpoint URL --model NAME`, or
`GRAPHIFY_SEARCH_ENDPOINT` / `GRAPHIFY_SEARCH_MODEL`, or `graphify-out/.graphify_search/config.json`.
That file travels with a copied graph, so an `endpoint` in it may name only `local` or a loopback
address; with any other value the command refuses until the key is removed from that file. A remote
server is given by the variable or the flag.
`GRAPHIFY_SEARCH_API_KEY`, when set, is sent as `Authorization: Bearer`.
Default: `http://localhost:1234/v1`, `text-embedding-nomic-embed-text-v1.5` (LM Studio).
`--endpoint local` (or `GRAPHIFY_SEARCH_ENDPOINT=local`) embeds in this process with the
`sentence-transformers` model `--model` names, installed by the `[local]` extra; it sends no
prefixes unless `config.json` sets them.

## Install the skill file

```bash
graphify-search install          # .claude/skills/graphify-search/SKILL.md
graphify-search install --global
graphify-search detect
graphify-search uninstall
```
