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

Read the answer before acting on it:

- `results[]` — places ranked best first: `rank`, `path`, `symbol`, `start` (the line to open;
  no end line is known), `kind` (`symbol` | `file` | `document`), `community`, `score`,
  optional `snippet` and one hop of `edges`.
- `mode` — `dense` (embeddings) or `bm25` (no vectors indexed, or the endpoint did not answer this
  query). `bm25` answers are weaker; say so.
- `index.stale: true` — the graph changed after indexing; run the refresh below first.
- `budget.dropped` — what is absent from the results: `snippet`, `edges` or `k`, either because
  the budget removed it or because you switched it off. Raise `--budget` (e.g. 9000) only when
  you did not pass the corresponding flag.

Options: `-k N` places (default 10), `--budget CHARS` (default 6000), `--no-snippets`,
`--no-edges`, `--require-dense` (refuse `bm25`).

On an `index` refresh where nothing changed, `--require-dense` trusts the cached vectors even if
the endpoint is down.

## Keep the index current

After code changes:

```bash
graphify update . && graphify-search index
```

`index` reads `graph.json` and the source tree, embeds only what changed, and prints a JSON
record. `graphify-search status` reports staleness and whether vectors exist.
If the endpoint is down during a refresh, the previous index is kept and the command fails; retry
when it answers. `--source PATH` names the source root when the graph's paths are not relative to
the graph directory's parent.

## Configuration

Embeddings come from an OpenAI-compatible endpoint: `--endpoint URL --model NAME`, or
`GRAPHIFY_SEARCH_ENDPOINT` / `GRAPHIFY_SEARCH_MODEL`, or `graphify-out/.graphify_search/config.json`.
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
