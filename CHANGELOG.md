# Changelog

## 0.6.0 — 2026-08-31

First public release.

- `config.json` beside a graph may name only `local` or a loopback `endpoint`; a remote one is refused,
  so a cloned repository cannot redirect source text and the API key. Redirects from the endpoint are
  not followed. Responses are capped at 64 MiB and a non-finite embedding is refused; rendered JSON never
  carries `NaN`. Every `index` and `query` record names the `endpoint` it used, and an `index` written
  without vectors after the endpoint refused carries `embedding_error`.
- A graph that is not a graphify graph (a node without `id`, a non-string `source_location`, a
  duplicate id, no place to index) is refused with one line and a hint instead of a traceback or an
  index that never loads. One `index` at a time per graph (`.graphify_search/lock`). Vectors are reused
  only when the endpoint matches too; `extends` edges are shown beside `inherits`.
- Words are matched in any script; in `bm25` a question with no word, or none the code contains, is
  refused with a reason instead of an empty list.
- SKILL.md documents every key of `query`, `status` and `index`, what `community` and `score` are, and
  that failures are plain text on stderr.
- Embeddings are checked after the cast to float32, so a value that overflows it is refused instead of
  stored as `NaN`; an answer that cannot be rendered as JSON is refused in plain text. `config.json`
  that is not a JSON object, and an endpoint the URL parser rejects, are refused with one line. The
  index lock names its holder's PID and a lock left by a killed process is taken over.
- `benchmark`: `build-symbols` returns files with no extractor and names them on stderr;
  `audit blind` says when a run holds nothing to check.
- SKILL.md and README carry a block asking the assistant to put its question in the language of the
  code before calling `query`; the README block is meant to be copied into `CLAUDE.md`.
- `query` takes `--exclude GLOB` (repeatable) to drop results whose path matches — so `tests/` and
  `docs/` rows whose names echo the question stop crowding out implementation code.
- `budget.dropped` names only what the budget or a `--no-snippets`/`--no-edges` flag removed; it no
  longer fires merely because the chosen rows (documents) carry no snippet.
- `index` refuses a `--source` under which no source file is found, instead of writing an index of
  empty snippets; its record adds `files_read`, and caps `dropped` at 20 ids with a `dropped_total`.
- A `document` row's `symbol` is always `null`; an empty source file gets no `file` row.
- A result names `start` only. The earlier derived `end_hint` undershot every class (its next code
  node is its own first method); a symbol's indexed body is now a fixed window from its start.
- `graphify-search index` / `query` / `status`: a search index beside `graphify-out/graph.json`, ranked
  by embeddings from any OpenAI-compatible `/v1/embeddings` endpoint or by BM25 when no vectors exist;
  every answer names the mode that ranked it.
- `--endpoint local`: in-process `sentence-transformers` embedding through the `[local]` extra.
- `--budget` ladder (ADR-0002) that drops snippets, then edges, then places, and reports what it dropped.
- `install` / `uninstall` / `detect` for the Claude Code skill file.
- `benchmark/`: a self-service harness — snapshot a corpus, freeze indexes, run baseline and
  driven rounds, summarise from the ledger — with the methodology it measures under.
