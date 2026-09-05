# graphify-search

## What this is

graphify — a companion tool, installed separately — reads a codebase and writes a map of it to
`graphify-out/graph.json`: every file, every named piece of code, and what calls what.
`graphify-search` makes that map searchable by meaning. You ask in ordinary words; it answers with a
machine-readable list of places — file, name, line number — best first, for you or your coding
assistant to open and read.

It works with nothing extra installed, by matching the words of your question. With an embedding
server — a program that turns text into numbers so that two sentences with the same meaning end up
close together — it ranks by meaning instead. It uses one when it was running at `index` time — start a
server later and re-run `graphify-search index` — and every
answer says which of the two ranked it.

## Install

Python 3.10 or newer. The package installs straight from this repository; pick one of the three
below. From a clone, `uv pip install .` (or `".[local]"`) does the same.

**Plain — word matching only.**

```bash
uv pip install "graphify-search @ git+https://github.com/kostenkoia/graphify-search.git"
# or: pip install "graphify-search @ git+https://github.com/kostenkoia/graphify-search.git"
```

Nothing to configure. Answers come back marked `"mode":"bm25"`: a place is ranked by the words it
shares with your question.

**A small model on your own machine.**

```bash
uv pip install "graphify-search[local] @ git+https://github.com/kostenkoia/graphify-search.git"
export GRAPHIFY_SEARCH_ENDPOINT=local
export GRAPHIFY_SEARCH_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The model runs inside the tool itself; its weights download once, and nothing leaves your machine.
It is loaded afresh on every `query`; for repeated questions a running embedding server answers
faster.

**An embedding server you already run.**

```bash
uv pip install "graphify-search @ git+https://github.com/kostenkoia/graphify-search.git"
export GRAPHIFY_SEARCH_ENDPOINT=http://localhost:1234/v1
export GRAPHIFY_SEARCH_MODEL=text-embedding-nomic-embed-text-v1.5
```

Any server that speaks the same `/v1/embeddings` request as OpenAI will do — LM Studio and Ollama
both do. Those two values are also the defaults. A key, if your server wants one, goes in
`GRAPHIFY_SEARCH_API_KEY` and is sent with each request. `index` sends the text of your source files
to this server and `query` sends your question, so point it at a server you trust; with `local`
nothing leaves the machine.

## The commands

graphify is a separate package — `uv tool install graphifyy`, command `graphify`. `index` and
`query` refuse without `graphify-out/graph.json`; `status` refuses until an index exists; `install`
needs neither.

```bash
graphify .                          # writes the map this tool searches — run first
graphify-search index               # read that map and the source, write the search index beside it
graphify-search query "how are uploaded files checked"   # answer one question as a list of places
graphify-search status              # is the index still current, and does it hold vectors
graphify-search install             # write the skill file, so Claude Code knows this tool
```

`query` also takes `-k N`, `--budget CHARS`, `--no-snippets`, `--no-edges`, `--require-dense` and
`--exclude GLOB` (repeatable — drop results whose path matches, e.g. `--exclude 'tests/*'` to search
implementation code only).
`index`, `query` and `status` take `--graph PATH` — the `graph.json` or its directory, defaulting to
`$GRAPHIFY_OUT` or `graphify-out` — and `index --full` rebuilds the index from scratch, which is the fix
the tool prints when an index is unreadable. `uninstall` removes the skill file and `detect` reports
where it is.

## How a coding assistant uses it

Every answer is one JSON document on stdout, and every failure is plain text on stderr with exit
code 1 — 2 for a bad flag.

After `graphify-search install`, Claude Code has a skill file that describes the tool. The assistant
runs one `query` call carrying your question as you wrote it, reads the list that comes back, and
opens the first few places in it, so it works from the real code rather than from a guess. The answer
also tells it which ranking was used, and whether the index has fallen behind the source. When the
assistant changes code it runs `graphify update .` and `graphify-search index` again, and the second
reprocesses only what moved.

If you write your own rules instead of installing the skill file, this block goes into `CLAUDE.md`;
replace "English" if your code is written in another language.

```markdown
## graphify-search: ask in the language of the code

Before calling `graphify-search query`, put the question in the language the code and its
comments are written in — for this repository, English. Keep names as they appear in the code
(`EmbeddingClient`, `--budget`), translate everything else. Word matching (`bm25`) finds nothing
across languages, and meaning-based ranking is weaker across them too. Report the answer in the
user's language.
```

## Measuring it, on your own code

This repository publishes no benchmark results. It publishes the instrument: `benchmark/` holds a
harness that puts one plain-language question to a code-search tool and decides, against places
written down in advance, whether the answer contains one — and records what the answer cost. The
record ships empty: no snapshot, no question and no recorded run is in this repository. Where a
design decision was settled by a number — a cap, an ADR's confirmation section — the number is
kept where it was used and says that its evidence is not published.

Running the benchmark needs `benchmark/lock/install`, which uses `sysadminctl`, `dscl`,
`chflags uchg`, `/Users/bench` and a python.org framework interpreter, so a run happens on macOS
only. The harness package and its tests run on Linux: CI exercises them on Python 3.10 to 3.13.

One run of the instrument, on the `sphinx/` package of sphinx-doc/sphinx, is shown in the
[v0.6.0 release](https://github.com/kostenkoia/graphify-search/releases/tag/v0.6.0) as the
harness rendered it. Its record was kept by the operator and is not published. It is a statement
about thirty questions on one snapshot, not a claim about your code.

[benchmark/PROTOCOL.md](benchmark/PROTOCOL.md) is the method: what counts as an answer, what must
be fixed in advance, what refuses each rule, and what a result of this design cannot show whatever
corpus it runs on. [benchmark/README.md](benchmark/README.md) is how to point it at a
codebase of your own, from freezing a snapshot to reading a cell. Nothing here says how this tool
would do on yours; running it is the only way to find out.

## Licence

Apache-2.0. See [LICENSE](LICENSE). The vendor documents frozen under `benchmark/systems/` carry their
own notices in [benchmark/systems/NOTICE.md](benchmark/systems/NOTICE.md).
