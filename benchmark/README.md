# Measure your own codebase

`PROTOCOL.md` beside this file says what is measured and what a figure from it can mean. This says
how to install the lock, point the tree at a codebase of your own, run it and read what it wrote.

Running the benchmark needs `benchmark/lock/install`, which uses `sysadminctl`, `dscl`,
`chflags uchg`, `/Users/bench` and a python.org framework interpreter, so a run happens on macOS
only. The harness package and its tests run on Linux: CI exercises them on Python 3.10 to 3.13.

The record ships empty: no corpus freeze, no question, no reference, and a ledger with no rows;
every attempt made before this protocol is in git history alone. §4 is the one instruction that
builds the run side of the tree from nothing — obtain a corpus, freeze it, build one index per
system, author and review the questions — and §0 is what is left once §4 and §3 are done. A
snapshot's `source/` and the tool environments under `envs/` are in no clone: they are yours.

Paths are from the repository root. `.venv` is this repository's own environment on Python 3.10 or
newer, made with `uv venv .venv` and `uv pip install --python .venv/bin/python -e ".[dev,bench]"`.

## 0. What is left before the first figure

§4 builds the corpus, the indexes and the questions; §3 installs the lock and seals the tree.
This is what remains for a reader who has done both, in this order, each step the owner's:

1. **Baselines** — every cell of your snapshot is missing both of its baseline rows:

   ```bash
   .venv/bin/python -m benchmark.harness run missing baseline --seconds 480   # until none remain
   ```

2. **The driven round** — one model, named on every line, over the questions you authored:

   ```bash
   for q in q001 q002 q003 q004 q005 q006 q007 q008 q009 q010; do
     for cell in "code-review-graph vector" "graphify default" "graphify-search dense" \
                 "graphify-search-bm25 bm25" "graphify-search-minilm minilm"; do
       .venv/bin/python -m benchmark.harness run driven $cell $q \
           --model <model> --effort <effort> --max-actions 12 --max-tokens 8192
     done
   done
   ```

3. **Summary, report, closing review** — §6 for the commands; `PROTOCOL.md` §9 for the reviewer.

Part of the suite skips itself until the tree holds what it reads: the tests that prepare a run or
check an expansion need a question with a built `graphify` index beside it, one ledger test needs a
`prepared_outputs.yaml` that the first attempt writes, and the four MCP tests need a built
`code-review-graph` index together with a `code-review-graph` launcher in this repository's own
`.venv/bin` — not the one under `envs/`, which they never read.

`.venv/bin/pip install "code-review-graph[embeddings]==2.3.7"` puts that launcher there; the
version is the pin `systems/code-review-graph/manifest.yaml` records, and the `embeddings` extra
is what `embed` and the `vector` configuration's query both need.

From the first row on, no row is removed (`PROTOCOL.md` §2): a later campaign is a new snapshot
beside the first, never a cleared record.

## 1. Layout

```
benchmark/
├── PROTOCOL.md          the law: what is measured, what may be done, what stops the rest
├── README.md            the manual: this file
├── INSTRUMENT.yaml      the seal: the sha256 of every instrument file, tracked
├── harness/             one package, one command, one verb per operation
├── lock/                install, unlock, lock, the pinned requirements, the signpost hook
├── systems/             per system: harness.yaml, manifest.yaml, docs/, models/
├── envs/                one environment per tool, and envs/harness (untracked)
└── record/              everything a run of the benchmark leaves, owned by bench
    ├── attempts.jsonl   the ledger, one row per attempt (tracked, no row when shipped)
    ├── summary.json     and SUMMARY.md, rebuilt from the ledger on demand (untracked)
    ├── runs/            per-run evidence (untracked)
    ├── evidence/        what a campaign keeps beside its report (tracked)
    ├── reports/         <snapshot>.md.in and the rendered <snapshot>.md (tracked)
    └── snapshots/       known_transitions.yaml, then one directory per corpus freeze:
        └── <snapshot>/  meta.yaml, fileset.sha256, symbols.sha256 — the freeze records
            ├── source/  the corpus, with symbols.jsonl beside it (both untracked)
            ├── indexes/ per index: build.yaml, prepared_outputs.yaml; artifacts untracked
            ├── questions/   q*.yaml, candidates/, review/<qid>.request.yaml and <qid>.yaml
            └── references/  q*.yaml, apart from questions/ on purpose: the blindness boundary
```

Everything belonging to a run is under `record/`, and nothing else is. The seal splits it
(`PROTOCOL.md` §3): the snapshot directories, their freeze records, `questions/` and `references/`
are instrument, written only between `unlock` and `lock`, so nothing a figure cites can move under
it. `source/`, `symbols.jsonl`, the index artifacts, the ledger, the runs and the reports are
`bench`'s.

Git carries no empty directory, so three of those appear only once something writes them. Closing
an attempt, `collect` creates `record/runs/` and copies the run into `record/runs/<run_id>/`;
`report draft` creates `record/reports/`. `record/evidence/` is written by no verb at all — it is
the tracked place for what a campaign keeps beside its report, and stays absent until something is
put there, as `bench`.

## 2. Verbs

Every operation is `python -m benchmark.harness <verb>`. The "runs as" column is what
`/etc/sudoers.d/bench-harness` allows: those six verbs are reached with `sudo -u bench` and the
`envs/harness` interpreter. `run` does that itself, for `attempt` only; §6 shows the elevated form
for `summary` and `report`, and §4 runs the setup verbs before `install`, as you.

| verb | runs as | does | reads | writes |
|---|---|---|---|---|
| `run` | owner | refuses or dispatches one attempt, then makes its single commit | the ledger, `systems/` | the ledger commit |
| `attempt` | `bench` | prepare, drive, audit, score and collect, in one process | the instrument and the snapshot | `record/runs/<run_id>/`, the ledger row |
| `abort` | `bench` | closes the row of a run the operating system killed, and frees its lock | the ledger, the tmp root | that row |
| `audit` | owner | `quotes`, `run`, `blind`, `attempts`, `rebaseline`, `stops`, `priors`, `expectations`, `recount` | git history and run evidence | `<run>/audit.json` for `audit run`; nothing for the rest |
| `expand` | owner | prints the mechanical expansion block of one question text | an index's graph file | stdout |
| `freeze-model` | `bench` | copies a weights directory and prints its `models:` block | the weights you point it at | `systems/<id>/models/<dir>/` |
| `build-symbols` | `bench` | builds the symbol universe of the snapshot directory it is given | that snapshot's `meta.yaml` and `source/` | `symbols.jsonl`, `symbols.sha256` beside them |
| `seal` | owner, or root inside `lock` | `--check`, `--paths <class>`, `--launchers`, or writes the seal | every instrument glob | `INSTRUMENT.yaml` |
| `questions` | owner | `author`, `review-request`, `review-check` | that snapshot's `questions/candidates/` | its `questions/`, `references/` and `questions/review/` |
| `summary` | `bench` | one row per group, deterministically | the ledger and every snapshot's `questions/` | `record/summary.json`, `record/SUMMARY.md` |
| `report` | `bench` | `draft` checks a template, `render` substitutes its figures | the template and the summary | `record/reports/<snapshot>.md.in` and `.md` |

## 3. Install

Once per machine, by the owner, after the tree is set up (§4) and every launcher a recipe names
lives under `benchmark/envs/`. Run it from a shell whose `umask` is `022`: files `install` creates
without an explicit mode take root's umask, and a permissive one leaves the record group- or
world-writable, which `seal --check` then refuses.

```bash
.venv/bin/python -m benchmark.harness seal      # without root: no seal, no machine facts yet
git add benchmark/INSTRUMENT.yaml
git commit -m "chore(benchmark): seal — first seal of the checkout"

umask 022
sudo benchmark/lock/install --dry-run    # prints every account, chown, chflags and sudoers action
sudo benchmark/lock/install
sudo benchmark/lock/unlock "initial seal"
sudo benchmark/lock/lock
```

The first three commands seal the checkout and commit the seal. Without root, `seal` refuses once
`INSTRUMENT.yaml` or `lock/machine.yaml` exists, and a clone taken after that commit carries the
seal already: rewriting instrument files for your machine (§4 step 1) makes `seal --check` fail
there. From the first `lock` on, `lock/machine.yaml` exists and only `lock` writes the seal.

The repair is to remove `benchmark/INSTRUMENT.yaml`, run `seal` again and commit it alone under the
same subject, byte for byte: the reason `write` stamps is a constant, and `audit attempts` reads a
seal commit's subject as that prefix plus the sealed reason, so the diff of `INSTRUMENT.yaml` is
what records the change, not the wording.

The CI seal job runs `seal --check` on a checkout: it proves the instrument files match the
committed seal, so it is green from that commit and red on an instrument file changed without a
matching re-seal.

The sandbox root every attempt runs under is `/private/tmp/bench-harness`; set `BENCH_TMP_ROOT` in
`install`'s environment to place it elsewhere, and `lock/machine.yaml` records the choice.

`install` is not re-runnable on a locked tree. `lock_instrument` in `benchmark/lock/common.sh`,
its first step after the ownership hand-over, runs `chmod` over every instrument path; on a path
already flagged `uchg` that returns `EPERM`, and `set -e` ends the script there. An `install` that
stopped part-way, or a machine that is to be set up again, goes through `uninstall` first.

### Uninstall

`benchmark/lock/uninstall` is the inverse of `install`, and like it prints its whole plan under
`--dry-run`:

```bash
sudo benchmark/lock/uninstall --dry-run
sudo benchmark/lock/uninstall
```

It refuses while a ledger row has no outcome. Then, in order: every instrument and lock path
loses `uchg` and goes back to the owner, writable; the record and the system under test are
chowned back to the owner as trees; `lock/machine.yaml`, `lock/UNLOCKED`,
`/etc/sudoers.d/bench-harness` and the sandbox root are removed; `envs/harness` is removed; the
signpost hook leaves `.claude/settings.json`; and last, so nothing above is left with an
orphaned uid, the account `bench`, the group `bench` and `/Users/bench` are deleted.

What it leaves alone is the record and the checkout: the ledger, every snapshot, the reports,
`record/runs/` — untracked, and the only place a figure about the *shape* of an answer can be
recounted from, so archive it before deleting anything — and the tool environments under
`envs/`, which are the owner's. The tree it leaves is one `install` accepts again.

Every step is guarded on what exists, so `uninstall` can be run again after an interruption
and finishes what is left.

Clearing a campaign from the disk is the owner's step, after `uninstall` and after archiving
`record/runs/`. `git checkout` of another branch removes only tracked files, and the ledger's
history stays whatever the working tree holds; `record/runs/`, `summary.json`, `SUMMARY.md` and a
snapshot's `source/`, `symbols.jsonl` and index artifacts are ignored by git and stay until
removed. From the repository root, the ignored files under `benchmark/` alone:

```bash
git clean -ndX benchmark     # list what would go
git clean -fdX benchmark
```

`unlock` before `lock` is not optional either. `benchmark` itself is one of the instrument globs, so
`install` leaves the directory flagged, and even root cannot create `INSTRUMENT.yaml` inside it.

What `install` does, in order. It creates the account `bench` with `sysadminctl` and the group
`bench` with `dscl`, giving the group a `PrimaryGroupID` — `chown` resolves a group through that
id and reports one created without it as `illegal group name` — and adding you to it once;
makes `/Users/bench` the account's own before anything writes there; builds `envs/harness/` from `lock/harness-requirements.txt`
and points it at this repository with a `.pth` file; warms the `tiktoken` cache under `bench`'s own
home, as `bench`; and writes `bench`'s git configuration.

Then it writes `lock/machine.yaml` mode `444`; writes `/etc/sudoers.d/bench-harness` from
`lock/sudoers.template` and validates it with `visudo -cf`; hands the record and the system under
test to `bench`; and locks the instrument and `lock/`.

It removes every `__pycache__` under `harness/` before locking, and `uchg` on those directories
refuses new ones, so the harness is parsed from source on each attempt. Precompiling first would
save that, and would leave owner-writable bytecode inside the locked tree; the trade is the owner's.

Then see the boundary hold. Each of these must fail, as you, and the last two are the point:

```bash
sed -i '' s/a/b/ benchmark/harness/score.py                       # Operation not permitted
touch benchmark/harness/x.py                                      # Operation not permitted
.venv/bin/python -c "open('benchmark/record/attempts.jsonl','a').write('x')"   # Permission denied
benchmark/envs/harness/bin/python -m benchmark.harness attempt baseline s c q001   # not bench
git checkout <a commit that changes a harness.yaml>               # cannot write a locked file
```

An `Edit` on `benchmark/harness/score.py` in an agent session prints the signpost `install` added to
`.claude/settings.json`. Nothing depends on that hook: delete it and the same writes still fail,
with a worse message. `lock` refuses a dirty instrument, and prints `nothing changed` when the seal
would be identical.

And each of these must pass, as you, on the same tree:

```bash
git status --short                                          # nothing: lock committed the seal
git log --oneline -1                                # chore(benchmark): seal — initial seal
.venv/bin/python -m benchmark.harness seal --check          # exit 0, silent
sudo -u bench benchmark/envs/harness/bin/python -m benchmark.harness summary   # sudoers holds
```

## 4. Set up your own corpus

Six steps take a clone that holds no snapshot at all to a tree that can run. Do them before
`install`, or afterwards from an unlocked instrument (`PROTOCOL.md` §4). Throughout, `<id>` is
the snapshot id you choose — the shape used so far is `<repository>-<short commit>`. `<repo>` is
the absolute path of this repository, and `<scratch>` a working directory outside it.

### What git keeps

Decisions and the record; whatever can be rebuilt, or would publish private code, stays out.
`.gitignore` already enforces the split, so your corpus stays out of your commits without your
doing anything.

| path | in git | why, or how to get it back |
|---|---|---|
| `record/attempts.jsonl` | yes | the ledger; every published figure is recounted from it |
| `record/snapshots/known_transitions.yaml` | yes | the only place an admitted `build.yaml` change is named, with its reason |
| `record/snapshots/<id>/questions/`, `references/` | yes | the experiment itself: the questions, the reviews that admit one before it runs, and the reference places they are scored against |
| `record/snapshots/<id>/meta.yaml`, `fileset.sha256`, `symbols.sha256` | yes | what the corpus is, and hashes so a copy can be checked |
| `record/snapshots/<id>/indexes/*/build.yaml`, `prepared_outputs.yaml`, `*ignore*.used` | yes | each index's freeze record, the expectations whose git history the audit reads, and the ignore file that defines the corpus, whose bytes the `build.yaml` of each index built under one hashes among its artifacts |
| `record/reports/<id>.md.in` and `<id>.md` | yes | the template a figure was checked against, and what `render` made of it |
| `systems/*/harness.yaml`, `manifest.yaml`, `docs/` | yes | how each tool is invoked, and the vendor sentences that authorise it |
| `record/snapshots/<id>/source/` | no | the corpus; `fileset.sha256` checks a tree you obtained elsewhere, it cannot produce one, and without `source/` no index can be rebuilt |
| `record/snapshots/<id>/symbols.jsonl`, `indexes/*/` artifacts | no | rebuilt by `build-symbols` and by each index's own `build.yaml` command |
| `record/summary.json`, `record/SUMMARY.md`, `record/runs/` | no | a function of the ledger, and the per-run evidence, all re-derived or re-recorded |
| `systems/*/models/`, `envs/` | no | frozen weights, re-frozen by `freeze-model`, and tool environments, rebuilt from the version pins below |

### The six systems that ship

Each is a template rather than a result: a `harness.yaml` saying how the system is invoked, the
grammar its arguments must satisfy and its fixed steps, and a `manifest.yaml` beside it.

That manifest records the system's `kind` — `own` for the four that are this repository's own
package, `external` for the two vendors — the version seen at build time, links to the vendor's
documentation, the workflow that documentation prescribes quoted verbatim, which recipe is primary
where the vendor's own documents disagree and why, steps excluded from measurement and why, and
what the system can and cannot return.

| system | what its recipe measures | what it needs | shipped `version.cli` |
|---|---|---|---|
| `graphify-search` | one `query` call carrying the question verbatim, `--require-dense`, over an index embedded through an HTTP endpoint | any server answering `/v1/embeddings` — LM Studio and Ollama both do — with the model its `harness.yaml` names; no weights are hashed for it | `0.6.0` |
| `graphify-search-minilm` | the same recipe over an index whose vectors came from a `sentence-transformers` model loaded into the query process itself | the wheel installed with its `[local]` extra, and MiniLM weights frozen under that system's `models/` | `0.6.0` |
| `graphify-search-bm25` | the same call without `--require-dense`, its endpoint pointed at a closed port, so the tool's own word matching answers — the tier a reader with no embedder is left with | the wheel alone | `0.6.0` |
| `graphify-search-rewrite` | nothing runnable: it adds `--rewrite`, which the package lost. Marked `status: reference`, kept as the record of a stopped experiment | — | `0.5.1` |
| `graphify` | the vendor's fixed-argument steps plus a mechanical expansion — not the workflow its skill prescribes, a departure declared under `deviations` in its manifest | `graphify` installed — the distribution is `graphifyy` — and its index built over `source/` | `0.9.27` |
| `code-review-graph` | its `vector` and `keyword` configurations; the only shipped cells needing `properties.paths_in_index: absolute` with `build_cwd` | `code-review-graph` and `sqlite3`; `vector` also needs an offline `embed` against MiniLM weights frozen under its `models/` | `2.3.7` |

The version column above holds the values shipped, not suggestions: each is what its
`harness.yaml` pins, and setting the wrong one is the `version.cli` failure described below.
Systems you will not run need no editing; nothing loads a `harness.yaml` you never name.

`systems/<id>/docs/` holds that system's documentation frozen at the revision its recipe was
written against, so `audit quotes` can pin the exact bytes that authorised a call.

For the three `graphify-search*` cells it is this repository's own `SKILL.md` as it read when the
recipes were frozen, named by hash in each `harness.yaml`: the frozen text told the agent that
`bm25` answers are weaker, where the shipped text asks only that the mode be named — a quality
claim the owner withdrew.

For the two external tools it is a verbatim copy of the vendor's, and `systems/NOTICE.md` carries
the attribution and licence notices redistribution requires. A vendor of your own means a notice
there too.

### 1. The tool environments

`install` builds `envs/harness` from `lock/harness-requirements.txt`. The tool environments are
yours: four of them, for the five systems that run, since `graphify-search-bm25` shares the
`graphify-search` one. All four must be under `benchmark/envs/` — `install` reads every launcher a
recipe names and refuses one outside that directory, because a launcher elsewhere stays writable by
your own user and cannot be sealed by ownership.

Each is a Python 3.14 environment: that is the version in every `site` path the runnable recipes
name. The pins below are the shipped ones, `version_seen` in each `manifest.yaml` and `version.cli`
in each `harness.yaml`. Installing another version and leaving the pin is the `version.cli` failure
described under "Point the tree at your machine"; moving the pin as well makes a new recipe, and
with it a new group and a `docs/` re-freeze.

```bash
python3.14 -m venv benchmark/envs/code-review-graph
benchmark/envs/code-review-graph/bin/pip install "code-review-graph[embeddings]==2.3.7"

python3.14 -m venv benchmark/envs/graphify
benchmark/envs/graphify/bin/pip install graphifyy==0.9.27

uv build --wheel --out-dir dist      # graphify-search is this repository; build the wheel once
python3.14 -m venv benchmark/envs/graphify-search
benchmark/envs/graphify-search/bin/pip install dist/graphify_search-0.6.0-py3-none-any.whl

uv venv --python 3.14 benchmark/envs/graphify-search-minilm
uv pip install --python benchmark/envs/graphify-search-minilm/bin/python \
    "dist/graphify_search-0.6.0-py3-none-any.whl[local]"
```

What separates the `bm25` cell from the `dense` one is the index and the endpoint, not the
package, which is why its `harness.yaml` names the same launcher. `graphify-search-rewrite` pins
`0.5.1` and carries `status: reference`: nothing installs or runs it, and its `harness.yaml` ships
a placeholder path rather than a real one.

#### Point the tree at your machine

Five of the six shipped `harness.yaml` carry this machine's absolute paths under
`invocation.package`; only `graphify-search-rewrite` ships a `/path/to/your/…` placeholder. Rewrite
`launcher`, `interpreter` and `site` in every system you will run — all three under
`benchmark/envs/`, or `install` refuses them — and the absolute `command` and `build_cwd` of each
`build.yaml`.

`prepare.verify_packages` is what refuses a path that does not exist here, with `launcher missing:`
before anything runs; `seal --check` does not look at these paths at all. `site` is not decoration
either: its walked hash lands in every row as `environment_sha256`, so a `site` pointing at another
environment records a hash describing that one.

`version.cli` fails later and louder. `score.records` refuses a run whose `version` step printed
something else — `version output '<what it printed>' does not carry version <what you declared>` —
and it does so after the run has already executed. Changing a vendor version means re-freezing that
system's `docs/` and the `sha256` beside each name under `docs:`.

### 2. Obtain a corpus

One commit of one repository, and the subtree you mean to measure:

```bash
git clone <url> <scratch>/corpus && ( cd <scratch>/corpus && git checkout <commit> )
S=benchmark/record/snapshots/<id>
mkdir -p "$S/source" && cp -R <scratch>/corpus/<subtree>/. "$S/source/" && rm -rf "$S/source/.git"
```

Write `$S/meta.yaml` by hand. Only its `universe` block is read by code: `extensions` (each parsed
by that suffix's extractor — `.py`, `.ts`, `.tsx`, `.sql` and `.sh` ship one), `doc_roots` (a
markdown file under one is one whole unit) and `skip_dirs` (dropped at any depth, by path
component: agent tooling, tool output, dependency and build directories).

The rest is for a reader, and worth writing anyway: `repo` with `name`, `url`, `commit` and
`subtree` is what lets someone else obtain the same tree, and a `note` there saying which subtree
`source/` holds is what tells them that every path in a reference is relative to it. A `tree`
block with the file and byte counts is the cheapest check that they got the same thing.

### 3. Freeze it

Clean tree. Write `fileset.sha256` once — regenerating it can order its lines differently, and its
own hash is what each `build.yaml` pins:

```bash
( cd "$S/source" && find . -type f | LC_ALL=C sort | tr '\n' '\0' | xargs -0 shasum -a 256 ) > "$S/fileset.sha256"
.venv/bin/python -m benchmark.harness build-symbols benchmark/record/snapshots/<id>
```

`build-symbols` writes `symbols.jsonl` and `symbols.sha256` beside `source/`, sorted by path, start
line and full name, numbered `sym_00000` upward; each record carries the sha256 of its own body
lines. Files it could not parse, and files whose declared extension has no extractor
(`no extractor for .go: cmd/main.go`), are named on the error stream: read those lines, they are the
honest edge of your universe.

`record/snapshots/known_transitions.yaml` starts as `transitions: []`. It stays that way until an
admitted `build.yaml` change needs an entry, and while it says that, nothing is admitted.

### 4. Build one index per system

An index is built with no language model in the loop. Run each index's own build command, then
write its `build.yaml` once; the harness never writes into one.

The three indexes built from a source tree — `graphify` and the two `code-review-graph` ones —
each keep the ignore file that build ran under, tracked as `*ignore*.used` and hashed among the
artifacts, so write it before the build. The three `graphify-search` cells index the node list of
`graph.json` instead: they declare `ignore_file: null`, list no such file, and one dropped there is
`unlisted file in master index: ignorefile.used` out of `prepare.verify_master`.

No command writes `ignorefile.used`; you do, into `<snapshot>/indexes/<index>/`, before the build
that reads it. It holds what that tool's own ignore file holds — one pattern per line, `#` opening
a comment — and it may exclude nothing, in which case it is the record that nothing was excluded.
The `graphify` one the prefect campaign ran under, kept in git history, is the whole file:

```text
# The corpus is the frozen source/ tree in full: prefect@3a128c2, src/ subtree.
# Nothing is excluded, and this file is the record that nothing was.
```

The two cells that embed locally need their weights frozen first, and `freeze-model` prints the
`models:` block to paste into that system's `harness.yaml`. `--dst` must not already exist:

```bash
.venv/bin/python -m benchmark.harness freeze-model \
    --src ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 \
    --dst benchmark/systems/graphify-search-minilm/models/models--sentence-transformers--all-MiniLM-L6-v2
.venv/bin/python -m benchmark.harness freeze-model \
    --src ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 \
    --dst benchmark/systems/code-review-graph/models/models--sentence-transformers--all-MiniLM-L6-v2
```

**`graphify` first**, because three other indexes read the `graph.json` it writes:

```bash
G=<scratch>/graphify
rm -rf "${G:?}"; mkdir -p "$G" "$S/indexes/graphify"     # ignorefile.used is written here first
cp -R "$S/source/." "$G/"
cp "$S/indexes/graphify/ignorefile.used" "$G/.graphifyignore"
( cd "$G" && <repo>/benchmark/envs/graphify/bin/graphify update . )
cp "$G"/graphify-out/{graph.json,GRAPH_REPORT.md,manifest.json,.graphify_labels.json,.graphify_root} \
   "$S/indexes/graphify/"
```

**`code-review-graph`**, the `keyword` configuration's index, and **`code-review-graph-vector`**,
which adds vectors to that same freshly built database — so the first build happens even though
`keyword` carries `status: declared` and no attempt runs it. Each checkpoint line folds the
write-ahead log back into `graph.db`, leaving one file to hash instead of a database and two
sidecars:

```bash
C=<scratch>/index
CRG=<repo>/benchmark/envs/code-review-graph/bin/code-review-graph
rm -rf "${C:?}"; mkdir -p "$C" "$S/indexes/code-review-graph" "$S/indexes/code-review-graph-vector"
cp -R "$S/source/." "$C/"
cp "$S/indexes/code-review-graph/ignorefile.used" "$C/.code-review-graphignore"
( cd "$C" && "$CRG" build --repo "$C" -q )
sqlite3 "$C/.code-review-graph/graph.db" "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;"
cp "$C/.code-review-graph/graph.db" "$S/indexes/code-review-graph/graph.db"

H=<scratch>/probe-model/home
rm -rf "${H:?}"; mkdir -p "$H/.cache/huggingface/hub"
cp -a benchmark/systems/code-review-graph/models/models--sentence-transformers--all-MiniLM-L6-v2 \
   "$H/.cache/huggingface/hub/"
( cd "$C" && env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$H" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1 "$CRG" embed --repo "$C" )
sqlite3 "$C/.code-review-graph/graph.db" "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;"
cp "$C/.code-review-graph/graph.db" "$S/indexes/code-review-graph-vector/graph.db"
cp "$S/indexes/code-review-graph/ignorefile.used" "$S/indexes/code-review-graph-vector/"
```

**The three `graphify-search` cells**, each over the same `graph.json` and differing only in what
answers the embedding request. `env -i` is deliberate: the child inherits nothing the recipe did
not name, so what embedded an index is on record rather than in your shell.

```bash
D=<scratch>/graphify-search
rm -rf "${D:?}"; mkdir -p "$D/graphify-out" "$S/indexes/graphify-search"
cp -R "$S/source/." "$D/"; cp "$S/indexes/graphify/graph.json" "$D/graphify-out/"
( cd "$D" && env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$D" \
    GRAPHIFY_SEARCH_ENDPOINT=http://localhost:1234/v1 \
    GRAPHIFY_SEARCH_MODEL=text-embedding-nomic-embed-text-v1.5 \
    <repo>/benchmark/envs/graphify-search/bin/graphify-search index --require-dense --full )
cp -R "$D/graphify-out/." "$S/indexes/graphify-search/"
```

```bash
B=<scratch>/graphify-search-bm25
rm -rf "${B:?}"; mkdir -p "$B/graphify-out" "$S/indexes/graphify-search-bm25"
cp -R "$S/source/." "$B/"; cp "$S/indexes/graphify/graph.json" "$B/graphify-out/"
( cd "$B" && env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$B" NO_PROXY='*' \
    GRAPHIFY_SEARCH_ENDPOINT=http://127.0.0.1:9/v1 \
    GRAPHIFY_SEARCH_MODEL=text-embedding-nomic-embed-text-v1.5 \
    <repo>/benchmark/envs/graphify-search/bin/graphify-search index --full )
cp -R "$B/graphify-out/." "$S/indexes/graphify-search-bm25/"
```

Port 9 is the discard port: nothing listens there, the one embedding request is refused at connect,
and the builder writes the word-matching part alone. `--require-dense` is absent for exactly that
reason — it would turn the refusal into a failed build.

```bash
M=<scratch>/graphify-search-minilm
rm -rf "${M:?}"; mkdir -p "$M/graphify-out" "$M/.cache/huggingface" "$S/indexes/graphify-search-minilm"
cp -R "$S/source/." "$M/"; cp "$S/indexes/graphify/graph.json" "$M/graphify-out/"
cp -R benchmark/systems/graphify-search-minilm/models "$M/.cache/huggingface/hub"
( cd "$M" && env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$M" NO_PROXY='*' \
    GRAPHIFY_SEARCH_ENDPOINT=local GRAPHIFY_SEARCH_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TQDM_DISABLE=1 \
    <repo>/benchmark/envs/graphify-search-minilm/bin/graphify-search index --require-dense --full )
cp -R "$M/graphify-out/." "$S/indexes/graphify-search-minilm/"
```

Then write each `build.yaml`. What it must declare of the build: `system`, `version`, `command` (the
recipe above, as it ran), `build_cwd`, `ignore_file`, `language_model_used`, `embedding` for a cell
whose index carries vectors, and `built_from` — the snapshot, the hash of whatever this index was
derived from, and one sentence saying how to obtain `source/` again.

Its load-bearing keys are `artifacts` (every file of the index with its sha256), `mutable` (exact
paths the tool rewrites on its own) and `excluded` (path prefixes, matched by segment), which
between them must name anything else in the directory, `vendor_writes` (artifacts the tool rewrites
when it opens the index), `properties.paths_in_index`, and `build_cwd`, which is required when
paths are absolute.

`prepare.verify_master` checks each artifact before every run: `artifact differs:` on a mismatch,
`artifact missing:` when it is gone, `unlisted file in master index: <rel>` for anything the lists
do not cover. A `vendor_writes` path is hash-checked at the master index before the copy and again
after the run, never during it, so it is dropped from the per-run artifact map rather than reported
as tampering.

`graphify-search index` writes four files under `.graphify_search/`, `vectors.npy` among them only
where the endpoint answered at index time, so the `graphify-search-bm25` cell lists three.

`system`, `version`, `command`, `built_from` and `reported` are documentation: no gate reads their
values, but `audit attempts` compares whole top-level keys, so changing any of them after the
freeze is a `build.yaml` transition needing an entry in `known_transitions.yaml`.
`reported.embedder` is the only record of which embedder answered for an index embedded over HTTP.

### 5. Author the questions

Candidates live in `$S/questions/candidates/`, in two kinds of file joined by the candidate number
`n`. `candidates.jsonl` is the sample itself, one row per symbol: `n`, `sym`, `path`, `fqname`,
`bare`, `kind`, `start`, `end`, `lines` and `body` — the place a reference will name. The
`authored-*.jsonl` files are your judgement of each: `n`, `verdict` (`keep`, or anything else to
skip it), `reason`, the `question` text, and the `why` sentence that says how the place answers it.

Draw the sample from `symbols.jsonl` with a seed and write the seed down; walk it in order; skip a
symbol whose bare name is not unique, and one whose body carries no responsibility of its own;
record how many you walked past and why. A record carries no bare name of its own: a code symbol's
is the last dotted segment of its `fqname`, while a doc unit's `fqname` is its whole path. This
names every bare name occurring more than once, commonest first:

```bash
.venv/bin/python -c 'import collections,json,sys; recs=[json.loads(x) for x in open(sys.argv[1],encoding="utf-8") if x.strip()]; c=collections.Counter(r["fqname"].rsplit(".",1)[-1] for r in recs if r["kind"]!="doc"); [print(n,name) for name,n in c.most_common() if n>1]' "$S/symbols.jsonl"
```

Then follow `PROTOCOL.md` §5, which is the procedure and the review that admit a question:

```bash
.venv/bin/python -m benchmark.harness questions author <id>
.venv/bin/python -m benchmark.harness questions review-request q001
# a reviewer at least as strong as Opus answers from the request file alone
.venv/bin/python -m benchmark.harness questions review-check q001
```

Authoring a snapshot that already has questions appends: an existing file is left alone when the
bytes match, and the call is refused, naming every file whose bytes would change, because every
recorded row pins its question by `question_sha256`.

A ledger naming a question no snapshot holds fails with `no snapshot holds question <qid>`:
`config.question_path` raises it, `summary` propagates it, and `questions review-check` hands it
back as its refusal. `prepare` refuses earlier with `question <qid> has no review`, and `questions
author` and `questions review-request` name the file they could not find.

To see what one question's expansion block will hold before you author it:

```bash
.venv/bin/python -m benchmark.harness expand --max-tokens 12 \
    --graph benchmark/record/snapshots/<id>/indexes/graphify/graph.json \
    "which function turns a cart into an invoice total"
```

It prints one token list per system whose recipe takes one — `graphify` and `code-review-graph`;
the three `graphify-search*` cells pass the question verbatim.

### 6. Install and lock

§3 in full, and nothing past it: the block there seals the checkout, commits that seal, installs the
lock and takes the first `unlock` and `lock`; `lock` is what writes the seal every later attempt is
checked against. `install` refuses a launcher outside `benchmark/envs/`, so step 1 comes first, and
the seal covers the instrument files as this step leaves them. The first `lock` rewrites the seal
with this machine's interpreter and machine facts.

§0 owns what follows, in its order: the first `run`, the driven round, then `summary` and
`report` (§6). The
record fills as they go: `collect` writes `record/runs/<run_id>/` as each attempt closes, `summary`
writes `record/summary.json` and `record/SUMMARY.md`, and `report draft` creates `record/reports/`.

## 5. Run

From a clean tracked tree, as the owner. `PROTOCOL.md` §6 has the two forms and what each refuses;
this is the third, which fills a whole round within a time budget and then reports what is left:

```bash
.venv/bin/python -m benchmark.harness run missing baseline --seconds 480
.venv/bin/python -m benchmark.harness run missing driven \
    --model <model> --effort <effort> --max-actions 12 --max-tokens 8192 --seconds 480
```

`missing` derives its cells from `systems/`, in directory order: every `(system, configuration)`
whose index exists for that question's snapshot, skipping a system marked `status: reference`, a
configuration marked `status: declared`, and any cell tried past its retry count that still has no
second completed row. Repeat it until it reports nothing remaining.

No verb that touches the record, the instrument or a run takes a path: `run`, `attempt`, `abort`,
`summary`, `report`, `seal`, `audit` and `questions` accept none. The benchmark root is the
package's own location, and the sandbox root and the server URL come from
`benchmark/lock/machine.yaml`.

The three setup verbs — `build-symbols`, `freeze-model` and `expand` — take the file they operate
on and no record path. One argument names the object under inspection rather than a root: `audit
run` and `audit blind` take the live sandbox run directory, which is not part of the record.

A driven round needs a chat-completions server on the `base_url` that file names; LM Studio serves
one. Name the model, the effort and both caps every time: they are inputs, like the recipe, and
they key the round a row belongs to.

## 6. Read

```bash
H=$PWD/benchmark/envs/harness/bin/python
sudo -u bench "$H" -m benchmark.harness summary
sudo -u bench "$H" -m benchmark.harness report draft <snapshot> < your-template.md
sudo -u bench "$H" -m benchmark.harness report render <snapshot>
```

All three write under `record/`, which `install` hands to `bench`, and the `NOPASSWD` rule in
`lock/sudoers.template` names that interpreter by its absolute path, which is why `$H` is one.
Before `install` there is no `bench` and no `envs/harness`, so the same three verbs run as you,
with `.venv/bin/python` in place of both the elevation and `$H`.

A ledger row is the smallest unit: `run_id`, the cell, `question`, `attempt` and `outcome`, then
three families of keys.

The hashes of what it ran against: `question_sha256`, `reference_sha256`, `review_sha256`,
`harness_sha256`, `build_yaml_sha256`, `prepared_sha256`, `instrument_sha256`, and the launcher,
interpreter and environment. The hashes of what it produced: `journal_sha256`, `records_sha256`,
`cost_sha256`, `audit_sha256`.

Then the verdict and the cost — `hit`, `hit_rank`, `tokens`, `system_calls`, `ceiling_calls` and
`model_served`, which is on every row and null on a baseline — and, on a driven row only, `model`,
`effort`, both caps, `backend`, `base_url`, `stop_hit`, `hit_by`, `runner_actions` and `refused`.

`record/summary.json` is one object per group with every column defined once; `record/SUMMARY.md`
is the same table in Markdown. Both are untracked and rebuilt on demand, so anyone holding your
ledger can recount them: `bench` writes them and no verb commits them, and they are a function of
the ledger, the snapshots' `questions/` and `INSTRUMENT.yaml` alone.

Check the record itself with the audit subcommands: `attempts` for provenance, `quotes` for the
frozen vendor sentences, `stops`, `priors`, `expectations` and `recount` for what the run evidence
still says. `recount` prints a table and always exits clean; the others exit 1 and name each
finding.

```bash
.venv/bin/python -m benchmark.harness audit attempts
```

Figures describing the *shape* of an answer — its mode, its budget, the kind of each place — are
not in the ledger. They are read from that step's own `.out` under `record/runs/<run_id>/`, and
none of it is tracked, so only a machine that still holds the run directories can recount them.

Each executed step leaves three files under that directory: `NN_name.cmd`, the call together with
the documentation sentence that authorises it; `NN_name.out`, the output verbatim; and
`NN_name.err`.

The step's number is its position in that recipe: `02_query.out` for the three `graphify-search*`
cells, `03_query.out` for `graphify`, whose recipe extracts a vocabulary first, and `04_search.out`
for `code-review-graph`, whose recipe has no step called `query` at all.

**Repairing a stranded row.** A run the operating system killed leaves a row without an outcome,
and that blocks every later attempt. The next `run` commits any such row the ledger holds and `HEAD`
does not, and for a row that never reached an outcome at all it refuses, naming this command
with the real run id and interpreter:

```bash
sudo -u bench "$H" -m benchmark.harness abort <run_id>
```

It refuses a row that already ended, and a lock whose process is still alive —
`<run_id> is still running (pid <pid>); kill it before aborting`. It removes no evidence: a stranded
`record/runs/<run_id>` is reported for you to inspect. Restoring the ledger with git is not
available: it belongs to `bench`.

Where the kill fell decides which command closes the row. A run killed before or inside the
drive holds `run.yaml` and no `records.jsonl`, and `abort` closes it. A run killed after `score`
wrote `records.jsonl` and before `collect` closed it is one `collect` can still finish, and
`abort` refuses it, naming the command; `collect` is not in the sudoers grant, so that command
takes the password:

```bash
sudo -u bench "$H" -m benchmark.harness.collect "$TMP_ROOT/<run_id>/run"
```

It closes the row as `void` when the audit is not valid, and as `completed` otherwise.

## 7. Limits of a setup

- **Every recipe names paths on one machine.** A launcher, an interpreter, a `site` directory and an
  absolute `build_cwd` describe an installation, not something this repository ships. Two machines
  agree on a cell only if both were pointed at the same thing, and the hashes in the row are what
  say whether they were.
- **An embedder behind an HTTP endpoint has no hashed identity.** The row records the endpoint and
  the model name in the child's environment, so two machines serving different weights, or a
  different quantisation, under one name record the same `environment_sha256`.

  Write the server's `/v1/models` reply, and the model file's sha256 where the file is reachable,
  under `reported.embedder` in that index's `build.yaml`. A cross-machine agreement on such a cell
  is agreement between two installations whose embedder identity rests on that note.
- **A vendor whose output order depends on the interpreter's hash seed cannot pass the repeat
  gate on every question.** `graphify query` prints the edges of its traversal in an order that
  changes between processes; the node lines, which are what scoring reads, do not. A repeat
  attempt of such a question aborts at `prepare` and the question scores on its completed rows.
  Setting `PYTHONHASHSEED` in that recipe's `environment` makes the order repeat; it is a new
  recipe, and every earlier row of the cell stays under its own group.
- **Attempts cannot overlap.** They share one sandbox root, and `prepare` takes a lock under it; a
  second attempt started while the first holds the lock is refused, not queued. An index whose
  `build.yaml` declares absolute paths must be measured under the same root it was built under, or
  no returned path can be compared with a reference.
