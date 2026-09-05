# Benchmark protocol

What is measured, what may be done, and what physically stops anything else. `README.md` beside
this file is the manual: how to install the lock, point the tree at a corpus, run it and read it.

## 0. Read this first

- **Sealed** — every file of the instrument hashes to what `benchmark/INSTRUMENT.yaml` records, and
  an attempt that finds one byte different refuses before it prepares anything.
- **Locked** — the instrument is owned by `root` and flagged `uchg`; the record is owned by the
  account `bench`. An agent runs as the owner's user and can write neither.
- **The key** is the owner's `sudo` password, typed at a terminal, in two commands:

  ```bash
  sudo benchmark/lock/unlock "why the instrument is being changed"
  sudo benchmark/lock/lock
  ```

- **The record** is `benchmark/record/` — every snapshot with its questions and references, the
  run evidence, the reports, and the ledger `attempts.jsonl`, one row per attempt, appended by the
  harness under `bench` and committed by `run` under the owner. No hand writes the ledger.
- **One command** carries every operation, and its verbs are `run`, `attempt`, `abort`, `audit`,
  `expand`, `freeze-model`, `build-symbols`, `seal`, `questions`, `summary` and `report`:

  ```bash
  .venv/bin/python -m benchmark.harness seal --check
  ```

- **Four rules** bind what may be said about a figure. They are §8, and they are read before any
  sentence carrying a number is written.

## 1. Definitions

| term | definition |
|---|---|
| **snapshot** | one corpus freeze and everything written from it: `record/snapshots/<id>/`, holding `meta.yaml`, `fileset.sha256`, `symbols.sha256`, `indexes/`, `questions/` and `references/` |
| **record** | the directory `benchmark/record/**` and what running the benchmark leaves in it: the ledger, every snapshot, the run evidence and the reports; §3 splits that directory into seal classes, under which a snapshot's freeze records, questions and references are instrument rather than record |
| **place** | a file path with a start line and an end line; a file without lines is not a place |
| **reference** | the places fixed in advance from the source, independent of any system; each carries a `path`, `start`, `end`, an optional `symbol` and `qualified_name`, and a `why` sentence |
| **hit** | a returned place whose path equals the reference's, whose start falls inside the reference's `[start, end]`, and — when the reference names a `symbol` — whose parsed symbol equals it |
| **named** | a driven attempt whose own final answer was a reference place; the row's `stop_hit` |
| **cost** | tokens delivered plus the number of calls, recorded on every attempt, hit or miss; tokens are `cl100k_base` (`tiktoken`) over every executed call's own `.out`, and a system call is an execution of the vendor's command or MCP tool, never a prescribed script |
| **cell** | one `(system, configuration)` pair |
| **recipe** | the bytes of that system's `harness.yaml`, recorded on each row as `harness_sha256` |
| **round** | `baseline`; or `driven/local/<model_served>/<effort>/<max_actions>/<max_tokens>`; or `driven/legacy/<driver>` for a driven row predating those keys |
| **group** | `(snapshot, system, configuration, harness_sha256, instrument_sha256, round)` — the unit every published figure belongs to, and the unit that is never summed with another |
| **attempt** | one prepare-to-collect pass in one process: one ledger row, one commit |
| **scored row** | every completed non-repeat row of a baseline group; for a driven group, the completed non-repeat row of each question with the lowest `attempt` number |
| **void** | a row `collect` closed without a valid audit — evidence, never a verdict |

## 2. Rules of measurement

Each rule names what refuses when it is broken. A rule with nothing in that column would be a wish
and is not written here.

| rule | held by |
|---|---|
| Only a place counts as an answer; an edge, a candidate or a bare file name is not one | `score._is_hit` |
| Cost is recorded on every attempt, over every executed call's own output | `score.cost` |
| A step's output is compared after its declared volatile keys are dropped, never byte for byte | `rules.canonical_hash` |
| Every question names the snapshot it was written from; there is no default | `config.question_snapshot` |
| A row is never removed. A new campaign is a new snapshot, and every earlier row stays under its own group; the harness appends a row and merges an outcome into one, and no verb takes a row out | `record/` owned by `bench`; `ledger.append_row`, `ledger.complete_row` |
| The universe a question is drawn from is whatever the snapshot's `meta.yaml` declares | `build_symbols.load_universe` |
| An index's artifacts hash to what its `build.yaml` froze, and no unlisted file sits beside them | `prepare.verify_master` |
| The launcher and the interpreter a recipe names must exist before anything runs | `prepare.verify_packages` |
| A run whose version step printed another version is refused rather than scored | `score.records` |
| A vendor's output goes straight to its own `.out` file, never through a pipe the harness drains into memory | `execute.execute` |
| A returned path outside the index root a `build.yaml` declares is recorded as unparsed, never scored as a miss | `adapters.normalize_paths` |
| A call outside the grammar its `harness.yaml` declares never reaches the vendor | `rules.check_call` |
| Every quote in a harness or a manifest is still a sentence of the frozen documents, verbatim once whitespace is folded | `audit.quotes.check_quotes` |
| A vendor step that hangs, the MCP handshake included, is cut off at `execute.VENDOR_TIMEOUT_S`, journalled, and ends the run | `execute.execute`, `mcp.Server` |
| A later attempt of a cell must reproduce the recorded expectation of every fixed step, keyed `<configuration>.<harness_sha256>.<question>.<step>` | `prepare.run_fixed_steps` |
| An expectation may be added and may never be dropped or rewritten | `audit.attempts.check_attempts` |
| A change to a frozen `build.yaml` counts only where `record/snapshots/known_transitions.yaml` admits it by name, with a reason the entry cannot be written without; absent the file nothing is admitted | `audit.attempts.check_attempts`, `audit.attempts._known_transitions` |
| Weights under `models:` hash to what `freeze-model` recorded, with no other file in the directory | `rules.check_models` |
| Where the snapshot's `source/` is on this machine, it matches `fileset.sha256`, and `symbols.jsonl` matches `symbols.sha256` | `rules.check_corpus` |
| Neither the authored prompt nor the tool definitions may name a reference place | `prepare.check_blind`, `audit.blind.check_blind` |
| A reference naming a symbol shorter than three characters is refused rather than checked for blindness | `blind._terms` |
| A mechanical expansion is re-derived on every attempt of a step declaring `checks: [expansion]` — `graphify`'s `vocab_extract` alone today — against the vocabulary that run itself wrote | `prepare._check_expansion` |
| Every file appearing in the sandbox is a declared artifact or is attributed to a journalled call, with no exemption for `build.yaml`'s `mutable` list | `audit.run.check_run` |
| An attempt starts only from a clean tracked tree with no unignored untracked file under `benchmark/` | `ledger.require_clean` |
| Every git call made as `bench` is a read passing `--no-optional-locks` — `require_clean`'s one `status`, and the two `ls-files` the seal check makes to enumerate its candidates — so `bench` never writes inside `.git/`; every git call that writes, the commits included, runs as the owner | `ledger.require_clean`, `seal._candidates` |
| An attempt runs only sealed, locked, as `bench`, under the tmp root `benchmark/lock/machine.yaml` names | `rules.require_sealed` |
| A question runs only after a review that passes and does not withdraw it | `questions.check_review`, `prepare.prepare` |
| Every measured cell is prepared twice under one recipe — `run missing` keeps asking until two completed baseline rows exist, and no more; held by `run` | `run.BASELINE_ATTEMPTS`, `run._todo` |
| A third completed baseline attempt of a cell, and an unrepeated second driven attempt of a round, are refused; held by `run` | `run._refusal` |
| A driven row must name what answered: a missing or differing `model_served` aborts it | `attempt.attempt` |
| Exactly one commit per ledger row, subject `chore(benchmark): attempt <run_id> <outcome>` | `audit.attempts.check_attempts` |
| An index declaring absolute paths must record `build_cwd`, or scoring refuses rather than reporting a miss | `score._path_prefix` |
| A report carries no figure the harness did not print | `report.check` |

Five rules of the same kind carry no refusal, and are stated here as obligations on whoever writes
a reference, an adapter or a recipe rather than as gates:

- An empty field stays empty. A missing line number is never filled from another source.
- The candidates of an ambiguous reply are not places, even where one carries a path and a line
  range inside the reference span. A candidate is the system saying it could not resolve the name;
  scoring one would credit a system for declining to answer, and which candidate a reader would
  have picked is not something the harness can know.
- An index is built without a language model in the loop.
- Where a recipe departs from a prescribed argument, the departure, its ground in the vendor's own
  documentation, and the part of it the vendor does not cover are written under `deviations` in that
  system's `manifest.yaml`. A departure that is not written there is a defect, not a setting.
- A configuration a `harness.yaml` declares but no attempt ever runs publishes no result at all. It
  carries `status: declared`, so the gap is a recorded choice rather than a hole a reader has to
  notice for themselves.

## 3. The lock

Ownership, not a filter: what the operating system refuses, no command, script, interpreter,
editor or git operation can do.

| class | paths | owner | the owner's user, and every agent, may |
|---|---|---|---|
| instrument | the globs of `seal.INSTRUMENT`: `benchmark` itself, `benchmark/__init__.py`, `harness/`, `systems/`, `record/snapshots/` and each snapshot directory, `known_transitions.yaml`, each snapshot's freeze records (`meta.yaml`, `fileset.sha256`, `symbols.sha256`, `indexes/` and each `indexes/*/build.yaml`), its `questions/**` and `references/**`, `INSTRUMENT.yaml`, both documents, `tests/benchmark/`, `tests/test_comment_convention.py`, `pyproject.toml`, `.gitignore`, `.github/` | `root`, files `444`, directories `555`, `uchg` | read |
| lock | `benchmark/lock/**` — never opened by `unlock` | `root` | read; execute through `sudo` |
| record | `benchmark/record/**`, and each index's `prepared_outputs.yaml` — `record/snapshots/*/indexes/*/prepared_outputs.yaml`, which `seal.classify` reaches through a glob narrower than the system-under-test one covering the directory around it | `bench` | read |
| system under test | `benchmark/envs/**`, `systems/*/models/**`, and inside each snapshot `source/`, `symbols.jsonl` and the index artifacts | `bench` | read, execute |
| everything else | `src/`, `docs/`, `.git/`, `.venv/`, the rest of `tests/` | the owner | read, write |

`uchg` on a directory refuses creating, renaming and removing entries inside it; on a file it
refuses writing, renaming and unlinking. Both are needed: a file flag alone would let an unlisted
file be created beside a locked one.

**Who writes the seal.** `lock`, as root, while `UNLOCKED` exists. The one exception is the first
seal of a checkout — a tree with no `INSTRUMENT.yaml` and no `lock/machine.yaml` — which the owner
writes without root, under the reason `first seal of the checkout`: the instrument's file hashes
alone, no interpreter and no machine facts. The presence of either file closes the exception, and on
a machine `lock/machine.yaml`, root-owned inside root-owned `555` `lock/`, closes it for good.

A checkout re-seals in the open: remove `benchmark/INSTRUMENT.yaml`, write it again with `seal`, and
commit it alone under `chore(benchmark): seal — first seal of the checkout`. That subject is not the
operator's to choose — `write` stamps every non-root seal with the same reason, and `audit attempts`
reads a seal commit's subject as the prefix plus the sealed reason — so what records the change is
the diff of `INSTRUMENT.yaml`, not the wording.

What the seal proves on a checkout is that the instrument files match the committed seal — a
consistency record, not a proof of authorship, since the file is the owner's to edit until the
machine is locked.

**What git cannot do while locked.** `checkout`, `merge`, `stash pop`, `reset --hard`, `apply`,
`am`, `cherry-pick` and `revert` fail on any commit that changes a locked file or a record file. A
branch switch that touches the instrument is an instrument change: unlock first. Reading history is
unaffected. `git checkout HEAD -- benchmark/record/attempts.jsonl` is not a repair the owner can
make; the repair is `run`, which commits a stranded row, or `abort` for a row with no outcome.

**What the lock does not hold** is §10's second list.

## 4. Changing the instrument

1. Open the instrument with the first of §0's two commands, giving the reason a later reader will
   read in the seal. It is refused while any ledger row has no outcome. It writes
   `benchmark/lock/UNLOCKED` with the reason and the time, and hands the instrument's files and
   directories back to the owner. `$`, a backtick, `"` and `\` in a reason are the operator's to
   avoid; `:` and `#` survive.
2. Edit and test. Attempts are refused for as long as `UNLOCKED` exists.
3. Commit the change as the owner, ordinarily. The seal is not part of that commit.
4. Close it with §0's second command. It refuses while any path under an instrument glob is dirty
   or untracked; writes `INSTRUMENT.yaml`; commits it alone under the subject
   `chore(benchmark): seal — <reason>`; restores root ownership and the flags; removes `UNLOCKED`.
   With nothing to seal it prints `nothing changed` and makes no commit.
5. Confirm the boundary:

   ```bash
   .venv/bin/python -m benchmark.harness seal --check
   ```

6. The next attempt records the new `instrument_sha256`. `audit attempts` then requires a lock
   commit between the two rows either side of the change, carrying that same reason:

   ```bash
   .venv/bin/python -m benchmark.harness audit attempts
   ```

## 5. Authoring questions

Authoring writes a snapshot's `questions/` and `references/`, both under
`record/snapshots/<snapshot>/` and both instrument paths, so it happens unlocked (§4) and the
questions are committed and sealed before the first attempt.

1. Commit `record/snapshots/<snapshot>/questions/candidates/`, then:

   ```bash
   .venv/bin/python -m benchmark.harness questions author <snapshot>
   ```

   It writes that snapshot's `questions/` and `references/`, derives each mechanical expansion,
   and refuses a question whose own text names its answer. Each question is written from the
   source, never from any system's answer.

   Authoring for a snapshot that already has questions is an **append**. Question ids are global
   across snapshots, because `run_id` does not carry the snapshot and two snapshots sharing an id
   would collide in the ledger. `questions.author` reuses this snapshot's committed ids
   positionally over the candidates whose verdict is `keep`, in `n` order, and gives each candidate
   past that count a number above every snapshot's highest (`questions._highest_question_number`).

   A second pass over an unchanged set renumbers nothing, and that holds while the kept set only
   grows by appending: a re-shaped set moves a committed id onto another question's bytes, and this
   refusal fails it closed. A file on disk is left untouched when the bytes match, and the call is
   refused whole, naming the file, when they differ: `question_sha256` on every recorded row pins
   the committed question, so a re-author is a refusal, not a correction.
2. Ask for the reviewer's file — the only text the reviewer is given:

   ```bash
   .venv/bin/python -m benchmark.harness questions review-request <qid>
   ```

   It writes that snapshot's `questions/review/<qid>.request.yaml`: the question and the
   reference verbatim, their sha256s, the snapshot id, and the fixed reviewer instruction.
3. The reviewer is **a model at least as strong as Opus**, given that request and read access to
   the snapshot's `source/`, and nothing else. It writes `questions/review/<qid>.yaml` beside the
   request, with `reference_is_right`, `question_is_ambiguous`, `note`, `reviewer_model`, and the
   two hashes as the request states them. `reviewer_model` names the model that answered.
4. Check every review, then commit questions, references, requests and reviews together:

   ```bash
   .venv/bin/python -m benchmark.harness questions review-check <qid>
   ```

5. Close the instrument with §0's second command. Only a sealed question runs.

A question found wrong after an attempt is never edited: an edit changes its hash, and
`review-check` then refuses it against the request the reviewer answered. A later review file
setting `reference_is_right: false`, or `question_is_ambiguous: true`, withdraws it; its rows stay.
A changed expansion is a new question id, not a correction of an old one.

## 6. Running

One attempt, from a clean tracked tree, as the owner:

```bash
.venv/bin/python -m benchmark.harness run baseline <system> <configuration> <qid>
.venv/bin/python -m benchmark.harness run driven <system> <configuration> <qid> \
    --model <model> --effort <effort> --max-actions 12 --max-tokens 8192
```

`run` commits any row the ledger holds that `HEAD` does not, refuses the attempt when §2's
attempt-count rules say so, calls `attempt` through `sudo -u bench`, then makes the attempt's one
commit and prints one JSON line: `run_id`, `outcome`, `stop`, `hit`, `hit_rank`, `tokens`,
`system_calls`, `ceiling_calls`, `runner_actions`, `refused`, `seconds`.

`attempt` runs `require_sealed`, `prepare`, the drive, `audit blind`, `audit run`, `score` and
`collect` in one process owned by another user, so there is no moment at which a verdict can be
read and the row then withheld. Any failing step marks the row `aborted` with `failed: <step>`.

**An aborted row is a result.** It stays; the next attempt is a new row under the same settings.
Different settings are a different cell and a different group. A row the operating system killed is
closed with `abort <run_id>`, which takes no path and refuses a row that already ended.

**First, not best.** A driven group scores each question's *first* completed attempt — the lowest
`attempt` number, never ledger order. `--repeat` marks a further attempt `repeat: true`, which
`summary` lists and scores in nothing. A figure cannot rise by trying again. Baseline rows agree by
construction, so `summary` keeps "all rows hit" for them.

The first attempt of a question under a cell and a recipe has no expectation to meet. `attempt`
passes `needs_record`, true in exactly that case, so the attempt records one itself and no operator
names `--record-prepared`.

`prepare.run_fixed_steps` refuses with `no expectation for <configuration>/<qid>/<step>; rerun with
--record-prepared` only where the question is recorded and that one step is not — the state a
first attempt killed or aborted between two steps leaves, since each step's expectation is written
as it passes — or where `prepare` is called directly.

### The driven round

A driven attempt writes `prompt.md` into the run directory and leaves the journal open for the
drive to close. The model, the effort and both caps are inputs like the recipe, and `run` requires
all four on its command line. No shipped system declares `runner_defaults`, so
`drive.runner_settings` has nothing to fall back on; `drive.DEFAULT_MAX_TOKENS`, 16000, is reached
only by calling the `drive` module directly, which no documented path does.

Blindness is shown three times, each time over every reference place the text names — the
qualified name, the path, the bare file name, the symbol and the place's own `why` sentence,
matched case-insensitively.

| when | what reads what |
|---|---|
| the prompt is written | `prepare.check_blind`, over the `prompt.md` it has just written |
| before the first request leaves | `prepare.check_blind` again, called from `drive._drive` with the tool definitions it is about to offer |
| after the drive | `audit.blind.check_blind`, over `request.json` — the record of what was sent, which a later edit of `prompt.md` cannot change |

Only the authored half is searched, so a fixed step that happens to print the reference is the
measurement and not a leak. Run before the drive, `audit blind` prints `no driven attempt in <run
directory>: nothing to check` and exits 0, which is also what it prints for a baseline run, whose
journal names no runner.

**Actions and the ceiling count different things.**

| figure | every |
|---|---|
| `runner_actions` | journalled call carrying `by: runner`, refused ones included |
| `refused` | call the grammar turned away — `rules.check_call` returning a refusal, which `execute.execute` stamps `action: false` — and nothing else |
| `ceiling_calls` | *executed* entry flagged `ceiling_call`: the fixed steps a `harness.yaml` marks so, alongside the runner calls that reached the vendor. Never `runner_actions` less `refused` |

A step declaring `ceiling_call: true` is counted the same way whether it is a command-line call or
an MCP tool call, and `prepare` stamps it from that step's own flag — which is why a baseline
attempt with no runner in it still records ceiling calls.

The declared `ceiling` reaches the runner's own calls alone. `drive.ceiling_tools` returns an empty
set where a system declares none, leaving `ceiling_calls` the fixed ceiling steps alone; all six
shipped systems declare one, so on those cells every runner call that reached the vendor counts.

Two things a runner spends sit outside all three figures:

- a second tool call in one turn, declined inside `drive.loop`, charged against `--max-actions` and
  answered `one call per turn: this call was not executed`, writing no journal entry;
- a tool call that raises, journalled by `CallRunner._tool` with `exit: null`, an `error` and no
  `.out`: it counts in `runner_actions`, it is not `refused` since it carries no `action` key at
  all, and `score._executed` drops that exact shape from every cost figure.

The `graphify` recipe measures graphify plus a mechanical expansion, not the workflow its skill
prescribes: the vendor's step 0 asks an agent to select the query tokens by meaning, and a baseline
round derives them by prefix instead, keeping a vocabulary token whose first `expand.STEM`
characters match the question word's.

Where that rule selects nothing the recipe does what step 0 prescribes: `prepare.halt_reason`
journals the step carrying `<expansion>` as halted, `action: false` and no output, and the attempt
completes as a miss rather than aborting. Only a driven round has a model making that selection
itself, as the vendor describes.

## 7. Summary and report

```bash
H=$PWD/benchmark/envs/harness/bin/python
sudo -u bench "$H" -m benchmark.harness summary
```

Both verbs of this section write under `record/`, which belongs to `bench`, and the `NOPASSWD` rule
names that interpreter by its absolute path, which is why `$H` is one. On a tree where `install`
has not run there is no `bench` and no `envs/harness`, and the same verbs run as the owner with
`.venv/bin/python` instead.

`summary` reads the ledger, every snapshot's `questions/` and `INSTRUMENT.yaml` — the last for
the seal reason it prints above the table — groups by §1's `group`, and writes
`record/summary.json` and `record/SUMMARY.md`. Two runs over one ledger produce identical bytes.

Its columns are defined once each: `hit`, `hit_at_1`, `hit_at_5`, `named`, `hit_by_runner`,
`refused`, the three `*_median` costs by `statistics.median_low`, `model_input_tokens`,
`model_output_tokens`, `ceiling_reached`, `agreement`, and `per_question` with `added_hit`.

```bash
H=$PWD/benchmark/envs/harness/bin/python
sudo -u bench "$H" -m benchmark.harness report draft <snapshot> < template.md
sudo -u bench "$H" -m benchmark.harness report render <snapshot>
```

`draft` reads the template on stdin and writes `record/reports/<snapshot>.md.in`; `render` writes
`record/reports/<snapshot>.md`. A figure is a placeholder,
`{{<column> <system> <configuration> <round>}}`, optionally qualified `@<seal8>/<recipe8>` and
optionally ending in `partial`.

Both verbs check the template *before* substitution and refuse: an unknown column or group; a
placeholder matching more than one group; a group whose `questions` is below its `sealed_questions`
without `partial`; a column that is null for the matched group; any digit outside a backtick span; a
backticked digit run outside the closed literal set; and any word of the number list outside a
backtick span.

## 8. What an agent says

1. A figure is quoted from `record/summary.json` or a rendered report, never computed by hand and
   never re-read from run evidence into a sentence.
2. A figure carries its group. Two groups are named separately or not at all; they are never summed,
   averaged, or presented as one cell.
3. A group whose `questions` is below its `sealed_questions` is quoted as partial, in the form
   `render` produces, or not quoted.
4. A comparison the render grammar cannot check — "most", "leads", "twice as often" in other
   words — is not asserted. It is left to §9.

## 9. Closing a campaign

A fresh reviewer, a model at least as strong as Opus with no context from the session that produced
the work, is given this file, `record/summary.json` and the rendered report. It reads for what the
grammar cannot catch:

- a comparative carried by words rather than digits;
- a figure quoted without the group it belongs to, or across two groups;
- a claim about a system where the evidence is a claim about one question set on one snapshot;
- an admitted `build.yaml` transition falling between a published cell's own attempts, and what it
  moved.

Its findings are recorded beside the report, not folded into it silently.

## 10. Limits

### What a figure from this harness cannot show

- **A verdict is a statement about one question set on one snapshot.** Nothing here supports a
  ranking of the systems in general. A question set large enough to separate two systems by
  retrieval quality is a different instrument; one small enough to author by hand can reorder two
  cells outright when the questions change.
- **`hit_rank` is not one scale across a row, and neither is `hit`.** A rank among every place one
  system prints and a rank among the handful another is capped at are different measurements, and
  `hit` — the reference anywhere in the printed list — inherits the bias. `hit_at_5` is the one
  hit figure on every shipped system's scale, since five is the shortest list any of them prints:
  `code-review-graph`'s search step declares `limit: 5`. Where a system's `hit@N` equals its find
  count, that is a property of its cap and not of its ranking.
- **The expansion is measured along with the system.** Where a recipe takes a token list rather than
  the question, part of what is compared is the expansion procedure. Deriving it mechanically
  removes a hand from any one question, not the procedure from the comparison.
- **A markdown place is reachable only by a system that indexes markdown**, and a reference naming a
  symbol can never be hit by a document place, however well ranked — before retrieval quality
  enters.
- **A reference's optional `symbol:` key changes what counts as a hit.** A node whose label is prose
  parses to no symbol and is passed over even inside the reference span.
- **A step's timeout kills the child the harness started and nothing below it.** A vendor that
  forks a worker and hangs is cut off on schedule while the worker keeps running, holding whatever
  it had open. No fixed step's vetted argv forks today; closing it needs a process group.
- **Where a recipe stops is part of the result.** A workflow whose later steps take arguments from
  earlier output is executed only as far as its fixed-argument prefix in the baseline round.
- **Two token counts are rarely one quantity.** They differ in what they cover and in the sandbox
  itself: an absolute path repeated through a reply is charged at the length of the path, which is a
  property of where the sandbox sits. The figure covers every executed call, the version probe
  included, and is wider than a vendor ceiling stated over the calls a model would consume.
- **A declared call ceiling is recorded, not enforced, during the fixed steps.** It binds the runner
  through `drive.ceiling_left`; a recipe whose own fixed steps exceed the vendor's number is a
  recipe to fix, and no check reports it.
- **The repeat check compares hashes, not behaviour.** Two completed attempts agree by construction,
  so `agreement` records that the gate held, not that replication was observed. A step with a
  language model inside the retrieval cannot be measured under this gate at all.
- **The adapter is not covered by the recipe hash.** A later parser leaves older runs holding
  records it would no longer write; their `.out` files are the evidence, their `records_sha256` is a
  hash of that day's adapter.
- **Blind wording does not make a benchmark independent.** Whoever writes the questions, the
  references and the recipes is the same party reporting the result.

### What the lock does not hold

- **Root.** Whoever holds the password can do anything; that is the design.
- **§2's attempt-count rules are held by `run` alone.** The sudoers grant names `attempt` itself,
  so `sudo -u bench <H> -m benchmark.harness attempt …` starts an attempt without passing `run`'s
  refusals — a third completed baseline of a cell, or an unrepeated second driven attempt of a
  round. Narrowing the grant would not close it: whoever may run `run` may type the same line. The
  ledger still records every such row, so `summary` sees it and `audit attempts` can be taught to
  count them (§11).
- **The local model server** runs as the owner. A driven row records the name the server returned
  and refuses when it is absent or differs from the one asked for; it cannot know which weights were
  loaded under that name.
- **Ignored files under an instrument glob stay owner-writable**, and ownership is applied to a
  shallow depth — a directory and its direct children.
- **`install` checks the launchers a recipe names, not the paths a `build.yaml` names.** It refuses
  a `launcher`, `interpreter` or `site` outside `benchmark/envs/`; a `command` or a `build_cwd`
  pointing elsewhere is the owner's to catch.
- **Nothing outside the repository is sealed.** `bench`'s home and caches are not owner-writable, but
  they are not hashed either.
- **A reviewer's freshness cannot be recorded.** §5 records what the reviewer was given and which
  model answered; no file can show what context that model had.
- **A quote proves the sentence exists, not that it authorises the call.** `audit quotes` shows
  that every quoted sentence is still verbatim in the frozen documents; what bounds a call is the
  grammar in `harness.yaml`, and reading a quote as permission is the reader's own step.
- **Attribution in the sandbox is by path only.** The sha256 a journal entry records beside a file is
  never re-compared to disk afterwards, so the audit sees that a known call wrote a path, not that
  the content is still what that call produced.
- **The absolute-path rule cannot tell a leak from a quotation.** Existence on this machine separates
  a real path from a fabricated one, never a leaked one from one the corpus itself quotes. A string
  occurring verbatim inside one of the run's own declared artifacts is corpus content and passes:
  that artifact is what the sandbox was laid out from, `prepare.verify_master` hashed it before the
  run, and the system under test cannot edit it.
- **`audit run` re-checks attribution against the live shared sandbox**, so it re-runs cleanly only
  for the configuration that ran last; for the others the hash-pinned `audit.json` is the record.
- **Nothing checks that a recipe change is substantive.** Any edit to a `harness.yaml` yields a new
  recipe and an empty baseline. `audit rebaseline <system>` compares the new recipe's expectations
  with the old, but the owner runs it; it is not a gate on `run`.
- **A row written before a key existed anchors nothing.** `_transitions` skips a row carrying no
  `build_yaml_sha256` or no `prepared_sha256` and joins the rows on either side, so each of those
  rules binds only from the first row that carries its key.
- **The `build.yaml` check compares whole top-level keys.** Correcting a provenance note between two
  attempts is indistinguishable to it from changing a hash or a count. Correcting the note is still
  right — an uncheckable citation is a defect — and admitting it in `known_transitions.yaml`,
  with the reason beside it, is what that costs.
- **`audit attempts` has three scope holes.** The first sealed row anchors nothing; a pair of
  consecutive rows carrying the same seal hash is skipped; and a seal changed after the last row is
  caught only by the working-tree-versus-`HEAD` comparison.
- **`_blob_sha256` (`audit/attempts.py:93`) runs `git show` with `check=False`.** A git failure
  yields empty bytes, so the hash becomes the sha256 of nothing, `e3b0c442…`. Against an on-disk
  seal that is a mismatch — a false alarm, the loud direction. In the seal-history walk, two
  consecutive failures hash equal and the same-hash-pair skip hides them: that is the silent
  direction.
- **The seal job in CI compares the instrument files with `INSTRUMENT.yaml`**; a branch that
  changes an instrument file and does not re-seal is red there.

### What the record as it stands does not carry

- **The record is empty.** Every attempt made before this protocol — 883 rows, none under a seal,
  173 of them driven through a backend that no longer exists, and every one over question reviews
  whose judges had seen the systems' output — was cleared, and its evidence with it, before the
  first lock. Git history holds them; no figure is taken from history, and `audit attempts`
  walks the ledger's history from the clearing commit `audit.RECORD_CLEARED` names.
- **No campaign ships.** `record/snapshots/` holds `known_transitions.yaml` and nothing else: a
  clone carries no corpus freeze, no question and no reference, and every figure a reader wants
  starts at README §4. A question authored there runs only after §5 — `prepare` refuses one with
  `question <qid> has no review` until the request, the review and `review-check` have run for it.
- `audit expectations` compares every prepared output against the **current** `build.yaml`, not the
  `build_yaml_sha256` each attempt recorded, so its findings shadow transitions
  `record/snapshots/known_transitions.yaml` already admits. `audit priors` reports corpus
  recognition, which is evidence about a runner's own query, not a verdict.
- Re-running `questions author` on a snapshot that already has questions writes nothing where the
  bytes match and refuses, naming every file whose bytes would change. A committed question is
  pinned by `question_sha256` on every row that ran it, so a rewrite is a refusal rather than a
  drift.
- A report template carries no fenced code block and no double-backtick span: the render grammar
  treats a single backtick pair as the only span, so a fenced block is checked as one span and the
  inside of a double-backtick span is checked as prose. It fails closed, refusing both.
- A backticked run of seven or more characters reads as an id only when it carries at least one
  `a-f`, so an abbreviated hash that happens to be all decimal digits is refused as a figure. It
  fails closed: the author lengthens the abbreviation until a letter appears in it.
- `report draft` and `report render` take a snapshot that must be one bare directory name under
  `record/snapshots/` — a name carrying `/`, or starting with `.`, is refused — and a
  placeholder resolves only among that snapshot's own groups. The `models:` literal class is the
  set of `models:` keys across every `systems/*/harness.yaml` with `models--` stripped.

## 11. Open

- `collect` in the sudoers grant beside `abort`, so a run killed between `score` and `collect`
  is closed without the password. `lock/sudoers.template` is lock class, never opened by
  `unlock`, so the grant moves only with `uninstall` and `install`.
- An `audit attempts` check that counts the completed baseline and driven rows of each cell and
  recipe, so an attempt started around `run` is reported by the record rather than only recorded
  in it.
- Whether `questions author` may leave a committed id whose bytes differ untouched and write only
  the ids it has never written, so a snapshot whose provenance wording has moved on can still take
  an append; today that call is refused whole and the snapshot's question set is closed where it
  stands.
- Thresholds over a question set: what count separates two cells, and on which column.
- Whether a shared reference set of questions can exist at all, given that a snapshot is private by
  construction.
- Whether the closing reviewer's findings should reach `report draft` as a second pass on stdin or a
  verb of their own. Preference: one verb, one stdin.
