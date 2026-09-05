"""Check the committed ledger's provenance, and one system's re-baseline, against git history."""

from __future__ import annotations

import hashlib
import re
import subprocess
from typing import TYPE_CHECKING

import yaml

from benchmark.harness import config, ledger, prepare, rules, seal
from benchmark.harness.audit._shared import _GIT

if TYPE_CHECKING:
    from pathlib import Path

# inv: every commit the harness itself writes to the ledger opens with this exact prefix
_ATTEMPT_PREFIX = "chore(benchmark): attempt "
# inv: every commit `lock` writes when it reseals INSTRUMENT.yaml opens with this exact prefix
_SEAL_PREFIX = "chore(benchmark): seal — "
_PREPARED_OUTPUTS_KEY = "prepared_outputs"
# NOT DERIVED: the sha of the commit that emptied the ledger on the owner's order; it buys the
# provenance walk a start, so the rows that commit deleted are history rather than a violation
RECORD_CLEARED: str = "32fabafbd67d50f4aefcd181288403d2776b52b0"
# inv: a sentinel distinct from every YAML value, so a dropped expectation cannot be mistaken
# for one whose recorded value happens to be null
_MISSING = object()


def _git_lines(benchmark: Path, *args: str) -> list[str]:
    proc = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
        [_GIT, "-c", "commit.gpgsign=false", "-c", "core.excludesFile=/dev/null",
         "-C", str(benchmark.parent), *args],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {list(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.splitlines()


_RUN_ID_RE = re.compile(r'"run_id":\s*"([^"]*)"')


def _row_diff(benchmark: Path, sha: str, *rels: str) -> tuple[set[str], set[str]]:
    """Return the run_ids of lines one commit adds to, and removes from, the ledger."""
    added: set[str] = set()
    removed: set[str] = set()
    # why: a pathspec naming only one side of a rename stops git from pairing it with the
    # other side, so the commit reads as a full add instead of a content-free rename; naming
    # every known name lets git pair them and collapse the diff to nothing
    for line in _git_lines(benchmark, "show", "--format=", "-p", sha, "--", *rels):
        if line.startswith(("+++", "---")):
            continue
        match = _RUN_ID_RE.search(line)
        if match is None:
            continue
        if line.startswith("+"):
            added.add(match.group(1))
        elif line.startswith("-"):
            removed.add(match.group(1))
    return added, removed


def _has_commit(benchmark: Path, sha: str) -> bool:
    """Return whether `sha` names a commit of the repository holding `benchmark`."""
    proc = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
        [_GIT, "-C", str(benchmark.parent), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode == 0


def _blob_sha256(benchmark: Path, rev: str, rel: str) -> str:
    """Return the sha256 of `rel` as it reads at `rev`."""
    blob = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
        [_GIT, "-C", str(benchmark.parent), "show", f"{rev}:{rel}"], capture_output=True, check=False).stdout
    return hashlib.sha256(blob).hexdigest()


def _blob_reason(benchmark: Path, sha: str) -> str:
    """Return the `reason:` field of INSTRUMENT.yaml as it reads at `sha`."""
    blob = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
        [_GIT, "-C", str(benchmark.parent), "show", f"{sha}:benchmark/INSTRUMENT.yaml"],
        capture_output=True, check=False).stdout
    return str((yaml.safe_load(blob) or {}).get("reason", ""))


def _seal_commits(benchmark: Path) -> list[tuple[str, str, str, list[str]]]:
    """Return (sha, subject, blob sha256, files touched) of every commit that changes INSTRUMENT.yaml, newest first."""
    rel = "benchmark/INSTRUMENT.yaml"
    out = []
    for sha in _git_lines(benchmark, "log", "--format=%H", "--", rel):
        files = [f for f in _git_lines(benchmark, "show", "--name-only", "--format=", sha) if f.strip()]
        subject = _git_lines(benchmark, "show", "-s", "--format=%s", sha)[0]
        out.append((sha, subject, _blob_sha256(benchmark, sha, rel), files))
    return out


def _between(benchmark: Path, older_row_sha: str, newer_row_sha: str) -> set[str]:
    """Return every commit sha strictly after `older_row_sha` up to and including `newer_row_sha`."""
    return set(_git_lines(benchmark, "rev-list", f"{older_row_sha}..{newer_row_sha}"))


def _known_transitions(benchmark: Path) -> set[tuple[str, str, str, str]]:
    """Return every build.yaml transition admitted by hand, as (system, configuration, before, after).

    Parameters
    ----------
    benchmark : Path
        Root whose `record/snapshots/` holds `known_transitions.yaml`.

    Returns
    -------
    set of tuple
        One entry per admitted transition; an entry without a reason is not admitted.

    Raises
    ------
    SystemExit
        When an entry carries no reason.
    """
    path = benchmark / "record" / "snapshots" / "known_transitions.yaml"
    if not path.is_file():
        return set()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = set()
    for entry in doc.get("transitions") or []:
        # why: silencing a transition costs an explanation the next reader can check against the diff
        if not str(entry.get("reason") or "").strip():
            raise SystemExit(f"known_transitions.yaml: {entry.get('before')}->{entry.get('after')} has no reason")
        known.add((entry["system"], entry["configuration"], entry["before"], entry["after"]))
    return known


def _transitions(rows: list[dict], key: str) -> list[tuple[str, str]]:
    """Return each consecutive change of `key` across rows, as (before, after) pairs.

    Parameters
    ----------
    rows : list of dict
        Ledger rows for one (system, configuration), in ledger order.
    key : str
        The row field whose changes make a transition.

    Returns
    -------
    list of tuple
        Pairs of hashes; rows missing the key contribute no transition.
    """
    seen: list[str] = []
    for r in rows:
        value = r.get(key)
        # inv: a row without the key predates it and can anchor nothing; skipping it joins the
        # rows on either side, which is a transition that did happen
        if isinstance(value, str) and (not seen or seen[-1] != value):
            seen.append(value)
    return list(zip(seen, seen[1:], strict=False))


def _transition_rows(rs: list[dict], key: str) -> list[tuple[dict, dict]]:
    """Return (older row, newer row) for every consecutive pair whose `key` values differ.

    Parameters
    ----------
    rs : list of dict
        Ledger rows for one (system, configuration), in ledger order.
    key : str
        The row field whose changes make a transition.

    Returns
    -------
    list of tuple
        Pairs of rows; rows missing the key contribute no pair.
    """
    carrying = [r for r in rs if r.get(key)]
    return [(a, b) for a, b in zip(carrying, carrying[1:], strict=False) if a[key] != b[key]]


def _index_rel(benchmark: Path, system: str, configuration: str, qid: str) -> str:
    h = config.load_harness(benchmark, system)
    snapshot = config.question_snapshot(config.load_question(benchmark, qid), qid)
    index_dir = config.snapshot_dir(benchmark, snapshot) / h.configurations[configuration]["index"]
    # why: config.snapshot_dir resolves symlinks, so the repository root it is made relative to
    # is resolved too -- an unresolved root would not be a prefix of the resolved index path
    return index_dir.relative_to(benchmark.resolve().parent).as_posix()


def _prepared_rel_for(benchmark: Path, row: dict) -> set[str]:
    """Return the row's cell's `prepared_outputs.yaml` path, or empty when it cannot be resolved.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/` and `record/snapshots/`.
    row : dict
        A ledger row, read for its `system`, `configuration` and `question`.

    Returns
    -------
    set of str
        One path, or an empty set when the row's system or question is not configured
        on this tree.
    """
    try:
        index_rel = _index_rel(benchmark, row["system"], row["configuration"], row["question"])
        return {f"{index_rel}/{prepare.PREPARED}"}
    except (OSError, KeyError, ValueError, config.ConfigError):
        return set()


def _content_by_hash(benchmark: Path, rel: str) -> dict[str, bytes]:
    """Map every sha256 `rel` has carried, across its full commit history, to its content bytes."""
    by_hash: dict[str, bytes] = {}
    for sha in _git_lines(benchmark, "log", "--format=%H", "--", rel):
        proc = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
            [_GIT, "-c", "commit.gpgsign=false", "-c", "core.excludesFile=/dev/null",
             "-C", str(benchmark.parent), "show", f"{sha}:{rel}"],
            capture_output=True, check=False,
        )
        if proc.returncode != 0:
            continue
        by_hash.setdefault(hashlib.sha256(proc.stdout).hexdigest(), proc.stdout)
    return by_hash


def _build_yaml_transition(before: bytes, after: bytes) -> tuple[set[str], bool, bool]:
    """Diff two `build.yaml` revisions' top-level keys.

    Parameters
    ----------
    before : bytes
        The revision's content before the transition.
    after : bytes
        The revision's content after the transition.

    Returns
    -------
    tuple of (set of str, bool, bool)
        Top-level keys other than `prepared_outputs` that changed; whether
        `prepared_outputs` is among the changed keys at all; and whether every
        expectation it already recorded survived the transition unchanged.
    """
    # inv: comparing parsed YAML, not raw bytes, so a rewrite that only reformats or drops a
    # comment (yaml.safe_dump does both) is never mistaken for a change to the file's content
    b, a = yaml.safe_load(before) or {}, yaml.safe_load(after) or {}
    changed = {k for k in set(b) | set(a) if b.get(k) != a.get(k)}
    other = changed - {_PREPARED_OUTPUTS_KEY}
    po_touched = _PREPARED_OUTPUTS_KEY in changed
    leaves_before, leaves_after = _prepared_leaves(b), _prepared_leaves(a)
    additive = all(leaves_after.get(k, _MISSING) == v for k, v in leaves_before.items())
    return other, po_touched, additive


def _prepared_leaves(doc: dict) -> dict[tuple, object]:
    """Flatten recorded expectations to one entry per `(configuration, recipe, question, step)`.

    Parameters
    ----------
    doc : dict
        A parsed `prepared_outputs.yaml`.

    Returns
    -------
    dict
        One entry per recorded expectation.

    Notes
    -----
    A level that is not the mapping it should be becomes one opaque leaf.
    """
    # why: an opaque leaf rather than an exception, so one malformed revision is compared and
    # reported instead of aborting the audit before every other violation it was about to raise
    leaves: dict[tuple, object] = {}
    if not isinstance(doc, dict):
        return {} if doc is None else {(_MISSING,): doc}
    for cfg, by_recipe in doc.items():
        if not isinstance(by_recipe, dict):
            leaves[(cfg, _MISSING)] = by_recipe
            continue
        for recipe, by_question in by_recipe.items():
            if not isinstance(by_question, dict):
                leaves[(cfg, recipe, _MISSING)] = by_question
                continue
            for qid, steps in by_question.items():
                if not isinstance(steps, dict):
                    leaves[(cfg, recipe, qid, _MISSING)] = steps
                else:
                    leaves.update(((cfg, recipe, qid, step), exp) for step, exp in steps.items())
    return leaves


def check_attempts(benchmark: Path) -> list[str]:
    """Check the committed ledger's provenance and internal consistency.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `systems/`; its parent is the git repository.

    Returns
    -------
    list of str
        One message per violated invariant; empty when the ledger is clean.
    """
    # why: imported here, not at module load -- which would cycle back through __init__ -- so
    # audit.RECORD_CLEARED, not this module's own binding, is the sha a caller can override
    from benchmark.harness import audit

    rel = ledger.rel(benchmark)
    # inv: an uncommitted edit gives the provenance walk below no commit to inspect, so a dirty
    # ledger is refused before that walk runs
    if _git_lines(benchmark, "status", "--porcelain", "--", rel):
        return [f"{rel}: uncommitted changes; commit or discard them before auditing the ledger"]
    problems: list[str] = []
    rows_ = ledger.rows(benchmark)
    row_by_id = {r["run_id"]: r for r in rows_}
    update_counts: dict[str, int] = {}
    # inv: a run_id's first_commit is the commit that introduced its row, not one that later
    # updated it -- the walk visits commits newest first, so recording it unconditionally on
    # every commit that adds the run_id leaves the oldest (introducing) commit as the last write
    first_commit: dict[str, str] = {}
    names = {rel, *ledger.HISTORICAL_RELS}
    # inv: this walk trusts git history as given and accepts a row on its commit's shape alone,
    # never against the evidence under benchmark/record/runs/
    # why: `--follow` is what makes the walk cross the ledger's rename; every known name is
    # then given to the diff so a rename between two of them shows as no content change,
    # rather than as a fresh add under whichever name the pathspec happened to restrict to
    # inv: the walk begins after the commit that cleared the record; a repository that does not
    # hold that commit -- a fixture, or a fork made before it -- is walked from its root
    span = [f"{audit.RECORD_CLEARED}..HEAD"] if _has_commit(benchmark, audit.RECORD_CLEARED) else []
    for sha in _git_lines(benchmark, "log", "--follow", "--format=%H", *span, "--", rel):
        # why: one --name-status call, not one --name-only plus one --name-status -M, feeds both
        # file_names and renamed below -- the second call this walk used to make on every commit.
        # --name-status and --name-only can disagree on a rename (the former also prints the
        # pre-image path); the one commit in this ledger's history where they do -- the rename
        # itself -- touches no row content, so `touched` is empty and file_names is never read
        # past the `if not touched: continue` guard below
        name_status = [line for line in
                       _git_lines(benchmark, "show", "--name-status", "--format=", "-M", sha) if line.strip()]
        file_names = [part for line in name_status for part in line.split("\t")[1:]]
        rel_at = next((name for name in file_names if name in names), rel)
        # inv: a rename that also carries row content makes the rename itself impossible to
        # verify as content-free, so it is refused rather than diffed
        renamed = [line for line in name_status
                   if line.startswith("R") and all(part in names for part in line.split("\t")[1:])]
        added, removed = _row_diff(benchmark, sha, *names)
        touched = added | removed
        for run_id in added:
            first_commit[run_id] = sha
        if renamed and touched:
            problems.append(f"commit {sha[:10]} renames the ledger and edits rows in the same commit")
            continue
        if not touched:
            continue
        if file_names != [rel_at]:
            # inv: a completing commit may also carry the row's own cell's prepared_outputs.yaml,
            # when --record-prepared wrote it during the same attempt; any other extra file is
            # still a violation, sealed row or not
            allowed = {rel_at}
            if len(touched) == 1:
                row = row_by_id.get(next(iter(touched)))
                if row is not None:
                    allowed |= _prepared_rel_for(benchmark, row)
            if set(file_names) != allowed:
                problems.append(f"commit {sha[:10]} touches {sorted(file_names)}, not only {rel_at}")
        vanished = removed - added
        if vanished:
            problems.append(f"commit {sha[:10]} deletes row(s) {sorted(vanished)} instead of updating them")
        for run_id in added & removed:
            update_counts[run_id] = update_counts.get(run_id, 0) + 1
        if len(touched) > 1:
            problems.append(f"commit {sha[:10]} touches more than one row in a single commit: {sorted(touched)}")
            continue
        (row_id,) = touched
        subject = _git_lines(benchmark, "show", "-s", "--format=%s", sha)[0]
        if not subject.startswith(f"{_ATTEMPT_PREFIX}{row_id}"):
            problems.append(f"commit {sha[:10]} message {subject!r} does not name {row_id}")
    problems.extend(f"{run_id}: row content updated {n} times; a row is completed exactly once"
                     for run_id, n in update_counts.items() if n > 1)
    # inv: an unsealed row is introduced without an outcome, so one carrying an outcome that was
    # never seen as an update was planted whole; a sealed row is exempt -- it is introduced
    # already complete, in the one commit its own rule requires, checked separately below
    problems.extend(f"{r['run_id']}: carries an outcome but was never seen as an update; "
                     f"a row must be introduced first, without one"
                     for r in rows_ if "outcome" in r and update_counts.get(r["run_id"], 0) == 0
                     and not r.get("instrument_sha256"))
    # inv: a sealed row is committed exactly once -- introduced already complete, subject and
    # files matching that one commit -- never introduced and later updated
    for run_id, row in row_by_id.items():
        if not row.get("instrument_sha256") or run_id not in first_commit:
            continue
        total_commits = 1 + update_counts.get(run_id, 0)
        if total_commits != 1:
            problems.append(f"{run_id}: sealed row touched by {total_commits} commits; "
                            f"a sealed row is committed exactly once")
            continue
        if "outcome" not in row:
            continue          # inv: still in flight -- not yet completed, nothing more to check here
        sha = first_commit[run_id]
        subject = _git_lines(benchmark, "show", "-s", "--format=%s", sha)[0]
        expected = f"{_ATTEMPT_PREFIX}{run_id} {row['outcome']}"
        if subject != expected:
            problems.append(f"commit {sha[:10]} message {subject!r} is not {expected!r} for sealed row {run_id}")
    # inv: the seal on disk is root-owned, so a HEAD that disagrees with it is history that was
    # rewritten, not a tree that was edited
    seal_rel = "benchmark/INSTRUMENT.yaml"
    seal_path = benchmark / "INSTRUMENT.yaml"
    if seal_path.is_file() and _blob_sha256(benchmark, "HEAD", seal_rel) != rules.sha256_file(seal_path):
        problems.append("INSTRUMENT.yaml on disk differs from HEAD")
    seals = _seal_commits(benchmark)          # newest first
    sealed_rows = [r for r in rows_ if r.get("instrument_sha256") and r["run_id"] in first_commit]
    for older, newer in zip(sealed_rows, sealed_rows[1:], strict=False):
        if older["instrument_sha256"] == newer["instrument_sha256"]:
            continue
        window = _between(benchmark, first_commit[older["run_id"]], first_commit[newer["run_id"]])
        inside = [s for s in seals if s[0] in window]
        # inv: every commit that changes the seal between two rows is a lock commit -- sole file,
        # the seal subject, the subject's reason equal to the blob's -- and the later row names the
        # last of them; two lock cycles between two rows are legitimate and pass
        shaped = bool(inside) and all(
            files == [seal_rel] and subject == f"{_SEAL_PREFIX}{_blob_reason(benchmark, sha)}"
            for sha, subject, _, files in inside)
        if not shaped or inside[0][2] != newer["instrument_sha256"]:
            problems.append(f"instrument changed without a lock commit between {older['run_id']} and {newer['run_id']}")
    by_triple: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows_:
        by_triple.setdefault((r["question"], r["system"], r["configuration"]), []).append(r)
    for triple, rs in by_triple.items():
        nums = [r["attempt"] for r in rs]
        if nums != list(range(1, len(nums) + 1)):
            problems.append(f"{triple}: attempt numbers not contiguous: {nums}")
        problems.extend(f"{r['run_id']}: no outcome" for r in rs[:-1] if "outcome" not in r)
    # inv: a row an agent commits by hand has no bench-owned evidence, so every completed row is
    # checked against the directory only the harness writes
    machine = benchmark / seal.MACHINE
    machine_doc = yaml.safe_load(machine.read_text(encoding="utf-8")) if machine.is_file() else None
    bench_uid = (machine_doc or {}).get("bench_uid")
    for r in rows_:
        if r.get("outcome") not in ("completed", "void"):
            continue
        run = benchmark / ledger.RECORD / "runs" / r["run_id"]
        if not run.is_dir():
            problems.append(f"row {r['run_id']}: no evidence directory")
            continue
        for name, key in (("journal.jsonl", "journal_sha256"), ("records.jsonl", "records_sha256"),
                          ("cost.json", "cost_sha256"), ("audit.json", "audit_sha256")):
            if key in r and (not (run / name).is_file() or rules.sha256_file(run / name) != r[key]):
                problems.append(f"row {r['run_id']}: {name} differs from its recorded hash")
        if bench_uid is not None and rules._owner(run) != int(bench_uid):
            problems.append(f"row {r['run_id']}: evidence not owned by bench")
    by_sc: dict[tuple[str, str], list[dict]] = {}
    for r in rows_:
        by_sc.setdefault((r["system"], r["configuration"]), []).append(r)
    known = _known_transitions(benchmark)
    observed: set[tuple[str, str, str, str]] = set()
    for sc, rs in by_sc.items():
        build_steps = _transitions(rs, "build_yaml_sha256")
        prepared_steps = _transitions(rs, "prepared_sha256")
        recorded = [r["build_yaml_sha256"] for r in rs if r.get("build_yaml_sha256")]
        # inv: resolving the index path reads the system's harness.yaml, so it waits until there is
        # something to check -- a group naming a system this benchmark does not declare has no file
        declared = (benchmark / "systems" / sc[0] / "harness.yaml").is_file()
        if not build_steps and not prepared_steps and not (recorded and declared):
            continue
        rel_dir = _index_rel(benchmark, sc[0], sc[1], rs[0]["question"])
        # inv: a change made after the last recorded attempt sits between no two rows, so it is
        # anchored by comparing the file as it stands now with what that attempt ran against
        current_file = benchmark.parent / rel_dir / "build.yaml"
        if recorded and current_file.is_file():
            current = rules.sha256_file(current_file)
            if current != recorded[-1]:
                build_steps = [*build_steps, (recorded[-1], current)]
        if not build_steps and not prepared_steps:
            continue
        # inv: build.yaml is a freeze record, so any parsed change between two attempts is drift
        # unless admitted by name
        for before, after in build_steps:
            content = _content_by_hash(benchmark, f"{rel_dir}/build.yaml")
            if before not in content or after not in content:
                problems.append(f"{sc}: build.yaml transition {before[:10]}->{after[:10]} not found in its history")
                continue
            observed.add((sc[0], sc[1], before[:10], after[:10]))
            if (sc[0], sc[1], before[:10], after[:10]) in known:
                pair = next(((a, b) for a, b in _transition_rows(rs, "build_yaml_sha256")
                             if a["build_yaml_sha256"] == before and b["build_yaml_sha256"] == after), None)
                # inv: build.yaml is locked, so an admitted change to it happened inside an unlock;
                # a transition no seal commit encloses was made by hand around the lock -- rows
                # recorded before the seal existed carry no instrument_sha256 and bind nothing here
                if (pair is not None and pair[0].get("instrument_sha256")
                        and all(r["run_id"] in first_commit for r in pair)):
                    older_c, newer_c = first_commit[pair[0]["run_id"]], first_commit[pair[1]["run_id"]]
                    if not any(s[0] in _between(benchmark, older_c, newer_c) for s in seals):
                        problems.append(f"{sc}: build.yaml transition {before[:10]}->{after[:10]} "
                                        f"outside any seal transition")
                continue
            b, a = yaml.safe_load(content[before]) or {}, yaml.safe_load(content[after]) or {}
            changed = sorted(k for k in set(b) | set(a) if b.get(k) != a.get(k))
            if changed:
                problems.append(f"{sc}: build.yaml transition {before[:10]}->{after[:10]} changed "
                                 f"{changed}; build.yaml does not change after the freeze")
        # inv: expectations may be gained and may never be lost or rewritten
        for before, after in prepared_steps:
            content = _content_by_hash(benchmark, f"{rel_dir}/{prepare.PREPARED}")
            if before not in content or after not in content:
                problems.append(f"{sc}: expectations transition {before[:10]}->{after[:10]} "
                                 f"not found in {prepare.PREPARED}'s history")
                continue
            leaves_b = _prepared_leaves(yaml.safe_load(content[before]) or {})
            leaves_a = _prepared_leaves(yaml.safe_load(content[after]) or {})
            if not all(leaves_a.get(k, _MISSING) == v for k, v in leaves_b.items()):
                problems.append(f"{sc}: expectations transition {before[:10]}->{after[:10]} rewrites "
                                 f"or drops one it had already recorded")
    problems.extend(f"admitted transition never observed: {s}/{c} {b}->{a}"
                     for (s, c, b, a) in sorted(known - observed))
    by_q: dict[str, set[str | None]] = {}
    for r in rows_:
        by_q.setdefault(r["question"], set()).add(r.get("question_sha256"))
    problems.extend(f"question {q}: question_sha256 differs across attempts" for q, hs in by_q.items() if len(hs) > 1)
    return problems


def check_rebaseline(benchmark: Path, system: str) -> list[str]:
    """Compare a system's two newest recipes' recorded expectations, step by step.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `systems/`.
    system : str
        System whose configurations are checked.

    Returns
    -------
    list of str
        One message per `(configuration, question, step)` where the newest recipe's
        output disagrees with the previous recipe's; empty when a configuration has
        fewer than two recipes, or every shared step agrees.
    """
    h = config.load_harness(benchmark, system)
    rows_ = ledger.rows(benchmark)
    problems: list[str] = []
    for configuration in h.configurations:
        rs = [r for r in rows_ if r["system"] == system and r["configuration"] == configuration]
        # inv: the recipe order is the order the ledger first mentions each harness_sha256, so a
        # re-baseline is always compared against the recipe it most recently replaced
        recipes: list[str] = []
        for r in rs:
            recipe = r.get("harness_sha256")
            if recipe and recipe not in recipes:
                recipes.append(recipe)
        if len(recipes) < 2:
            continue
        previous, newest = recipes[-2], recipes[-1]
        index_dir = benchmark.parent / _index_rel(benchmark, system, configuration, rs[0]["question"])
        by_recipe = prepare.load_prepared(index_dir).get(configuration, {})
        previous_by_q, newest_by_q = by_recipe.get(previous, {}), by_recipe.get(newest, {})
        for qid, steps in previous_by_q.items():
            for step, expectation in steps.items():
                after = newest_by_q.get(qid, {}).get(step)
                if after is not None and after.get("out") != expectation.get("out"):
                    problems.append(f"rebaseline {system}/{configuration} {qid} {step}: "
                                     f"output differs from the previous recipe")
    return problems
