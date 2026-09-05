import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.harness import audit, ledger, prepare, rules
from tests.benchmark.conftest import snapshot_dir, write_question


def _snapshots_root(bench: Path) -> Path:
    """Return `record/snapshots/` under `bench`, creating it.

    Parameters
    ----------
    bench : Path
        A test tree's `benchmark/` directory.

    Returns
    -------
    Path
        The directory holding `known_transitions.yaml` and every snapshot.
    """
    root = bench / "record" / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _row(run_id: str, *, question: str = "q", system: str = "s", configuration: str = "c", attempt: int = 1,
         question_sha256: str = "a", build_yaml_sha256: str = "b",
         prepared_sha256: str | None = None) -> dict:
    row = {"run_id": run_id, "question": question, "system": system, "configuration": configuration,
           "attempt": attempt, "question_sha256": question_sha256, "build_yaml_sha256": build_yaml_sha256}
    # inv: a row without the key predates the expectations file and anchors no transition, which
    # is what a first recording run actually writes
    if prepared_sha256 is not None:
        row["prepared_sha256"] = prepared_sha256
    return row


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.excludesFile=/dev/null",
                    "-C", str(root), *args], check=True, capture_output=True)


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _evidence(bench: Path, run_id: str) -> dict:
    """Commit `run_id`'s four evidence files under `record/runs/` and return their hashes.

    Committed in a run of their own -- mirroring `_attempt()`'s choreography, which keeps
    evidence out of the ledger's own commits -- so a later commit that touches the ledger
    is never also seen touching evidence.
    """
    run_dir = bench / "record" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, key in (("journal.jsonl", "journal_sha256"), ("records.jsonl", "records_sha256"),
                      ("cost.json", "cost_sha256"), ("audit.json", "audit_sha256")):
        p = run_dir / name
        p.write_text(name + "\n", encoding="utf-8")
        hashes[key] = rules.sha256_file(p)
    _git(bench.parent, "add", "-A")
    _git(bench.parent, "commit", "-qm", f"chore(benchmark): attempt {run_id} evidence")
    return hashes


def _append(bench: Path, row: dict) -> None:
    """Append and commit one row, the way `prepare.prepare` does."""
    ledger.append_row(bench, row)
    ledger.commit_rows(bench, f"chore(benchmark): attempt {row['run_id']} prepared")


def _complete(bench: Path, run_id: str, update: dict) -> None:
    """Complete and commit one row, the way `collect.collect` does."""
    ledger.complete_row(bench, run_id, update)
    ledger.commit_rows(bench, f"chore(benchmark): attempt {run_id} {update.get('outcome', 'updated')}")


def _hand_commit(repo_root: Path, message: str, *, touch: str | None = None) -> None:
    """Commit whatever is on disk directly, bypassing `ledger`'s own commit path."""
    if touch is not None:
        (repo_root / touch).write_text("x", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", message)


def test_attempts_audit_passes_then_flags_gap_and_tamper(git_bench: Path):
    row = {"run_id": "q-s-c-a01", "question": "q", "system": "s", "configuration": "c", "attempt": 1,
           "question_sha256": "a", "build_yaml_sha256": "b"}
    _append(git_bench, row)
    _evidence(git_bench, "q-s-c-a01")
    _complete(git_bench, "q-s-c-a01", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == []
    _append(git_bench, {**row, "run_id": "q-s-c-a03", "attempt": 3})
    assert any("contiguous" in m for m in audit.check_attempts(git_bench))
    # inv: a row edited by hand is introduced by a commit the harness did not write
    p = git_bench / "record" / "attempts.jsonl"
    p.write_text(p.read_text().replace('"question_sha256": "a"', '"question_sha256": "z"', 1))
    (git_bench.parent / "other.txt").write_text("x")
    subprocess.run(["git", "-C", str(git_bench.parent), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(git_bench.parent), "commit", "-qm", "tamper"], check=True)
    msgs = audit.check_attempts(git_bench)
    assert any("question_sha256" in m for m in msgs)
    # inv: dropping either the touches-only check or the message-shape check alone must still
    # leave the other producing a violation -- neither half may hide behind an `or`
    assert any("touches" in m for m in msgs)
    assert any("message" in m for m in msgs)


def test_check_attempts_flags_gap_in_attempt_numbers(git_bench: Path):
    _append(git_bench, _row("q-s-c-a01", attempt=1))
    _evidence(git_bench, "q-s-c-a01")
    _complete(git_bench, "q-s-c-a01", {"outcome": "completed"})
    _append(git_bench, _row("q-s-c-a03", attempt=3))
    _evidence(git_bench, "q-s-c-a03")
    _complete(git_bench, "q-s-c-a03", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == ["('q', 's', 'c'): attempt numbers not contiguous: [1, 3]"]


def test_check_attempts_flags_missing_outcome_on_non_last_attempt(git_bench: Path):
    _append(git_bench, _row("q-s-c-a01", attempt=1))
    _append(git_bench, _row("q-s-c-a02", attempt=2))
    _evidence(git_bench, "q-s-c-a02")
    _complete(git_bench, "q-s-c-a02", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == ["q-s-c-a01: no outcome"]


def test_check_attempts_flags_question_sha256_drift(git_bench: Path):
    _append(git_bench, _row("q-s-c-a01", attempt=1, question_sha256="a"))
    _evidence(git_bench, "q-s-c-a01")
    _complete(git_bench, "q-s-c-a01", {"outcome": "completed"})
    _append(git_bench, _row("q-s-c-a02", attempt=2, question_sha256="different"))
    _evidence(git_bench, "q-s-c-a02")
    _complete(git_bench, "q-s-c-a02", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == ["question q: question_sha256 differs across attempts"]


def _seed_index(git_bench: Path) -> Path:
    """Commit a minimal harness.yaml, question and build.yaml for system `s`, configuration `c`."""
    repo_root = git_bench.parent
    harness_dir = git_bench / "systems" / "s"
    harness_dir.mkdir(parents=True)
    (harness_dir / "harness.yaml").write_text(yaml.safe_dump({
        "adapter": "a", "version": {}, "invocation": {}, "fixed_steps": [],
        "default_configuration": "c", "configurations": {"c": {"index": "indexes/c"}},
        "sandbox_layout": {}, "environment": {}, "docs": {},
    }), encoding="utf-8")
    for qid in ("q1", "q2"):
        write_question(git_bench, qid, {"id": qid, "snapshot": "snap"})
    index_dir = snapshot_dir(git_bench, "snap") / "indexes" / "c"
    index_dir.mkdir(parents=True)
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "fixtures: harness and questions")
    return index_dir / "build.yaml"


def _commit_build_yaml(repo_root: Path, build_path: Path, content: dict, message: str) -> str:
    build_path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", message)
    return hashlib.sha256(build_path.read_bytes()).hexdigest()


def _commit_prepared(repo_root: Path, index_dir: Path, content: dict, message: str) -> str:
    path = index_dir / prepare.PREPARED
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", message)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_check_attempts_tolerates_the_first_recording_of_expectations(git_bench: Path):
    build_path = _seed_index(git_bench)
    repo_root, idx = git_bench.parent, build_path.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    p1 = _commit_prepared(repo_root, idx, {"c": {"r1": {"q1": {"version": {"out": "A"}}}}},
                          "chore(benchmark): record prepared outputs for s/c")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2,
                                      build_yaml_sha256=h1, prepared_sha256=p1))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == []


def test_check_attempts_allows_added_expectations_and_flags_a_changed_build_key(git_bench: Path):
    # inv: expectations are gained freely in their own file; build.yaml is a freeze record, so any
    # change to it after the freeze is drift, whatever key it touches
    build_path = _seed_index(git_bench)
    repo_root, idx = git_bench.parent, build_path.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    p1 = _commit_prepared(repo_root, idx, {"c": {"r1": {"q1": {"version": {"out": "A"}}}}}, "chore: record")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1, prepared_sha256=p1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    p2 = _commit_prepared(repo_root, idx,
                          {"c": {"r1": {"q1": {"version": {"out": "A"}}, "q2": {"version": {"out": "B"}}}}},
                          "chore: record a second question")
    _append(git_bench, _row("q2-s-c-a01", question="q2", build_yaml_sha256=h1, prepared_sha256=p2))
    _evidence(git_bench, "q2-s-c-a01")
    _complete(git_bench, "q2-s-c-a01", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == []
    h2 = _commit_build_yaml(repo_root, build_path, {"artifacts": {"graph.db": "zzz"}}, "chore: rehash")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2,
                                      build_yaml_sha256=h2, prepared_sha256=p2))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == [
        (f"('s', 'c'): build.yaml transition {h1[:10]}->{h2[:10]} "
         f"changed ['artifacts']; build.yaml does not change after the freeze")]


def test_check_attempts_flags_an_artifact_hash_change(git_bench: Path):
    build_path = _seed_index(git_bench)
    repo_root = git_bench.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {"graph.db": "aaa"}}, "fixtures: initial build.yaml")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    h2 = _commit_build_yaml(repo_root, build_path, {"artifacts": {"graph.db": "bbb"}},
                             "chore(benchmark): rehash graph.db")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2, build_yaml_sha256=h2))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    expected = (f"('s', 'c'): build.yaml transition {h1[:10]}->{h2[:10]} "
                f"changed ['artifacts']; build.yaml does not change after the freeze")
    assert audit.check_attempts(git_bench) == [expected]


def test_check_attempts_flags_a_rewrite_of_an_existing_expectation(git_bench: Path):
    build_path = _seed_index(git_bench)
    repo_root, idx = git_bench.parent, build_path.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    p1 = _commit_prepared(repo_root, idx, {"c": {"r1": {"q1": {"v1": {"out": "aaa"}}}}}, "chore: record")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1, prepared_sha256=p1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    p2 = _commit_prepared(repo_root, idx, {"c": {"r1": {"q1": {"v1": {"out": "bbb"}}}}},
                          "chore: rewrite a recorded expectation")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2,
                                      build_yaml_sha256=h1, prepared_sha256=p2))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == [
        (f"('s', 'c'): expectations transition {p1[:10]}->{p2[:10]} rewrites "
         f"or drops one it had already recorded")]


def test_check_attempts_reports_an_expectations_transition_it_cannot_find(git_bench: Path):
    # inv: the additive check reads revisions out of git history, so an untracked copy leaves
    # every transition unfindable and must be reported, not passed in silence
    build_path = _seed_index(git_bench)
    repo_root, idx = git_bench.parent, build_path.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    (repo_root / ".gitignore").write_text(f"{prepare.PREPARED}\n", encoding="utf-8")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-qm", "fixtures: leave the expectations file untracked")
    path = idx / prepare.PREPARED
    hashes = []
    for attempt, out in ((1, "aaa"), (2, "bbb")):
        path.write_text(yaml.safe_dump({"c": {"r1": {"q1": {"v1": {"out": out}}}}}), encoding="utf-8")
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        run_id = f"q1-s-c-a0{attempt}"
        _append(git_bench, _row(run_id, question="q1", attempt=attempt,
                                          build_yaml_sha256=h1, prepared_sha256=hashes[-1]))
        _evidence(git_bench, run_id)
        _complete(git_bench, run_id, {"outcome": "completed"})
    p1, p2 = hashes
    assert audit.check_attempts(git_bench) == [
        (f"('s', 'c'): expectations transition {p1[:10]}->{p2[:10]} "
         f"not found in {prepare.PREPARED}'s history")]


def test_every_recorded_expectations_file_is_tracked(bench: Path):
    # inv: the rewrite the additive check exists to catch is invisible to it unless git holds the
    # file's earlier revisions, so tracking is the check's precondition, not a preference
    found = sorted(bench.glob(f"record/snapshots/*/indexes/*/{prepare.PREPARED}"))
    if not found:
        pytest.skip(f"no {prepare.PREPARED} in this tree")
    for path in found:
        rel = path.relative_to(bench.parent).as_posix()
        proc = subprocess.run(["git", "-C", str(bench.parent), "ls-files", "--error-unmatch", "--", rel],
                              capture_output=True, check=False)
        assert proc.returncode == 0, f"{rel} is not tracked; the additive check cannot read its history"


def test_check_attempts_names_the_key_beyond_prepared_outputs(git_bench: Path):
    build_path = _seed_index(git_bench)
    repo_root = git_bench.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    h2 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}, "vendor_writes": ["graph.db"]},
                             "chore(benchmark): declare graph.db vendor-written")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2, build_yaml_sha256=h2))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    msgs = audit.check_attempts(git_bench)
    assert len(msgs) == 1
    assert "vendor_writes" in msgs[0]
    assert h1[:10] in msgs[0]
    assert h2[:10] in msgs[0]
    assert "does not change after the freeze" in msgs[0]


def test_check_attempts_flags_deleted_last_attempt_row(git_bench: Path):
    _append(git_bench, _row("q-s-c-a01", attempt=1))
    _evidence(git_bench, "q-s-c-a01")
    _complete(git_bench, "q-s-c-a01", {"outcome": "completed"})
    _append(git_bench, _row("q-s-c-a02", attempt=2))
    _evidence(git_bench, "q-s-c-a02")
    _complete(git_bench, "q-s-c-a02", {"outcome": "completed"})
    path = git_bench / "record" / "attempts.jsonl"
    rows = [r for r in _read_rows(path) if r["run_id"] != "q-s-c-a02"]
    _write_rows(path, rows)
    _hand_commit(git_bench.parent, "chore(benchmark): attempt q-s-c-a02 void")
    msgs = audit.check_attempts(git_bench)
    assert len(msgs) == 1
    assert "deletes row(s)" in msgs[0]
    assert "q-s-c-a02" in msgs[0]


def test_check_attempts_flags_deleted_only_row(git_bench: Path):
    _append(git_bench, _row("q-s-c-a01", attempt=1))
    _complete(git_bench, "q-s-c-a01", {"outcome": "completed"})
    _write_rows(git_bench / "record" / "attempts.jsonl", [])
    _hand_commit(git_bench.parent, "chore(benchmark): attempt q-s-c-a01 void")
    msgs = audit.check_attempts(git_bench)
    assert len(msgs) == 1
    assert "deletes row(s)" in msgs[0]
    assert "q-s-c-a01" in msgs[0]


def test_check_attempts_flags_commit_touching_multiple_rows(git_bench: Path):
    _append(git_bench, _row("q1-s-c-a01", question="q1", attempt=1))
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed", "hit": True})
    _append(git_bench, _row("q2-s-c-a01", question="q2", attempt=1))
    _complete(git_bench, "q2-s-c-a01", {"outcome": "completed", "hit": True})
    path = git_bench / "record" / "attempts.jsonl"
    rows = _read_rows(path)
    for r in rows:
        r["hit"] = False
    _write_rows(path, rows)
    _hand_commit(git_bench.parent, "chore(benchmark): attempt q1-s-c-a01 updated")
    msgs = audit.check_attempts(git_bench)
    assert any("touches more than one row" in m for m in msgs)


def test_check_attempts_flags_message_naming_the_wrong_row(git_bench: Path):
    # inv: the disclosed residual -- a commit shaped like a legitimate single-row update, with
    # a message plausible for a different row -- must be caught by the id it actually touches
    _append(git_bench, _row("q1-s-c-a01", question="q1", attempt=1))
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    _append(git_bench, _row("q2-s-c-a01", question="q2", attempt=1))
    _complete(git_bench, "q2-s-c-a01", {"outcome": "completed"})
    path = git_bench / "record" / "attempts.jsonl"
    rows = _read_rows(path)
    for r in rows:
        if r["run_id"] == "q2-s-c-a01":
            r["hit"] = False
    _write_rows(path, rows)
    _hand_commit(git_bench.parent, "chore(benchmark): attempt q1-s-c-a01 updated")
    msgs = audit.check_attempts(git_bench)
    assert any("does not name" in m and "q2-s-c-a01" in m for m in msgs)


def test_check_attempts_flags_row_updated_more_than_once(git_bench: Path):
    # inv: a row is completed exactly once -- a second edit to an already-completed row must be
    # caught even under a message that plausibly names the row it actually changed
    _append(git_bench, _row("q-s-c-a01", attempt=1))
    _evidence(git_bench, "q-s-c-a01")
    _complete(git_bench, "q-s-c-a01", {"outcome": "completed"})
    path = git_bench / "record" / "attempts.jsonl"
    rows = _read_rows(path)
    for r in rows:
        r["outcome"] = "void"
    _write_rows(path, rows)
    _hand_commit(git_bench.parent, "chore(benchmark): attempt q-s-c-a01 void")
    expected = "q-s-c-a01: row content updated 2 times; a row is completed exactly once"
    assert audit.check_attempts(git_bench) == [expected]


def test_check_attempts_flags_commit_touching_another_file(git_bench: Path):
    _append(git_bench, _row("q-s-c-a01", attempt=1))
    path = git_bench / "record" / "attempts.jsonl"
    rows = _read_rows(path)
    for r in rows:
        r["question_sha256"] = "changed"
    _write_rows(path, rows)
    _hand_commit(git_bench.parent, "chore(benchmark): attempt q-s-c-a01 updated", touch="other.txt")
    msgs = audit.check_attempts(git_bench)
    assert len(msgs) == 1
    assert "touches" in msgs[0]
    assert "not only" in msgs[0]
    assert "other.txt" in msgs[0]


def test_check_attempts_flags_a_row_introduced_already_carrying_an_outcome(git_bench: Path):
    # inv: a row appears first without an outcome, so one introduced already complete must not
    # pass on a well-formed commit subject alone
    _evidence(git_bench, "q-s-c-a01")
    row = {**_row("q-s-c-a01"), "outcome": "completed", "hit": True}
    _write_rows(git_bench / "record" / "attempts.jsonl", [row])
    _hand_commit(git_bench.parent, "chore(benchmark): attempt q-s-c-a01 prepared")
    msgs = audit.check_attempts(git_bench)
    assert len(msgs) == 1
    assert "q-s-c-a01" in msgs[0]
    assert "never seen as an update" in msgs[0]


def test_check_attempts_flags_uncommitted_ledger_changes(git_bench: Path):
    _append(git_bench, _row("q-s-c-a01", attempt=1))
    path = git_bench / "record" / "attempts.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    msgs = audit.check_attempts(git_bench)
    assert len(msgs) == 1
    assert "uncommitted" in msgs[0]


def test_check_attempts_accepts_a_second_recipe_and_still_flags_a_dropped_one(git_bench: Path):
    build_path = _seed_index(git_bench)
    repo_root, idx = git_bench.parent, build_path.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    one = {"c": {"aa" * 32: {"q1": {"v1": {"out": "aaa"}}}}}
    p1 = _commit_prepared(repo_root, idx, one, "chore: one recipe recorded")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1, prepared_sha256=p1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    # inv: a new recipe's expectations are new keys, so recording them cannot overwrite the
    # baseline the earlier attempts were verified against -- the danger the rule exists for
    both = {"c": {"aa" * 32: {"q1": {"v1": {"out": "aaa"}}}, "bb" * 32: {"q1": {"v1": {"out": "bbb"}}}}}
    p2 = _commit_prepared(repo_root, idx, both, "chore: a second recipe recorded")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2,
                                      build_yaml_sha256=h1, prepared_sha256=p2))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == []
    p3 = _commit_prepared(repo_root, idx, {"c": {"bb" * 32: {"q1": {"v1": {"out": "bbb"}}}}},
                          "chore: drop the first recipe")
    _append(git_bench, _row("q1-s-c-a03", question="q1", attempt=3,
                                      build_yaml_sha256=h1, prepared_sha256=p3))
    _evidence(git_bench, "q1-s-c-a03")
    _complete(git_bench, "q1-s-c-a03", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == [
        (f"('s', 'c'): expectations transition {p2[:10]}->{p3[:10]} rewrites "
         f"or drops one it had already recorded")]


def test_check_attempts_flags_a_question_dropped_from_the_expectations(git_bench: Path):
    # inv: every level of the key is a level of the guarantee -- dropping one question's
    # expectations is losing a record, exactly as dropping one step or one recipe is
    build_path = _seed_index(git_bench)
    repo_root, idx = git_bench.parent, build_path.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    two = {"c": {"r1": {"q1": {"v1": {"out": "A"}}, "q2": {"v1": {"out": "B"}}}}}
    p1 = _commit_prepared(repo_root, idx, two, "chore: two questions recorded")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1, prepared_sha256=p1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    p2 = _commit_prepared(repo_root, idx, {"c": {"r1": {"q1": {"v1": {"out": "A"}}}}},
                          "chore: drop the second question")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2,
                                      build_yaml_sha256=h1, prepared_sha256=p2))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == [
        (f"('s', 'c'): expectations transition {p1[:10]}->{p2[:10]} rewrites "
         f"or drops one it had already recorded")]


@pytest.mark.parametrize("shape", [[1, 2], "oops", {"c": "oops"},
                                   {"c": {"r": "oops"}}, {"c": {"r": {"q1": "oops"}}}])
def test_prepared_leaves_survives_a_malformed_shape(shape: Any):
    # inv: one malformed historical revision must not abort the whole ledger audit, which would
    # suppress every other violation it was about to report
    assert isinstance(audit._prepared_leaves(shape), dict)


def _admit(bench: Path, before: str, after: str, reason: str = "a recorded reason") -> None:
    (_snapshots_root(bench) / "known_transitions.yaml").write_text(yaml.safe_dump(
        {"transitions": [{"system": "s", "configuration": "c", "before": before[:10],
                          "after": after[:10], "reason": reason}]}), encoding="utf-8")


def _two_attempts_with_a_changed_key(git_bench: Path) -> tuple[str, str]:
    build_path = _seed_index(git_bench)
    repo_root = git_bench.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    h2 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}, "vendor_writes": ["graph.db"]},
                            "chore(benchmark): declare graph.db vendor-written")
    _append(git_bench, _row("q1-s-c-a02", question="q1", attempt=2, build_yaml_sha256=h2))
    _evidence(git_bench, "q1-s-c-a02")
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    return h1, h2


def test_check_attempts_admits_only_the_transition_the_list_names(git_bench: Path):
    h1, h2 = _two_attempts_with_a_changed_key(git_bench)
    assert len(audit.check_attempts(git_bench)) == 1
    _admit(git_bench, h1, h2)
    assert audit.check_attempts(git_bench) == []
    # inv: the pair names one diff; admitting a different pair must leave this one reported --
    # and the fabricated admission itself is reported as never observed
    _admit(git_bench, "0" * 10, "1" * 10)
    msgs = audit.check_attempts(git_bench)
    assert any(f"{h1[:10]}->{h2[:10]}" in m for m in msgs)
    assert any("admitted transition never observed: s/c 0000000000->1111111111" in m for m in msgs)
    assert len(msgs) == 2


def test_check_attempts_refuses_an_admission_without_a_reason(git_bench: Path):
    h1, h2 = _two_attempts_with_a_changed_key(git_bench)
    _admit(git_bench, h1, h2, reason="   ")
    # inv: silencing a transition costs an explanation the next reader can check against the diff
    with pytest.raises(SystemExit, match="has no reason"):
        audit.check_attempts(git_bench)


def test_check_attempts_flags_a_build_yaml_changed_after_the_last_attempt(git_bench: Path):
    # inv: a change made after the last recorded attempt sits between no two rows, so nothing
    # anchored it -- the file as it stands is compared with what that attempt ran against
    build_path = _seed_index(git_bench)
    repo_root = git_bench.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == []
    h2 = _commit_build_yaml(repo_root, build_path, {"artifacts": {"graph.db": "zzz"}},
                            "chore: edit the freeze record after the last attempt")
    assert audit.check_attempts(git_bench) == [
        (f"('s', 'c'): build.yaml transition {h1[:10]}->{h2[:10]} "
         f"changed ['artifacts']; build.yaml does not change after the freeze")]


def test_ledger_lives_under_record(git_bench):
    assert ledger.path(git_bench) == git_bench / "record" / "attempts.jsonl"
    assert ledger.rel(git_bench) == "benchmark/record/attempts.jsonl"
    assert ledger.HISTORICAL_RELS == ("benchmark/attempts.jsonl",)


def test_a_trailing_build_yaml_change_can_be_admitted_by_name(git_bench: Path):
    build_path = _seed_index(git_bench)
    repo_root = git_bench.parent
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    _append(git_bench, _row("q1-s-c-a01", question="q1", build_yaml_sha256=h1))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    h2 = _commit_build_yaml(repo_root, build_path, {"artifacts": {"graph.db": "zzz"}}, "chore: edit")
    (_snapshots_root(git_bench) / "known_transitions.yaml").write_text(yaml.safe_dump({"transitions": [
        {"system": "s", "configuration": "c", "before": h1[:10], "after": h2[:10],
         "reason": "admitted by this fixture"}]}), encoding="utf-8")
    assert audit.check_attempts(git_bench) == []


def test_attempts_audit_walks_from_the_commit_that_cleared_the_record(
        git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = git_bench.parent
    _append(git_bench, _row("q1-s-c-a01", question="q1"))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    shutil.rmtree(git_bench / "record" / "runs")
    ledger.path(git_bench).write_text("", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "record(benchmark): clear the record")
    cleared = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()

    # inv: the module's own sha names no commit of this fixture, so the walk starts at the root
    assert any("deletes row(s) ['q1-s-c-a01']" in m for m in audit.check_attempts(git_bench))
    monkeypatch.setattr(audit, "RECORD_CLEARED", cleared)
    assert audit.check_attempts(git_bench) == []

    # inv: the cleared record numbers attempts from one again, so the first new row reuses a01
    _append(git_bench, _row("q1-s-c-a01", question="q1"))
    _evidence(git_bench, "q1-s-c-a01")
    _complete(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    assert audit.check_attempts(git_bench) == []
    ledger.path(git_bench).write_text("", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "record(benchmark): clear the record again")
    assert any("deletes row(s) ['q1-s-c-a01']" in m for m in audit.check_attempts(git_bench))


def test_attempts_audit_sees_commits_from_before_a_rename(git_bench: Path):
    repo_root = git_bench.parent
    # the ledger under its historical name, with one row introduced and completed there
    old = git_bench / "attempts.jsonl"
    ledger.path(git_bench).unlink()
    old.write_text("", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "ledger at its old name")
    old.write_text(json.dumps(_row("q1-s-c-a01", question="q1", system="s", configuration="c", attempt=1)) + "\n",
                   encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 prepared")
    _evidence(git_bench, "q1-s-c-a01")
    done = {**_row("q1-s-c-a01", question="q1", system="s", configuration="c", attempt=1), "outcome": "completed"}
    old.write_text(json.dumps(done) + "\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 completed")
    # the rename, as this plan performs it
    _git(repo_root, "mv", "benchmark/attempts.jsonl", "benchmark/record/attempts.jsonl")
    _git(repo_root, "commit", "-qm", "bench: everything the harness writes lives under benchmark/record/")

    assert audit.check_attempts(git_bench) == []


def test_require_clean_refuses_an_untracked_question_and_accepts_an_untracked_run(git_bench):
    (git_bench / "record" / "runs").mkdir(parents=True, exist_ok=True)
    (git_bench / "record" / "runs" / "q-s-c-a01").mkdir()
    (git_bench / "record" / "runs" / "q-s-c-a01" / "journal.jsonl").write_text("{}\n", encoding="utf-8")
    (git_bench.parent / ".gitignore").write_text("benchmark/record/runs/\n", encoding="utf-8")
    _git(git_bench.parent, "add", ".gitignore")
    _git(git_bench.parent, "commit", "-qm", "ignore runs")
    ledger.require_clean(git_bench)          # an ignored run directory is not dirt

    write_question(git_bench, "q999", {"id": "q999", "snapshot": "snap"})
    with pytest.raises(ledger.LedgerError, match="untracked file under benchmark/: "
                                                 "benchmark/record/snapshots/snap/questions/q999.yaml"):
        ledger.require_clean(git_bench)


def _seal_commit(repo_root: Path, b: Path, reason: str, files_extra: dict | None = None) -> str:
    """Write INSTRUMENT.yaml by hand in the shape lock produces, commit it alone, and return its blob sha256."""
    doc = {"reason": reason, "sealed_at_commit": "x",
           "files": {"benchmark/harness/x.py": "0" * 64, **(files_extra or {})},
           "interpreter": None, "machine_sha256": None}
    (b / "INSTRUMENT.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    _git(repo_root, "add", "benchmark/INSTRUMENT.yaml")
    _git(repo_root, "commit", "-qm", f"chore(benchmark): seal — {reason}")
    return rules.sha256_file(b / "INSTRUMENT.yaml")


def _sealed_row(b: Path, row: dict) -> None:
    """Introduce one sealed row already complete, in the single commit the sealed-row rule requires."""
    all_rows = [r for r in ledger.rows(b) if r["run_id"] != row["run_id"]] + [row]
    ledger.rewrite(b, all_rows)
    ledger.commit_rows(b, f"chore(benchmark): attempt {row['run_id']} {row['outcome']}")


def _attempt(repo_root: Path, b: Path, run_id: str, seal_sha: str, attempt: int) -> None:
    """Introduce one sealed row already complete, with its evidence under record/runs/<run_id>.

    Evidence is committed on its own first, exactly as `_evidence` does for an unsealed row,
    so the row's own commit -- the sealed row's one and only -- touches only the ledger.
    """
    run_dir = b / "record" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name, key in (("journal.jsonl", "journal_sha256"), ("records.jsonl", "records_sha256"),
                      ("cost.json", "cost_sha256"), ("audit.json", "audit_sha256")):
        p = run_dir / name
        p.write_text("{}\n", encoding="utf-8")
        evidence[key] = rules.sha256_file(p)
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", f"chore(benchmark): attempt {run_id} evidence")
    row = {**_row(run_id, question="q1", system="s", configuration="c", attempt=attempt),
           "instrument_sha256": seal_sha, "outcome": "completed", **evidence}
    _sealed_row(b, row)


def test_audit_accepts_one_and_two_lock_cycles_between_rows_and_refuses_a_bare_change(git_bench):
    b = git_bench
    repo_root = b.parent
    s1 = _seal_commit(repo_root, b, "initial seal")
    _attempt(repo_root, b, "q1-s-c-a01", s1, 1)
    assert audit.check_attempts(b) == []

    _seal_commit(repo_root, b, "fix one")
    s3 = _seal_commit(repo_root, b, "fix two")
    _attempt(repo_root, b, "q1-s-c-a02", s3, 2)
    assert audit.check_attempts(b) == []          # two lock cycles between rows are legitimate

    # a change to the seal that is not a lock commit: wrong subject
    doc = yaml.safe_load((b / "INSTRUMENT.yaml").read_text(encoding="utf-8"))
    doc["reason"] = "planted"
    (b / "INSTRUMENT.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "tidy")
    s4 = rules.sha256_file(b / "INSTRUMENT.yaml")
    _attempt(repo_root, b, "q1-s-c-a03", s4, 3)
    msgs = audit.check_attempts(b)
    assert any("instrument changed without a lock commit between q1-s-c-a02 and q1-s-c-a03" in m for m in msgs)


def test_audit_refuses_a_working_tree_seal_that_differs_from_head(git_bench):
    b = git_bench
    s1 = _seal_commit(b.parent, b, "initial seal")
    _attempt(b.parent, b, "q1-s-c-a01", s1, 1)
    doc = yaml.safe_load((b / "INSTRUMENT.yaml").read_text(encoding="utf-8"))
    doc["reason"] = "edited on disk"
    (b / "INSTRUMENT.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert any("INSTRUMENT.yaml on disk differs from HEAD" in m for m in audit.check_attempts(b))


def test_known_transition_must_be_observed(git_bench):
    b = git_bench
    repo_root = b.parent
    (_snapshots_root(b) / "known_transitions.yaml").write_text(yaml.safe_dump({"transitions": [
        {"system": "s", "configuration": "c", "before": "aaaaaaaaaa", "after": "bbbbbbbbbb",
         "reason": "never happened"}]}), encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "admit")
    s1 = _seal_commit(repo_root, b, "initial seal")
    _attempt(repo_root, b, "q1-s-c-a01", s1, 1)
    msgs = audit.check_attempts(b)
    assert any("admitted transition never observed: s/c aaaaaaaaaa->bbbbbbbbbb" in m for m in msgs)


def test_ledger_walk_refuses_a_rename_that_also_edits_rows(git_bench):
    b = git_bench
    repo_root = b.parent
    old = b / "attempts.jsonl"
    ledger.path(b).unlink()
    row = _row("q1-s-c-a01", question="q1", system="s", configuration="c", attempt=1)
    old.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 prepared")
    _git(repo_root, "mv", "benchmark/attempts.jsonl", "benchmark/record/attempts.jsonl")
    ledger.path(b).write_text(json.dumps({**row, "outcome": "completed"}) + "\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "move and edit")
    msgs = audit.check_attempts(b)
    assert any("renames the ledger and edits rows in the same commit" in m for m in msgs)


def test_a_build_yaml_transition_inside_known_transitions_must_also_fall_inside_a_seal_transition(git_bench: Path):
    # inv: build.yaml is locked, so an admitted change to it must have happened inside an
    # unlock/reseal cycle -- an admitted transition with no seal commit in its window was made
    # by hand around the lock
    b = git_bench
    repo_root = b.parent
    build_path = _seed_index(b)
    s0 = _seal_commit(repo_root, b, "initial seal")
    h1 = _commit_build_yaml(repo_root, build_path, {"artifacts": {}}, "fixtures: initial build.yaml")
    _sealed_row(b, {**_row("q1-s-c-a01", question="q1", build_yaml_sha256=h1),
                    "instrument_sha256": s0, "outcome": "completed"})
    h2 = _commit_build_yaml(repo_root, build_path, {"artifacts": {"graph.db": "zzz"}}, "chore: edit one")
    # no seal commit between q1-s-c-a01 and q1-s-c-a02: the instrument stays s0 on both rows
    row2 = {**_row("q1-s-c-a02", question="q1", attempt=2, build_yaml_sha256=h2),
            "instrument_sha256": s0, "outcome": "completed"}
    _sealed_row(b, row2)
    _admit(b, h1, h2)
    assert any(f"build.yaml transition {h1[:10]}->{h2[:10]} outside any seal transition" in m
               for m in audit.check_attempts(b))

    # a fresh transition, admitted the same way, but with a seal commit inside its window: silent
    s1 = _seal_commit(repo_root, b, "reseal for the second edit")
    h3 = _commit_build_yaml(repo_root, build_path, {"artifacts": {"graph.db": "zzz"}, "vendor_writes": ["x"]},
                            "chore: edit two")
    row3 = {**_row("q1-s-c-a03", question="q1", attempt=3, build_yaml_sha256=h3),
            "instrument_sha256": s1, "outcome": "completed"}
    _sealed_row(b, row3)
    _admit(b, h2, h3, reason="admitted the second edit")
    assert not any("outside any seal transition" in m for m in audit.check_attempts(b))


def test_audit_checks_completed_rows_against_their_evidence(git_bench, monkeypatch):
    b = git_bench
    repo_root = b.parent
    # why: record/runs/ is gitignored on the real tree, so evidence never rides along in a row's
    # own commit -- mirrored here so `git add -A` below stages the row alone, as it does for real
    (repo_root / ".gitignore").write_text("benchmark/record/runs/\n", encoding="utf-8")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "fixtures: ignore run evidence")
    run = b / "record" / "runs" / "q1-s-c-a01"
    run.mkdir(parents=True)
    for name in ("journal.jsonl", "records.jsonl", "cost.json", "audit.json"):
        (run / name).write_text(name + "\n", encoding="utf-8")
    evidence_names = ("journal.jsonl", "records.jsonl", "cost.json", "audit.json")
    hashes = {f"{n.split('.')[0]}_sha256": rules.sha256_file(run / n) for n in evidence_names}
    row = _row("q1-s-c-a01", question="q1", system="s", configuration="c", attempt=1)
    _write_rows(ledger.path(b), [row])
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 prepared")
    _write_rows(ledger.path(b), [{**row, "outcome": "completed", **hashes}])
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 completed")
    assert audit.check_attempts(b) == []

    (run / "cost.json").write_text("tampered\n", encoding="utf-8")
    assert any("row q1-s-c-a01: cost.json differs from its recorded hash" in m for m in audit.check_attempts(b))
    (run / "cost.json").write_text("cost.json\n", encoding="utf-8")

    import shutil
    shutil.rmtree(run)
    assert any("row q1-s-c-a01: no evidence directory" in m for m in audit.check_attempts(b))


def test_audit_skips_evidence_for_aborted_rows(git_bench):
    b = git_bench
    row = _row("q1-s-c-a01", question="q1", system="s", configuration="c", attempt=1)
    _write_rows(ledger.path(b), [row])
    _git(b.parent, "add", "-A")
    _git(b.parent, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 prepared")
    _write_rows(ledger.path(b), [{**row, "outcome": "aborted"}])
    _git(b.parent, "add", "-A")
    _git(b.parent, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 aborted")
    assert audit.check_attempts(b) == []


def test_audit_flags_evidence_not_owned_by_bench(git_bench, monkeypatch):
    b = git_bench
    row = _row("q1-s-c-a01", question="q1", system="s", configuration="c", attempt=1)
    _write_rows(ledger.path(b), [row])
    _git(b.parent, "add", "-A")
    _git(b.parent, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 prepared")
    hashes = _evidence(b, "q1-s-c-a01")
    _write_rows(ledger.path(b), [{**row, "outcome": "completed", **hashes}])
    _git(b.parent, "add", "-A")
    _git(b.parent, "commit", "-qm", "chore(benchmark): attempt q1-s-c-a01 completed")
    (b / "lock").mkdir()
    (b / "lock" / "machine.yaml").write_text(yaml.safe_dump({"bench_uid": 1}), encoding="utf-8")
    # why: only the run dir's ownership is under test, so every path resolves to a fixed non-bench uid
    monkeypatch.setattr(rules, "_owner", lambda p: 2)
    assert audit.check_attempts(b) == ["row q1-s-c-a01: evidence not owned by bench"]


def test_rebaseline_compares_the_two_newest_recipes(git_bench):
    b = git_bench
    idx = snapshot_dir(b, "snap") / "indexes" / "i"
    idx.mkdir(parents=True)
    (b / "systems" / "s").mkdir(parents=True)
    (b / "systems" / "s" / "harness.yaml").write_text(yaml.safe_dump({
        "adapter": "a", "version": {"cli": "1"}, "invocation": {"package": {}}, "fixed_steps": [],
        "default_configuration": "c", "configurations": {"c": {"index": "indexes/i"}},
        "sandbox_layout": {}, "environment": {}, "docs": {}}), encoding="utf-8")
    old, new = "a" * 64, "b" * 64
    (idx / "prepared_outputs.yaml").write_text(yaml.safe_dump({"c": {
        old: {"q001": {"02_query": {"out": "x"}, "01_version": {"out": "v"}}},
        new: {"q001": {"02_query": {"out": "x"}, "01_version": {"out": "v"}}, "q002": {"02_query": {"out": "y"}}}}}),
        encoding="utf-8")
    rows = [{**_row("q001-s-c-a01", question="q001", system="s", configuration="c", attempt=1),
             "harness_sha256": old, "outcome": "completed"},
            {**_row("q001-s-c-a02", question="q001", system="s", configuration="c", attempt=2),
             "harness_sha256": new, "outcome": "completed"}]
    _write_rows(ledger.path(b), rows)
    write_question(b, "q001", {"id": "q001", "snapshot": "snap", "text": "t"})
    assert audit.check_rebaseline(b, "s") == []
    doc = yaml.safe_load((idx / "prepared_outputs.yaml").read_text(encoding="utf-8"))
    doc["c"][new]["q001"]["02_query"]["out"] = "z"
    (idx / "prepared_outputs.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert audit.check_rebaseline(b, "s") == ["rebaseline s/c q001 02_query: output differs from the previous recipe"]


def test_commit_rows_folds_a_changed_extra_path_into_the_same_commit(git_bench: Path):
    (git_bench / "extra.txt").write_text("x", encoding="utf-8")
    _git(git_bench.parent, "add", "-A")
    _git(git_bench.parent, "commit", "-qm", "fixtures: track extra.txt")
    ledger.append_row(git_bench, _row("q-s-c-a01", attempt=1))
    (git_bench / "extra.txt").write_text("y", encoding="utf-8")
    sha = ledger.commit_rows(git_bench, "chore(benchmark): attempt q-s-c-a01 prepared",
                             extra_paths=("benchmark/extra.txt",))
    touched = sorted(subprocess.run(
        ["git", "-C", str(git_bench.parent), "show", "--name-only", "--format=", sha],
        capture_output=True, text=True, check=True).stdout.split())
    assert touched == ["benchmark/extra.txt", "benchmark/record/attempts.jsonl"]


def test_commit_rows_omits_an_extra_path_that_did_not_change(git_bench: Path):
    (git_bench / "extra.txt").write_text("x", encoding="utf-8")
    _git(git_bench.parent, "add", "-A")
    _git(git_bench.parent, "commit", "-qm", "fixtures: track extra.txt")
    ledger.append_row(git_bench, _row("q-s-c-a01", attempt=1))
    sha = ledger.commit_rows(git_bench, "chore(benchmark): attempt q-s-c-a01 prepared",
                             extra_paths=("benchmark/extra.txt",))
    touched = subprocess.run(
        ["git", "-C", str(git_bench.parent), "show", "--name-only", "--format=", sha],
        capture_output=True, text=True, check=True).stdout.split()
    assert touched == ["benchmark/record/attempts.jsonl"]


def test_commit_content_records_the_blob_it_is_given_not_the_working_file(git_bench: Path):
    # inv: the commit is made from a staged blob, so a ledger the working file has already grown
    # past can still be committed one row at a time without the file being rewritten
    first, second = _row("q-s-c-a01", attempt=1), _row("q-s-c-a02", attempt=2)
    ledger.append_row(git_bench, first)
    ledger.append_row(git_bench, second)
    whole = ledger.path(git_bench).read_bytes()
    sha = ledger.commit_content(git_bench, "chore(benchmark): attempt q-s-c-a01 prepared",
                                ledger.as_text([first]))
    committed = subprocess.run(
        ["git", "-C", str(git_bench.parent), "show", f"{sha}:{ledger.rel(git_bench)}"],
        capture_output=True, text=True, check=True).stdout
    assert [json.loads(line)["run_id"] for line in committed.splitlines() if line.strip()] == ["q-s-c-a01"]
    assert ledger.path(git_bench).read_bytes() == whole


def test_commit_content_folds_a_changed_extra_path_into_the_same_commit(git_bench: Path):
    (git_bench / "extra.txt").write_text("x", encoding="utf-8")
    _git(git_bench.parent, "add", "-A")
    _git(git_bench.parent, "commit", "-qm", "fixtures: track extra.txt")
    row = _row("q-s-c-a01", attempt=1)
    ledger.append_row(git_bench, row)
    (git_bench / "extra.txt").write_text("y", encoding="utf-8")
    sha = ledger.commit_content(git_bench, "chore(benchmark): attempt q-s-c-a01 prepared",
                                ledger.as_text([row]), extra_paths=("benchmark/extra.txt",))
    touched = sorted(subprocess.run(
        ["git", "-C", str(git_bench.parent), "show", "--name-only", "--format=", sha],
        capture_output=True, text=True, check=True).stdout.split())
    assert touched == ["benchmark/extra.txt", "benchmark/record/attempts.jsonl"]


def test_commit_content_refuses_when_the_index_already_holds_a_staged_change(git_bench: Path):
    # inv: the whole index is committed, so anything a caller left staged would ride along under
    # an attempt's own subject; it is refused instead, and nothing is staged by the refusal
    (git_bench / "extra.txt").write_text("x", encoding="utf-8")
    _git(git_bench.parent, "add", "-A")
    _git(git_bench.parent, "commit", "-qm", "fixtures: track extra.txt")
    (git_bench / "extra.txt").write_text("y", encoding="utf-8")
    _git(git_bench.parent, "add", "--", "benchmark/extra.txt")
    row = _row("q-s-c-a01", attempt=1)
    ledger.append_row(git_bench, row)
    with pytest.raises(ledger.LedgerError, match="index already holds a staged change"):
        ledger.commit_content(git_bench, "chore(benchmark): attempt q-s-c-a01 prepared",
                              ledger.as_text([row]))


def test_commit_content_refuses_on_a_detached_head(git_bench: Path):
    _git(git_bench.parent, "checkout", "-q", "--detach")
    row = _row("q-s-c-a01", attempt=1)
    ledger.append_row(git_bench, row)
    with pytest.raises(ledger.LedgerError, match="HEAD is detached"):
        ledger.commit_content(git_bench, "chore(benchmark): attempt q-s-c-a01 prepared",
                              ledger.as_text([row]))


def test_append_row_and_complete_row_no_longer_commit(git_bench: Path):
    # inv: the caller commits now -- append_row and complete_row only ever write the file
    ledger.append_row(git_bench, _row("q-s-c-a01", attempt=1))
    assert _git_status(git_bench.parent)
    ledger.commit_rows(git_bench, "chore(benchmark): attempt q-s-c-a01 prepared")
    assert not _git_status(git_bench.parent)
    ledger.complete_row(git_bench, "q-s-c-a01", {"outcome": "completed"})
    assert _git_status(git_bench.parent)
    ledger.commit_rows(git_bench, "chore(benchmark): attempt q-s-c-a01 completed")
    assert not _git_status(git_bench.parent)


def _git_status(repo_root: Path) -> str:
    return subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                          capture_output=True, text=True, check=True).stdout


def test_a_sealed_row_committed_once_passes_and_the_old_two_commit_shape_is_now_flagged(git_bench: Path):
    s0 = _seal_commit(git_bench.parent, git_bench, "initial seal")
    # the new rule's happy path: a sealed row introduced already complete, in one commit
    _evidence(git_bench, "q1-s-c-a01")
    _sealed_row(git_bench, {**_row("q1-s-c-a01", question="q1"), "instrument_sha256": s0, "outcome": "completed"})
    assert audit.check_attempts(git_bench) == []
    # a sealed row still following the old introduce-then-update shape: two commits, now a violation
    _evidence(git_bench, "q1-s-c-a02")
    _append(git_bench, {**_row("q1-s-c-a02", question="q1", attempt=2), "instrument_sha256": s0})
    _complete(git_bench, "q1-s-c-a02", {"outcome": "completed"})
    msgs = audit.check_attempts(git_bench)
    assert any("q1-s-c-a02: sealed row touched by 2 commits; a sealed row is committed exactly once" in m
               for m in msgs)


def test_a_sealed_row_committed_once_with_the_wrong_subject_is_flagged(git_bench: Path):
    s0 = _seal_commit(git_bench.parent, git_bench, "initial seal")
    row = {**_row("q1-s-c-a01", question="q1"), "instrument_sha256": s0, "outcome": "completed"}
    ledger.rewrite(git_bench, [*ledger.rows(git_bench), row])
    sha = ledger.commit_rows(git_bench, "chore(benchmark): attempt q1-s-c-a01 finished")  # wrong outcome word
    msgs = audit.check_attempts(git_bench)
    expected = (f"commit {sha[:10]} message 'chore(benchmark): attempt q1-s-c-a01 finished' is not "
               f"'chore(benchmark): attempt q1-s-c-a01 completed' for sealed row q1-s-c-a01")
    assert any(expected in m for m in msgs)


def test_a_sealed_row_still_in_flight_carries_no_violation(git_bench: Path):
    # inv: a sealed row not yet collected has no outcome yet; the one-commit rule leaves it alone
    s0 = _seal_commit(git_bench.parent, git_bench, "initial seal")
    _append(git_bench, {**_row("q1-s-c-a01", question="q1"), "instrument_sha256": s0})
    assert audit.check_attempts(git_bench) == []


def test_a_sealed_rows_one_commit_may_also_carry_its_own_cells_prepared_outputs(git_bench: Path):
    _seed_index(git_bench)
    s0 = _seal_commit(git_bench.parent, git_bench, "initial seal")
    _evidence(git_bench, "q1-s-c-a01")
    index_dir = snapshot_dir(git_bench, "snap") / "indexes" / "c"
    (index_dir / prepare.PREPARED).write_text("c: {}\n", encoding="utf-8")
    row = {**_row("q1-s-c-a01", question="q1"), "instrument_sha256": s0, "outcome": "completed"}
    ledger.rewrite(git_bench, [*ledger.rows(git_bench), row])
    prepared_rel = f"{index_dir.relative_to(git_bench.parent).as_posix()}/{prepare.PREPARED}"
    ledger.commit_rows(git_bench, "chore(benchmark): attempt q1-s-c-a01 completed",
                       extra_paths=(prepared_rel,))
    assert audit.check_attempts(git_bench) == []


def test_a_sealed_rows_one_commit_flags_an_unrelated_file_under_the_same_index_dir(git_bench: Path):
    _seed_index(git_bench)
    s0 = _seal_commit(git_bench.parent, git_bench, "initial seal")
    _evidence(git_bench, "q1-s-c-a01")
    index_dir = snapshot_dir(git_bench, "snap") / "indexes" / "c"
    (index_dir / "junk.txt").write_text("stray\n", encoding="utf-8")
    row = {**_row("q1-s-c-a01", question="q1"), "instrument_sha256": s0, "outcome": "completed"}
    ledger.rewrite(git_bench, [*ledger.rows(git_bench), row])
    stray_rel = f"{index_dir.relative_to(git_bench.parent).as_posix()}/junk.txt"
    ledger.commit_rows(git_bench, "chore(benchmark): attempt q1-s-c-a01 completed", extra_paths=(stray_rel,))
    msgs = audit.check_attempts(git_bench)
    assert any("touches" in m and "not only" in m and "junk.txt" in m for m in msgs)
