import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.harness import collect, ledger, prepare, rules
from tests.benchmark.conftest import _reseal, references_dir, review_dir, snapshot_dir, write_question


def test_clean_sandbox_removes_only_unlisted(tmp_path: Path):
    sb = tmp_path / "index"
    (sb / "graphify-out" / "cache").mkdir(parents=True)
    (sb / "graphify-out" / "graph.json").write_text("{}")
    (sb / "graphify-out" / "cache" / "stamp").write_text("1")
    (sb / "graphify-out" / ".vocab.txt").write_text("a")
    removed = collect.clean_sandbox(sb, {"graphify-out/graph.json": "x"})
    assert sorted(removed) == ["graphify-out/.vocab.txt", "graphify-out/cache/stamp"]
    assert (sb / "graphify-out" / "graph.json").exists()
    assert not (sb / "graphify-out" / "cache").exists()


@pytest.mark.parametrize("name", ["03_query.out", "03_query.cmd", "journal.jsonl", "build.yaml"])
def test_clean_sandbox_refuses_evidence(tmp_path: Path, name: str):
    sb = tmp_path / "index"
    sb.mkdir()
    (sb / name).write_text("x")
    with pytest.raises(SystemExit, match=name):
        collect.clean_sandbox(sb, {})


def _setup_system(bench: Path) -> None:
    # why: systems/** is instrument-classified, so a system built after sealed_bench sealed the
    # tree must be resealed, or every collect() below meets require_sealed's own drift check
    sysdir = bench / "systems" / "s"
    sysdir.mkdir(parents=True)
    harness = {"adapter": "graphify", "version": {"cli": "x"}, "invocation": {}, "fixed_steps": [],
               "default_configuration": "d", "configurations": {"d": {"index": "idx"}},
               "sandbox_layout": {"layout": "<artifacts>"}, "environment": {}, "docs": {}}
    (sysdir / "harness.yaml").write_text(yaml.safe_dump(harness), encoding="utf-8")
    index_dir = snapshot_dir(bench, "snap") / "idx"
    index_dir.mkdir(parents=True)
    (index_dir / "build.yaml").write_text(yaml.safe_dump({"artifacts": {}}), encoding="utf-8")
    # why: every path collect resolves for a run goes through the question the run names, so a
    # tree without that question exercises a shape no prepared run can have
    write_question(bench, "q", {"id": "q", "snapshot": "snap", "text": "does not matter"})
    _reseal(bench)


def _build_run(bench: Path, tmp_root: Path, run_id: str, *, valid: bool, with_records: bool,
               with_artifacts_key: bool = True, driven: dict | None = None,
               with_review: bool = True) -> tuple[Path, dict]:
    run_dir = tmp_root / run_id / "run"
    run_dir.mkdir(parents=True)
    build_hash = rules.sha256_file(snapshot_dir(bench, "snap") / "idx" / "build.yaml")
    row = {"run_id": run_id, "question": "q", "system": "s", "configuration": "d", "attempt": 1,
           "question_sha256": "a", "build_yaml_sha256": build_hash}
    ledger.append_row(bench, row)
    review_path = review_dir(bench, "snap") / f"{row['question']}.yaml"
    if with_review and not review_path.is_file():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(yaml.safe_dump({"reviewer_model": "test"}, sort_keys=False), encoding="utf-8")
        _reseal(bench)
    run_yaml = {**row, "tmp_root": str(tmp_root), "snapshot": "snap", "fixed_steps": 0, "outcome": "prepared"}
    if with_artifacts_key:
        run_yaml["artifacts"] = {}
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run_yaml), encoding="utf-8")
    entries = [{"n": 0, "kind": "header", "rules_version": 1},
               {"n": 1, "kind": "stop", "by": "harness", "reason": "harness"}]
    (run_dir / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    (run_dir / "audit.json").write_text(json.dumps({"valid": valid, "violations": [], "stop": "harness"}),
                                         encoding="utf-8")
    (run_dir / "cost.json").write_text(
        json.dumps({"tokens": 3, "system_calls": 1, "ceiling_calls": 2}), encoding="utf-8")
    if with_records:
        (run_dir / "records.jsonl").write_text("", encoding="utf-8")
    # inv: collect() takes hits in memory, so the fixture hands the mapping straight to it
    # rather than writing it to a file collect would have to find
    result = {"hit": True, "hit_rank": 1, "hit_entry": 1, **(driven or {})}
    return run_dir, result


def test_collect_completed_when_audit_valid_and_records_exist(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    prepare.take_lock(tmp_root, run_id)
    outcome = collect.collect(sealed_bench, run_dir, hits)
    assert outcome == "completed"
    assert ledger.rows(sealed_bench)[0]["outcome"] == "completed"
    assert (sealed_bench / "record" / "runs" / run_id).exists()
    assert not (tmp_root / "sandbox" / "lock").exists()


def test_collect_void_when_audit_invalid(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=False, with_records=True)
    assert collect.collect(sealed_bench, run_dir, hits) == "void"
    assert ledger.rows(sealed_bench)[0]["outcome"] == "void"


def test_collect_void_when_records_missing(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=False)
    assert collect.collect(sealed_bench, run_dir, hits) == "void"


def test_collect_standalone_closes_a_run_with_no_records_as_void(sealed_bench: Path):
    # inv: called with no hits and no records.jsonl, as the command line does after a kill inside
    # the drive, collect closes the row as void instead of failing inside score
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, _hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=False)
    assert collect.collect(sealed_bench, run_dir) == "void"


def test_collect_writes_exactly_the_fifteen_specified_ledger_fields(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True)
    before_keys = set(ledger.rows(sealed_bench)[0])
    collect.collect(sealed_bench, run_dir, hits)
    after_keys = set(ledger.rows(sealed_bench)[0])
    assert after_keys - before_keys == {
        "outcome", "stop", "hit", "hit_rank", "hit_entry", "tokens", "system_calls",
        "ceiling_calls", "canonical", "journal_sha256", "records_sha256", "cost_sha256",
        "audit_sha256", "model_served", "review_sha256",
    }


def test_collect_carries_the_ceiling_call_count_into_the_ledger(sealed_bench: Path):
    # inv: the ledger is the durable record, so a figure quoted about ceiling calls has to be
    # recheckable from it rather than from gitignored results/
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True)
    collect.collect(sealed_bench, run_dir, hits)
    assert ledger.rows(sealed_bench)[0]["ceiling_calls"] == 2


def test_collect_refuses_to_overwrite_existing_evidence(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    collect.collect(sealed_bench, run_dir, hits)
    with pytest.raises(SystemExit, match="evidence is never overwritten"):
        collect.collect(sealed_bench, run_dir, hits)


def test_collect_refuses_missing_artifacts_key_instead_of_wiping_the_sandbox(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True,
                               with_artifacts_key=False)
    sandbox_index = tmp_root / "sandbox" / "index"
    sandbox_index.mkdir(parents=True)
    (sandbox_index / "graph.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="artifacts"):
        collect.collect(sealed_bench, run_dir, hits)
    assert (sandbox_index / "graph.json").exists()
    assert not (sealed_bench / "record" / "runs" / "q-s-d-a01").exists()


def test_collect_rolls_back_evidence_and_releases_lock_when_writing_the_row_fails(
    monkeypatch: pytest.MonkeyPatch, sealed_bench: Path,
):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    prepare.take_lock(tmp_root, run_id)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated ledger commit failure")

    monkeypatch.setattr(collect.ledger, "complete_row", _boom)
    with pytest.raises(RuntimeError, match="simulated"):
        collect.collect(sealed_bench, run_dir, hits)
    assert not (sealed_bench / "record" / "runs" / run_id).exists()
    assert not (tmp_root / "sandbox" / "lock").exists()


def test_collect_refuses_a_dest_published_after_its_own_check_without_deleting_it(
        sealed_bench: Path, monkeypatch: pytest.MonkeyPatch):
    # inv: the dest check and the copy are seconds apart, so a dest that appears between them
    # holds another invocation's evidence -- this one refuses and takes nothing back out
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    # inv: collect only reaches clean_sandbox when the sandbox exists, so the interposition below
    # stands where collect actually stands rather than never running
    (tmp_root / "sandbox" / "index").mkdir(parents=True)
    prepare.take_lock(tmp_root, run_id)
    dest = sealed_bench / "record" / "runs" / run_id
    real_clean = collect.clean_sandbox

    def publish_then_clean(sandbox: Path, artifacts: dict) -> list[str]:
        """Stand where collect stands between its dest check and its copytree."""
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "journal.jsonl").write_text("the other invocation's evidence\n", encoding="utf-8")
        return real_clean(sandbox, artifacts)

    monkeypatch.setattr(collect, "clean_sandbox", publish_then_clean)
    with pytest.raises(SystemExit, match="evidence is never overwritten"):
        collect.collect(sealed_bench, run_dir, hits)
    assert (dest / "journal.jsonl").read_text() == "the other invocation's evidence\n"


def test_collect_retried_after_a_failed_commit_publishes_what_the_first_clean_removed(
        sealed_bench: Path, monkeypatch: pytest.MonkeyPatch):
    # inv: the second clean finds an already-empty sandbox and returns nothing, so the record of
    # what was scrubbed has to accumulate rather than be rewritten from the retry alone
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    sandbox = tmp_root / "sandbox" / "index"
    sandbox.mkdir(parents=True)
    (sandbox / "junk.tmp").write_text("vendor droppings", encoding="utf-8")
    prepare.take_lock(tmp_root, run_id)
    real_complete = collect.ledger.complete_row
    calls = {"n": 0}

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated ledger commit failure")
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(collect.ledger, "complete_row", fail_once)
    with pytest.raises(RuntimeError, match="simulated"):
        collect.collect(sealed_bench, run_dir, hits)
    assert json.loads((run_dir / "removed.json").read_text()) == ["junk.tmp"]
    assert collect.collect(sealed_bench, run_dir, hits) == "completed"
    assert json.loads((sealed_bench / "record" / "runs" / run_id / "removed.json").read_text()) == ["junk.tmp"]


def test_a_copy_that_dies_partway_leaves_no_tree_behind(sealed_bench: Path):
    # inv: only a dest that already existed is refused without a rollback; a copy this invocation
    # started and did not finish is taken back out, or every retry meets that refusal instead
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    prepare.take_lock(tmp_root, run_id)
    dest = sealed_bench / "record" / "runs" / run_id

    def half_copy(src: Path, dst: Path) -> None:
        Path(dst).mkdir(parents=True)
        (Path(dst) / "journal.jsonl").write_text("half a tree\n", encoding="utf-8")
        raise shutil.Error("the copy died partway")

    # why: a scoped context, not the shared `monkeypatch` fixture -- exiting it must not also
    # undo the seal patches sealed_bench set on that fixture for the rest of this test, and it
    # must undo cleanly even if an assertion inside the block raises
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(collect.shutil, "copytree", half_copy)
        with pytest.raises(shutil.Error):
            collect.collect(sealed_bench, run_dir, hits)
        assert not dest.exists()
    prepare.take_lock(tmp_root, run_id)
    assert collect.collect(sealed_bench, run_dir, hits) == "completed"


def test_a_removed_record_a_kill_truncated_does_not_stop_the_retry(sealed_bench: Path):
    # inv: collect is the retry command, so a record half-written by a kill must not crash the
    # attempt that would replace it -- and by then this attempt's own clean has already run
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    sandbox = tmp_root / "sandbox" / "index"
    sandbox.mkdir(parents=True)
    (sandbox / "junk.tmp").write_text("droppings", encoding="utf-8")
    (run_dir / "removed.json").write_text('[\n "junk', encoding="utf-8")
    prepare.take_lock(tmp_root, run_id)
    assert collect.collect(sealed_bench, run_dir, hits) == "completed"
    assert json.loads((sealed_bench / "record" / "runs" / run_id / "removed.json").read_text()) == ["junk.tmp"]
    # inv: the record is written through a rename, so no part file is left beside it
    assert not list(run_dir.glob("*.part"))


def test_a_retry_records_what_each_clean_removed(sealed_bench: Path, monkeypatch: pytest.MonkeyPatch):
    # inv: the record is a union, so a retry that finds something new adds it instead of replacing
    # the first attempt's list with its own
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    sandbox = tmp_root / "sandbox" / "index"
    sandbox.mkdir(parents=True)
    (sandbox / "first.tmp").write_text("x", encoding="utf-8")
    prepare.take_lock(tmp_root, run_id)
    real_complete = collect.ledger.complete_row
    calls = {"n": 0}

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated ledger commit failure")
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(collect.ledger, "complete_row", fail_once)
    with pytest.raises(RuntimeError, match="simulated"):
        collect.collect(sealed_bench, run_dir, hits)
    (sandbox / "second.tmp").write_text("y", encoding="utf-8")
    assert collect.collect(sealed_bench, run_dir, hits) == "completed"
    assert json.loads((sealed_bench / "record" / "runs" / run_id / "removed.json").read_text()) == \
        ["first.tmp", "second.tmp"]


def test_collect_refuses_when_run_yaml_and_the_row_name_different_questions(sealed_bench: Path):
    # inv: the hit is scored against the question run.yaml names and published under the one the
    # row names; no hash covers run.yaml, so a disagreement would publish one verdict as another's
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    meta["question"] = "q999"
    (run_dir / "run.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    prepare.take_lock(tmp_root, run_id)
    with pytest.raises(SystemExit, match="disagree"):
        collect.collect(sealed_bench, run_dir, hits)
    assert not (sealed_bench / "record" / "runs" / run_id).exists()


def test_collect_refuses_a_lock_it_cannot_read(sealed_bench: Path):
    # inv: take_lock stops for a human on an unreadable lock, so cleaning a sandbox whose owner
    # cannot be determined stops here too rather than deleting the lock and proceeding
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    sandbox = tmp_root / "sandbox" / "index"
    sandbox.mkdir(parents=True)
    (sandbox / "left_by_someone.bin").write_text("x", encoding="utf-8")
    (tmp_root / "sandbox" / "lock").write_text('{"pid": 12', encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable sandbox lock"):
        collect.collect(sealed_bench, run_dir, hits)
    assert (tmp_root / "sandbox" / "lock").exists()
    assert (sandbox / "left_by_someone.bin").exists()


def test_collect_refuses_a_sandbox_a_different_run_holds(sealed_bench: Path):
    # inv: the sandbox is shared, so cleaning it against this run's artifact list is only correct
    # while this run holds it -- a lock naming another run means a later one is using it
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    sandbox = tmp_root / "sandbox" / "index"
    sandbox.mkdir(parents=True)
    (sandbox / "belongs_to_the_other_run.bin").write_text("x", encoding="utf-8")
    (tmp_root / "sandbox" / "lock").write_text(
        json.dumps({"run_id": "SOMEBODY-ELSE", "pid": 999999}), encoding="utf-8")
    with pytest.raises(SystemExit, match="held by SOMEBODY-ELSE"):
        collect.collect(sealed_bench, run_dir, hits)
    assert (sandbox / "belongs_to_the_other_run.bin").exists()
    assert not (sealed_bench / "record" / "runs" / run_id).exists()


DRIVEN = {"stop": "answer_met", "stop_hit": True, "hit_by": "runner", "runner_actions": 3,
          "refused": 1, "model_usage": {"input_tokens": 10, "output_tokens": 2}}


def test_a_driven_run_carries_what_the_runner_did_into_the_ledger(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True, driven=DRIVEN)
    collect.collect(sealed_bench, run_dir, hits)
    row = ledger.rows(sealed_bench)[0]
    assert row["hit_by"] == "runner"
    assert row["runner_actions"] == 3
    assert row["refused"] == 1
    assert row["model_usage"] == {"input_tokens": 10, "output_tokens": 2}


def test_a_baseline_row_does_not_grow_the_runners_fields(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True)
    collect.collect(sealed_bench, run_dir, hits)
    row = ledger.rows(sealed_bench)[0]
    # inv: a baseline row keeps the shape every published figure was written from
    assert not {"hit_by", "runner_actions", "refused", "model_usage", "stop_hit"} & set(row)


def test_collect_computes_hits_itself_when_none_are_given(sealed_bench: Path):
    # inv: collect must still run standalone -- when the caller has no hits in hand, it scores
    # the run itself, exactly as `score.hits` would
    _setup_system(sealed_bench)
    (references_dir(sealed_bench, "snap") / "q.yaml").write_text(
        yaml.safe_dump({"places": [{"path": "pkg/a.py", "symbol": "f", "start": 1}]}), encoding="utf-8")
    # inv: a snapshot's references/ is instrument-classified, so a fresh one must be resealed
    _reseal(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, _ = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True)
    (run_dir / "records.jsonl").write_text("", encoding="utf-8")
    assert collect.collect(sealed_bench, run_dir) == "completed"
    row = ledger.rows(sealed_bench)[0]
    assert row["hit"] is False


def test_collect_carries_the_model_served_from_run_yaml(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True)
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    meta["model_served"] = "qwen3-8b"
    (run_dir / "run.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    collect.collect(sealed_bench, run_dir, hits)
    assert ledger.rows(sealed_bench)[0]["model_served"] == "qwen3-8b"


def test_collect_hashes_the_request_only_for_a_driven_run(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True)
    (run_dir / "request.json").write_text('{"model": "m"}', encoding="utf-8")
    collect.collect(sealed_bench, run_dir, hits)
    row = ledger.rows(sealed_bench)[0]
    assert row["request_sha256"] == rules.sha256_file(run_dir / "request.json")


def test_collect_omits_the_request_hash_for_a_baseline_run(sealed_bench: Path):
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True)
    collect.collect(sealed_bench, run_dir, hits)
    assert "request_sha256" not in ledger.rows(sealed_bench)[0]


def test_collect_hashes_the_review_file_when_it_exists(sealed_bench: Path):
    _setup_system(sealed_bench)
    (review_dir(sealed_bench, "snap") / "q.yaml").write_text("verdict: ok\n", encoding="utf-8")
    # inv: a snapshot's questions/ is instrument-classified, so a fresh review must be resealed
    _reseal(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_dir, hits = _build_run(sealed_bench, tmp_root, "q-s-d-a01", valid=True, with_records=True)
    collect.collect(sealed_bench, run_dir, hits)
    row = ledger.rows(sealed_bench)[0]
    assert row["review_sha256"] == rules.sha256_file(review_dir(sealed_bench, "snap") / "q.yaml")


def test_collect_raises_when_there_is_no_review_file(sealed_bench: Path):
    # inv: prepare refuses a question with no review, so collect finding none here is a defect
    # elsewhere in the pipeline, not a shape collect should tolerate; the sandbox lock must still
    # come back, or every later run would meet "the sandbox is held by ..." because of this failure
    _setup_system(sealed_bench)
    tmp_root = sealed_bench.parent / "tmp"
    run_id = "q-s-d-a01"
    run_dir, hits = _build_run(sealed_bench, tmp_root, run_id, valid=True, with_records=True,
                               with_review=False)
    prepare.take_lock(tmp_root, run_id)
    with pytest.raises(SystemExit, match="has no review"):
        collect.collect(sealed_bench, run_dir, hits)
    assert not (tmp_root / "sandbox" / "lock").exists()


def test_prepared_outputs_rel_names_the_cells_expectations_file(sealed_bench: Path):
    # inv: run._prepared_rel calls this to fold the path into the attempt's one commit exactly
    # when git sees it changed; collect itself no longer commits, so this derivation is what
    # run reuses rather than repeats
    _setup_system(sealed_bench)
    meta = {"system": "s", "configuration": "d", "snapshot": "snap"}
    assert collect.prepared_outputs_rel(sealed_bench, meta) == \
        "benchmark/record/snapshots/snap/idx/prepared_outputs.yaml"
