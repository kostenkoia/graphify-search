import json
import subprocess
from pathlib import Path

import pytest
import yaml

from benchmark.harness import config, ledger, rules, run
from tests.benchmark.conftest import _reseal, snapshot_dir, write_question, write_review


def _row(run_id: str, **extra: object) -> dict:
    base = {"run_id": run_id, "question": "q1", "system": "s", "configuration": "c", "attempt": 1}
    return {**base, **extra}


def _write_harness(bench: Path, system: str, content: str) -> str:
    d = bench / "systems" / system
    d.mkdir(parents=True, exist_ok=True)
    (d / "harness.yaml").write_text(content, encoding="utf-8")
    return rules.sha256_file(d / "harness.yaml")


# --- stranded-row repair -----------------------------------------------------------------


def _fake_machine(bench: Path) -> dict:
    return {"repo_root": str(bench.parent)}


def test_repair_stranded_commits_a_completed_row_head_does_not_have(git_bench: Path):
    ledger.append_row(git_bench, _row("q1-s-c-a01", outcome="completed"))
    assert ledger.changed(git_bench, ledger.rel(git_bench))
    repaired = run._repair_stranded(git_bench, _fake_machine(git_bench))
    assert repaired == ["q1-s-c-a01"]
    assert not ledger.changed(git_bench, ledger.rel(git_bench))
    log = ledger.rows(git_bench)
    assert log[-1]["outcome"] == "completed"


def test_repair_stranded_refuses_a_row_with_no_outcome_yet(git_bench: Path):
    # inv: bench marks; a row with no outcome yet was never even marked -- `run` never invents
    # one, it sends the operator to `abort` first
    ledger.append_row(git_bench, _row("q1-s-c-a01"))
    with pytest.raises(SystemExit, match="q1-s-c-a01 has no outcome.*abort q1-s-c-a01"):
        run._repair_stranded(git_bench, _fake_machine(git_bench))
    # inv: a refused repair makes no commit -- the row is left exactly as found
    assert ledger.changed(git_bench, ledger.rel(git_bench))


def test_repair_stranded_is_a_no_op_when_the_ledger_already_matches_head(git_bench: Path):
    assert run._repair_stranded(git_bench, _fake_machine(git_bench)) == []


def _subjects(bench: Path, count: int) -> list[str]:
    out = subprocess.run(["git", "-C", str(bench.parent), "log", f"-{count}", "--format=%s"],
                         capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def _rows_at(bench: Path, revision: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(bench.parent), "show", f"{revision}:{ledger.rel(bench)}"],
                         capture_output=True, text=True, check=True).stdout
    return [json.loads(line)["run_id"] for line in out.splitlines() if line.strip()]


def _watch_commits(monkeypatch, bench: Path, fail_on: int | None = None) -> list[bytes]:
    """Record the ledger file's bytes as each `git commit` is issued; optionally fail the nth."""
    seen: list[bytes] = []
    real_git = ledger._git

    def _spy(benchmark: Path, *args: str, **kwargs: str | None) -> str:
        if args and args[0] == "commit":
            seen.append(ledger.path(bench).read_bytes())
            if fail_on is not None and len(seen) == fail_on:
                raise RuntimeError("git ['commit'] failed: simulated")
        return real_git(benchmark, *args, **kwargs)

    monkeypatch.setattr(ledger, "_git", _spy)
    return seen


def test_repair_stranded_gives_each_stranded_row_its_own_commit(
    git_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    # inv: audit attempts refuses a commit touching more than one row, and the ledger's history
    # is never rewritten, so two rows repaired together must still land as two commits
    ledger.append_row(git_bench, _row("q1-s-c-a01", outcome="completed"))
    ledger.append_row(git_bench, _row("q1-s-c-a02", attempt=2, outcome="aborted"))
    whole = ledger.path(git_bench).read_bytes()
    at_commit = _watch_commits(monkeypatch, git_bench)
    repaired = run._repair_stranded(git_bench, _fake_machine(git_bench))
    monkeypatch.undo()
    assert repaired == ["q1-s-c-a01", "q1-s-c-a02"]
    assert _subjects(git_bench, 2) == ["chore(benchmark): attempt q1-s-c-a02 aborted",
                                       "chore(benchmark): attempt q1-s-c-a01 completed"]
    assert _rows_at(git_bench, "HEAD~1") == ["q1-s-c-a01"]
    assert _rows_at(git_bench, "HEAD") == ["q1-s-c-a01", "q1-s-c-a02"]
    # inv: each commit records a blob, so the working file holds every row at every moment of
    # the repair -- it is never truncated to the prefix a commit is about to take
    assert at_commit == [whole, whole]
    assert ledger.path(git_bench).read_bytes() == whole
    assert not ledger.changed(git_bench, ledger.rel(git_bench))


def test_repair_stranded_leaves_the_rest_stranded_when_a_commit_fails(
    git_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    # inv: a failed or killed repair loses no row -- the working file holds every row throughout,
    # so whatever HEAD did not take is stranded again for the next `run`
    ledger.append_row(git_bench, _row("q1-s-c-a01", outcome="completed"))
    ledger.append_row(git_bench, _row("q1-s-c-a02", attempt=2, outcome="aborted"))
    whole = ledger.path(git_bench).read_bytes()
    at_commit = _watch_commits(monkeypatch, git_bench, fail_on=2)
    with pytest.raises(RuntimeError, match="simulated"):
        run._repair_stranded(git_bench, _fake_machine(git_bench))
    monkeypatch.undo()
    assert at_commit == [whole, whole]
    assert ledger.path(git_bench).read_bytes() == whole
    assert _rows_at(git_bench, "HEAD") == ["q1-s-c-a01"]
    assert [r["run_id"] for r in run._stranded_rows(git_bench)] == ["q1-s-c-a02"]


def test_repair_stranded_refuses_every_stranded_row_when_one_has_no_outcome(git_bench: Path):
    # inv: the guard reads every stranded row, not just the last one, so an older row with no
    # outcome cannot be committed under a younger row's subject
    ledger.append_row(git_bench, _row("q1-s-c-a01"))
    ledger.append_row(git_bench, _row("q1-s-c-a02", attempt=2, outcome="completed"))
    with pytest.raises(SystemExit) as exc:
        run._repair_stranded(git_bench, _fake_machine(git_bench))
    assert "q1-s-c-a01" in str(exc.value)
    assert "q1-s-c-a02" in str(exc.value)
    assert "abort q1-s-c-a01" in str(exc.value)
    assert ledger.changed(git_bench, ledger.rel(git_bench))
    assert [r["run_id"] for r in ledger.rows(git_bench)] == ["q1-s-c-a01", "q1-s-c-a02"]


def test_head_rows_reads_through_the_hardened_git_helper(git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    # inv: every git call in the package goes through ledger._git, which pins gpgsign and the
    # excludes file and passes --no-optional-locks; _head_rows is not an exception
    seen: list[tuple[str, ...]] = []

    def _spy(benchmark: Path, *args: str) -> str:
        seen.append(args)
        return ""

    monkeypatch.setattr(ledger, "_git", _spy)
    assert run._head_rows(git_bench) == []
    assert seen == [("show", f"HEAD:{ledger.rel(git_bench)}")]


def test_head_rows_is_empty_when_head_does_not_hold_the_ledger(git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    def _fail(benchmark: Path, *args: str) -> str:
        raise RuntimeError("git ['show'] failed: does not exist")

    monkeypatch.setattr(ledger, "_git", _fail)
    assert run._head_rows(git_bench) == []


def test_prepared_rel_says_on_stderr_why_it_could_not_fold_the_expectations(
    git_bench: Path, capsys: pytest.CaptureFixture,
):
    # inv: the fold is best-effort, so the reason is printed rather than raised; silence would
    # leave the untracked expectations file to surface one attempt later as generic dirt
    assert run._prepared_rel(git_bench, _row("q1-s-c-a01", outcome="completed")) is None
    assert "q1-s-c-a01" in capsys.readouterr().err


def test_finish_success_names_a_run_id_the_ledger_does_not_hold(git_bench: Path):
    # inv: a named refusal, because StopIteration out of a generator expression says nothing
    # about which attempt the commit was owed for
    with pytest.raises(SystemExit, match="no row for q1-s-c-a01"):
        run._finish_success(git_bench, "q1-s-c-a01")


# --- the four refusals --------------------------------------------------------------------


def test_a_third_completed_baseline_under_the_current_recipe_is_refused_but_old_ones_are_grandfathered(
    git_bench: Path,
):
    old_recipe = _write_harness(git_bench, "s", "v1\n")
    for i in range(3):
        ledger.append_row(git_bench, _row(f"q1-s-c-a0{i}", outcome="completed", harness_sha256=old_recipe))
    # inv: three completed rows exist for this cell, but all under an old recipe -- grandfathered,
    # and a fresh attempt under today's recipe sees none of them
    current_recipe = _write_harness(git_bench, "s", "v2\n")
    assert run._refusal(git_bench, "baseline", "s", "c", "q1", None, False) is None
    ledger.append_row(git_bench, _row("q1-s-c-a10", outcome="completed", harness_sha256=current_recipe))
    ledger.append_row(git_bench, _row("q1-s-c-a11", outcome="completed", harness_sha256=current_recipe))
    refusal = run._refusal(git_bench, "baseline", "s", "c", "q1", None, False)
    assert refusal is not None
    assert "already has 2 completed baselines" in refusal


def test_a_second_completed_driven_row_of_the_same_round_is_refused_without_repeat(git_bench: Path):
    recipe = _write_harness(git_bench, "s", "v1\n")
    driver = {"model": "m", "effort": "high", "max_actions": 4, "max_tokens": 100}
    ledger.append_row(git_bench, _row("q1-s-c-a01", outcome="completed", harness_sha256=recipe,
                                      runner=True, **driver))
    refusal = run._refusal(git_bench, "driven", "s", "c", "q1", driver, False)
    assert refusal is not None
    assert "already has a completed driven attempt" in refusal
    assert run._refusal(git_bench, "driven", "s", "c", "q1", driver, True) is None


def test_a_second_completed_driven_row_of_a_different_round_is_not_refused(git_bench: Path):
    recipe = _write_harness(git_bench, "s", "v1\n")
    driver_a = {"model": "m", "effort": "high", "max_actions": 4, "max_tokens": 100}
    driver_b = {"model": "m", "effort": "low", "max_actions": 4, "max_tokens": 100}
    ledger.append_row(git_bench, _row("q1-s-c-a01", outcome="completed", harness_sha256=recipe,
                                      runner=True, **driver_a))
    assert run._refusal(git_bench, "driven", "s", "c", "q1", driver_b, False) is None


def test_a_baseline_cell_tried_past_the_backoff_budget_is_refused(git_bench: Path):
    recipe = _write_harness(git_bench, "s", "v1\n")
    for i in range(run.BASELINE_ATTEMPTS + run.BASELINE_RETRIES):
        ledger.append_row(git_bench, _row(f"q1-s-c-a0{i}", outcome="aborted", harness_sha256=recipe))
    refusal = run._refusal(git_bench, "baseline", "s", "c", "q1", None, False)
    assert refusal is not None
    assert "failing, not pending" in refusal


def test_a_driven_round_tried_past_the_backoff_budget_is_refused_even_with_repeat(git_bench: Path):
    recipe = _write_harness(git_bench, "s", "v1\n")
    driver = {"model": "m", "effort": "high", "max_actions": 4, "max_tokens": 100}
    for i in range(run.BASELINE_ATTEMPTS + run.BASELINE_RETRIES):
        ledger.append_row(git_bench, _row(f"q1-s-c-a0{i}", outcome="aborted", harness_sha256=recipe,
                                          runner=True, **driver))
    refusal = run._refusal(git_bench, "driven", "s", "c", "q1", driver, True)
    assert refusal is not None
    assert "failing, not pending" in refusal


def test_no_such_system_is_refused_rather_than_raising(git_bench: Path):
    assert run._refusal(git_bench, "baseline", "no-such-system", "c", "q1", None, False) is not None


# --- missing: cell derivation ---------------------------------------------------------------


def _harness_doc(*, status: str | None = None, cfg_status: str | None = None, index: str = "indexes/a") -> dict:
    cfg: dict = {"index": index}
    if cfg_status is not None:
        cfg["status"] = cfg_status
    doc = {"adapter": "graphify", "version": {"cli": "x"}, "invocation": {}, "fixed_steps": [],
          "default_configuration": "c", "configurations": {"c": cfg},
          "sandbox_layout": {"layout": "<artifacts>"}, "environment": {}, "docs": {}}
    if status is not None:
        doc["status"] = status
    return doc


def _write_system(bench: Path, name: str, doc: dict) -> None:
    d = bench / "systems" / name
    d.mkdir(parents=True)
    (d / "harness.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


def test_cells_for_skips_reference_systems_declared_configurations_and_missing_indexes(git_bench: Path):
    write_question(git_bench, "q1", {"id": "q1", "snapshot": "snap"})
    _write_system(git_bench, "runnable", _harness_doc(index="indexes/runnable"))
    (snapshot_dir(git_bench, "snap") / "indexes" / "runnable").mkdir(parents=True)
    _write_system(git_bench, "no-index", _harness_doc(index="indexes/absent"))
    _write_system(git_bench, "a-reference", _harness_doc(status="reference", index="indexes/refd"))
    (snapshot_dir(git_bench, "snap") / "indexes" / "refd").mkdir(parents=True)
    _write_system(git_bench, "declared-cfg", _harness_doc(cfg_status="declared", index="indexes/declared"))
    (snapshot_dir(git_bench, "snap") / "indexes" / "declared").mkdir(parents=True)
    cells = run._cells_for(git_bench, "q1")
    assert cells == [("runnable", "c")]


def test_todo_admits_only_cells_with_no_refusal(git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    write_question(git_bench, "q1", {"id": "q1", "snapshot": "snap"}, {"id": "q1", "places": []})
    write_review(git_bench, "q1")
    _write_system(git_bench, "s", _harness_doc(index="indexes/a"))
    (snapshot_dir(git_bench, "snap") / "indexes" / "a").mkdir(parents=True)
    assert run._todo(git_bench, "baseline", None) == [("s", "c", "q1")]
    monkeypatch.setattr(run, "_refusal", lambda *a, **kw: "refused for the test")
    assert run._todo(git_bench, "baseline", None) == []


def test_todo_skips_a_question_with_no_review(git_bench: Path):
    write_question(git_bench, "q1", {"id": "q1", "snapshot": "snap"})
    _write_system(git_bench, "s", _harness_doc(index="indexes/a"))
    (snapshot_dir(git_bench, "snap") / "indexes" / "a").mkdir(parents=True)
    assert run._todo(git_bench, "baseline", None) == []


def test_todo_skips_a_question_its_review_withdrew(git_bench: Path):
    write_question(git_bench, "q1", {"id": "q1", "snapshot": "snap"}, {"id": "q1", "places": []})
    write_review(git_bench, "q1")
    _write_system(git_bench, "s", _harness_doc(index="indexes/a"))
    (snapshot_dir(git_bench, "snap") / "indexes" / "a").mkdir(parents=True)
    path = config.review_path(git_bench, "q1")
    review = yaml.safe_load(path.read_text(encoding="utf-8"))
    review["question_is_ambiguous"] = True
    path.write_text(yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    assert run._todo(git_bench, "baseline", None) == []


# --- missing: runs within budget, per cell ---------------------------------------------------


def test_run_missing_stops_immediately_when_the_budget_is_already_spent(
    git_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(run, "_todo", lambda *a, **kw: [("s", "c", "q1")])
    calls: list[object] = []
    monkeypatch.setattr(run, "_run_one", lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(rules, "_ROOT", git_bench)
    result = run._run_missing("baseline", model=None, effort=None, max_actions=None, max_tokens=None,
                              seconds=-1.0)
    assert result["ran"] == 0
    assert calls == []


def test_run_missing_runs_every_pending_cell_within_a_generous_budget(
    git_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    pending = [("s", "c", "q1"), ("s", "c", "q2")]

    def _fake_todo(*_a: object, **_kw: object) -> list:
        return list(pending)

    def _fake_run_one(kind: str, system: str, configuration: str, qid: str, **_kw: object) -> dict:
        pending.remove((system, configuration, qid))
        return {"run_id": f"{qid}-{system}-{configuration}-a01"}

    monkeypatch.setattr(run, "_todo", _fake_todo)
    monkeypatch.setattr(run, "_run_one", _fake_run_one)
    monkeypatch.setattr(rules, "_ROOT", git_bench)
    result = run._run_missing("baseline", model=None, effort=None, max_actions=None, max_tokens=None,
                              seconds=30.0)
    assert result["ran"] == 2
    assert result["completed"] == 2
    assert result["remaining_attempts"] == 0
    assert result["failure_count"] == 0


def test_run_missing_keeps_going_past_a_failure_that_reached_bench(
    git_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    # inv: a failure that still added a ledger row (attempt marked it aborted, `run` committed
    # it) is a cell that genuinely failed -- not a reason to stop the whole budget
    pending = [("s", "c", "q1"), ("s", "c", "q2")]

    def _fake_todo(*_a: object, **_kw: object) -> list:
        return list(pending)

    def _flaky_run_one(kind: str, system: str, configuration: str, qid: str, **_kw: object) -> dict:
        pending.remove((system, configuration, qid))
        if qid == "q1":
            ledger.append_row(git_bench, _row(f"{qid}-{system}-{configuration}-a01", outcome="aborted"))
            ledger.commit_rows(git_bench, f"chore(benchmark): attempt {qid}-{system}-{configuration}-a01 aborted")
            raise SystemExit("simulated failure")
        return {"run_id": "ok"}

    monkeypatch.setattr(run, "_todo", _fake_todo)
    monkeypatch.setattr(run, "_run_one", _flaky_run_one)
    monkeypatch.setattr(rules, "_ROOT", git_bench)
    result = run._run_missing("baseline", model=None, effort=None, max_actions=None, max_tokens=None,
                              seconds=30.0)
    assert result["ran"] == 2
    assert result["completed"] == 1
    assert result["failure_count"] == 1
    assert result["failures"][0]["question"] == "q1"
    assert result["halted"] is False


def test_run_missing_halts_when_a_failure_adds_no_ledger_row(git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    # inv: a failure that reaches no ledger row never reached bench at all -- a refusal, a sudo
    # or seal problem, or a dead server; _todo can never advance past it, so retrying it until
    # the budget runs out would only hide a machine-level problem behind a wall of retries
    pending = [("s", "c", "q1"), ("s", "c", "q2")]

    def _fake_todo(*_a: object, **_kw: object) -> list:
        return list(pending)

    def _dead_machine_run_one(kind: str, system: str, configuration: str, qid: str, **_kw: object) -> dict:
        pending.remove((system, configuration, qid))
        raise SystemExit("simulated machine-level failure")

    monkeypatch.setattr(run, "_todo", _fake_todo)
    monkeypatch.setattr(run, "_run_one", _dead_machine_run_one)
    monkeypatch.setattr(rules, "_ROOT", git_bench)
    result = run._run_missing("baseline", model=None, effort=None, max_actions=None, max_tokens=None,
                              seconds=30.0)
    assert result["ran"] == 1
    assert result["completed"] == 0
    assert result["failure_count"] == 1
    assert result["halted"] is True


# --- the sudo boundary: monkeypatched, per the brief -----------------------------------------


def test_run_one_refuses_before_reaching_invoke_attempt(
    sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    called = []
    monkeypatch.setattr(run, "_invoke_attempt", lambda *a, **kw: called.append(1))
    monkeypatch.setattr(rules, "_ROOT", sealed_bench)
    recipe = _write_harness(sealed_bench, "s", "v1\n")
    _reseal(sealed_bench)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "commit", "-qm", "fixtures"], check=True)
    for i in range(run.BASELINE_ATTEMPTS):
        ledger.append_row(sealed_bench, _row(f"q1-s-c-a0{i}", outcome="completed", harness_sha256=recipe))
    with pytest.raises(SystemExit, match="already has 2 completed baselines"):
        run._run_one("baseline", "s", "c", "q1")
    assert called == []


def test_run_one_dispatches_through_the_invoke_attempt_seam_and_makes_the_one_commit(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    from benchmark.harness import prepare
    from tests.benchmark.test_prepare import _seed_runnable

    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _stub_invoke(machine: dict, argv: list[str]) -> tuple[int, str, str]:
        from benchmark.harness import attempt as attempt_module

        code = attempt_module.attempt(sealed_bench, argv[1], argv[2], argv[3], tmp_root=tmp_root,
                                      base_url="http://localhost:1234/v1")
        return 0, json.dumps(code, default=str) + "\n", ""

    monkeypatch.setattr(run, "_invoke_attempt", _stub_invoke)
    try:
        result = run._run_one("baseline", "graphify", "default", qid)
    finally:
        prepare.release_lock(tmp_root)
    assert result["outcome"] in ("completed", "void")
    assert not ledger.changed(sealed_bench, ledger.rel(sealed_bench))
    from benchmark.harness import audit

    assert audit.check_attempts(sealed_bench) == []
    # inv: _seed_runnable deletes prepared_outputs.yaml, so needs_record wrote a fresh one this
    # attempt -- the fold must land in the same commit as the ledger row; a mutant dropping
    # extra_paths would leave it uncommitted and this assertion would catch it
    log = subprocess.run(["git", "-C", str(sealed_bench.parent), "show", "-1", "--name-only", "--format="],
                         capture_output=True, text=True, check=True).stdout.split()
    assert ledger.rel(sealed_bench) in log
    assert any(name.endswith("prepared_outputs.yaml") for name in log)
    assert len(log) == 2


def test_run_one_repairs_an_aborted_row_and_leaves_the_ledger_clean_after_a_failure(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    from benchmark.harness import audit, prepare
    from tests.benchmark.test_prepare import _seed_runnable

    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _stub_invoke(machine: dict, argv: list[str]) -> tuple[int, str, str]:
        from benchmark.harness import attempt as attempt_module

        try:
            attempt_module.attempt(sealed_bench, argv[1], argv[2], argv[3], tmp_root=tmp_root,
                                   base_url="http://localhost:1234/v1")
        except SystemExit as exc:
            return 1, "", str(exc)
        raise AssertionError("expected the pipeline to fail")

    def _violating(*_args: object, **_kwargs: object) -> dict:
        return {"violations": ["simulated violation"]}

    monkeypatch.setattr(audit, "check_run", _violating)
    monkeypatch.setattr(run, "_invoke_attempt", _stub_invoke)
    with pytest.raises(SystemExit, match="failed: audit run: simulated violation"):
        run._run_one("baseline", "graphify", "default", qid)
    prepare.release_lock(tmp_root)
    # inv: bench marked the row aborted; run's own repair, on the nonzero exit, is what commits
    # it -- the ledger ends clean, and the sealed-row rule sees exactly one commit
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    assert not ledger.changed(sealed_bench, ledger.rel(sealed_bench))
    assert audit.check_attempts(sealed_bench) == []
    # inv: require_clean reproduces the reviewer's own finding when the fold is missing --
    # prepare wrote a fresh prepared_outputs.yaml before the failure, and the aborted commit
    # must carry it too, or this call raises on the untracked expectations file left behind
    ledger.require_clean(sealed_bench)
    log = subprocess.run(["git", "-C", str(sealed_bench.parent), "show", "-1", "--name-only", "--format="],
                         capture_output=True, text=True, check=True).stdout.split()
    assert any(name.endswith("prepared_outputs.yaml") for name in log)


def test_run_one_machine_check_fires_before_any_argument_work(monkeypatch: pytest.MonkeyPatch):
    # inv: a monkeypatched refusal fires before system/configuration/qid are ever touched, so a
    # nonsense system never reaches _refusal's own, gentler "no such system" message first
    def _refuse(benchmark: Path) -> dict:
        raise SystemExit("refused for the test")

    monkeypatch.setattr(run, "_machine_facts", _refuse)
    with pytest.raises(SystemExit, match="refused for the test"):
        run._run_one("baseline", "no-such-system", "no-such-config", "not-a-real-qid")
