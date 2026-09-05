import json
from pathlib import Path

import pytest

from benchmark.harness import attempt, drive, ledger, rules
from benchmark.harness import audit as audit_module
from benchmark.harness import collect as collect_module
from benchmark.harness import score as score_module
from tests.benchmark.test_prepare import _a_runnable_question, _seed_runnable

DRIVER = {"model": "qwen3-8b", "effort": "high", "max_actions": 4, "max_tokens": 16000}


def _fake_drive_run(benchmark: Path, run_id: str, tmp_root: Path, backend: object, *,
                    model: str, effort: str, max_actions: int, max_tokens: int) -> dict:
    from benchmark.harness import config as config_module
    from benchmark.harness import drive as drive_module

    run_dir = tmp_root / run_id / "run"
    run_yaml = config_module.load_yaml(run_dir / "run.yaml")
    harness = config_module.load_harness(benchmark, run_yaml["system"])
    ctx = drive_module.context_from_run(benchmark, harness, run_yaml, tmp_root)
    # inv: a real request.json, built from the same prompt.md prepare wrote and already
    # cleared prepare's own blind check -- so this replay of the blind check passes too
    prompt_text = (run_dir / "prompt.md").read_text(encoding="utf-8")
    (run_dir / "request.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": prompt_text}], "tools": []}),
        encoding="utf-8")
    outcome = {"reason": "no_further_action", "actions": 0, "ceiling_calls": 0}
    drive_module.finish(ctx, outcome)
    return {**outcome, "model_served": model}


def test_attempt_baseline_completes_and_leaves_the_row_uncommitted(bench: Path, sealed_bench: Path):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)
    try:
        row = attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                              base_url="http://localhost:1234/v1")
    finally:
        from benchmark.harness import prepare

        prepare.release_lock(tmp_root)
    assert row["outcome"] in ("completed", "void")
    # inv: attempt makes no commit on success -- run makes the attempt's one commit
    assert ledger.changed(sealed_bench, ledger.rel(sealed_bench))
    assert (sealed_bench / "record" / "runs" / row["run_id"]).exists()


def test_attempt_aborts_with_failed_audit_when_check_run_finds_a_violation(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _violating(*_args: object, **_kwargs: object) -> dict:
        return {"violations": ["simulated violation"]}

    monkeypatch.setattr(audit_module, "check_run", _violating)
    with pytest.raises(SystemExit, match="failed: audit run: simulated violation"):
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1")
    # inv: bench marks, kia commits -- attempt's own failure handling never writes inside .git/,
    # so the mark stays uncommitted for `run` to read back and close
    assert ledger.changed(sealed_bench, ledger.rel(sealed_bench))
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    assert not (tmp_root / "sandbox" / "lock").exists()


def test_attempt_aborts_with_failed_audit_blind_when_the_request_told_the_runner(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    # inv: a blindness violation and a sandbox-attribution violation are two different failures,
    # so the row's step distinguishes them rather than reading `audit` for both
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _not_blind(*_args: object, **_kwargs: object) -> list[str]:
        return ["simulated leak"]

    monkeypatch.setattr(drive, "drive_run", _fake_drive_run)
    monkeypatch.setattr(audit_module, "check_blind", _not_blind)
    with pytest.raises(SystemExit, match="failed: audit blind: simulated leak"):
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1", driver=DRIVER)
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"


def test_attempt_names_the_missing_row_rather_than_raising_stopiteration(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    # inv: the row lookup sits inside the marking try, so a ledger that no longer holds the row
    # ends in a named refusal rather than a bare StopIteration
    from benchmark.harness import abort as abort_module

    qid, tmp_root = _seed_runnable(bench, sealed_bench)
    real_collect = collect_module.collect

    def _collect_then_drop_the_row(benchmark: Path, run_dir: Path, hits: dict | None = None) -> str:
        outcome = real_collect(benchmark, run_dir, hits)
        ledger.rewrite(benchmark, [])
        return outcome

    monkeypatch.setattr(collect_module, "collect", _collect_then_drop_the_row)
    monkeypatch.setattr(abort_module, "mark_aborted", lambda *_a, **_k: None)
    try:
        with pytest.raises(SystemExit, match="no row for"):
            attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                            base_url="http://localhost:1234/v1")
    finally:
        from benchmark.harness import prepare

        prepare.release_lock(tmp_root)


def test_attempt_aborts_with_failed_score_when_records_raises(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise score_module.UnparsedError("simulated parse failure")

    monkeypatch.setattr(score_module, "records", _boom)
    with pytest.raises(SystemExit, match="failed: score: simulated parse failure"):
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1")
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    # inv: bench only marks -- the row stays uncommitted until `run` reads it back
    assert ledger.changed(sealed_bench, ledger.rel(sealed_bench))


def test_attempt_aborts_with_failed_collect_when_collect_refuses(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise SystemExit("simulated collect refusal")

    monkeypatch.setattr(collect_module, "collect", _boom)
    with pytest.raises(SystemExit, match="failed: collect: simulated collect refusal"):
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1")
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    assert ledger.changed(sealed_bench, ledger.rel(sealed_bench))


def test_attempt_driven_aborts_with_failed_model_served_on_a_mismatch(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _served_someone_else(*_args: object, **_kwargs: object) -> dict:
        return {"reason": "answer_met", "actions": 1, "ceiling_calls": 0, "model_served": "a-different-model"}

    monkeypatch.setattr(drive, "drive_run", _served_someone_else)
    with pytest.raises(SystemExit, match="failed: model_served"):
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1", driver=DRIVER)
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    assert ledger.changed(sealed_bench, ledger.rel(sealed_bench))


def test_attempt_driven_aborts_with_failed_model_served_when_none_answered(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _served_nobody(*_args: object, **_kwargs: object) -> dict:
        return {"reason": "answer_met", "actions": 1, "ceiling_calls": 0, "model_served": None}

    monkeypatch.setattr(drive, "drive_run", _served_nobody)
    with pytest.raises(SystemExit, match="failed: model_served"):
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1", driver=DRIVER)
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    assert ledger.changed(sealed_bench, ledger.rel(sealed_bench))


def test_attempt_driven_aborts_with_failed_drive_when_drive_run_raises(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    def _boom(*_args: object, **_kwargs: object) -> dict:
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(drive, "drive_run", _boom)
    with pytest.raises(SystemExit, match="failed: drive: simulated network failure"):
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1", driver=DRIVER)
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    assert ledger.changed(sealed_bench, ledger.rel(sealed_bench))


def test_attempt_driven_happy_path_runs_the_blind_check_and_completes(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch,
):
    # inv: a driven attempt that never reaches audit.check_blind's real call site is
    # unfalsifiable evidence for the blind check -- this drives the pipeline through it for
    # real, with only the network call itself (drive_run's own loop) stood in for
    qid, tmp_root = _seed_runnable(bench, sealed_bench)

    monkeypatch.setattr(drive, "drive_run", _fake_drive_run)
    row = attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                          base_url="http://localhost:1234/v1", driver=DRIVER)
    from benchmark.harness import prepare

    prepare.release_lock(tmp_root)
    assert row["outcome"] in ("completed", "void")
    assert row["model_served"] == DRIVER["model"]


def test_needs_record_is_true_with_no_expectation_and_false_once_one_is_recorded(bench: Path, sealed_bench: Path):
    from benchmark.harness import config

    qid, tmp_root = _seed_runnable(bench, sealed_bench)
    h = config.load_harness(sealed_bench, "graphify")
    assert attempt.needs_record(sealed_bench, h, "default", qid) is True
    try:
        attempt.attempt(sealed_bench, "graphify", "default", qid, tmp_root=tmp_root,
                        base_url="http://localhost:1234/v1")
    finally:
        from benchmark.harness import prepare

        prepare.release_lock(tmp_root)
    assert attempt.needs_record(sealed_bench, h, "default", qid) is False


def test_main_require_sealed_fires_before_any_argument_work(monkeypatch: pytest.MonkeyPatch):
    # inv: a monkeypatched refusal fires before system/configuration/qid are ever touched, so a
    # nonsense qid never reaches config.load_question ahead of the machine check
    def _refuse() -> dict:
        raise SystemExit("refused for the test")

    monkeypatch.setattr(rules, "machine_facts", _refuse)
    with pytest.raises(SystemExit, match="refused for the test"):
        attempt.main(["baseline", "no-such-system", "no-such-config", "not-a-real-qid"])


def test_main_prints_the_completed_rows_json_line(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)
    machine = {"home": str(tmp_root / "home"), "tmpdir": str(tmp_root / "tmp"),
              "tiktoken_cache": str(tmp_root / "tiktoken"), "tmp_root": str(tmp_root),
              "base_url": "http://localhost:1234/v1"}
    monkeypatch.setattr(rules, "machine_facts", lambda: machine)
    try:
        assert attempt.main(["baseline", "graphify", "default", qid]) == 0
    finally:
        from benchmark.harness import prepare

        prepare.release_lock(tmp_root)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    row = json.loads(line)
    assert row["run_id"].startswith(f"{qid}-graphify-default-a")
    assert row["outcome"] in ("completed", "void")


def test_main_puts_the_backend_and_its_base_url_on_a_driven_row(
    bench: Path, sealed_bench: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
):
    # inv: an aborted driven row still has to say which server it was pointed at, so the row's
    # base_url is the same machine fact LocalBackend is constructed with
    qid, tmp_root = _seed_runnable(bench, sealed_bench)
    machine = {"home": str(tmp_root / "home"), "tmpdir": str(tmp_root / "tmp"),
              "tiktoken_cache": str(tmp_root / "tiktoken"), "tmp_root": str(tmp_root),
              "base_url": "http://localhost:4321/v1"}
    monkeypatch.setattr(rules, "machine_facts", lambda: machine)
    monkeypatch.setattr(drive, "drive_run", _fake_drive_run)
    try:
        assert attempt.main(["driven", "graphify", "default", qid,
                             "--model", str(DRIVER["model"]), "--effort", str(DRIVER["effort"]),
                             "--max-actions", str(DRIVER["max_actions"]),
                             "--max-tokens", str(DRIVER["max_tokens"])]) == 0
    finally:
        from benchmark.harness import prepare

        prepare.release_lock(tmp_root)
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["backend"] == "local"
    assert row["base_url"] == machine["base_url"]


def test_attempt_cli_shape_baseline_and_driven(bench: Path):
    ap = attempt._parser()
    args = ap.parse_args(["baseline", "s", "c", "q"])
    assert (args.kind, args.system, args.configuration, args.qid) == ("baseline", "s", "c", "q")
    args = ap.parse_args(["driven", "s", "c", "q", "--model", "m", "--effort", "high",
                          "--max-actions", "3", "--max-tokens", "100", "--repeat"])
    assert args.model == "m"
    assert args.max_actions == 3
    assert args.repeat is True
    # inv: attempt takes no path arguments -- machine.yaml names the tmp root and the benchmark
    # root is the package's own location
    with pytest.raises(SystemExit):
        ap.parse_args(["baseline", "s", "c", "q", "--tmp-root", "/tmp"])


def test_a_runnable_question_exists(bench: Path):
    # why: the helper skips when no question has a built index beside it, so this reports a broken
    # helper only on a machine whose tree holds one
    qid, snapshot, index = _a_runnable_question(bench)
    assert qid
    assert snapshot
    assert index.is_dir()
