import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.harness import abort, config, execute, ledger, prepare, prompt, rules
from tests.benchmark.conftest import (
    _reseal,
    questions_dir,
    references_dir,
    snapshot_dir,
    write_question,
    write_review,
)

PY = sys.executable


def _a_runnable_question(bench: Path) -> tuple[str, str, Path]:
    """Return the first question this tree can actually run, its snapshot and its graphify index."""
    for qid in config.question_ids(bench):
        qpath = config.question_path(bench, qid)
        question = yaml.safe_load(qpath.read_text(encoding="utf-8")) or {}
        snapshot = question.get("snapshot")
        if not isinstance(snapshot, str):
            continue
        index = config.snapshot_dir(bench, snapshot) / "indexes" / "graphify"
        if (index / "graph.json").is_file() and config.reference_path(bench, qid).is_file():
            return qid, snapshot, index
    pytest.skip("no question with a built graphify index beside it")


def seed_question(bench: Path, target: Path, qid: str, snapshot: str, index: Path) -> Path:
    """Copy one shipped question, its reference and its graphify index into `target`'s record.

    Parameters
    ----------
    bench : Path
        The tree this repository ships, read for the question, reference and index.
    target : Path
        The test tree's `benchmark/` directory the copies land under.
    qid : str
        The question id.
    snapshot : str
        The snapshot the question names.
    index : Path
        The built graphify index directory to copy.

    Returns
    -------
    Path
        The index directory inside `target`.
    """
    shutil.copy(config.question_path(bench, qid),
                questions_dir(target, snapshot) / f"{qid}.yaml")
    # inv: a real benchmark always holds the reference beside the question -- the row records its
    # hash, so a fixture without it exercises a shape prepare can never meet
    shutil.copy(config.reference_path(bench, qid),
                references_dir(target, snapshot) / f"{qid}.yaml")
    idx = snapshot_dir(target, snapshot) / index.relative_to(config.snapshot_dir(bench, snapshot))
    shutil.copytree(index, idx)
    # why: the copy carries expectations recorded outside this test, and any assertion below would
    # pass off those instead of what this run writes
    (idx / prepare.PREPARED).unlink(missing_ok=True)
    return idx



def test_ledger_append_then_commit_rows_numbers_and_commits(git_bench: Path):
    assert ledger.next_attempt(git_bench, "q001", "graphify", "default") == 1
    ledger.append_row(git_bench, {"run_id": "q001-graphify-default-a01", "question": "q001",
                                  "system": "graphify", "configuration": "default", "attempt": 1})
    sha = ledger.commit_rows(git_bench, "chore(benchmark): attempt q001-graphify-default-a01 prepared")
    assert len(sha) == 40  # inv: a git object id is 40 hex characters
    assert ledger.next_attempt(git_bench, "q001", "graphify", "default") == 2
    with pytest.raises(ledger.LedgerError, match="outcome"):
        ledger.require_clean(git_bench)
    ledger.complete_row(git_bench, "q001-graphify-default-a01", {"outcome": "completed"})
    ledger.commit_rows(git_bench, "chore(benchmark): attempt q001-graphify-default-a01 completed")
    ledger.require_clean(git_bench)
    log = subprocess.run(
        ["git", "-C", str(git_bench.parent), "log", "--format=%s"], capture_output=True, text=True,
    ).stdout
    assert "chore(benchmark): attempt q001-graphify-default-a01 prepared" in log


def test_lock_refuses_live_pid(tmp_path: Path):
    prepare.take_lock(tmp_path, "r1")
    with pytest.raises(SystemExit, match="live run"):
        prepare.take_lock(tmp_path, "r2")
    prepare.release_lock(tmp_path)
    prepare.take_lock(tmp_path, "r3")
    prepare.release_lock(tmp_path)


def test_require_clean_refuses_dirty_tracked_files(git_bench: Path):
    (git_bench / "record" / "attempts.jsonl").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match="uncommitted"):
        ledger.require_clean(git_bench)


def test_verify_master_accepts_listed_mutable_excluded(tmp_path: Path):
    idx = tmp_path / "idx"
    (idx / "cache").mkdir(parents=True)
    (idx / "graph.db").write_bytes(b"db")
    (idx / "graph.db-wal").write_bytes(b"")
    (idx / "cache" / "stamp").write_text("1")
    (idx / "build.yaml").write_text("x")
    build = {"artifacts": {"graph.db": hashlib.sha256(b"db").hexdigest()},
             "mutable": ["graph.db-wal"], "excluded": ["cache/"]}
    prepare.verify_master(idx, build)
    (idx / "stray").write_text("x")
    with pytest.raises(SystemExit, match="stray"):
        prepare.verify_master(idx, build)


def test_verify_master_refuses_missing_artifact(tmp_path: Path):
    idx = tmp_path / "idx2"
    idx.mkdir()
    (idx / "build.yaml").write_text("x")
    build = {"artifacts": {"graph.db": "deadbeef"}}
    with pytest.raises(SystemExit, match="missing"):
        prepare.verify_master(idx, build)


def test_verify_master_refuses_differing_artifact(tmp_path: Path):
    idx = tmp_path / "idx3"
    idx.mkdir()
    (idx / "graph.db").write_bytes(b"tampered")
    (idx / "build.yaml").write_text("x")
    build = {"artifacts": {"graph.db": hashlib.sha256(b"original").hexdigest()}}
    with pytest.raises(SystemExit, match="differs"):
        prepare.verify_master(idx, build)


def test_make_sandbox_refuses_pre_existing_registry_json(tmp_path: Path):
    # inv: this vendor's data-directory registry can redirect where it reads and writes an
    # index; a run must never inherit one left behind by anything else
    h = config.Harness(system="s", adapter="a", invocation={}, fixed_steps=[], configurations={},
                       default_configuration="default", environment={}, sandbox_layout={"x": "<artifacts>"}, docs={})
    home = tmp_path / "home"
    (home / ".code-review-graph").mkdir(parents=True)
    (home / ".code-review-graph" / "registry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="registry.json"):
        prepare.make_sandbox(h, {}, tmp_path / "idx", tmp_path / "sandbox", home, tmp_path)


def test_check_expansion_refuses_more_than_the_token_cap(tmp_path: Path):
    h = config.Harness(system="s", adapter="a", invocation={}, fixed_steps=[], configurations={},
                       default_configuration="default", environment={}, sandbox_layout={}, docs={})
    tokens = [str(i) for i in range(prepare.MAX_EXPANSION_TOKENS + 1)]
    question = {"expansion": {"s": {"tokens": tokens}}}
    with pytest.raises(SystemExit, match=str(prepare.MAX_EXPANSION_TOKENS)):
        prepare._check_expansion(h, question, tmp_path)


def _graphify_harness() -> config.Harness:
    return config.Harness(system="graphify", adapter="a", invocation={}, fixed_steps=[],
                          configurations={}, default_configuration="default", environment={},
                          sandbox_layout={}, docs={})


def _sandbox_with_vocabulary(tmp_path: Path, words: list[str]) -> Path:
    out = tmp_path / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)
    (out / ".vocab.txt").write_text("\n".join(sorted(words)), encoding="utf-8")
    return tmp_path


def test_check_expansion_refuses_mechanical_tokens_the_rule_does_not_produce(tmp_path: Path):
    sandbox = _sandbox_with_vocabulary(tmp_path, ["score", "scorer"])
    question = {"id": "q900", "text": "score", "rule": "mechanical",
                "expansion": {"graphify": {"tokens": ["scorer"]}}}
    with pytest.raises(SystemExit, match="the rule produces"):
        prepare._check_expansion(_graphify_harness(), question, sandbox)


def test_check_expansion_accepts_mechanical_tokens_the_rule_produces(tmp_path: Path):
    sandbox = _sandbox_with_vocabulary(tmp_path, ["score", "scorer"])
    question = {"id": "q900", "text": "score", "rule": "mechanical",
                "expansion": {"graphify": {"tokens": ["score", "scorer"]}}}
    prepare._check_expansion(_graphify_harness(), question, sandbox)


def test_check_expansion_refuses_a_mechanical_question_with_no_vocabulary_to_recheck(tmp_path: Path):
    question = {"id": "q900", "text": "score", "rule": "mechanical",
                "expansion": {"graphify": {"tokens": ["score"]}}}
    with pytest.raises(SystemExit, match="vocabulary"):
        prepare._check_expansion(_graphify_harness(), question, tmp_path)


def test_check_expansion_leaves_a_question_without_the_rule_alone(tmp_path: Path):
    sandbox = _sandbox_with_vocabulary(tmp_path, ["score", "scorer"])
    question = {"id": "q001", "text": "score", "expansion": {"graphify": {"tokens": ["score"]}}}
    prepare._check_expansion(_graphify_harness(), question, sandbox)


def test_ctx_artifacts_drops_vendor_writes_only():
    build = {"vendor_writes": ["graph.db"]}
    artifacts = {".code-review-graph/graph.db": "deadbeef", ".code-review-graph/ignorefile.used": "cafe"}
    assert prepare.ctx_artifacts(build, ".code-review-graph", artifacts) == {
        ".code-review-graph/ignorefile.used": "cafe",
    }
    assert prepare.ctx_artifacts({}, ".code-review-graph", artifacts) == artifacts


def test_substitute():
    q = {"text": "how", "expansion": {"graphify": {"tokens": ["a", "b"]}}}
    assert prepare.substitute(["graphify", "query", "<expansion>"], q, "graphify") == ["graphify", "query", "a b"]
    assert prepare.substitute({"task": "debug: <question>"}, q, "graphify") == {"task": "debug: how"}


def test_prepared_expectations_round_trip_through_their_own_file(tmp_path: Path):
    # inv: expectations live in their own file, so writing them cannot touch build.yaml, which
    # is a freeze record
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "build.yaml").write_text("system: s  # a hand-written note\n", encoding="utf-8")
    assert prepare.load_prepared(idx) == {}
    payload = {"vector": {"r1": {"q001": {"version": {"out": "aaa", "files": {}}}}}}
    prepare.write_prepared(idx, payload)
    assert prepare.load_prepared(idx) == payload
    assert (idx / "build.yaml").read_text(encoding="utf-8") == "system: s  # a hand-written note\n"


def test_keep_for_prepared_outputs():
    excluded, mutable = ["cache/"], ["graph.db-wal"]
    assert prepare.keep_for_prepared("sandbox/graphify-out/.vocab.txt", excluded, mutable)
    assert not prepare.keep_for_prepared("sandbox/graphify-out/cache/last_query_stamp", excluded, mutable)
    assert not prepare.keep_for_prepared("sandbox/.code-review-graph/graph.db-wal", excluded, mutable)
    assert prepare.keep_for_prepared("home/.x", excluded, mutable)


class _FakeServer:
    """A stand-in MCP server whose `call` writes into the sandbox, as the real vendor does."""

    def __init__(self, ctx: execute.Context) -> None:
        self._ctx = ctx

    def call(self, tool: str, args: dict) -> str:
        (self._ctx.sandbox / "written_during_call.txt").write_text("x")
        return "reply"


def test_run_step_takes_the_before_listing_ahead_of_the_tool_call(tmp_path: Path):
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for d in (run, sandbox, home):
        d.mkdir()
    ctx = execute.Context(run_dir=run, sandbox=sandbox, home=home, environment={"PATH": "/usr/bin:/bin"},
                          invocation={"tools": {"t": {"keys": {}}}}, volatile=[], artifacts={})
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    h = config.Harness(system="s", adapter="a", invocation={}, fixed_steps=[], configurations={},
                       default_configuration="default", environment={}, sandbox_layout={}, docs={})
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    step = {"name": "t", "tool": "t", "args": {}}
    entry = prepare._run_step(ctx, h, step, question, tmp_path,
                              _FakeServer(ctx))  # type: ignore[arg-type]  # why: the double implements the one method _run_step calls, and typing it as mcp.Server would pull fastmcp into a test that needs none
    assert [f["path"] for f in entry["files"]] == ["sandbox/written_during_call.txt"]


def _fixed_step_ctx(tmp_path: Path) -> tuple[execute.Context, config.Harness, str]:
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for d in (run, sandbox, home):
        d.mkdir()
    sysdir = tmp_path / "systems" / "s"
    sysdir.mkdir(parents=True)
    (sysdir / "harness.yaml").write_text("adapter: a\n", encoding="utf-8")
    inv = {"package": {"launcher": PY, "interpreter": "/nonexistent/python", "site": str(tmp_path)},
           "subcommands": {"-c": {"positional": 1, "flags": {}}}, "rejected_subcommands": []}
    ctx = execute.Context(run_dir=run, sandbox=sandbox, home=home, environment={"PATH": "/usr/bin:/bin"},
                          invocation=inv, volatile=[], artifacts={})
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    h = config.Harness(system="s", adapter="a", invocation=inv,
                       fixed_steps=[{"name": "probe", "argv": [PY, "-c", "print('out')"], "quote": None}],
                       configurations={}, default_configuration="default", environment={}, sandbox_layout={}, docs={})
    return ctx, h, rules.sha256_file(sysdir / "harness.yaml")


@pytest.mark.parametrize(("declared", "expected"), [(True, True), (False, False)])
def test_a_command_line_step_carries_the_ceiling_flag_it_declares(tmp_path: Path, declared, expected):
    # inv: drive.ceiling_left subtracts the fixed steps' counted calls from the vendor's ceiling,
    # so an argv step that declares one must journal one -- otherwise a runner is handed a budget
    # the vendor never offered, and every command-line cell records ceiling_calls: 0
    ctx, h, _recipe = _fixed_step_ctx(tmp_path)
    step = {"name": "probe", "argv": [PY, "-c", "print('out')"], "quote": None,
            "ceiling_call": declared}
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    entry = prepare._run_step(ctx, h, step, question, tmp_path, None)
    assert entry["ceiling_call"] is expected


def test_run_fixed_steps_records_a_fresh_prepared_output(tmp_path: Path):
    ctx, h, recipe = _fixed_step_ctx(tmp_path)
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    build: dict = {}
    idx = tmp_path / "idx"
    idx.mkdir()
    prepare.run_fixed_steps(ctx, h, question, tmp_path, build, idx, "cfg", True, None)
    # inv: the question is part of the key, so a second question records beside this one instead
    # of meeting its expectation and aborting
    assert prepare.load_prepared(idx)["cfg"][recipe]["q1"]["probe"]["files"] == {}


def test_run_fixed_steps_raises_when_output_differs_from_prepared_outputs(tmp_path: Path):
    # inv: this comparison is the only thing that makes a second, unrecorded attempt mean
    # anything -- a step whose output no longer matches the recorded one must halt the run
    ctx, h, recipe = _fixed_step_ctx(tmp_path)
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    build: dict = {}
    idx = tmp_path / "idx"
    idx.mkdir()
    prepare.write_prepared(idx, {"cfg": {recipe: {"q1": {"probe": {"out": "x", "files": {}}}}}})
    with pytest.raises(SystemExit, match="differs from the expectation recorded"):
        prepare.run_fixed_steps(ctx, h, question, tmp_path, build, idx, "cfg", False, None)


def test_run_fixed_steps_passes_when_output_matches_prepared_outputs(tmp_path: Path):
    ctx, h, recipe = _fixed_step_ctx(tmp_path)
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    canonical = execute.rules.canonical_hash("out\n", [])
    build: dict = {}
    idx = tmp_path / "idx"
    idx.mkdir()
    prepare.write_prepared(idx, {"cfg": {recipe: {"q1": {"probe": {"out": canonical, "files": {}}}}}})
    prepare.run_fixed_steps(ctx, h, question, tmp_path, build, idx, "cfg", False, None)


@pytest.mark.slow
def test_prepare_graphify_end_to_end(bench: Path, sealed_bench: Path):
    qid, snapshot, index = _a_runnable_question(bench)
    shutil.copytree(bench / "systems" / "graphify", sealed_bench / "systems" / "graphify")
    idx = seed_question(bench, sealed_bench, qid, snapshot, index)
    write_review(sealed_bench, qid)
    # why: the copy above adds systems, question, reference and index files the fixture's own seal
    # never listed, so this attempt must reseal or meet require_sealed's own drift check -- before
    # the commit below, so the committed tree and the tracked seal agree and require_clean passes
    _reseal(sealed_bench)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "commit", "-qm", "fixtures"], check=True)
    tmp_root = sealed_bench.parent / "tmp"
    try:
        run_dir = prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=True)
    finally:
        prepare.release_lock(tmp_root)
    entries = [json.loads(line) for line in (run_dir / "journal.jsonl").read_text().splitlines()]
    assert [e.get("name") for e in entries if e["kind"] == "call"] == ["version", "vocab_extract", "query"]
    assert entries[-1]["kind"] == "stop"
    rows = ledger.rows(sealed_bench)
    assert rows[-1]["run_id"] == f"{qid}-graphify-default-a01"
    # inv: the reference defines the verdict, so an attempt that does not record which reference
    # it ran against cannot later show that the one on disk is still the one that scored it
    assert rows[-1]["reference_sha256"] == \
        rules.sha256_file(references_dir(sealed_bench, snapshot) / f"{qid}.yaml")
    # inv: the recipe is an input, so the row must carry its hash; without it two attempts under
    # different fixed_steps merge into one aggregate and read as a nondeterminism neither has
    recipe = rules.sha256_file(sealed_bench / "systems" / "graphify" / "harness.yaml")
    assert rows[-1]["harness_sha256"] == recipe
    # inv: the row names the seal it ran under, so a run made against a drifted tree is
    # distinguishable from one made against the seal this attempt actually checked
    assert rows[-1]["instrument_sha256"] == rules.sha256_file(sealed_bench / "INSTRUMENT.yaml")
    # inv: build.yaml is a freeze record and the run must not have touched it
    assert rules.sha256_file(idx / "build.yaml") == rules.sha256_file(index / "build.yaml")
    prepared = prepare.load_prepared(idx)["default"][recipe][qid]
    assert set(prepared) == {"version", "vocab_extract", "query"}
    # inv: the row records the expectations as they stood when the attempt began, so the run that
    # first records them carries None -- the file it went on to write is the next attempt's anchor
    assert rows[-1]["prepared_sha256"] is None
    assert (idx / prepare.PREPARED).is_file()
    # inv: last_query_stamp sits under the excluded cache/ prefix, so it must never be recorded
    assert set(prepared["vocab_extract"]["files"]) == {"sandbox/graphify-out/.vocab.txt"}
    assert prepared["query"]["files"] == {}


def test_run_step_journals_a_tool_call_that_raises_before_reraising(tmp_path: Path):
    # inv: server.call raises before execute.execute runs, so without an entry here a timed-out
    # or errored tool call leaves the run with no journal line naming which step failed
    ctx, h, _recipe = _fixed_step_ctx(tmp_path)

    class _Boom:
        def call(self, tool: str, args: dict) -> str:
            raise RuntimeError("Timed out while waiting for response")

    step = {"name": "search", "tool": "t", "args": {"q": "x"}, "quote": None, "ceiling_call": True}
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    with pytest.raises(RuntimeError, match="Timed out"):
        prepare._run_step(ctx, h, step, question, tmp_path,
                          _Boom())  # type: ignore[arg-type]  # why: the same double, raising -- the point of the test is a server whose call fails
    entry = json.loads((ctx.run_dir / "journal.jsonl").read_text().splitlines()[-1])
    assert entry["name"] == "search"
    assert entry["exit"] is None
    assert "Timed out" in entry["error"]
    assert entry["ceiling_call"] is True
    # inv: the failed attempt holds its own n, so the next call cannot reuse the stem
    assert execute.next_n(ctx) == 2


def test_run_fixed_steps_refuses_to_record_a_baseline_without_being_asked(tmp_path: Path):
    # inv: without this gate a run with no recorded expectation records itself as its own
    # baseline, which is the one thing prepared_outputs exists to prevent
    ctx, h, _recipe = _fixed_step_ctx(tmp_path)
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    build: dict = {}
    with pytest.raises(SystemExit, match="rerun with --record-prepared"):
        prepare.run_fixed_steps(ctx, h, question, tmp_path, build, tmp_path, "cfg", False, None)


def test_ledger_refuses_to_commit_on_a_detached_head(git_bench: Path):
    # inv: a detached-HEAD commit is reachable from nothing, so the next checkout would discard
    # the row while `git status` still reads clean
    subprocess.run(["git", "-C", str(git_bench.parent), "checkout", "-q", "--detach"], check=True)
    ledger.append_row(git_bench, {"run_id": "q1-s-c-a01", "question": "q1", "system": "s",
                                  "configuration": "c", "attempt": 1})
    with pytest.raises(ledger.LedgerError, match="detached"):
        ledger.commit_rows(git_bench, "chore(benchmark): attempt q1-s-c-a01 prepared")
    # inv: the refusal is about the commit; the row is written before it and is deliberately
    # left on disk, where the next require_clean reports it loudly
    committed = subprocess.run(["git", "-C", str(git_bench.parent), "show", "HEAD:benchmark/record/attempts.jsonl"],
                               capture_output=True, text=True, check=True).stdout
    assert "q1-s-c-a01" not in committed


def test_release_lock_leaves_a_lock_another_run_holds(tmp_path: Path):
    # inv: a late collect must not unlock the sandbox a live run is using; the lock names its
    # holder, so a release that does not name the same run is a no-op
    prepare.take_lock(tmp_path, "live-run")
    prepare.release_lock(tmp_path, "some-older-run")
    assert (tmp_path / "sandbox" / "lock").exists()
    prepare.release_lock(tmp_path, "live-run")
    assert not (tmp_path / "sandbox" / "lock").exists()


def _killed_run(git_bench: Path, tmp_path: Path, run_id: str = "q1-s-c-a01") -> None:
    ledger.append_row(git_bench, {"run_id": run_id, "question": "q1", "system": "s",
                                  "configuration": "c", "attempt": 1})
    prepare.take_lock(tmp_path, run_id)
    # inv: a killed process leaves the lock naming a pid that is gone, which is what makes the
    # wreckage distinguishable from a run still going
    lock = tmp_path / "sandbox" / "lock"
    held = json.loads(lock.read_text())
    held["pid"] = 999999
    lock.write_text(json.dumps(held))


def test_abort_marks_the_row_and_releases_the_lock(git_bench: Path, tmp_path: Path):
    _killed_run(git_bench, tmp_path)
    done = abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert ledger.rows(git_bench)[-1]["outcome"] == "aborted"
    assert not (tmp_path / "sandbox" / "lock").exists()
    assert any("marked aborted" in line for line in done)
    # inv: bench marks; kia commits -- abort never writes inside .git/, so the mark stays
    # uncommitted here, for `run` to read back and close with the attempt's one commit
    assert ledger.changed(git_bench, ledger.rel(git_bench))


def test_abort_main_takes_no_path_flags_and_reads_them_from_machine_facts(sealed_bench: Path):
    # inv: abort is a record verb -- it takes no path arguments; the benchmark root and the
    # tmp root it recovers a lock under come from the same machine facts attempt reads
    tmp_root = Path(rules.machine_facts()["tmp_root"])
    _killed_run(sealed_bench, tmp_root)
    assert abort.main(["q1-s-c-a01"]) == 0
    assert ledger.rows(sealed_bench)[-1]["outcome"] == "aborted"
    with pytest.raises(SystemExit):
        abort.main(["q1-s-c-a01", "--tmp-root", str(tmp_root)])


def test_abort_releases_the_lock_when_complete_row_fails_after_marking_the_row(
    git_bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _killed_run(git_bench, tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(abort.ledger, "complete_row", _boom)
    with pytest.raises(RuntimeError, match="simulated"):
        abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    # inv: release_lock sits in mark_aborted's own finally, so a write that fails here still
    # gives up the sandbox rather than leaving it held by a process about to exit
    assert not (tmp_path / "sandbox" / "lock").exists()


def test_abort_refuses_a_run_that_already_ended_and_left_no_lock(git_bench: Path, tmp_path: Path):
    _killed_run(git_bench, tmp_path)
    (tmp_path / "sandbox" / "lock").unlink()
    ledger.complete_row(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    with pytest.raises(SystemExit, match="already ended as completed"):
        abort.abort(git_bench, "q1-s-c-a01", tmp_path)


def test_abort_finishes_a_half_done_abort_without_touching_the_row(git_bench: Path, tmp_path: Path):
    # inv: marking the row and releasing the lock are two writes, so a kill between them leaves a
    # lock this command must still release -- and the row, already ended, must not be written again
    _killed_run(git_bench, tmp_path)
    ledger.complete_row(git_bench, "q1-s-c-a01", {"outcome": "aborted"})
    ledger.commit_rows(git_bench, "chore(benchmark): attempt q1-s-c-a01 aborted")
    updates_before = len(ledger.rows(git_bench))
    done = abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert not (tmp_path / "sandbox" / "lock").exists()
    assert any("already ended as aborted" in line for line in done)
    assert any("lock released" in line for line in done)
    assert len(ledger.rows(git_bench)) == updates_before
    assert ledger.rows(git_bench)[-1]["outcome"] == "aborted"
    # inv: the point of the command is that the next run may start
    ledger.require_clean(git_bench)


def test_abort_says_so_when_it_finds_no_lock_under_the_given_tmp_root(git_bench: Path, tmp_path: Path):
    # inv: a run prepared under another --tmp-root leaves its lock where this command never looks,
    # so silence here would read as a completed recovery
    _killed_run(git_bench, tmp_path)
    elsewhere = tmp_path / "another-root"
    elsewhere.mkdir()
    done = abort.abort(git_bench, "q1-s-c-a01", elsewhere)
    assert any("no sandbox lock at" in line and "--tmp-root" in line for line in done)
    assert (tmp_path / "sandbox" / "lock").exists()


def test_abort_refuses_a_run_that_is_still_going(git_bench: Path, tmp_path: Path):
    ledger.append_row(git_bench, {"run_id": "q1-s-c-a01", "question": "q1", "system": "s",
                                  "configuration": "c", "attempt": 1})
    prepare.take_lock(tmp_path, "q1-s-c-a01")  # this process is alive
    with pytest.raises(SystemExit, match="still running"):
        abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert "outcome" not in ledger.rows(git_bench)[-1]


def test_abort_releases_a_lock_taken_before_any_row_existed(git_bench: Path, tmp_path: Path):
    # inv: a kill between take_lock and append_row leaves no row to mark, only the lock
    prepare.take_lock(tmp_path, "q1-s-c-a01")
    lock = tmp_path / "sandbox" / "lock"
    held = json.loads(lock.read_text())
    held["pid"] = 999999
    lock.write_text(json.dumps(held))
    done = abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert not lock.exists()
    assert any("no ledger row" in line for line in done)


def test_load_question_refuses_an_id_that_walks_out_of_questions(tmp_path: Path):
    bench = tmp_path / "benchmark"
    write_question(bench, "q001", {"id": "q001", "snapshot": "snap"})
    (bench / "systems" / "graphify").mkdir(parents=True)
    (bench / "systems" / "graphify" / "harness.yaml").write_text("adapter: graphify\n", encoding="utf-8")
    # inv: a qid is an identifier, not a path; `../../../../systems/graphify/harness` names a
    # real YAML that would otherwise be read as a question and run under an id describing none
    # of it
    assert config.load_question(bench, "q001")["id"] == "q001"
    with pytest.raises(config.ConfigError, match="not a bare identifier"):
        config.load_question(bench, "../../../../systems/graphify/harness")


@pytest.mark.parametrize("qid", ["../escape", "sub/dir", "q001/../../x"])
def test_prepare_places_no_run_for_a_question_id_carrying_a_separator(qid: str, sealed_bench: Path):
    # inv: nothing is created outside the named tmp tree, whichever of the two gates refuses first
    tmp_root = sealed_bench.parent / "tmp"
    with pytest.raises((SystemExit, config.ConfigError, FileNotFoundError)):
        prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=False)
    assert list(tmp_root.rglob("run.yaml")) == []


def _prepared_not_yet_collected(git_bench: Path, tmp_path: Path, run_id: str = "q1-s-c-a01") -> None:
    ledger.append_row(git_bench, {"run_id": run_id, "question": "q1", "system": "s",
                                  "configuration": "c", "attempt": 1})
    prepare.take_lock(tmp_path, run_id)
    lock = tmp_path / "sandbox" / "lock"
    held = json.loads(lock.read_text())
    held["pid"] = 999999
    lock.write_text(json.dumps(held))
    (tmp_path / run_id / "run").mkdir(parents=True)
    (tmp_path / run_id / "run" / "run.yaml").write_text(f"run_id: {run_id}\noutcome: prepared\n")
    (tmp_path / run_id / "run" / "records.jsonl").write_text("")


def test_abort_refuses_a_run_that_only_finished_preparing(git_bench: Path, tmp_path: Path):
    # inv: run.yaml is the only thing that tells a finished prepare from a kill, and aborting one
    # would make collect update its row a second time
    _prepared_not_yet_collected(git_bench, tmp_path)
    with pytest.raises(SystemExit, match="waiting for collect"):
        abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert "outcome" not in ledger.rows(git_bench)[-1]
    assert (tmp_path / "sandbox" / "lock").exists()


def test_abort_closes_a_run_killed_inside_the_drive(git_bench: Path, tmp_path: Path):
    # inv: run.yaml without records.jsonl is a run killed after prepare and before score, which
    # collect cannot close; abort marks its row and frees the sandbox
    _prepared_not_yet_collected(git_bench, tmp_path)
    (tmp_path / "q1-s-c-a01" / "run" / "records.jsonl").unlink()
    abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert ledger.rows(git_bench)[-1]["outcome"] == "aborted"
    assert not (tmp_path / "sandbox" / "lock").exists()


def test_take_lock_sends_a_run_killed_inside_the_drive_to_abort(git_bench: Path, tmp_path: Path):
    _prepared_not_yet_collected(git_bench, tmp_path)
    (tmp_path / "q1-s-c-a01" / "run" / "records.jsonl").unlink()
    with pytest.raises(SystemExit, match="abort"):
        prepare.take_lock(tmp_path, "q1-s-c-a02")


def test_take_lock_sends_a_prepared_run_to_collect_and_a_killed_one_to_abort(git_bench: Path,
                                                                             tmp_path: Path):
    _prepared_not_yet_collected(git_bench, tmp_path)
    with pytest.raises(SystemExit, match="prepared and waiting.*collect it"):
        prepare.take_lock(tmp_path, "q1-s-c-a02")
    (tmp_path / "q1-s-c-a01" / "run" / "run.yaml").unlink()
    with pytest.raises(SystemExit, match="killed before it finished preparing"):
        prepare.take_lock(tmp_path, "q1-s-c-a02")


def test_take_lock_names_an_unreadable_lock_instead_of_raising_from_inside(tmp_path: Path):
    (tmp_path / "sandbox").mkdir(parents=True)
    (tmp_path / "sandbox" / "lock").write_text("")
    with pytest.raises(SystemExit, match="unreadable sandbox lock"):
        prepare.take_lock(tmp_path, "q1-s-c-a01")


def test_abort_clears_a_lock_that_names_nobody(git_bench: Path, tmp_path: Path):
    # inv: a lock written by a crash mid-write is wreckage even though it identifies no run
    (tmp_path / "sandbox").mkdir(parents=True)
    (tmp_path / "sandbox" / "lock").write_text("")
    done = abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert not (tmp_path / "sandbox" / "lock").exists()
    assert any("no ledger row" in line for line in done)


def test_abort_releases_the_lock_a_kill_left_after_a_completed_collect(git_bench: Path, tmp_path: Path):
    # inv: a run whose row already ended is collected, so its run.yaml is wreckage like the lock
    # beside it -- refusing here sent the operator back to collect, which refuses in turn
    _killed_run(git_bench, tmp_path)
    ledger.complete_row(git_bench, "q1-s-c-a01", {"outcome": "completed"})
    run_dir = tmp_path / "q1-s-c-a01" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text("outcome: prepared\n", encoding="utf-8")
    done = abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert not (tmp_path / "sandbox" / "lock").exists()
    assert any("already ended as completed" in line for line in done)
    assert ledger.rows(git_bench)[-1]["outcome"] == "completed"


def test_abort_clears_an_unreadable_lock_beside_a_row_that_already_ended(git_bench: Path, tmp_path: Path):
    # inv: a lock naming nobody is wreckage to clear, so the refusal keys on the lock being absent
    # and not on it being unreadable -- otherwise abort reports "no lock" with one sitting there
    _killed_run(git_bench, tmp_path)
    ledger.complete_row(git_bench, "q1-s-c-a01", {"outcome": "aborted"})
    lock = tmp_path / "sandbox" / "lock"
    lock.write_text("{ truncated by a kill", encoding="utf-8")
    done = abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert not lock.exists()
    assert any("lock released" in line for line in done)


def test_abort_recovers_a_kill_that_landed_between_the_copy_and_the_commit(git_bench: Path, tmp_path: Path):
    # inv: with the evidence already copied, collect refuses and can never finish the run, so
    # abort must act instead of sending the operator to a command that points back at it
    _killed_run(git_bench, tmp_path)
    run_dir = tmp_path / "q1-s-c-a01" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text("outcome: prepared\n", encoding="utf-8")
    dest = git_bench / "record" / "runs" / "q1-s-c-a01"
    dest.mkdir(parents=True)
    (dest / "journal.jsonl").write_text("copied before the kill\n", encoding="utf-8")
    done = abort.abort(git_bench, "q1-s-c-a01", tmp_path)
    assert not (tmp_path / "sandbox" / "lock").exists()
    assert ledger.rows(git_bench)[-1]["outcome"] == "aborted"
    # inv: a recovery command never removes evidence; it reports the copy and leaves it
    assert any("unaccounted for" in line for line in done)
    assert (dest / "journal.jsonl").exists()


def _runner_ctx(tmp_path: Path) -> execute.Context:
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for directory in (run, sandbox, home):
        directory.mkdir()
    ctx = execute.Context(run_dir=run, sandbox=sandbox, home=home, environment={},
                          invocation={"package": {}}, volatile=[], artifacts={})
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    return ctx


def _journal_lines(ctx: execute.Context) -> list[dict]:
    return [json.loads(line) for line in (ctx.run_dir / "journal.jsonl").read_text().splitlines()]


def test_a_run_no_runner_takes_on_is_closed_by_the_harness(tmp_path: Path):
    ctx = _runner_ctx(tmp_path)
    prepare.hand_over(ctx, runner=False)
    assert _journal_lines(ctx)[-1] == {"n": 1, "kind": "stop", "by": "harness", "reason": "harness"}


def test_a_run_a_runner_takes_on_is_left_open(tmp_path: Path):
    ctx = _runner_ctx(tmp_path)
    prepare.hand_over(ctx, runner=True)
    assert [e["kind"] for e in _journal_lines(ctx)] == ["header"]


def test_the_prompt_steps_are_the_executed_calls_in_order(tmp_path: Path):
    ctx = _runner_ctx(tmp_path)
    (ctx.run_dir / "01_version.cmd").write_text("kind: act\ncall: v", encoding="utf-8")
    (ctx.run_dir / "01_version.out").write_text("toy 1.0\n", encoding="utf-8")
    (ctx.run_dir / "02_query.cmd").write_text("kind: act\ncall: q", encoding="utf-8")
    (ctx.run_dir / "02_query.out").write_text("NODE thing\n", encoding="utf-8")
    execute.append(ctx, {"n": 1, "kind": "call", "name": "version", "exit": 0})
    execute.append(ctx, {"n": 2, "kind": "call", "name": "query", "exit": 0})
    # why: the refused call is given an output file on purpose, so the test turns on the
    # refusal itself rather than on a file that happens to be absent
    (ctx.run_dir / "03_gone.cmd").write_text("c", encoding="utf-8")
    (ctx.run_dir / "03_gone.out").write_text("must not be shown\n", encoding="utf-8")
    execute.append(ctx, {"n": 3, "kind": "call", "name": "gone", "action": False, "refused": "no"})
    execute.append(ctx, {"n": 4, "kind": "server", "event": "stop"})
    execute.append(ctx, {"n": 5, "kind": "call", "name": "crashed", "exit": None, "error": "boom"})
    steps = prepare.prompt_steps(ctx.run_dir, _journal_lines(ctx))
    assert [step["name"] for step in steps] == ["version", "query"]
    assert steps[1]["out"] == "NODE thing\n"
    assert steps[0]["cmd"] == "kind: act\ncall: v"


def test_the_written_prompt_carries_the_question_the_expansion_and_every_step(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "systems" / "toy").mkdir(parents=True)
    (benchmark / "systems" / "toy" / "manifest.yaml").write_text(
        "id: toy\nprescribed_workflow:\n  source: SKILL.md\n  steps:\n"
        "    - {id: 1, name: traversal, quote: 'Build the expanded query string'}\n",
        encoding="utf-8")
    ctx = _runner_ctx(tmp_path)
    (ctx.run_dir / "01_query.cmd").write_text("kind: act\ncall: q", encoding="utf-8")
    (ctx.run_dir / "01_query.out").write_text("NODE thing\n", encoding="utf-8")
    execute.append(ctx, {"n": 1, "kind": "call", "name": "query", "exit": 0})
    harness = config.Harness(system="toy", adapter="graphify", invocation={}, fixed_steps=[],
                             configurations={}, default_configuration="default", environment={},
                             sandbox_layout={}, docs={})
    # why: the tokens are deliberately words the question does not contain, or the assertion
    # below would pass through the question text and never touch the expansion
    question = {"id": "q001", "text": "how is the score calculated",
                "expansion": {"toy": {"tokens": ["invoice", "weights"]}}}
    written = prepare.write_prompt(benchmark, ctx, harness, question)
    text = written.read_text(encoding="utf-8")
    assert written.name == "prompt.md"
    assert "how is the score calculated" in text
    assert "invoice weights" in text
    assert "Build the expanded query string" in text
    assert "NODE thing" in text


def test_the_written_prompt_keeps_the_step_outputs_below_the_heading(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "systems" / "toy").mkdir(parents=True)
    (benchmark / "systems" / "toy" / "manifest.yaml").write_text("id: toy\n", encoding="utf-8")
    ctx = _runner_ctx(tmp_path)
    (ctx.run_dir / "01_query.cmd").write_text("c", encoding="utf-8")
    (ctx.run_dir / "01_query.out").write_text("SECRET NODE\n", encoding="utf-8")
    execute.append(ctx, {"n": 1, "kind": "call", "name": "query", "exit": 0})
    harness = config.Harness(system="toy", adapter="graphify", invocation={}, fixed_steps=[],
                             configurations={}, default_configuration="default", environment={},
                             sandbox_layout={}, docs={})
    question = {"id": "q001", "text": "t", "expansion": {"toy": {"tokens": ["a"]}}}
    above, below = prompt.split(prepare.write_prompt(benchmark, ctx, harness, question)
                                .read_text(encoding="utf-8"))
    assert "SECRET NODE" not in above
    assert "SECRET NODE" in below


def test_prepare_for_a_runner_leaves_a_prompt_and_an_open_journal(bench: Path, sealed_bench: Path):
    qid, snapshot, index = _a_runnable_question(bench)
    shutil.copytree(bench / "systems" / "graphify", sealed_bench / "systems" / "graphify")
    seed_question(bench, sealed_bench, qid, snapshot, index)
    write_review(sealed_bench, qid)
    # why: reseal before the commit, so the committed tree and the tracked seal agree and the
    # next require_clean inside prepare() passes
    _reseal(sealed_bench)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "commit", "-qm", "fixtures"], check=True)
    tmp_root = sealed_bench.parent / "tmp"
    try:
        run_dir = prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=True, runner=True)
    finally:
        prepare.release_lock(tmp_root)
    entries = [json.loads(line) for line in (run_dir / "journal.jsonl").read_text().splitlines()]
    # inv: the driver owns the stop of a run it is handed, so preparation must leave none
    assert [e["kind"] for e in entries if e["kind"] == "stop"] == []
    text = (run_dir / "prompt.md").read_text(encoding="utf-8")
    question = config.load_question(sealed_bench, qid)
    above, below = prompt.split(text)
    assert question["text"] in above
    assert " ".join(question["expansion"]["graphify"]["tokens"]) in above
    # inv: what the vendor printed reaches the runner only below the heading, where the blind
    # check does not reach and the record of a journaled action does
    assert "NODE" in below
    reference = config.load_yaml(references_dir(sealed_bench, snapshot) / f"{qid}.yaml")
    for place in reference["places"]:
        assert place["symbol"] not in above


def _seed_runnable(bench: Path, sealed_bench: Path) -> tuple[str, Path]:
    """Seed `sealed_bench` with a runnable question and its index; return `(qid, tmp_root)`."""
    qid, snapshot, index = _a_runnable_question(bench)
    shutil.copytree(bench / "systems" / "graphify", sealed_bench / "systems" / "graphify")
    seed_question(bench, sealed_bench, qid, snapshot, index)
    write_review(sealed_bench, qid)
    # why: reseal before the commit, so the committed tree and the tracked seal agree and the
    # next require_clean inside prepare() passes
    _reseal(sealed_bench)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "commit", "-qm", "fixtures"], check=True)
    return qid, sealed_bench.parent / "tmp"


DRIVER_SETTINGS = {"model": "qwen3-8b", "effort": "high", "max_actions": 4, "max_tokens": 16000,
                   "backend": "local", "base_url": "http://localhost:1234/v1"}


def test_prepare_writes_the_runner_flag_and_the_drivers_settings_into_the_row(bench: Path, sealed_bench: Path):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)
    try:
        run_dir = prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=True,
                                  runner=True, driver=DRIVER_SETTINGS)
    finally:
        prepare.release_lock(tmp_root)
    row = ledger.rows(sealed_bench)[-1]
    assert row["runner"] is True
    assert {k: row[k] for k in DRIVER_SETTINGS} == DRIVER_SETTINGS
    run_yaml = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert {k: run_yaml[k] for k in DRIVER_SETTINGS} == DRIVER_SETTINGS
    # inv: prepare holds no commit sha, so run.yaml cannot carry one -- the ledger's git history
    # carries it instead
    assert "ledger_commit" not in run_yaml


def test_prepare_without_a_driver_still_writes_the_runner_flag_and_no_driver_keys(
        bench: Path, sealed_bench: Path):
    qid, tmp_root = _seed_runnable(bench, sealed_bench)
    try:
        prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=True)
    finally:
        prepare.release_lock(tmp_root)
    row = ledger.rows(sealed_bench)[-1]
    assert row["runner"] is False
    assert not set(DRIVER_SETTINGS) & set(row)


def _blind_tree(tmp_path: Path, prompt_body: str) -> tuple[Path, Path]:
    benchmark = tmp_path / "bench"
    write_question(benchmark, "q001", {"id": "q001", "snapshot": "snap", "text": "t"})
    (references_dir(benchmark, "snap") / "q001.yaml").write_text(
        "places:\n  - {path: pkg/logic.py, symbol: render_invoice, start: 63, end: 149}\n",
        encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "prompt.md").write_text(
        f"{prompt_body}\n{prompt.HEADING}\n\nNODE render_invoice src=pkg/logic.py\n",
        encoding="utf-8")
    return benchmark, run_dir


def test_a_prompt_that_kept_the_answer_out_passes_the_blind_check(tmp_path: Path):
    benchmark, run_dir = _blind_tree(tmp_path, "how is the score calculated")
    prepare.check_blind(benchmark, run_dir, "q001")


def test_a_prompt_that_names_the_answer_stops_the_preparation(tmp_path: Path):
    benchmark, run_dir = _blind_tree(tmp_path, "look at render_invoice")
    with pytest.raises(SystemExit, match="render_invoice"):
        prepare.check_blind(benchmark, run_dir, "q001")


def test_what_the_vendor_printed_below_the_heading_is_not_a_leak(tmp_path: Path):
    benchmark, run_dir = _blind_tree(tmp_path, "how is the score calculated")
    text = (run_dir / "prompt.md").read_text(encoding="utf-8")
    # inv: the answer sits below the heading in this fixture, which is the whole point --
    # a journaled action may return it, the owner's own words may not
    assert "render_invoice" in text
    prepare.check_blind(benchmark, run_dir, "q001")


def test_the_written_prompt_does_not_invite_a_call_the_grammar_refuses(bench: Path, tmp_path: Path):
    benchmark = tmp_path / "bench"
    (benchmark / "systems" / "graphify").mkdir(parents=True)
    shutil.copy(bench / "systems" / "graphify" / "manifest.yaml",
                benchmark / "systems" / "graphify" / "manifest.yaml")
    ctx = _runner_ctx(tmp_path)
    (ctx.run_dir / "01_query.cmd").write_text("c", encoding="utf-8")
    (ctx.run_dir / "01_query.out").write_text("NODE thing\n", encoding="utf-8")
    execute.append(ctx, {"n": 1, "kind": "call", "name": "query", "exit": 0})
    real = config.load_harness(bench, "graphify")
    # why: the grammar here is invented, not read from the file -- what this pins is that
    # preparation hands the grammar to the prompt, not which subcommands the owner has opened
    invocation = {**real.invocation, "rejected_subcommands": ["explain", "path"]}
    harness = config.Harness(system="graphify", adapter="graphify", invocation=invocation,
                             fixed_steps=[], configurations={}, default_configuration="default",
                             environment={}, sandbox_layout={}, docs={})
    question = {"id": "q", "text": "t", "expansion": {"graphify": {"tokens": ["a"]}}}
    text = prepare.write_prompt(benchmark, ctx, harness, question).read_text(encoding="utf-8")
    assert "graphify explain" not in text
    assert prompt.WITHHELD_NOTE in text


def test_substitute_without_an_expansion_block_for_this_system():
    q = {"text": "how", "expansion": {"graphify": {"tokens": ["a"]}}}
    assert prepare.substitute(["gs", "query", "<question>"], q, "graphify-search") == ["gs", "query", "how"]
    with pytest.raises(KeyError):
        prepare.substitute(["gs", "query", "<expansion>"], q, "graphify-search")


def test_the_written_prompt_of_a_system_with_no_expansion_block_carries_no_expansion(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "systems" / "toy").mkdir(parents=True)
    (benchmark / "systems" / "toy" / "manifest.yaml").write_text("id: toy\n", encoding="utf-8")
    ctx = _runner_ctx(tmp_path)
    (ctx.run_dir / "01_query.cmd").write_text("c", encoding="utf-8")
    (ctx.run_dir / "01_query.out").write_text("NODE thing\n", encoding="utf-8")
    execute.append(ctx, {"n": 1, "kind": "call", "name": "query", "exit": 0})
    harness = config.Harness(system="toy", adapter="graphify", invocation={}, fixed_steps=[],
                             configurations={}, default_configuration="default", environment={},
                             sandbox_layout={}, docs={})
    # why: the block names another system, so this pins the lookup by system rather than the
    # absence of the key altogether
    question: dict[str, Any] = {"id": "q001", "text": "how is the score calculated",
                                "expansion": {"other": {"tokens": ["invoice"]}}}
    above, below = prompt.split(prepare.write_prompt(benchmark, ctx, harness, question)
                                .read_text(encoding="utf-8"))
    assert "The expansion the harness prepared" not in above
    assert "invoice" not in above
    assert question["text"] in above
    assert "NODE thing" in below


def _expansion_step_ctx(tmp_path: Path, system: str) -> tuple[execute.Context, config.Harness, str]:
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for d in (run, sandbox, home):
        d.mkdir()
    sysdir = tmp_path / "systems" / system
    sysdir.mkdir(parents=True)
    (sysdir / "harness.yaml").write_text("adapter: a\n", encoding="utf-8")
    inv = {"package": {"launcher": PY, "interpreter": "/nonexistent/python", "site": str(tmp_path)},
           "subcommands": {"-c": {"positional": 1, "flags": {}}}, "rejected_subcommands": []}
    ctx = execute.Context(run_dir=run, sandbox=sandbox, home=home, environment={"PATH": "/usr/bin:/bin"},
                          invocation=inv, volatile=[], artifacts={})
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    step = {"name": "query", "argv": [PY, "-c", "print('<expansion>')"], "quote": "q", "ceiling_call": True}
    h = config.Harness(system=system, adapter="a", invocation=inv, fixed_steps=[step],
                       configurations={}, default_configuration="default", environment={}, sandbox_layout={}, docs={})
    return ctx, h, rules.sha256_file(sysdir / "harness.yaml")


def _journal(ctx: execute.Context) -> list[dict]:
    return [json.loads(line) for line in (ctx.run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()]


def test_run_fixed_steps_halts_a_graphify_query_whose_expansion_is_empty(tmp_path: Path):
    # inv: graphify's step 0 stops when no vocabulary token matches, so the harness journals the
    # query step as halted instead of running the vendor with an empty string
    ctx, h, recipe = _expansion_step_ctx(tmp_path, "graphify")
    question = {"id": "q1", "text": "q", "expansion": {"graphify": {"tokens": []}}}
    idx = tmp_path / "idx"
    idx.mkdir()
    prepare.run_fixed_steps(ctx, h, question, tmp_path, {}, idx, "cfg", True, None)
    call = _journal(ctx)[-1]
    assert call["kind"] == "call"
    assert call["name"] == "query"
    assert call["by"] == "harness"
    assert call["action"] is False
    assert "empty expansion" in call["halted"]
    assert call["system_call"] is False
    assert call["ceiling_call"] is False
    assert not list(ctx.run_dir.glob("*_query.out"))
    assert prepare.load_prepared(idx)["cfg"][recipe]["q1"]["query"] == {"out": None, "files": {}}


def test_run_fixed_steps_runs_a_graphify_query_that_has_tokens(tmp_path: Path):
    ctx, h, _recipe = _expansion_step_ctx(tmp_path, "graphify")
    question = {"id": "q1", "text": "q", "expansion": {"graphify": {"tokens": ["alpha", "beta"]}}}
    idx = tmp_path / "idx"
    idx.mkdir()
    prepare.run_fixed_steps(ctx, h, question, tmp_path, {}, idx, "cfg", True, None)
    call = _journal(ctx)[-1]
    assert call.get("action") is not False
    assert (ctx.run_dir / f"{call['n']:02d}_query.out").read_text(encoding="utf-8") == "alpha beta\n"


def test_run_fixed_steps_runs_an_empty_expansion_for_a_system_without_the_halt_rule(tmp_path: Path):
    ctx, h, _recipe = _expansion_step_ctx(tmp_path, "s")
    question = {"id": "q1", "text": "q", "expansion": {"s": {"tokens": []}}}
    idx = tmp_path / "idx"
    idx.mkdir()
    prepare.run_fixed_steps(ctx, h, question, tmp_path, {}, idx, "cfg", True, None)
    call = _journal(ctx)[-1]
    assert call.get("action") is not False
    assert (ctx.run_dir / f"{call['n']:02d}_query.out").read_text(encoding="utf-8") == "\n"


def test_a_halted_step_is_recorded_once_and_matches_its_own_expectation(tmp_path: Path):
    ctx, h, _recipe = _expansion_step_ctx(tmp_path, "graphify")
    question = {"id": "q1", "text": "q", "expansion": {"graphify": {"tokens": []}}}
    idx = tmp_path / "idx"
    idx.mkdir()
    prepare.run_fixed_steps(ctx, h, question, tmp_path, {}, idx, "cfg", True, None)
    (tmp_path / "second").mkdir()
    ctx2, h2, _ = _expansion_step_ctx(tmp_path / "second", "graphify")
    prepare.run_fixed_steps(ctx2, h2, question, tmp_path / "second", {}, idx, "cfg", False, None)
    assert _journal(ctx2)[-1]["action"] is False
