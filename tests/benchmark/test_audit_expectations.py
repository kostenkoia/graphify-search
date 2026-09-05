import json

import yaml

from benchmark.harness import ledger
from benchmark.harness.audit import expectations
from tests.benchmark.conftest import snapshot_dir


def _seed(bench, *, prepared_out: str = "x", excluded: list[str] | None = None,
          entry_files: list[dict] | None = None, prepared_files: dict | None = None,
          prepared_steps: dict | None = None) -> None:
    idx = snapshot_dir(bench, "snap") / "indexes" / "i"
    idx.mkdir(parents=True, exist_ok=True)
    (bench / "systems" / "s").mkdir(parents=True, exist_ok=True)
    (bench / "systems" / "s" / "harness.yaml").write_text(yaml.safe_dump({
        "adapter": "a", "version": {"cli": "1"}, "invocation": {"package": {}}, "fixed_steps": [],
        "default_configuration": "c", "configurations": {"c": {"index": "indexes/i"}},
        "sandbox_layout": {}, "environment": {}, "docs": {}}), encoding="utf-8")
    (idx / "build.yaml").write_text(yaml.safe_dump({"excluded": excluded or [], "mutable": []}), encoding="utf-8")
    recipe = "r" * 64
    steps = prepared_steps if prepared_steps is not None else {
        "query": {"out": prepared_out, "files": prepared_files or {}}}
    (idx / "prepared_outputs.yaml").write_text(yaml.safe_dump({"c": {recipe: {"q001": steps}}}), encoding="utf-8")
    row = {"run_id": "q001-s-c-a01", "question": "q001", "system": "s", "configuration": "c",
           "attempt": 1, "outcome": "completed", "harness_sha256": recipe}
    ledger.path(bench).write_text(json.dumps(row) + "\n", encoding="utf-8")
    run = bench / "record" / "runs" / "q001-s-c-a01"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run.yaml").write_text(yaml.safe_dump({
        "run_id": "q001-s-c-a01", "system": "s", "configuration": "c", "snapshot": "snap",
        "question": "q001"}), encoding="utf-8")
    entries = [{"n": 1, "kind": "call", "by": "harness", "name": "query", "canonical_sha256": "x",
                "files": entry_files or []}]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


def test_expectations_compares_a_journal_with_the_disk(git_bench):
    _seed(git_bench, prepared_out="x")
    assert expectations.check(git_bench) == []

    _seed(git_bench, prepared_out="bent")
    assert expectations.check(git_bench) == [
        "q001-s-c-a01/query: expectation on disk differs from this run's journal"]


def test_expectations_applies_the_current_build_yaml_exclusion_to_the_recorded_expectation(git_bench):
    # a cache file the run's own journal carries, and the expectation on disk also records --
    # so the two would agree, except build.yaml's current excluded list drops it from the
    # observed side but not from the recorded expectation
    entry_files = [{"path": "sandbox/graphify-out/cache/last_query_stamp", "sha256": "c" * 64}]
    prepared_files = {"sandbox/graphify-out/cache/last_query_stamp": "c" * 64}

    _seed(git_bench, excluded=["cache/"], entry_files=entry_files, prepared_files=prepared_files)
    assert expectations.check(git_bench) == [
        "q001-s-c-a01/query: expectation on disk differs from this run's journal"]

    _seed(git_bench, excluded=[], entry_files=entry_files, prepared_files=prepared_files)
    assert expectations.check(git_bench) == []


def test_expectations_flags_a_step_with_no_recorded_expectation(git_bench):
    _seed(git_bench, prepared_steps={})
    problems = expectations.check(git_bench)
    assert len(problems) == 1
    assert problems[0].startswith("q001-s-c-a01/query: no expectation recorded under recipe ")
