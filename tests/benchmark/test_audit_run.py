import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from benchmark.harness import audit, prepare, prompt, rules
from tests.benchmark.conftest import references_dir, snapshot_dir, write_question

INV = {"package": {"launcher": "/x/graphify", "interpreter": "/x/py", "site": "/x/site"},
       "subcommands": {"--version": {"positional": 0, "flags": {}}}, "rejected_subcommands": []}


def _run_yaml(run_id: str, tmp_root: Path, **extra) -> dict:
    base = {"run_id": run_id, "system": "s", "tmp_root": str(tmp_root), "artifacts": {},
            "environment_sha256": "skip", "baseline_listing": "skip", "outcome": "prepared"}
    return {**base, **extra}


def _run(tmp_path: Path, out_text: str) -> Path:
    run = tmp_path / "q-s-d-a01" / "run"
    run.mkdir(parents=True)
    # inv: prepare.make_sandbox creates both watched roots for every real run, so a fixture
    # without them exercises a state the harness cannot produce and hides the attribution check
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a01" / "home").mkdir(parents=True, exist_ok=True)
    (run / "01_version.out").write_text(out_text)
    (run / "01_version.err").write_text("")
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "version", "by": "harness", "argv": ["/x/graphify", "--version"],
         "quote": None, "exit": 0, "out_sha256": rules.sha256_file(run / "01_version.out"),
         "canonical_sha256": rules.canonical_hash(out_text, []), "system_call": True,
         "ceiling_call": False, "files": []},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a01", tmp_path, fixed_steps=1)))
    return run


def test_audit_run_passes_clean_run(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert result["valid"] is True
    assert result["violations"] == []


def test_audit_run_flags_hash_and_path_leak(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    (run / "01_version.out").write_text("tampered\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("out_sha256" in v for v in result["violations"])
    run2 = _run(tmp_path / "b", "/home/someone/projects/demo/x\n")
    result2 = audit.check_run(run2, INV, skip_environment=True)
    assert any("absolute path" in v for v in result2["violations"])


def test_audit_run_flags_a_grammar_violation(tmp_path: Path):
    # inv: a journaled call the invocation grammar refuses must surface, not pass silently
    run = tmp_path / "q-s-d-a02" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a02" / "home").mkdir(parents=True, exist_ok=True)
    (run / "01_bad.out").write_text("noise\n")
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "bad", "by": "harness", "argv": ["/x/graphify", "frobnicate"],
         "quote": None, "exit": 0, "out_sha256": rules.sha256_file(run / "01_bad.out"),
         "canonical_sha256": rules.canonical_hash("noise\n", []), "system_call": True,
         "ceiling_call": False, "files": []},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a02", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("unknown subcommand: frobnicate" in v for v in result["violations"])


def test_audit_run_flags_unattributed_home_file_but_exempts_model_paths(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    home = tmp_path / "q-s-d-a01" / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "model.bin").write_text("frozen weights\n")
    (home / "rogue.bin").write_text("nobody attributed this\n")
    result = audit.check_run(run, INV, skip_environment=True, model_paths={"model.bin"})
    assert not any("model.bin" in v for v in result["violations"])
    assert any("rogue.bin" in v for v in result["violations"])


def test_audit_run_exempts_sandbox_file_named_in_an_entrys_files(tmp_path: Path):
    # inv: attribution is by-entry-`files`, never path-shape guessing -- this exempts an
    # otherwise-unattributed sandbox file only because some entry's `files` names it
    run = _run(tmp_path, "graphify 0.9.27\n")
    sandbox = tmp_path / "sandbox" / "index" / "layout"
    sandbox.mkdir(parents=True)
    (sandbox / "output.bin").write_text("written during the query call\n")
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    entries[1]["files"] = [{"path": "sandbox/layout/output.bin", "sha256": rules.sha256_file(sandbox / "output.bin")}]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    result = audit.check_run(run, INV, skip_environment=True)
    assert result["violations"] == []


def _bench(tmp_path: Path, mutable: list[str], n_fixed_steps: int = 1) -> Path:
    bench = tmp_path / "bench"
    sysdir = bench / "systems" / "s"
    sysdir.mkdir(parents=True)
    fixed_steps = [{"name": f"step{i}", "argv": ["s"], "quote": None} for i in range(n_fixed_steps)]
    harness = {"adapter": "graphify", "version": {"cli": "x"}, "invocation": {},
               "fixed_steps": fixed_steps, "default_configuration": "d", "configurations": {"d": {"index": "idx"}},
               "sandbox_layout": {"layout": "<artifacts>"}, "environment": {}, "docs": {}}
    (sysdir / "harness.yaml").write_text(yaml.safe_dump(harness), encoding="utf-8")
    index_dir = snapshot_dir(bench, "snap") / "idx"
    index_dir.mkdir(parents=True)
    (index_dir / "build.yaml").write_text(yaml.safe_dump({"mutable": mutable}), encoding="utf-8")
    return bench


def test_audit_run_flags_an_unattributed_sandbox_file_build_yaml_calls_mutable(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    bench = _bench(tmp_path, ["foo.txt"])
    meta = yaml.safe_load((run / "run.yaml").read_text())
    meta.update({"configuration": "d", "snapshot": "snap"})
    (run / "run.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    sandbox = tmp_path / "sandbox" / "index" / "layout"
    sandbox.mkdir(parents=True)
    # why: `foo.txt` is in this build.yaml's `mutable` list, and asserting the exact message keeps
    # the unrelated master-index violation from passing for this one
    (sandbox / "foo.txt").write_text("vendor-recreated file\n")
    (sandbox / "rogue.txt").write_text("nobody attributed this\n")
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert "sandbox file not attributed: layout/foo.txt" in result["violations"]
    assert "sandbox file not attributed: layout/rogue.txt" in result["violations"]


def test_audit_run_without_benchmark_still_flags_unattributed_sandbox_file(tmp_path: Path):
    # inv: attribution is decided by the journal alone, so omitting `benchmark` -- which only
    # supplies the fixed-step authority and the master-index recheck -- cannot widen it
    run = _run(tmp_path, "graphify 0.9.27\n")
    sandbox = tmp_path / "sandbox" / "index" / "layout"
    sandbox.mkdir(parents=True)
    (sandbox / "rogue.txt").write_text("nobody attributed this\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("rogue.txt" in v for v in result["violations"])


def test_audit_run_refuses_to_re_audit_collected_evidence(tmp_path: Path):
    # inv: collect.py has already cleaned the sandbox a run under benchmark/record/runs/ was judged
    # against, so re-auditing there must refuse
    run = _run(tmp_path, "graphify 0.9.27\n")
    bench = tmp_path / "bench"
    collected = bench / "record" / "runs" / "q-s-d-a01" / "run"
    collected.parent.mkdir(parents=True)
    shutil.copytree(run, collected)
    with pytest.raises(SystemExit, match="collected evidence"):
        audit.check_run(collected, INV, skip_environment=True, benchmark=bench)


def test_audit_run_still_audits_a_run_directory_outside_benchmark_runs(tmp_path: Path):
    # inv: the refusal above must be narrow -- a run still under its tmp root, mid-checklist,
    # must audit normally even when a benchmark root is given
    run = _run(tmp_path, "graphify 0.9.27\n")
    bench = _bench(tmp_path, [])
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert result["valid"] is True


def test_audit_run_computes_master_index_changed_from_the_current_master(tmp_path: Path):
    # inv: with a benchmark root given, check_run recomputes master_index_changed against the
    # live master rather than hard-coding it False
    run = _run(tmp_path, "graphify 0.9.27\n")
    bench = _bench(tmp_path, [])
    meta = yaml.safe_load((run / "run.yaml").read_text())
    meta.update({"configuration": "d", "snapshot": "snap"})
    (run / "run.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    index_dir = snapshot_dir(bench, "snap") / "idx"
    build = yaml.safe_load((index_dir / "build.yaml").read_text())
    build["artifacts"] = {"graph.db": "deadbeef"}
    (index_dir / "build.yaml").write_text(yaml.safe_dump(build), encoding="utf-8")
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert result["master_index_changed"] is True
    assert any("master index" in v for v in result["violations"])


def test_audit_run_master_index_changed_stays_false_without_a_benchmark_root(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert result["master_index_changed"] is False


def test_audit_run_fixed_steps_check_prefers_harness_yaml_over_run_yaml(tmp_path: Path):
    # inv: run.yaml and journal.jsonl both live in the untracked scratch tree and could be
    # edited together; harness.yaml is git-tracked, so it is the authority when it is available,
    # even though run.yaml's own copy would have agreed with the journal here
    run = _run(tmp_path, "graphify 0.9.27\n")
    bench = _bench(tmp_path, [], n_fixed_steps=2)
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert any("harness.yaml" in v and "fixed_steps=2" in v and "1 harness call entries" in v
               for v in result["violations"])



def test_audit_run_reads_a_refused_entrys_planted_output(tmp_path: Path):
    # inv: a refusal never runs anything, but any output planted at its stem must still be
    # scanned for a leaked absolute path -- refusal is not an exemption from the leak scan
    run = tmp_path / "q-s-d-a03" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a03" / "home").mkdir(parents=True, exist_ok=True)
    (run / "01_bad.out").write_text("/home/someone/projects/demo/leak\n")
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "bad", "by": "harness", "argv": ["/x/graphify", "explain"],
         "quote": None, "action": False, "refused": "unknown subcommand: explain", "system_call": False},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a03", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("absolute path" in v for v in result["violations"])


def test_audit_run_flags_an_orphan_out_file(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    (run / "99_extra.out").write_text("orphaned output no entry claims\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("99_extra.out" in v and "not claimed" in v for v in result["violations"])


def test_audit_run_flags_missing_call_entries_against_fixed_steps(tmp_path: Path):
    # inv: deleting a journal line (and its .out) must not silently pass as a shorter run
    run = tmp_path / "q-s-d-a04" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a04" / "home").mkdir(parents=True, exist_ok=True)
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a04", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("fixed_steps" in v for v in result["violations"])


def test_audit_run_flags_a_stop_that_is_not_last(tmp_path: Path):
    run = tmp_path / "q-s-d-a05" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a05" / "home").mkdir(parents=True, exist_ok=True)
    (run / "02_version.out").write_text("graphify 0.9.27\n")
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "stop", "by": "harness", "reason": "harness"},
        {"n": 2, "kind": "call", "name": "version", "by": "harness", "argv": ["/x/graphify", "--version"],
         "quote": None, "exit": 0, "out_sha256": rules.sha256_file(run / "02_version.out"),
         "canonical_sha256": rules.canonical_hash("graphify 0.9.27\n", []), "system_call": True,
         "ceiling_call": False, "files": []},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a05", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert "journal must carry exactly one stop entry, last" in result["violations"]


def test_audit_run_reports_a_clobbered_sandbox_root_without_crashing(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    sandbox_parent = tmp_path / "sandbox"
    shutil.rmtree(sandbox_parent / "index")
    (sandbox_parent / "index").write_text("not a directory\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("not a directory" in v for v in result["violations"])
    assert (run / "audit.json").exists()


def test_audit_run_flags_a_nonzero_exit(tmp_path: Path):
    run = tmp_path / "q-s-d-a06" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a06" / "home").mkdir(parents=True, exist_ok=True)
    (run / "01_version.out").write_text("")
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "version", "by": "harness", "argv": ["/x/graphify", "--version"],
         "quote": None, "exit": 1, "out_sha256": rules.sha256_file(run / "01_version.out"),
         "canonical_sha256": rules.canonical_hash("", []), "system_call": True,
         "ceiling_call": False, "files": []},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a06", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("exit 1" in v for v in result["violations"])


@pytest.mark.skipif(sys.platform != "darwin", reason="/tmp is a symlink to /private/tmp only on macOS")
def test_audit_run_resolves_the_macos_tmp_symlink_before_flagging_a_path(tmp_path: Path):
    # inv: the sandbox root may be spelled /private/tmp while a vendor line spells the macOS
    # symlink form /tmp -- both must resolve to the same root before being compared
    run = tmp_path / "q-s-d-a07" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a07" / "home").mkdir(parents=True, exist_ok=True)
    tmp_root = Path("/private/tmp/graphify-bench-test-a07")
    text = "/tmp/graphify-bench-test-a07/sandbox/index/x seen via the macOS symlink\n"
    (run / "01_version.out").write_text(text)
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "version", "by": "harness", "argv": ["/x/graphify", "--version"],
         "quote": None, "exit": 0, "out_sha256": rules.sha256_file(run / "01_version.out"),
         "canonical_sha256": rules.canonical_hash(text, []), "system_call": True,
         "ceiling_call": False, "files": []},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a07", tmp_root, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert not any("absolute path" in v for v in result["violations"])


def test_audit_run_flags_a_lookalike_sibling_of_the_sandbox(tmp_path: Path):
    # inv: a leaked path is judged by path ancestry, not string prefix -- a sibling directory
    # whose name merely starts with "sandbox" must never be accepted as inside it
    run = tmp_path / "q-s-d-a08" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a08" / "home").mkdir(parents=True, exist_ok=True)
    leak_path = tmp_path / "sandbox-evil" / "leak"
    text = f"{leak_path}\n"
    (run / "01_version.out").write_text(text)
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "version", "by": "harness", "argv": ["/x/graphify", "--version"],
         "quote": None, "exit": 0, "out_sha256": rules.sha256_file(run / "01_version.out"),
         "canonical_sha256": rules.canonical_hash(text, []), "system_call": True,
         "ceiling_call": False, "files": []},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a08", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("absolute path" in v for v in result["violations"])



def test_audit_run_exempts_a_failed_exec_entry_from_hash_grammar_and_exit_checks(tmp_path: Path):
    # inv: check_run skips a failed-exec entry -- error, exit None, no .out -- the same way
    # score._executed does
    run = tmp_path / "q-s-d-a09" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a09" / "home").mkdir(parents=True, exist_ok=True)
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "query", "by": "harness", "argv": ["/x/graphify", "query", "a"],
         "quote": None, "exit": None, "error": "OSError: no such file or directory", "system_call": True,
         "ceiling_call": False},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a09", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert result["violations"] == []



def _bad_call_run(root: Path, run_id: str, *, tagged: bool) -> Path:
    run = root / run_id / "run"
    run.mkdir(parents=True)
    (run / "01_bad.out").write_text("noise\n")
    entry = {"n": 1, "kind": "call", "name": "bad", "by": "harness", "argv": ["/x/graphify", "frobnicate"],
             "quote": None, "exit": 0, "out_sha256": rules.sha256_file(run / "01_bad.out"),
             "canonical_sha256": rules.canonical_hash("noise\n", []), "system_call": True,
             "ceiling_call": False, "files": []}
    if tagged:
        entry["error"] = "fabricated to probe the exemption"
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        entry,
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml(run_id, root, fixed_steps=1)))
    return run


def test_audit_run_error_tag_does_not_bypass_checks_when_out_exists(tmp_path: Path):
    # inv: the "error" exemption keys on the absent .out, so a real .out with a planted "error"
    # key is still checked
    control = _bad_call_run(tmp_path, "q-s-d-a10", tagged=False)
    tagged = _bad_call_run(tmp_path, "q-s-d-a11", tagged=True)
    control_result = audit.check_run(control, INV, skip_environment=True)
    tagged_result = audit.check_run(tagged, INV, skip_environment=True)
    assert any("unknown subcommand: frobnicate" in v for v in control_result["violations"])
    assert any("unknown subcommand: frobnicate" in v for v in tagged_result["violations"])



def test_audit_run_error_tag_does_not_exempt_an_off_shape_exit(tmp_path: Path):
    # inv: execute.py only ever pairs "error" with exit: None; an "error" entry carrying any
    # other exit is already off-shape and must not be waved through just because .out is missing
    run = tmp_path / "q-s-d-a12" / "run"
    run.mkdir(parents=True)
    entries = [
        {"n": 0, "kind": "header", "rules_version": 1},
        {"n": 1, "kind": "call", "name": "bad", "by": "harness", "argv": ["/x/graphify", "frobnicate"],
         "quote": None, "exit": 0, "error": "off-shape: exit is not None", "system_call": True,
         "ceiling_call": False},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a12", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("unknown subcommand: frobnicate" in v for v in result["violations"])


def test_audit_run_flags_bad_server_start_argv(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    server_entry = {"n": 5, "kind": "server", "event": "start", "by": "harness",
                    "argv": ["/x/graphify", "serve", "--http", "--repo", "/somewhere/else"], "files": []}
    entries.insert(-1, server_entry)
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("--http" in v for v in result["violations"])
    assert any("--repo" in v for v in result["violations"])


def test_audit_run_flags_a_step_whose_output_no_longer_matches_prepared_outputs(tmp_path: Path):
    # inv: prepare compares each fixed step against prepared_outputs while the vendor runs;
    # check_run replays that comparison, so evidence edited after the run stops matching
    run = _run(tmp_path, "graphify 0.9.27\n")
    bench = _bench(tmp_path, [])
    recipe = "aa" * 32
    meta = yaml.safe_load((run / "run.yaml").read_text())
    meta.update({"configuration": "d", "snapshot": "snap", "harness_sha256": recipe, "question": "q1"})
    (run / "run.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    idx = snapshot_dir(tmp_path / "bench", "snap") / "idx"
    honest = rules.canonical_hash("graphify 0.9.27\n", [])
    prepare.write_prepared(idx, {"d": {recipe: {"q1": {"version": {"out": honest, "files": {}}}}}})
    assert audit.check_run(run, INV, skip_environment=True, benchmark=bench)["violations"] == []
    (run / "01_version.out").write_text("graphify 9.9.9\n")
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    entries[1]["out_sha256"] = rules.sha256_file(run / "01_version.out")
    entries[1]["canonical_sha256"] = rules.canonical_hash("graphify 9.9.9\n", [])
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert "entry 1: version differs from prepared_outputs" in result["violations"]
    assert result["valid"] is False


def test_audit_run_reports_an_absent_watched_root_instead_of_skipping_it(tmp_path: Path):
    # inv: an absent root means the attribution loop could not look; reporting valid with it
    # skipped is the one outcome that loop exists to prevent
    run = _run(tmp_path, "graphify 0.9.27\n")
    shutil.rmtree(tmp_path / "sandbox" / "index")
    result = audit.check_run(run, INV, skip_environment=True)
    assert any("sandbox root absent" in v for v in result["violations"])
    assert result["valid"] is False


def test_audit_run_passes_a_plain_refused_entry(tmp_path: Path):
    # inv: a refused call produced no output and is exempt from the hash, grammar and exit checks
    # -- counting it would void every run in which the model attempted one refused call
    run = tmp_path / "q-s-d-a01" / "run"
    run.mkdir(parents=True)
    (tmp_path / "sandbox" / "index").mkdir(parents=True, exist_ok=True)
    (tmp_path / "q-s-d-a01" / "home").mkdir(parents=True, exist_ok=True)
    entries = [{"n": 0, "kind": "header", "rules_version": 1},
               {"n": 1, "kind": "call", "name": "bad", "by": "model", "action": False,
                "refused": "unknown subcommand: bad", "system_call": False},
               {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"}]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    # inv: a refused call is still a call entry, so the step count names it
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a01", tmp_path, fixed_steps=1)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert result["violations"] == []
    assert result["valid"] is True


def test_audit_main_refuses_an_invocation_that_names_no_subcommand():
    # inv: exit 0 from the audit is the green light, so a mistyped invocation must not produce one
    with pytest.raises(SystemExit) as excinfo:
        audit.main([])
    assert excinfo.value.code == 2


def _sent_run(tmp_path: Path, prompt_body: str, tool_definitions: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "run.yaml").write_text(
        "run_id: r1\nsystem: graphify\nconfiguration: default\nquestion: q001\n", encoding="utf-8")
    (run / "request.json").write_text(json.dumps({
        "model": "m", "tools": tool_definitions,
        "messages": [{"role": "user", "content": f"{prompt_body}\n{prompt.HEADING}\n\nNODE it\n"}],
    }), encoding="utf-8")
    # inv: a run carrying a request is a run a model drove, and its journal says so; a fixture
    # without one exercises a shape the harness cannot produce
    (run / "journal.jsonl").write_text(json.dumps(
        {"n": 1, "kind": "call", "by": "runner", "name": "act"}) + "\n", encoding="utf-8")
    return run


def _blind_bench(tmp_path: Path) -> Path:
    benchmark = tmp_path / "bench"
    write_question(benchmark, "q001", {"id": "q001", "snapshot": "snap", "text": "t"})
    (references_dir(benchmark, "snap") / "q001.yaml").write_text(
        "places:\n  - {path: pkg/logic.py, symbol: render_invoice, start: 63}\n",
        encoding="utf-8")
    return benchmark


def test_the_blind_audit_reads_the_request_that_was_actually_sent(tmp_path: Path):
    benchmark = _blind_bench(tmp_path)
    run = _sent_run(tmp_path, "how is the score calculated", [{"name": "act", "description": "d"}])
    assert audit.check_blind(benchmark, run) == []


def test_the_blind_audit_catches_a_leak_in_the_prompt_that_was_sent(tmp_path: Path):
    benchmark = _blind_bench(tmp_path)
    run = _sent_run(tmp_path, "start at render_invoice", [{"name": "act", "description": "d"}])
    assert any("render_invoice" in message for message in audit.check_blind(benchmark, run))


def test_the_blind_audit_catches_a_leak_in_a_tool_that_was_sent(tmp_path: Path):
    benchmark = _blind_bench(tmp_path)
    run = _sent_run(tmp_path, "how is the score calculated",
                    [{"name": "act", "description": "call render_invoice"}])
    assert any("act" in message for message in audit.check_blind(benchmark, run))


def test_the_blind_audit_refuses_a_driven_run_that_recorded_no_request(tmp_path: Path):
    benchmark = _blind_bench(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.yaml").write_text("run_id: r1\nsystem: graphify\nquestion: q001\n", encoding="utf-8")
    (run / "journal.jsonl").write_text(json.dumps(
        {"n": 1, "kind": "call", "by": "runner", "name": "act"}) + "\n", encoding="utf-8")
    assert any("request.json" in message for message in audit.check_blind(benchmark, run))


def _driven_run(tmp_path: Path, runner_calls: int) -> Path:
    """A run whose fixed step ran and whose runner then made calls of its own."""
    run = _run(tmp_path, "graphify 0.9.27\n")
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    fixed = [e for e in entries if e.get("kind") != "stop"]
    n = max(e["n"] for e in fixed)
    for _ in range(runner_calls):
        n += 1
        out = run / f"{n:02d}_act.out"
        out.write_text("NODE thing\n")
        (run / f"{n:02d}_act.err").write_text("")
        (run / f"{n:02d}_act.cmd").write_text("kind: act\n")
        fixed.append({"n": n, "kind": "call", "name": "act", "by": "runner",
                      "argv": ["/x/graphify", "--version"], "quote": "q", "exit": 0,
                      "out_sha256": rules.sha256_file(out),
                      "canonical_sha256": rules.canonical_hash("NODE thing\n", []),
                      "system_call": True, "ceiling_call": False, "files": []})
    fixed.append({"n": n + 1, "kind": "stop", "by": "runner", "reason": "answer_met"})
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in fixed))
    return run


def test_a_runner_may_make_calls_beyond_the_fixed_steps(tmp_path: Path):
    run = _driven_run(tmp_path, runner_calls=3)
    result = audit.check_run(run, INV, skip_environment=True)
    assert not [v for v in result["violations"] if "fixed_steps" in v]


def test_the_wrong_number_of_harness_calls_is_still_a_violation(tmp_path: Path):
    run = _driven_run(tmp_path, runner_calls=3)
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    # why: the fixed steps are the harness's own, and one of them going missing must still be
    # caught however many calls the runner went on to make
    entries = [e for e in entries if not (e.get("kind") == "call" and e.get("by") == "harness")]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    result = audit.check_run(run, INV, skip_environment=True)
    assert [v for v in result["violations"] if "fixed_steps" in v]


def test_a_driven_run_carries_its_stop_last(tmp_path: Path):
    run = _driven_run(tmp_path, runner_calls=2)
    result = audit.check_run(run, INV, skip_environment=True)
    assert not [v for v in result["violations"] if "stop entry" in v]


def test_a_run_no_model_drove_has_no_blindness_to_show(tmp_path: Path):
    # why: a baseline has no runner, so there is nobody to have been told the answer; calling
    # that a violation would fail every baseline ever collected
    benchmark = _blind_bench(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.yaml").write_text("run_id: r1\nsystem: graphify\nquestion: q001\n", encoding="utf-8")
    (run / "journal.jsonl").write_text(json.dumps(
        {"n": 1, "kind": "call", "by": "harness", "name": "query"}) + "\n", encoding="utf-8")
    assert audit.check_blind(benchmark, run) == []


def test_a_run_a_model_drove_must_still_show_the_request_it_was_sent(tmp_path: Path):
    benchmark = _blind_bench(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.yaml").write_text("run_id: r1\nsystem: graphify\nquestion: q001\n", encoding="utf-8")
    (run / "journal.jsonl").write_text(json.dumps(
        {"n": 1, "kind": "call", "by": "runner", "name": "act"}) + "\n", encoding="utf-8")
    assert any("request.json" in m for m in audit.check_blind(benchmark, run))


def test_a_route_inside_a_sentence_is_not_a_leak(tmp_path: Path):
    # why: a docstring like "Scope filter tests for GET /invoices/totals." reaches the scan as a
    # node label, and its route scans as absolute while naming no path on this machine
    run = _run(tmp_path, "NODE Scope filter tests for GET /invoices/totals. [src=a.py loc=L1]\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert [v for v in result["violations"] if "absolute path" in v] == []


def test_a_real_path_outside_the_run_is_still_a_leak(tmp_path: Path):
    # inv: this is what the check exists for -- a path that exists on this machine and is not
    # part of the run names something the vendor was never given
    leaked = str(Path(__file__).resolve().parent)
    run = _run(tmp_path, f"NODE thing [src={leaked}/test_audit_run.py loc=L1]\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert [v for v in result["violations"] if "absolute path" in v]


def test_a_path_that_names_nothing_here_is_not_reported(tmp_path: Path):
    run = _run(tmp_path, "see /no/such/place/at/all for details\n")
    result = audit.check_run(run, INV, skip_environment=True)
    assert [v for v in result["violations"] if "absolute path" in v] == []


def _artifact_bench(tmp_path: Path, artifact_text: str) -> Path:
    """A benchmark root whose index declares one artifact holding `artifact_text`."""
    bench = _bench(tmp_path, [])
    index_dir = snapshot_dir(bench, "snap") / "idx"
    artifact = index_dir / "nodes.jsonl"
    artifact.write_text(artifact_text, encoding="utf-8")
    (index_dir / "build.yaml").write_text(
        yaml.safe_dump({"mutable": [], "artifacts": {"nodes.jsonl": rules.sha256_file(artifact)}}),
        encoding="utf-8")
    return bench


def _artifact_run(tmp_path: Path, out_text: str) -> Path:
    run = _run(tmp_path, out_text)
    meta = yaml.safe_load((run / "run.yaml").read_text())
    meta.update({"configuration": "d", "snapshot": "snap"})
    (run / "run.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return run


def test_a_sandbox_root_path_is_a_leak_even_with_artifacts_to_check_against(tmp_path: Path):
    # inv: the exemption is a substring test against the frozen artifact, so a path of this
    # machine that the artifact does not carry is flagged exactly as before
    leaked = str(Path(__file__).resolve().parent)
    run = _artifact_run(tmp_path, f"NODE thing [src={leaked}/test_audit_run.py loc=L1]\n")
    bench = _artifact_bench(tmp_path, '{"label": "#!/usr/bin/env python3"}\n')
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert [v for v in result["violations"] if "absolute path" in v]


def test_a_path_the_runs_own_frozen_artifact_carries_is_not_a_leak(tmp_path: Path):
    # why: /usr/bin/env exists on this host and is not under the run's roots, so existence alone
    # calls it a leak; the frozen artifact holds the same string, which makes it corpus content
    run = _artifact_run(tmp_path, "NODE snippet [src=a.py loc=L1]\n#!/usr/bin/env python3\n")
    bench = _artifact_bench(tmp_path, '{"label": "#!/usr/bin/env python3", "path": "a.py"}\n')
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert [v for v in result["violations"] if "absolute path" in v] == []


def test_the_same_path_absent_from_the_artifacts_is_still_a_leak(tmp_path: Path):
    # inv: the artifact is the whole of the exemption -- the identical output against an index
    # whose artifact does not carry the string is refused
    run = _artifact_run(tmp_path, "NODE snippet [src=a.py loc=L1]\n#!/usr/bin/env python3\n")
    bench = _artifact_bench(tmp_path, '{"label": "def main() -> None:", "path": "a.py"}\n')
    result = audit.check_run(run, INV, skip_environment=True, benchmark=bench)
    assert any("/usr/bin/env" in v for v in result["violations"] if "absolute path" in v)


def test_audit_run_counts_a_halted_step_as_a_harness_call_and_checks_no_output_for_it(tmp_path: Path):
    run = _run(tmp_path, "graphify 0.9.27\n")
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    halted = {"n": 2, "kind": "call", "name": "query", "by": "harness", "argv": ["/x/graphify", "query", ""],
              "quote": "q", "action": False, "halted": "empty expansion", "system_call": False,
              "ceiling_call": False, "files": []}
    entries = [*entries[:-1], halted, {"n": 3, "kind": "stop", "by": "harness", "reason": "harness"}]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(_run_yaml("q-s-d-a01", tmp_path, fixed_steps=2)))
    result = audit.check_run(run, INV, skip_environment=True)
    assert result["violations"] == []
    assert result["valid"] is True
    (run / "01_version.cmd").write_text("graphify --version\n")
    assert [s["name"] for s in prepare.prompt_steps(run, entries)] == ["version"]


def test_a_run_no_model_drove_says_on_stderr_that_there_was_nothing_to_check(
        tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    # inv: an empty problem list means "blind" everywhere else, so the one case where it means
    # "not asked" has to say so, or a recipe run reads as a passed blindness check
    benchmark = _blind_bench(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.yaml").write_text("run_id: r1\nsystem: graphify\nquestion: q001\n", encoding="utf-8")
    (run / "journal.jsonl").write_text(json.dumps(
        {"n": 1, "kind": "call", "by": "harness", "name": "query"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(rules, "_ROOT", benchmark)
    assert audit.main(["blind", str(run)]) == 0
    assert f"no driven attempt in {run}: nothing to check" in capsys.readouterr().err


def test_a_driven_run_that_was_blind_says_nothing_on_stderr(
        tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    benchmark = _blind_bench(tmp_path)
    run = _sent_run(tmp_path, "how is the score calculated", [{"name": "act", "description": "d"}])
    monkeypatch.setattr(rules, "_ROOT", benchmark)
    assert audit.main(["blind", str(run)]) == 0
    assert capsys.readouterr().err == ""
