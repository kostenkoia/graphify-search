import json
import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("tiktoken")

from benchmark.harness import config, score
from tests.benchmark.conftest import snapshot_dir, write_question


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "r" / "run"
    run.mkdir(parents=True)
    out = ("Traversal: BFS depth=2 | Start: ['a'] | 1 nodes found\n\n"
           "NODE alpha() [src=pkg/a.py loc=L10 community=a]\n"
           "NODE Doc node [src=pkg/a.py loc=L12 community=a]\n")
    (run / "03_query.out").write_text(out)
    (run / "01_version.out").write_text("graphify 0.9.27\n")
    entries = [
        {"n": 0, "kind": "header"},
        {"n": 1, "kind": "call", "name": "version", "argv": ["/x/graphify", "--version"], "exit": 0,
         "system_call": True, "ceiling_call": False, "files": []},
        {"n": 3, "kind": "call", "name": "query", "argv": ["/x/graphify", "query", "a"], "exit": 0,
         "system_call": True, "ceiling_call": False, "files": []},
        {"n": 4, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump({"run_id": "q1-graphify-default-a01", "system": "graphify",
                                                  "question": "q1", "configuration": "default"}))
    return run


def test_records_and_cost(tmp_path: Path, bench: Path):
    run = _run(tmp_path)
    recs = score.records(bench, run)
    assert [r["kind"] for r in recs] == ["place", "place"]
    assert (run / "records.jsonl").exists()
    c = score.cost(run)
    assert c["system_calls"] == 2
    assert c["actions"] == 2
    assert c["tokens"] > 0


def test_hits(tmp_path: Path, bench: Path):
    run = _run(tmp_path)
    score.records(bench, run)
    ref = tmp_path / "q1.yaml"
    place = {"path": "pkg/a.py", "symbol": "alpha", "start": 9, "end": 20}
    ref.write_text(yaml.safe_dump({"id": "q1", "places": [place]}))
    assert score.first_hit(run, ref) == {"hit": True, "hit_rank": 1, "hit_entry": 3}
    place = {"path": "pkg/a.py", "symbol": None, "start": 12, "end": 12}
    ref.write_text(yaml.safe_dump({"id": "q1", "places": [place]}))
    assert score.first_hit(run, ref)["hit_rank"] == 2
    place = {"path": "pkg/a.py", "symbol": "other", "start": 9, "end": 20}
    ref.write_text(yaml.safe_dump({"id": "q1", "places": [place]}))
    assert score.first_hit(run, ref)["hit"] is False


def test_records_and_cost_skip_a_failed_exec_entry(tmp_path: Path, bench: Path):
    # inv: a failed-exec entry carries "error" and wrote no .out; scoring it must skip it,
    # not crash trying to open output that was never written
    run = _run(tmp_path)
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    entries.insert(-1, {"n": 5, "kind": "call", "name": "boom", "argv": ["/x/graphify", "query", "b"],
                        "exit": None, "error": "OSError: no such file", "system_call": True, "ceiling_call": False})
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    recs = score.records(bench, run)
    assert [r["kind"] for r in recs] == ["place", "place"]
    c = score.cost(run)
    assert c["actions"] == 2


def test_records_and_cost_do_not_skip_a_tagged_entry_with_a_real_out(tmp_path: Path, bench: Path):
    # inv: the "error" exemption is keyed on the absence of a .out, never on the tag alone --
    # an entry with a real .out must still be scored even if "error" is also present
    run = _run(tmp_path)
    (run / "05_boom.out").write_text("graphify 0.9.27\n")
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    entries.insert(-1, {"n": 5, "kind": "call", "name": "boom", "argv": ["/x/graphify", "--version"],
                        "exit": 0, "error": "fabricated to probe the exemption", "system_call": True,
                        "ceiling_call": False})
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    score.records(bench, run)
    c = score.cost(run)
    assert c["actions"] == 3


def test_records_and_cost_do_not_exempt_an_off_shape_exit(tmp_path: Path, bench: Path):
    # inv: execute.py only ever pairs "error" with exit: None; an "error" entry carrying any
    # other exit is already off-shape and must not be silently skipped just because .out is missing
    run = _run(tmp_path)
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    entries.insert(-1, {"n": 5, "kind": "call", "name": "boom", "argv": ["/x/graphify", "--version"],
                        "exit": 0, "error": "off-shape: exit is not None", "system_call": True,
                        "ceiling_call": False})
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    with pytest.raises(FileNotFoundError):
        score.cost(run)


def test_main_exits_1_on_an_unparsed_record(tmp_path: Path, bench: Path):
    # inv: main() surfaces an UnparsedError as a non-zero exit, never exit 0 on a shape the
    # adapter was not written for
    run = _run(tmp_path)
    (run / "03_query.out").write_text("garbage the adapter cannot place\n")
    assert score.main([str(run), "--benchmark", str(bench)]) == 1


def test_an_unparsed_output_of_the_runners_own_call_is_kept_and_does_not_raise(tmp_path: Path, bench: Path):
    # inv: a fixed step's output must parse; a call the runner chose may print anything the
    # grammar admits, and its unreadable output is a record scored as no place, never a refusal
    run = _run(tmp_path)
    (run / "05_act.out").write_text("vocab: 2446 tokens\n")
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    entries.insert(-1, {"n": 5, "kind": "call", "name": "act", "by": "runner",
                        "argv": ["/x/graphify", "query", "b"], "exit": 0,
                        "system_call": True, "ceiling_call": True, "files": []})
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    recs = score.records(bench, run)
    assert [r for r in recs if r["kind"] == "unparsed"] == [
        {"run": "q1-graphify-default-a01", "n": 5, "by": "runner", "kind": "unparsed", "text": "vocab: 2446 tokens"}]


def test_records_raises_on_unknown_configuration(tmp_path: Path, bench: Path):
    run = _run(tmp_path)
    meta = yaml.safe_load((run / "run.yaml").read_text())
    meta["configuration"] = "bogus"
    (run / "run.yaml").write_text(yaml.safe_dump(meta))
    with pytest.raises(KeyError):
        score.records(bench, run)


# inv: the root the crg fixtures' absolute file_path values carry, which the synthetic
# build.yaml below declares as build_cwd so the adapter has a prefix to strip
_CRG_INDEX_ROOT = "/tmp/bench-sandbox/index"


def _crg_bench(tmp_path: Path, bench: Path) -> Path:
    out = tmp_path / "bench"
    (out / "systems").mkdir(parents=True)
    shutil.copytree(bench / "systems" / "code-review-graph", out / "systems" / "code-review-graph")
    index = snapshot_dir(out, "snap") / "indexes" / "code-review-graph-vector"
    index.mkdir(parents=True)
    (index / "build.yaml").write_text(
        yaml.safe_dump({"properties": {"paths_in_index": "absolute"}, "build_cwd": _CRG_INDEX_ROOT}),
        encoding="utf-8")
    return out


def _crg_run(tmp_path: Path, reply: str) -> Path:
    run = tmp_path / "c" / "run"
    run.mkdir(parents=True)
    (run / "01_version.out").write_text("code-review-graph 2.3.7\n")
    (run / "02_search.out").write_text(reply)
    entries = [
        {"n": 0, "kind": "header"},
        {"n": 1, "kind": "call", "name": "version", "argv": ["/x/code-review-graph", "--version"], "exit": 0,
         "system_call": True, "ceiling_call": False, "files": []},
        {"n": 2, "kind": "call", "name": "search", "tool": "semantic_search_nodes_tool", "args": {}, "exit": 0,
         "system_call": True, "ceiling_call": True, "files": []},
        {"n": 3, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump(
        {"run_id": "q1-code-review-graph-vector-a01", "system": "code-review-graph", "question": "q1",
         "configuration": "vector", "snapshot": "snap"}))
    return run


def test_records_strip_the_index_root_declared_by_build_yaml(tmp_path: Path, bench: Path):
    fixture = Path(__file__).parent / "fixtures" / "crg" / "search_semantic.json"
    recs = score.records(_crg_bench(tmp_path, bench), _crg_run(tmp_path, fixture.read_text()))
    # inv: the reference names corpus-relative paths, so a record still carrying the index root
    # could never match one; the root comes from this index's own build_cwd, not from the run
    assert [r["path"] for r in recs] == ["pkg/a.py", "pkg/b.py"]


def test_records_reject_a_vendor_path_outside_the_declared_index_root(tmp_path: Path, bench: Path):
    reply = json.dumps({"status": "ok", "search_mode": "semantic", "summary": "",
                        "results": [{"name": "Alpha", "kind": "Class", "file_path": "/elsewhere/pkg/a.py",
                                     "line_start": 3, "score": 0.5}]})
    with pytest.raises(score.UnparsedError, match="not a file under index root"):
        score.records(_crg_bench(tmp_path, bench), _crg_run(tmp_path, reply))


def test_path_prefix_refuses_an_absolute_index_that_never_recorded_where_it_was_built(tmp_path: Path):
    bench = tmp_path / "bench"
    sysdir = bench / "systems" / "s"
    sysdir.mkdir(parents=True)
    (sysdir / "harness.yaml").write_text(yaml.safe_dump(
        {"adapter": "code_review_graph", "version": {"cli": "x"}, "invocation": {}, "fixed_steps": [],
         "default_configuration": "d", "configurations": {"d": {"index": "idx"}},
         "sandbox_layout": {"layout": "<artifacts>"}, "environment": {}, "docs": {}}), encoding="utf-8")
    index_dir = snapshot_dir(bench, "snap") / "idx"
    index_dir.mkdir(parents=True)
    (index_dir / "build.yaml").write_text(yaml.safe_dump({"properties": {"paths_in_index": "absolute"}}),
                                          encoding="utf-8")
    h = config.load_harness(bench, "s")
    with pytest.raises(config.ConfigError, match="without build_cwd"):
        score._path_prefix(bench, h, {"snapshot": "snap"}, "d")
    (index_dir / "build.yaml").write_text(yaml.safe_dump({"properties": {"symbol_end_line": False}}),
                                          encoding="utf-8")
    assert score._path_prefix(bench, config.load_harness(bench, "s"), {"snapshot": "snap"}, "d") is None


def test_is_hit_requires_a_place_not_merely_a_matching_location():
    ref = {"path": "pkg/a.py", "start": 10, "end": 20, "symbol": None}
    place = {"kind": "place", "path": "pkg/a.py", "start": 12}
    # inv: the protocol's first rule of measurement is that only a place counts as an answer; an edge or a
    # candidate at the same location is evidence, never a hit
    assert score._is_hit(place, ref) is True
    for kind in ("edge", "candidate", "file", "no_results"):
        assert score._is_hit({**place, "kind": kind}, ref) is False


def test_is_hit_refuses_a_place_that_names_another_file():
    # inv: every clause stands on its own -- a place with a valid line range in the wrong file is
    # not the answer, and this is the benchmark's headline metric
    ref = {"path": "src/right.py", "start": 1, "end": 10, "symbol": "f"}
    assert score._is_hit({"kind": "place", "path": "src/right.py", "start": 5, "symbol": "f"}, ref)
    for wrong in ({"kind": "place", "path": "src/other.py", "start": 5, "symbol": "f"},
                  {"kind": "file", "path": "src/right.py", "start": 5, "symbol": "f"},
                  {"kind": "place", "path": "src/right.py", "start": None, "symbol": "f"},
                  {"kind": "place", "path": "src/right.py", "start": 11, "symbol": "f"},
                  {"kind": "place", "path": "src/right.py", "start": 0, "symbol": "f"},
                  {"kind": "place", "path": "src/right.py", "start": 5, "symbol": "g"}):
        assert not score._is_hit(wrong, ref), wrong


def test_is_hit_takes_the_reference_range_inclusively():
    # inv: the bounds are inclusive at both ends, so a place starting exactly at the reference's
    # first or last line is the answer
    ref = {"path": "a.py", "start": 10, "end": 20, "symbol": None}
    for start, want in ((9, False), (10, True), (20, True), (21, False)):
        assert score._is_hit({"kind": "place", "path": "a.py", "start": start}, ref) is want, start


def _runner_run(tmp_path: Path, *, place: dict, reason: str = "answer_met") -> Path:
    run = tmp_path / "d" / "run"
    run.mkdir(parents=True)
    (run / "01_query.out").write_text("NODE alpha() [src=pkg/a.py loc=L10 community=a]\n")
    entries = [
        {"n": 0, "kind": "header"},
        {"n": 1, "kind": "call", "name": "query", "by": "harness", "argv": ["/x/graphify", "query", "a"],
         "exit": 0, "system_call": True, "ceiling_call": False, "files": []},
        {"n": 2, "kind": "call", "name": "act", "by": "runner", "argv": ["/x/graphify", "query", "b"],
         "action": False, "refused": "unknown subcommand"},
        {"n": 3, "kind": "call", "name": "act", "by": "runner", "argv": ["/x/graphify", "query", "c"],
         "exit": 0, "system_call": True, "ceiling_call": False, "files": []},
        {"n": 4, "kind": "stop", "by": "runner", "reason": reason, "place": place},
    ]
    (run / "03_act.out").write_text("NODE beta() [src=pkg/b.py loc=L20 community=b]\n")
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (run / "run.yaml").write_text(yaml.safe_dump({"run_id": "q1-graphify-default-a02",
                                                  "system": "graphify", "question": "q1",
                                                  "configuration": "default"}))
    api = run / "api"
    api.mkdir()
    (api / "01.json").write_text(json.dumps({"request": {}, "response": {
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 5}}}))
    (api / "02.json").write_text(json.dumps({"request": {}, "response": {
        "usage": {"input_tokens": 200, "output_tokens": 20}}}))
    return run


def _reference(tmp_path: Path) -> Path:
    path = tmp_path / "ref.yaml"
    path.write_text(yaml.safe_dump({"places": [
        {"path": "pkg/a.py", "symbol": "alpha", "start": 10, "end": 14}]}))
    return path


def test_the_stop_reason_is_read_from_the_journal(tmp_path: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "alpha", "start": 10})
    assert score.stop_reason(run) == "answer_met"


def test_a_runner_naming_the_reference_place_is_the_one_credited(tmp_path: Path, bench: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "alpha", "start": 10})
    score.records(bench, run)
    verdict = score.runner_verdict(run, _reference(tmp_path))
    assert verdict["stop_hit"] is True
    assert verdict["hit_by"] == "runner"


def test_a_runner_naming_the_wrong_place_leaves_the_credit_with_the_output(tmp_path: Path, bench: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/z.py", "symbol": "zeta", "start": 1})
    score.records(bench, run)
    verdict = score.runner_verdict(run, _reference(tmp_path))
    assert verdict["stop_hit"] is False
    assert verdict["hit_by"] == "harness"


def test_a_run_that_found_nothing_credits_no_one(tmp_path: Path, bench: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/z.py", "symbol": "zeta", "start": 1})
    score.records(bench, run)
    reference = tmp_path / "other.yaml"
    reference.write_text(yaml.safe_dump({"places": [{"path": "pkg/nowhere.py", "symbol": "n", "start": 1}]}))
    verdict = score.runner_verdict(run, reference)
    assert verdict["hit_by"] is None


def test_a_run_the_runner_never_answered_has_no_place_to_score(tmp_path: Path, bench: Path):
    run = tmp_path / "e" / "run"
    run.mkdir(parents=True)
    (run / "journal.jsonl").write_text(json.dumps(
        {"n": 1, "kind": "stop", "by": "runner", "reason": "no_further_action"}) + "\n")
    (run / "records.jsonl").write_text("")
    (run / "run.yaml").write_text(yaml.safe_dump({"run_id": "r", "system": "graphify",
                                                  "question": "q1", "configuration": "default"}))
    verdict = score.runner_verdict(run, _reference(tmp_path))
    assert verdict["stop_hit"] is False
    assert verdict["stop"] == "no_further_action"


def test_the_runner_actions_are_counted_apart_from_the_harness_own(tmp_path: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "alpha", "start": 10})
    assert score.runner_actions(run) == {"runner_actions": 2, "refused": 1}


def test_the_model_usage_is_summed_over_every_recorded_exchange(tmp_path: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "alpha", "start": 10})
    assert score.model_usage(run) == {"input_tokens": 300, "output_tokens": 30,
                                      "cache_read_input_tokens": 5}


def test_a_run_no_model_drove_reports_no_usage(tmp_path: Path):
    run = _run(tmp_path)
    assert score.model_usage(run) == {}


def test_a_baseline_result_keeps_the_shape_it_has_always_had(tmp_path: Path, bench: Path):
    run = _run(tmp_path)
    score.records(bench, run)
    benchmark = tmp_path / "b"
    write_question(benchmark, "q1", {"id": "q1", "snapshot": "snap", "text": "t"},
                   {"places": [{"path": "pkg/a.py", "symbol": "alpha",
                                "start": 10, "end": 14}]})
    assert set(score.hits(benchmark, run)) == {"hit", "hit_rank", "hit_entry"}


def test_scoring_a_run_writes_nothing_under_the_benchmark_root(tmp_path: Path, bench: Path):
    # inv: hits returns its mapping to its caller and leaves no file behind, so no scoring run
    # can leave untracked dirt that the next require_clean would refuse an attempt over
    run = _run(tmp_path)
    score.records(bench, run)
    benchmark = tmp_path / "b"
    write_question(benchmark, "q1", {"id": "q1", "snapshot": "snap", "text": "t"},
                   {"places": [{"path": "pkg/a.py", "symbol": "alpha",
                                "start": 10, "end": 14}]})
    before = sorted(p.relative_to(benchmark).as_posix() for p in benchmark.rglob("*"))
    score.hits(benchmark, run)
    assert sorted(p.relative_to(benchmark).as_posix() for p in benchmark.rglob("*")) == before
    assert not (benchmark / "results").exists()


def test_a_driven_result_carries_what_the_runner_did(tmp_path: Path, bench: Path):
    import shutil

    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "alpha", "start": 10})
    score.records(bench, run)
    benchmark = tmp_path / "b"
    # inv: a real benchmark always holds the system beside the reference, and scoring a driven
    # run reads the system to learn whether its index stores absolute paths
    (benchmark / "systems" / "graphify").mkdir(parents=True)
    shutil.copy(bench / "systems" / "graphify" / "harness.yaml",
                benchmark / "systems" / "graphify" / "harness.yaml")
    write_question(benchmark, "q1", {"id": "q1", "snapshot": "snap", "text": "t"},
                   {"places": [{"path": "pkg/a.py", "symbol": "alpha",
                                "start": 10, "end": 14}]})
    result = score.hits(benchmark, run)
    assert result["stop"] == "answer_met"
    assert result["hit_by"] == "runner"
    assert result["runner_actions"] == 2
    assert result["refused"] == 1
    assert result["model_usage"]["output_tokens"] == 30



def test_a_runner_repeating_the_vendors_own_label_still_names_the_place(tmp_path: Path, bench: Path):
    # why: the vendor prints render_invoice() and the harness strips that decoration when it
    # parses the vendor's own output; not stripping it from the runner's answer would score one
    # place two ways depending on who named it
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "alpha()", "start": 10})
    score.records(bench, run)
    assert score.runner_verdict(run, _reference(tmp_path))["stop_hit"] is True


def test_a_leading_dot_the_other_vendor_writes_is_stripped_too(tmp_path: Path, bench: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": ".alpha", "start": 10})
    score.records(bench, run)
    assert score.runner_verdict(run, _reference(tmp_path))["stop_hit"] is True


def test_a_runner_naming_a_different_symbol_still_misses(tmp_path: Path, bench: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "beta()", "start": 10})
    score.records(bench, run)
    assert score.runner_verdict(run, _reference(tmp_path))["stop_hit"] is False


ROOT = "/tmp/bench/sandbox/index"


def test_a_runner_repeating_the_vendors_absolute_path_still_names_the_place(tmp_path: Path, bench: Path):
    # why: this vendor stores absolute paths and the harness strips the index root when it parses
    # the vendor's output; not stripping it from the runner's answer scores one place two ways
    run = _runner_run(tmp_path, place={"path": f"{ROOT}/pkg/a.py", "symbol": "alpha", "start": 10})
    score.records(bench, run)
    verdict = score.runner_verdict(run, _reference(tmp_path), path_prefix=ROOT)
    assert verdict["stop_hit"] is True


def test_a_runner_naming_a_path_outside_the_index_root_misses(tmp_path: Path, bench: Path):
    run = _runner_run(tmp_path, place={"path": "/elsewhere/pkg/a.py", "symbol": "alpha", "start": 10})
    score.records(bench, run)
    assert score.runner_verdict(run, _reference(tmp_path), path_prefix=ROOT)["stop_hit"] is False


def test_an_index_that_stores_relative_paths_leaves_the_answer_alone(tmp_path: Path, bench: Path):
    run = _runner_run(tmp_path, place={"path": "pkg/a.py", "symbol": "alpha", "start": 10})
    score.records(bench, run)
    assert score.runner_verdict(run, _reference(tmp_path), path_prefix=None)["stop_hit"] is True


def _version_only_bench(tmp_path: Path, declared: str | None) -> Path:
    bench = tmp_path / "bench"
    sysdir = bench / "systems" / "s"
    sysdir.mkdir(parents=True)
    harness = {"adapter": "graphify_search", "version": {"cli": declared}, "invocation": {},
               "fixed_steps": [], "default_configuration": "d",
               "configurations": {"d": {"index": "idx", "search_mode": ["dense"]}},
               "sandbox_layout": {"graphify-out": "<artifacts>"}, "environment": {}, "docs": {}}
    (sysdir / "harness.yaml").write_text(yaml.safe_dump(harness), encoding="utf-8")
    index_dir = snapshot_dir(bench, "snap") / "idx"
    index_dir.mkdir(parents=True)
    (index_dir / "build.yaml").write_text(yaml.safe_dump({"properties": {"paths_in_index": "relative"}}),
                                          encoding="utf-8")
    return bench


def _version_only_run(tmp_path: Path, printed: str) -> Path:
    run = tmp_path / "r" / "run"
    run.mkdir(parents=True)
    (run / "01_version.out").write_text(printed, encoding="utf-8")
    entries = [
        {"n": 0, "kind": "header"},
        {"n": 1, "kind": "call", "name": "version", "argv": ["/x/graphify-search", "--version"], "exit": 0,
         "system_call": True, "ceiling_call": False, "files": []},
        {"n": 2, "kind": "stop", "by": "harness", "reason": "harness"},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    (run / "run.yaml").write_text(yaml.safe_dump({"run_id": "q1-s-d-a01", "system": "s", "question": "q1",
                                                  "configuration": "d", "snapshot": "snap"}), encoding="utf-8")
    return run


def test_version_step_is_checked_against_the_system_not_the_adapter(tmp_path: Path):
    # inv: one adapter serves cells frozen at different wheels, so a cell's own version.cli decides
    # what its recorded version output must carry
    bench = _version_only_bench(tmp_path, "0.5.1")
    assert score.records(bench, _version_only_run(tmp_path / "a", "graphify-search 0.5.1\n")) == []
    with pytest.raises(score.UnparsedError, match="does not carry version 0.5.1"):
        score.records(bench, _version_only_run(tmp_path / "b", "graphify-search 0.4.0\n"))


def test_version_step_falls_back_to_the_adapter_when_the_system_declares_none(tmp_path: Path):
    from benchmark.harness.scoring import adapters

    bench = _version_only_bench(tmp_path, None)
    printed = f"graphify-search {adapters.load('graphify_search').VERSION}\n"
    assert score.records(bench, _version_only_run(tmp_path / "a", printed)) == []


