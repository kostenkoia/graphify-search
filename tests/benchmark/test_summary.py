import json
from pathlib import Path

import pytest
import yaml

from benchmark.harness import summary
from tests.benchmark.conftest import references_dir, write_question, write_review

# inv: a completed driven row carries both `model` (what prepare asked for) and `model_served`
# (what collect saw answer); an aborted one, built separately below, carries only `model`
_LOCAL = {"model": "m", "model_served": "m", "effort": "high", "max_actions": 4, "max_tokens": 8192}
_LOCAL_ROUND = "driven/local/m/high/4/8192"
_LEGACY_ROUND = "driven/legacy/claude-code-subagent"


def _question(bench: Path, qid: str, snapshot: str) -> None:
    write_question(bench, qid, {"id": qid, "snapshot": snapshot, "text": "does not matter"})


def _write_ledger(bench: Path, rows: list[dict]) -> None:
    (bench / "record").mkdir(parents=True, exist_ok=True)
    (bench / "record" / "attempts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(run_id: str, question: str, **extra) -> dict:
    base = {
        "run_id": run_id, "question": question, "system": "s", "configuration": "c",
        "attempt": 1, "harness_sha256": "h" * 64, "outcome": "completed", "runner": False,
        "hit": True, "hit_rank": 5, "canonical": ["a", "b"], "tokens": 100,
        "system_calls": 2, "ceiling_calls": 1, "stop": "harness",
    }
    base.update(extra)
    return base


def _driven_row(run_id: str, question: str, **extra) -> dict:
    base = _row(run_id, question, runner=True, stop="answer_met", stop_hit=True, hit_by="runner",
               refused=0, model_usage={"input_tokens": 10, "output_tokens": 5}, **_LOCAL)
    base.update(extra)
    return base


def _legacy_row(run_id: str, question: str, **extra) -> dict:
    base = _row(run_id, question, runner=True, stop="answer_met", driver="claude-code-subagent",
               stop_hit=True, hit_by="runner", refused=0,
               model_usage={"input_tokens": 1, "output_tokens": 1})
    base.update(extra)
    return base


def _one_group(bench: Path, rows: list[dict]) -> dict:
    _write_ledger(bench, rows)
    doc = summary.summarize(bench)
    assert len(doc["groups"]) == 1
    return doc["groups"][0]


def test_round_of_baseline_local_and_legacy():
    assert summary.round_of({"runner": False}) == "baseline"
    assert summary.round_of({"runner": True, **_LOCAL}) == _LOCAL_ROUND
    assert summary.round_of({"runner": True, "driver": "claude-code-subagent"}) == _LEGACY_ROUND


def test_round_of_refuses_a_row_with_no_runner_key():
    with pytest.raises(ValueError, match="runner"):
        summary.round_of({"run_id": "x"})


def test_round_of_an_aborted_driven_row_uses_the_requested_model():
    # inv: prepare writes `model`/`effort`/`max_actions`/`max_tokens` before drive or collect
    # ever run; a row a kill left aborted carries them and nothing collect would have added
    row = {"runner": True, "model": "m", "effort": "high", "max_actions": 4, "max_tokens": 8192}
    assert summary.round_of(row) == _LOCAL_ROUND


def test_round_of_refuses_a_driven_row_with_neither_driver_nor_model():
    with pytest.raises(SystemExit, match="names no driver and no model"):
        summary.round_of({"runner": True})


def test_an_aborted_driven_row_is_classified_and_counted_in_its_local_group(git_bench: Path):
    _question(git_bench, "q1", "snap")
    aborted_row = {
        "run_id": "q1-s-c-a01", "question": "q1", "system": "s", "configuration": "c",
        "attempt": 1, "harness_sha256": "h" * 64, "outcome": "aborted", "runner": True,
        "model": "m", "effort": "high", "max_actions": 4, "max_tokens": 8192,
    }
    _write_ledger(git_bench, [aborted_row, _driven_row("q1-s-c-a02", "q1", attempt=2)])
    doc = summary.summarize(git_bench)
    assert len(doc["groups"]) == 1
    group = doc["groups"][0]
    assert group["round"] == _LOCAL_ROUND
    assert group["aborted"] == 1
    assert group["attempts"] == 1


def test_two_recipes_of_the_same_cell_split_into_two_groups(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [
        _row("q1-graphify-default-a01", "q1", system="graphify", configuration="default",
            harness_sha256="aa" * 32),
        _row("q1-graphify-default-b01", "q1", system="graphify", configuration="default",
            harness_sha256="bb" * 32),
    ])
    doc = summary.summarize(git_bench)
    recipes = {g["harness_sha256"]: g["questions"] for g in doc["groups"]}
    assert recipes == {"aa" * 32: 1, "bb" * 32: 1}


def test_legacy_and_local_driven_rows_stay_in_separate_rounds(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [_legacy_row("q1-s-c-a01", "q1"),
                              _driven_row("q1-s-c-a02", "q1", attempt=2)])
    doc = summary.summarize(git_bench)
    assert {g["round"] for g in doc["groups"]} == {_LEGACY_ROUND, _LOCAL_ROUND}


def test_a_driven_group_scores_the_lowest_attempt_not_the_best(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [
        _driven_row("q1-s-c-a03", "q1", attempt=3, hit=True),
        _driven_row("q1-s-c-a04", "q1", attempt=4, hit=False),
    ])
    assert group["per_question"]["q1"]["attempts"] == ["q1-s-c-a03"]
    assert group["hit"] == 1


def test_repeat_rows_count_as_attempts_but_are_not_scored(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [
        _row("q1-s-c-a01", "q1"),
        _row("q1-s-c-a02", "q1", attempt=2, repeat=True),
    ])
    assert group["attempts"] == 2
    assert group["repeats"] == 1
    assert group["questions"] == 1
    assert group["per_question"]["q1"]["attempts"] == ["q1-s-c-a01"]


def test_a_hit_with_no_rank_publishes_null_hit_rank_and_is_not_counted_in_hit_at_5(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [_row("q1-s-c-a01", "q1", hit=True, hit_rank=None)])
    assert group["per_question"]["q1"]["hit_rank"] is None
    assert group["hit"] == 1
    assert group["hit_at_5"] == 0


def test_hit_at_1_and_hit_at_5_use_their_own_thresholds(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _question(git_bench, "q2", "snap")
    group = _one_group(git_bench, [
        _row("q1-s-c-a01", "q1", hit=True, hit_rank=1),
        _row("q2-s-c-a01", "q2", hit=True, hit_rank=3),
    ])
    assert group["hit_at_1"] == 1
    assert group["hit_at_5"] == 2


def test_hit_requires_every_scored_row_to_hit_not_just_one(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [
        _row("q1-s-c-a01", "q1", hit=True),
        _row("q1-s-c-a02", "q1", attempt=2, hit=False, hit_rank=None),
    ])
    assert group["hit"] == 0
    assert group["per_question"]["q1"]["hit"] is False


def test_hit_at_5_is_false_when_any_scored_rows_rank_exceeds_the_cut(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [
        _row("q1-s-c-a01", "q1", hit=True, hit_rank=3),
        _row("q1-s-c-a02", "q1", attempt=2, hit=True, hit_rank=8),
    ])
    assert group["hit"] == 1
    assert group["hit_at_5"] == 0


def test_hit_at_is_null_when_any_scored_row_carries_no_rank():
    rows = [{"hit_rank": None}, {"hit_rank": 3}]
    assert summary._hit_at(rows, True, 5) is None


def test_hit_at_does_not_depend_on_row_order():
    forward = [{"hit_rank": 3}, {"hit_rank": 8}]
    backward = list(reversed(forward))
    assert summary._hit_at(forward, True, 5) is summary._hit_at(backward, True, 5) is False


def test_added_hit_is_null_when_no_baseline_group_exists(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [_driven_row("q1-s-c-a01", "q1")])
    assert group["per_question"]["q1"]["added_hit"] is None


def test_added_hit_is_true_when_the_drive_hits_where_the_baseline_missed(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [
        _row("q1-s-c-a01", "q1", hit=False, hit_rank=None),
        _driven_row("q1-s-c-a02", "q1", attempt=2, hit=True),
    ])
    doc = summary.summarize(git_bench)
    driven = next(g for g in doc["groups"] if g["round"] == _LOCAL_ROUND)
    assert driven["per_question"]["q1"]["added_hit"] is True


def test_added_hit_is_false_when_the_drive_only_repeats_the_baseline(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [
        _row("q1-s-c-a01", "q1", hit=True),
        _driven_row("q1-s-c-a02", "q1", attempt=2, hit=True),
    ])
    doc = summary.summarize(git_bench)
    driven = next(g for g in doc["groups"] if g["round"] == _LOCAL_ROUND)
    assert driven["per_question"]["q1"]["added_hit"] is False


def test_tokens_median_is_median_low_on_an_even_count(git_bench: Path):
    for qid in ("q1", "q2", "q3", "q4"):
        _question(git_bench, qid, "snap")
    group = _one_group(git_bench, [
        _row("q1-s-c-a01", "q1", tokens=10), _row("q2-s-c-a01", "q2", tokens=20),
        _row("q3-s-c-a01", "q3", tokens=30), _row("q4-s-c-a01", "q4", tokens=40),
    ])
    # inv: median_low of an even-length list is the lower of the two middle values, 20 here,
    # never an interpolated 25 that no run actually recorded
    assert group["tokens_median"] == 20


def test_driven_only_columns_are_null_on_a_baseline_group(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [_row("q1-s-c-a01", "q1")])
    assert group["round"] == "baseline"
    assert group["named"] is None
    assert group["hit_by_runner"] is None
    assert group["refused"] is None
    assert group["model_input_tokens"] is None
    assert group["model_output_tokens"] is None
    assert group["agreement"] == 1


def test_agreement_is_null_on_a_driven_group(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [_driven_row("q1-s-c-a01", "q1")])
    assert group["agreement"] is None
    assert group["named"] == 1
    assert group["hit_by_runner"] == 1
    assert group["model_input_tokens"] == 10
    assert group["model_output_tokens"] == 5


def test_model_usage_sums_are_null_when_every_scored_row_carries_no_usage(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [_driven_row("q1-s-c-a01", "q1", model_usage={})])
    assert group["model_input_tokens"] is None
    assert group["model_output_tokens"] is None


def test_model_usage_sums_add_across_every_scored_row(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _question(git_bench, "q2", "snap")
    group = _one_group(git_bench, [
        _driven_row("q1-s-c-a01", "q1", model_usage={"input_tokens": 10, "output_tokens": 5}),
        _driven_row("q2-s-c-a01", "q2", model_usage={"input_tokens": 20, "output_tokens": 7}),
    ])
    assert group["model_input_tokens"] == 30
    assert group["model_output_tokens"] == 12


def test_agreement_counts_questions_whose_completed_rows_share_one_canonical_list(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _question(git_bench, "q2", "snap")
    _question(git_bench, "q3", "snap")
    group = _one_group(git_bench, [
        _row("q1-s-c-a01", "q1", canonical=["a"]),
        _row("q1-s-c-a02", "q1", attempt=2, canonical=["a"]),
        _row("q2-s-c-a01", "q2", canonical=["a"]),
        _row("q2-s-c-a02", "q2", attempt=2, canonical=["b"]),
        # inv: q3's one scored (non-repeat) row trivially agrees with itself; agreement still
        # reads its repeat's different canonical, because agreement is a fact about every
        # completed attempt, repeats included, not only the ones summary scores
        _row("q3-s-c-a01", "q3", canonical=["a"]),
        _row("q3-s-c-a02", "q3", attempt=2, canonical=["b"], repeat=True),
    ])
    assert group["agreement"] == 1
    assert group["per_question"]["q3"]["attempts"] == ["q3-s-c-a01"]


def test_a_completed_row_carrying_no_canonical_list_counts_as_disagreement(git_bench: Path):
    # inv: two rows that both lack the list have nothing to compare, so they must not read as a
    # match; agreement records that the repeat gate held, never that nobody looked
    _question(git_bench, "q1", "snap")
    rows = [_row("q1-s-c-a01", "q1"), _row("q1-s-c-a02", "q1", attempt=2)]
    for row in rows:
        del row["canonical"]
    group = _one_group(git_bench, rows)
    assert group["agreement"] == 0


def test_ceiling_reached_counts_questions_whose_scored_rows_include_one(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [_row("q1-s-c-a01", "q1", stop="ceiling_reached")])
    assert group["ceiling_reached"] == 1


def test_voided_and_aborted_rows_are_counted_apart_from_attempts(git_bench: Path):
    _question(git_bench, "q1", "snap")
    group = _one_group(git_bench, [
        _row("q1-s-c-a01", "q1"),
        _row("q1-s-c-a02", "q1", attempt=2, outcome="void"),
        _row("q1-s-c-a03", "q1", attempt=3, outcome="aborted"),
    ])
    assert group["attempts"] == 1
    assert group["voided"] == 1
    assert group["aborted"] == 1


def test_seal_reason_is_null_while_unsealed(git_bench: Path):
    assert summary.summarize(git_bench)["seal_reason"] is None


def test_sealed_questions_counts_only_files_that_pass_review(git_bench: Path):
    _question(git_bench, "q1", "snap")
    (references_dir(git_bench, "snap") / "q1.yaml").write_text(
        yaml.safe_dump({"places": []}), encoding="utf-8")
    write_review(git_bench, "q1")
    _question(git_bench, "q2", "snap")
    group = _one_group(git_bench, [_row("q1-s-c-a01", "q1")])
    assert group["sealed_questions"] == 1


def test_render_refuses_when_two_harness_sha256_collide_on_their_table_prefix(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _question(git_bench, "q2", "snap")
    _write_ledger(git_bench, [
        _row("q1-s-c-a01", "q1", harness_sha256="aaaaaaaa" + "1" * 56),
        _row("q2-s-c2-a01", "q2", system="s", configuration="c2", harness_sha256="aaaaaaaa" + "2" * 56),
    ])
    doc = summary.summarize(git_bench)
    with pytest.raises(SystemExit, match="share the table prefix"):
        summary.render(doc)


def test_render_prints_the_seal_reason_bare_and_empty_when_unsealed(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [_row("q1-s-c-a01", "q1")])
    doc = summary.summarize(git_bench)
    assert doc["seal_reason"] is None
    lines = summary.render(doc).splitlines()
    assert lines[2] == "seal reason: "


def test_summarize_is_deterministic_across_two_calls(git_bench: Path):
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [_row("q1-s-c-a01", "q1")])
    first = json.dumps(summary.summarize(git_bench), indent=2, sort_keys=True, ensure_ascii=False)
    second = json.dumps(summary.summarize(git_bench), indent=2, sort_keys=True, ensure_ascii=False)
    assert first == second


def test_main_writes_byte_identical_files_on_a_second_run(git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [_row("q1-s-c-a01", "q1")])
    monkeypatch.setattr(summary, "DEFAULT_BENCHMARK", git_bench)
    assert summary.main([]) == 0
    json_first = (git_bench / "record" / "summary.json").read_bytes()
    md_first = (git_bench / "record" / "SUMMARY.md").read_bytes()
    assert summary.main([]) == 0
    assert (git_bench / "record" / "summary.json").read_bytes() == json_first
    assert (git_bench / "record" / "SUMMARY.md").read_bytes() == md_first
    assert json_first.endswith(b"\n")
    assert md_first.startswith(b"# summary")


def test_main_writes_neither_file_when_rendering_refuses(git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    # inv: the json and the md are written from one rendered string, so a refusal in render
    # cannot leave a fresh json beside a stale table
    _question(git_bench, "q1", "snap")
    _write_ledger(git_bench, [_row("q1-s-c-a01", "q1")])
    monkeypatch.setattr(summary, "DEFAULT_BENCHMARK", git_bench)
    assert summary.main([]) == 0
    json_first = (git_bench / "record" / "summary.json").read_bytes()
    md_first = (git_bench / "record" / "SUMMARY.md").read_bytes()

    def _refuse(doc: dict) -> str:
        raise SystemExit("two hashes share the table prefix")

    monkeypatch.setattr(summary, "render", _refuse)
    _write_ledger(git_bench, [_row("q1-s-c-a01", "q1"), _row("q1-s-c-a02", "q1", attempt=2)])
    with pytest.raises(SystemExit):
        summary.main([])
    assert (git_bench / "record" / "summary.json").read_bytes() == json_first
    assert (git_bench / "record" / "SUMMARY.md").read_bytes() == md_first


def test_main_takes_no_arguments(git_bench: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(summary, "DEFAULT_BENCHMARK", git_bench)
    with pytest.raises(SystemExit):
        summary.main(["--benchmark", str(git_bench)])
