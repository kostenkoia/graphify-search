import json
from pathlib import Path

import pytest
import yaml

from benchmark.harness import report
from tests.benchmark.conftest import snapshot_dir, write_question, write_review

HARNESS_YAML = (
    "adapter: graphify\n"
    "version: {cli: '1.2.3'}\n"
    "invocation: {package: {launcher: /x/graphify, interpreter: /x/python, site: /x/site}, subcommands: {}}\n"
    "fixed_steps: []\n"
    "default_configuration: cfg1\n"
    "configurations: {cfg1: {index: indexes/sys1, search_mode: null}}\n"
    "sandbox_layout: {out: '<artifacts>'}\n"
    "environment: {PATH: /usr/bin:/bin}\n"
    "docs: {}\n"
)

SNAPSHOT = "snap1"


def _bench(tmp_path: Path, snapshot: str = SNAPSHOT) -> Path:
    b = tmp_path / "benchmark"
    (b / "systems" / "sys1").mkdir(parents=True)
    (b / "record").mkdir(exist_ok=True)
    (b / "record" / "attempts.jsonl").write_text("", encoding="utf-8")
    (b / "systems" / "sys1" / "harness.yaml").write_text(HARNESS_YAML, encoding="utf-8")
    snapshot_dir(b, snapshot)
    return b


def _sealed_question(bench: Path, qid: str, snapshot: str = SNAPSHOT) -> None:
    write_question(bench, qid, {"id": qid, "snapshot": snapshot, "text": "does not matter"},
                   {"answer": []})
    write_review(bench, qid)


def _row(run_id: str, question: str, **extra: object) -> dict:
    base = {
        "run_id": run_id, "question": question, "system": "sys1", "configuration": "cfg1",
        "attempt": 1, "harness_sha256": "a" * 64, "instrument_sha256": None,
        "outcome": "completed", "runner": False,
        "hit": True, "hit_rank": 1, "canonical": ["x"], "tokens": 10,
        "system_calls": 1, "ceiling_calls": 1, "stop": "harness",
    }
    base.update(extra)
    return base


def _write_ledger(bench: Path, rows: list[dict]) -> None:
    (bench / "record" / "attempts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _one_group_bench(tmp_path: Path, n: int = 3, snapshot: str = SNAPSHOT) -> Path:
    """A `sys1`/`cfg1`/`baseline` group of `n` fully sealed, fully run, hit questions."""
    b = _bench(tmp_path, snapshot)
    qids = [f"q{i:03d}" for i in range(1, n + 1)]
    for qid in qids:
        _sealed_question(b, qid, snapshot)
    _write_ledger(b, [_row(f"{qid}-sys1-cfg1-a01", qid) for qid in qids])
    return b


# --- unknown column / unknown group -----------------------------------------------------------

def test_unknown_column_is_refused(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "{{no_such_column sys1 cfg1 baseline}}")
    assert any("unknown column" in p for p in problems)


def test_unknown_group_is_refused(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "{{hit no-such-system cfg1 baseline}}")
    assert any("unknown group" in p for p in problems)


def test_the_unknown_group_message_names_the_snapshot_it_looked_under(tmp_path: Path):
    # inv: a placeholder resolves only among one snapshot's groups, so the refusal has to say
    # which snapshot it looked under or it reads as "this group exists nowhere"
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "{{hit no-such-system cfg1 baseline}}")
    assert any(f"unknown group under {SNAPSHOT}" in p for p in problems)


def test_unknown_snapshot_is_an_unknown_group(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, "no-such-snapshot", "{{hit sys1 cfg1 baseline}}")
    assert any("unknown group" in p for p in problems)


# --- ambiguity and disambiguation -------------------------------------------------------------

def _two_recipe_bench(tmp_path: Path) -> Path:
    """Two `sys1`/`cfg1`/`baseline` groups, differing only in `harness_sha256`."""
    b = _bench(tmp_path)
    _sealed_question(b, "q001")
    _write_ledger(b, [
        _row("q001-sys1-cfg1-a01", "q001", harness_sha256="a" * 64),
        _row("q001-sys1-cfg1-b01", "q001", harness_sha256="b" * 64),
    ])
    return b


def test_ambiguous_match_is_refused_without_a_recipe_prefix(tmp_path: Path):
    b = _two_recipe_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline}}")
    assert any("ambiguous" in p for p in problems)


def test_ambiguous_message_names_the_recipe_fix_when_one_would_work(tmp_path: Path):
    b = _two_recipe_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline}}")
    assert any("add @<seal8> or /<recipe8>" in p for p in problems)


def test_recipe_prefix_disambiguates_two_groups(tmp_path: Path):
    b = _two_recipe_bench(tmp_path)
    assert report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline @/bbbbbbbb}}") == []
    assert report.render(b, SNAPSHOT, "{{hit sys1 cfg1 baseline @/bbbbbbbb}}") == "1"


def test_seal_prefix_matches_by_instrument_sha256(tmp_path: Path):
    b = _bench(tmp_path)
    _sealed_question(b, "q001")
    _write_ledger(b, [
        _row("q001-sys1-cfg1-a01", "q001", instrument_sha256="c" * 64),
        _row("q001-sys1-cfg1-b01", "q001", instrument_sha256="d" * 64, harness_sha256="b" * 64),
    ])
    assert report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline @cccccccc}}") == []
    assert report.render(b, SNAPSHOT, "{{hit sys1 cfg1 baseline @cccccccc}}") == "1"


def test_two_snapshots_sharing_every_hash_are_disambiguated_by_snapshot(tmp_path: Path):
    """Two snapshots' groups that would collide on every other field never reach ambiguity."""
    b = _bench(tmp_path, "snapA")
    snapshot_dir(b, "snapB")
    _sealed_question(b, "q001", "snapA")
    _sealed_question(b, "q002", "snapB")
    _write_ledger(b, [
        _row("q001-sys1-cfg1-a01", "q001", harness_sha256="a" * 64),
        _row("q002-sys1-cfg1-a01", "q002", harness_sha256="a" * 64),
    ])
    assert report.check(b, "snapA", "{{hit sys1 cfg1 baseline}}") == []
    assert report.render(b, "snapA", "{{hit sys1 cfg1 baseline}}") == "1"
    assert report.render(b, "snapB", "{{hit sys1 cfg1 baseline}}") == "1"


# --- partial -------------------------------------------------------------------------------

def test_partial_group_is_refused_without_the_partial_token(tmp_path: Path):
    b = _bench(tmp_path)
    for qid in ("q001", "q002", "q003"):
        _sealed_question(b, qid)
    _write_ledger(b, [_row("q001-sys1-cfg1-a01", "q001")])
    problems = report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline}}")
    assert any("partial" in p for p in problems)


def test_partial_group_renders_value_of_questions_run_of_sealed(tmp_path: Path):
    b = _bench(tmp_path)
    for qid in ("q001", "q002", "q003"):
        _sealed_question(b, qid)
    _write_ledger(b, [_row("q001-sys1-cfg1-a01", "q001")])
    assert report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline partial}}") == []
    assert report.render(b, SNAPSHOT, "{{hit sys1 cfg1 baseline partial}}") == "1 of 1 run of 3"


# --- null column (I3) -----------------------------------------------------------------------

def test_a_null_column_on_the_matched_group_is_refused(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    # inv: `named` is only populated for a driven round; a baseline group's `named` is None
    problems = report.check(b, SNAPSHOT, "{{named sys1 cfg1 baseline}}")
    assert any("is null" in p for p in problems)
    with pytest.raises(SystemExit):
        report.render(b, SNAPSHOT, "{{named sys1 cfg1 baseline}}")


# --- Nd / No characters outside backticks ------------------------------------------------------

def test_a_decimal_digit_outside_backticks_is_refused(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "The system answered 3 questions.")
    assert any("outside backticks" in p for p in problems)


@pytest.mark.parametrize("char", ["²", "½"])
def test_a_no_category_character_outside_backticks_is_refused(tmp_path: Path, char: str):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, f"The system answered {char} of the questions.")
    assert any("outside backticks" in p for p in problems)


def test_digits_inside_placeholders_are_exempt(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    assert report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline}}") == []


def test_the_outside_backtick_message_windows_the_offending_character(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    prefix = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt "
    prefix += "ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation "
    template = prefix + "and 7 hits total, more filler text after it to pad the paragraph out further."
    assert len(template) > 200
    problems = report.check(b, SNAPSHOT, template)
    matches = [p for p in problems if "outside backticks" in p]
    assert len(matches) == 1
    assert "7" in matches[0]


# --- backtick literal set ------------------------------------------------------------------

def test_a_backticked_digit_not_in_the_literal_set_is_refused(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "It answered `42` times.")
    assert any("backtick digit" in p for p in problems)


@pytest.mark.parametrize("content", ["x²", "½"])
def test_a_no_category_character_inside_backticks_is_refused(tmp_path: Path, content: str):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, f"It answered `{content}` times.")
    assert any("backtick digit" in p for p in problems)


def test_q_stem_in_backticks_passes(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    assert report.check(b, SNAPSHOT, "See `q001`.") == []


def test_hyphen_range_of_two_q_stems_in_backticks_passes(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    assert report.check(b, SNAPSHOT, "Covers `q001-q003`.") == []


def test_en_dash_range_of_two_q_stems_in_backticks_passes(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    assert report.check(b, SNAPSHOT, "Covers `q001–q003`.") == []


def test_systems_directory_name_in_backticks_passes(tmp_path: Path):
    b = _bench(tmp_path)
    (b / "systems" / "sys2x9").mkdir()
    assert report.check(b, SNAPSHOT, "Ran on `sys2x9`.") == []


def test_configurations_key_in_backticks_passes(tmp_path: Path):
    b = _bench(tmp_path)
    (b / "systems" / "sys9").mkdir()
    (b / "systems" / "sys9" / "harness.yaml").write_text(
        HARNESS_YAML.replace("cfg1", "cfg9"), encoding="utf-8")
    assert report.check(b, SNAPSHOT, "Configuration `cfg9`.") == []


def test_version_cli_in_backticks_passes(tmp_path: Path):
    b = _bench(tmp_path)
    assert report.check(b, SNAPSHOT, "Version `1.2.3`.") == []


def test_harness_models_map_key_in_backticks_passes(tmp_path: Path):
    """The `models:` literal class reads a `harness.yaml` map, never a `models/` directory."""
    b = _bench(tmp_path)
    with_models = HARNESS_YAML + (
        "models:\n"
        "  models--org--m7b:\n"
        "    files: {}\n"
        "    links: {}\n"
    )
    (b / "systems" / "sys1" / "harness.yaml").write_text(with_models, encoding="utf-8")
    assert not (b / "systems" / "sys1" / "models").exists()
    assert not (b / "models").exists()
    assert report.check(b, SNAPSHOT, "Model `org--m7b`.") == []


def test_ledger_model_served_in_backticks_passes(tmp_path: Path):
    b = _bench(tmp_path)
    _sealed_question(b, "q001")
    _write_ledger(b, [_row("q001-sys1-cfg1-a01", "q001", runner=True, model="m7", model_served="m7",
                          effort="high", max_actions=4, max_tokens=8192)])
    assert report.check(b, SNAPSHOT, "Served by `m7`.") == []


def test_ledger_run_id_in_backticks_passes(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    assert report.check(b, SNAPSHOT, "Attempt `q001-sys1-cfg1-a01`.") == []


def test_seven_or_more_hex_characters_in_backticks_passes(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    assert report.check(b, SNAPSHOT, "Hash `abcdef1`.") == []


def test_a_run_of_seven_decimal_digits_in_backticks_is_refused(tmp_path: Path):
    # inv: an abbreviated id carries at least one a-f, so an all-decimal run is a hand-typed
    # figure, not an id, however long it is
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "Hash `1234567`.")
    assert any("backtick digit" in p for p in problems)


# --- number words outside backticks ---------------------------------------------------------

@pytest.mark.parametrize("phrase", ["Twenty questions ran.", "Half of them hit."])
def test_a_number_word_outside_backticks_is_refused(tmp_path: Path, phrase: str):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, phrase)
    assert any("number word" in p for p in problems)


@pytest.mark.parametrize("phrase", ["A percentage of them hit.", "It ran halfway through."])
def test_a_non_exact_number_word_form_passes(tmp_path: Path, phrase: str):
    b = _one_group_bench(tmp_path)
    assert report.check(b, SNAPSHOT, phrase) == []


# --- malformed placeholders (M1) -------------------------------------------------------------

def test_too_few_tokens_is_a_malformed_placeholder(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "{{hit sys1 cfg1}}")
    assert any("malformed placeholder" in p for p in problems)


def test_an_unrecognised_trailing_token_is_a_malformed_placeholder(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "{{hit sys1 cfg1 baseline bogus}}")
    assert any("malformed placeholder" in p for p in problems)


# --- duplicate occurrences are not deduplicated (M4) ------------------------------------------

def test_two_occurrences_of_the_same_refusal_both_appear(tmp_path: Path):
    b = _one_group_bench(tmp_path)
    problems = report.check(b, SNAPSHOT, "It answered `42` here and `42` there.")
    matches = [p for p in problems if "backtick digit" in p]
    assert len(matches) == 2


# --- draft / render CLI plumbing -----------------------------------------------------------

def test_draft_writes_md_in_only_when_the_check_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    b = _one_group_bench(tmp_path, n=3)
    monkeypatch.setattr(report, "DEFAULT_BENCHMARK", b)
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: "{{hit sys1 cfg1 baseline}}")})())
    assert report.main(["draft", SNAPSHOT]) == 0
    out = b / "record" / "reports" / f"{SNAPSHOT}.md.in"
    assert out.read_text(encoding="utf-8") == "{{hit sys1 cfg1 baseline}}"


def test_draft_refuses_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    b = _one_group_bench(tmp_path, n=3)
    monkeypatch.setattr(report, "DEFAULT_BENCHMARK", b)
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: "3 hits")})())
    with pytest.raises(SystemExit):
        report.main(["draft", SNAPSHOT])
    assert not (b / "record" / "reports").exists()


def test_render_writes_md_with_every_placeholder_substituted(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    in_dir = b / "record" / "reports"
    in_dir.mkdir(parents=True)
    (in_dir / f"{SNAPSHOT}.md.in").write_text("hit: {{hit sys1 cfg1 baseline}}", encoding="utf-8")
    report._render_snapshot(b, SNAPSHOT)
    assert (in_dir / f"{SNAPSHOT}.md").read_text(encoding="utf-8") == "hit: 3"


def test_render_refuses_and_writes_no_md(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    in_dir = b / "record" / "reports"
    in_dir.mkdir(parents=True)
    (in_dir / f"{SNAPSHOT}.md.in").write_text("{{no_such_column sys1 cfg1 baseline}}", encoding="utf-8")
    with pytest.raises(SystemExit):
        report._render_snapshot(b, SNAPSHOT)
    assert not (in_dir / f"{SNAPSHOT}.md").exists()


# --- snapshot directory / missing draft (M5) --------------------------------------------------

def test_draft_refuses_when_the_snapshot_has_no_directory(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    with pytest.raises(SystemExit, match="no-such-snapshot"):
        report._draft(b, "no-such-snapshot")
    assert not (b / "record" / "reports").exists()


@pytest.mark.parametrize("snapshot", ["../questions", ".hidden", "a/b"])
def test_a_snapshot_that_is_not_a_bare_name_is_refused(tmp_path: Path, snapshot: str):
    # inv: the snapshot names one directory under record/snapshots/, so a separator or a
    # leading dot would steer report's own output out of record/reports/
    b = _one_group_bench(tmp_path, n=3)
    with pytest.raises(SystemExit, match="not a snapshot name"):
        report._draft(b, snapshot)
    assert not (b / "record" / "reports").exists()


def test_render_refuses_when_the_snapshot_has_no_directory(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    with pytest.raises(SystemExit, match="no-such-snapshot"):
        report._render_snapshot(b, "no-such-snapshot")


def test_render_refuses_with_a_sentence_when_no_draft_exists(tmp_path: Path):
    b = _one_group_bench(tmp_path, n=3)
    with pytest.raises(SystemExit, match="no draft"):
        report._render_snapshot(b, SNAPSHOT)


# --- a frozen weights name reads as a name, not a figure -------------------------------------

_MINILM = "sentence-transformers--all-MiniLM-L6-v2"


def test_a_frozen_weights_name_in_backticks_passes(tmp_path: Path):
    # inv: a weights directory name carries digits, so it must reach the closed literal set from
    # the system's own `models:` keys or every report quoting one would be refused as a figure
    b = _one_group_bench(tmp_path, n=1)
    harness = yaml.safe_load(HARNESS_YAML)
    harness["models"] = {f"models--{_MINILM}": {"revision": "abc"}}
    (b / "systems" / "sys1" / "harness.yaml").write_text(yaml.safe_dump(harness), encoding="utf-8")
    assert report.check(b, SNAPSHOT, f"weights `{_MINILM}`") == []
