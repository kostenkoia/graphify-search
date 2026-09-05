from pathlib import Path

import pytest
import yaml

from benchmark.harness import config

MINIMAL = (
    "adapter: graphify\n"
    "version: {cli: '0.9.27'}\n"
    "invocation: {package: {launcher: /x/graphify, interpreter: /x/python, site: /x/site}, subcommands: {}}\n"
    "fixed_steps: []\n"
    "default_configuration: default\n"
    "configurations: {default: {index: indexes/graphify, search_mode: null}}\n"
    "sandbox_layout: {graphify-out: '<artifacts>'}\n"
    "environment: {PATH: /usr/bin:/bin}\n"
    "docs: {}\n"
)


def test_load_harness_reads_required_keys(tmp_path: Path):
    sysdir = tmp_path / "systems" / "demo"
    sysdir.mkdir(parents=True)
    (sysdir / "harness.yaml").write_text(MINIMAL, encoding="utf-8")
    h = config.load_harness(tmp_path, "demo")
    assert h.system == "demo"
    assert h.adapter == "graphify"
    assert h.configurations["default"]["index"] == "indexes/graphify"
    assert h.models == {}
    assert h.volatile == []


def test_load_harness_missing_key_fails(tmp_path: Path):
    sysdir = tmp_path / "systems" / "demo"
    sysdir.mkdir(parents=True)
    (sysdir / "harness.yaml").write_text("adapter: graphify\n", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="invocation"):
        config.load_harness(tmp_path, "demo")


def test_harness_status_is_read_and_defaults_to_none(tmp_path):
    bench = tmp_path / "benchmark"
    sysdir = bench / "systems" / "s"
    sysdir.mkdir(parents=True)
    minimal = {
        "adapter": "a", "version": {"cli": "1"}, "invocation": {"package": {}},
        "fixed_steps": [], "default_configuration": "c",
        "configurations": {"c": {"index": "indexes/c"}, "d": {"index": "indexes/d", "status": "declared"}},
        "sandbox_layout": {}, "environment": {}, "docs": {},
    }
    (sysdir / "harness.yaml").write_text(yaml.safe_dump(minimal), encoding="utf-8")
    h = config.load_harness(bench, "s")
    assert h.status is None
    assert h.configurations["d"]["status"] == "declared"
    assert "status" not in h.configurations["c"]

    (sysdir / "harness.yaml").write_text(yaml.safe_dump({**minimal, "status": "reference"}), encoding="utf-8")
    assert config.load_harness(bench, "s").status == "reference"


def test_harness_status_refuses_an_unknown_value(tmp_path):
    bench = tmp_path / "benchmark"
    sysdir = bench / "systems" / "s"
    sysdir.mkdir(parents=True)
    minimal = {
        "adapter": "a", "version": {"cli": "1"}, "invocation": {"package": {}},
        "fixed_steps": [], "default_configuration": "c",
        "configurations": {"c": {"index": "indexes/c"}},
        "sandbox_layout": {}, "environment": {}, "docs": {}, "status": "retired",
    }
    (sysdir / "harness.yaml").write_text(yaml.safe_dump(minimal), encoding="utf-8")
    with pytest.raises(config.ConfigError, match="status"):
        config.load_harness(bench, "s")


# ---------------------------------------------------------------------------
# Path helpers: snapshot_dir, question_path, reference_path, review_path,
# question_ids.
# ---------------------------------------------------------------------------

def _write_question(benchmark: Path, snapshot: str, qid: str) -> Path:
    """Create `record/snapshots/<snapshot>/questions/<qid>.yaml` under `benchmark`."""
    questions_dir = benchmark / "record" / "snapshots" / snapshot / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    path = questions_dir / f"{qid}.yaml"
    path.write_text(f"id: {qid}\nsnapshot: {snapshot}\n", encoding="utf-8")
    return path


class TestSnapshotDir:
    def test_returns_the_snapshot_directory(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        assert config.snapshot_dir(bench, "prefect-3a128c2") == (
            bench / "record" / "snapshots" / "prefect-3a128c2"
        )

    def test_refuses_a_multi_segment_id(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        with pytest.raises(config.ConfigError):
            config.snapshot_dir(bench, "a/b")

    def test_refuses_a_parent_escape(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        with pytest.raises(config.ConfigError):
            config.snapshot_dir(bench, "../x")

    def test_refuses_an_empty_id(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        with pytest.raises(config.ConfigError):
            config.snapshot_dir(bench, "")


class TestQuestionPath:
    def test_resolves_the_one_snapshot_holding_the_qid(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        expected = _write_question(bench, "snap-a", "q001")
        assert config.question_path(bench, "q001") == expected

    @pytest.mark.parametrize("qid", ["../x", "a/b", ""])
    def test_refuses_a_non_bare_id(self, tmp_path: Path, qid: str):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        with pytest.raises(config.ConfigError):
            config.question_path(bench, qid)

    def test_refuses_a_non_bare_id_before_record_snapshots_exists(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        with pytest.raises(config.ConfigError, match="not a bare identifier"):
            config.question_path(bench, "../x")

    def test_refuses_a_non_bare_id_when_snapshots_holds_only_files(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        snapshots_dir = bench / "record" / "snapshots"
        snapshots_dir.mkdir(parents=True)
        (snapshots_dir / "known_transitions.yaml").write_text("[]\n", encoding="utf-8")
        with pytest.raises(config.ConfigError, match="not a bare identifier"):
            config.question_path(bench, "../x")

    def test_refuses_zero_matches_and_names_the_qid(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        with pytest.raises(config.ConfigError, match="q999"):
            config.question_path(bench, "q999")

    def test_refuses_two_matches_and_names_both_files(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        first = _write_question(bench, "snap-a", "q001")
        second = _write_question(bench, "snap-b", "q001")
        with pytest.raises(config.ConfigError) as exc_info:
            config.question_path(bench, "q001")
        message = str(exc_info.value)
        assert str(first) in message
        assert str(second) in message

    @pytest.mark.parametrize("qid", ["q?01", "q*", "q00[0-9]"])
    def test_a_glob_shaped_id_never_matches_a_different_question(self, tmp_path: Path, qid: str):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        with pytest.raises(config.ConfigError, match="no snapshot holds"):
            config.question_path(bench, qid)


class TestReferencePath:
    def test_returns_the_reference_beside_the_question(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        assert config.reference_path(bench, "q001") == (
            bench / "record" / "snapshots" / "snap-a" / "references" / "q001.yaml"
        )

    def test_propagates_question_path_failures(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        with pytest.raises(config.ConfigError):
            config.reference_path(bench, "q001")


class TestReviewPath:
    def test_returns_the_review_file_beside_the_question(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        assert config.review_path(bench, "q001") == (
            bench / "record" / "snapshots" / "snap-a" / "questions" / "review" / "q001.yaml"
        )

    def test_propagates_question_path_failures(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        with pytest.raises(config.ConfigError):
            config.review_path(bench, "q001")


class TestQuestionIds:
    def test_sorted_across_two_snapshots(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q002")
        _write_question(bench, "snap-a", "q001")
        _write_question(bench, "snap-b", "q003")
        assert config.question_ids(bench) == ["q001", "q002", "q003"]

    def test_refuses_a_duplicate_id_across_snapshots(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        _write_question(bench, "snap-b", "q001")
        with pytest.raises(config.ConfigError, match="q001"):
            config.question_ids(bench)

    def test_empty_tree_returns_empty_list(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        assert config.question_ids(bench) == []


class TestLoadQuestion:
    def test_reads_the_question_from_its_snapshot(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        question = config.load_question(bench, "q001")
        assert question["id"] == "q001"
        assert question["snapshot"] == "snap-a"

    def test_refuses_an_escaping_id(self, tmp_path: Path):
        bench = tmp_path / "benchmark"
        _write_question(bench, "snap-a", "q001")
        with pytest.raises(config.ConfigError):
            config.load_question(bench, "../q001")
