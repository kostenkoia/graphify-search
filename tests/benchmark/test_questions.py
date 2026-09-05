import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from benchmark.harness import config, prepare, questions, rules
from tests.benchmark.conftest import (
    _reseal,
    questions_dir,
    references_dir,
    review_dir,
    write_question,
    write_review,
)
from tests.benchmark.test_prepare import _a_runnable_question, seed_question


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _seed_candidates(tmp_path: Path, authored: list[dict], candidates: list[dict],
                      snapshot: str = "snap") -> Path:
    """Build a `benchmark/` with one snapshot's authored candidates; return the benchmark root."""
    bench = tmp_path / "benchmark"
    cand_dir = config.snapshot_dir(bench, snapshot) / "questions" / "candidates"
    cand_dir.mkdir(parents=True)
    _write_jsonl(cand_dir / "authored-1.jsonl", authored)
    _write_jsonl(cand_dir / "candidates.jsonl", candidates)
    graph_dir = config.snapshot_dir(bench, snapshot) / "indexes" / "graphify"
    graph_dir.mkdir(parents=True)
    (graph_dir / "graph.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")
    return bench


def test_author_refuses_a_candidate_that_leaks_its_own_reference(tmp_path: Path):
    authored = [{"n": 1, "verdict": "keep", "reason": None,
                "question": "how does bar_symbol_z9x work", "why": "because bar_symbol_z9x does it"}]
    candidates = [{"n": 1, "path": "pkg/mod.py", "fqname": "pkg.mod.bar_symbol_z9x",
                  "bare": "bar_symbol_z9x", "start": 1, "end": 5}]
    bench = _seed_candidates(tmp_path, authored, candidates)
    with pytest.raises(SystemExit, match="leaks its own reference"):
        questions.author(bench, "snap")
    assert not (config.snapshot_dir(bench, "snap") / "questions" / "q001.yaml").exists()
    assert not (config.snapshot_dir(bench, "snap") / "references" / "q001.yaml").exists()


def test_author_writes_a_question_and_reference_for_a_clean_candidate(tmp_path: Path):
    authored = [{"n": 1, "verdict": "keep", "reason": None,
                "question": "how is the score calculated", "why": "it sums the weighted terms"}]
    candidates = [{"n": 1, "path": "pkg/mod.py", "fqname": "pkg.mod.compute_score",
                  "bare": "compute_score", "start": 1, "end": 5}]
    bench = _seed_candidates(tmp_path, authored, candidates)
    assert questions.author(bench, "snap") == 0
    question = yaml.safe_load(
        (config.snapshot_dir(bench, "snap") / "questions" / "q001.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load(
        (config.snapshot_dir(bench, "snap") / "references" / "q001.yaml").read_text(encoding="utf-8"))
    assert question["text"] == "how is the score calculated"
    assert question["rule"] == "mechanical"
    assert reference["places"][0]["symbol"] == "compute_score"


def test_author_names_the_candidates_file_it_could_not_find(tmp_path: Path):
    # inv: a missing input is named in one line; a traceback out of open() tells the operator
    # nothing about which snapshot's file is missing
    bench = _seed_candidates(tmp_path, [], [])
    with pytest.raises(SystemExit, match="candidates.jsonl"):
        questions.author(bench, "no-such-snapshot")


def test_review_request_refuses_an_id_no_snapshot_holds(tmp_path: Path):
    bench = tmp_path / "benchmark"
    questions_dir(bench, "snap")
    with pytest.raises(SystemExit, match="no snapshot holds question q999"):
        questions.review_request(bench, "q999")


def test_author_leaves_an_identical_question_alone_on_a_second_pass(tmp_path: Path):
    # inv: authoring more questions for a snapshot is an append, so a re-author of an unchanged
    # candidate must not rewrite the file whose sha256 every recorded row pins
    authored = [{"n": 1, "verdict": "keep", "reason": None,
                "question": "how is the score calculated", "why": "it sums the weighted terms"}]
    candidates = [{"n": 1, "path": "pkg/mod.py", "fqname": "pkg.mod.compute_score",
                  "bare": "compute_score", "start": 1, "end": 5}]
    bench = _seed_candidates(tmp_path, authored, candidates)
    assert questions.author(bench, "snap") == 0
    before = {name: (config.snapshot_dir(bench, "snap") / name / "q001.yaml").stat().st_mtime_ns
              for name in ("questions", "references")}
    assert questions.author(bench, "snap") == 0
    assert {name: (config.snapshot_dir(bench, "snap") / name / "q001.yaml").stat().st_mtime_ns
            for name in ("questions", "references")} == before


def test_author_refuses_to_rewrite_a_question_whose_bytes_would_change(tmp_path: Path):
    authored = [{"n": 1, "verdict": "keep", "reason": None,
                "question": "how is the score calculated", "why": "it sums the weighted terms"}]
    candidates = [{"n": 1, "path": "pkg/mod.py", "fqname": "pkg.mod.compute_score",
                  "bare": "compute_score", "start": 1, "end": 5}]
    bench = _seed_candidates(tmp_path, authored, candidates)
    assert questions.author(bench, "snap") == 0
    snap_dir = config.snapshot_dir(bench, "snap")
    reference_before = (snap_dir / "references" / "q001.yaml").read_bytes()
    (snap_dir / "questions" / "q001.yaml").write_text("id: q001\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="questions/q001.yaml"):
        questions.author(bench, "snap")
    assert (snap_dir / "questions" / "q001.yaml").read_text(encoding="utf-8") == "id: q001\n"
    assert (snap_dir / "references" / "q001.yaml").read_bytes() == reference_before


def test_author_skips_a_candidate_the_authoring_pass_did_not_keep(tmp_path: Path):
    authored = [{"n": 1, "verdict": "skip", "reason": "cannot be described without naming it",
                "question": None, "why": None}]
    candidates = [{"n": 1, "path": "pkg/mod.py", "fqname": "pkg.mod.x", "bare": "x", "start": 1, "end": 2}]
    bench = _seed_candidates(tmp_path, authored, candidates)
    assert questions.author(bench, "snap") == 0
    snap_dir = config.snapshot_dir(bench, "snap")
    assert list((snap_dir / "questions").glob("q*.yaml")) == []
    assert list((snap_dir / "references").glob("q*.yaml")) == []


def test_author_numbers_from_one_past_the_highest_id_across_all_snapshots(tmp_path: Path):
    # why: q9 and q10 sort as strings with q9 after q10, so a numbering rule that trusted sort
    # order over the parsed number would allocate q10 again here instead of q11
    bench = tmp_path / "benchmark"
    write_question(bench, "q9", {"id": "q9", "snapshot": "old-a", "text": "already authored"})
    write_question(bench, "q10", {"id": "q10", "snapshot": "old-b", "text": "already authored too"})
    authored = [{"n": 1, "verdict": "keep", "reason": None,
                "question": "how is the score calculated", "why": "it sums the weighted terms"}]
    candidates = [{"n": 1, "path": "pkg/mod.py", "fqname": "pkg.mod.compute_score",
                  "bare": "compute_score", "start": 1, "end": 5}]
    _seed_candidates(tmp_path, authored, candidates, snapshot="snap")
    assert questions.author(bench, "snap") == 0
    snap_dir = config.snapshot_dir(bench, "snap")
    assert (snap_dir / "questions" / "q011.yaml").is_file()
    assert not (snap_dir / "questions" / "q001.yaml").exists()
    assert not (snap_dir / "questions" / "q010.yaml").exists()


def test_review_request_writes_the_question_and_reference_verbatim(tmp_path: Path):
    bench = tmp_path / "benchmark"
    (questions_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump({"id": "q001", "snapshot": "snap", "text": "how does x work"}), encoding="utf-8")
    (references_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump({"id": "q001", "places": [{"path": "a.py", "symbol": "x"}]}), encoding="utf-8")
    assert questions.review_request(bench, "q001") == 0
    request = yaml.safe_load(
        (review_dir(bench, "snap") / "q001.request.yaml").read_text(encoding="utf-8"))
    assert request["qid"] == "q001"
    assert request["snapshot"] == "snap"
    assert request["question"] == "how does x work"
    assert request["question_sha256"] == rules.sha256_file(questions_dir(bench, "snap") / "q001.yaml")
    assert request["reference_sha256"] == rules.sha256_file(references_dir(bench, "snap") / "q001.yaml")
    assert request["instruction"] == questions.REVIEW_INSTRUCTION


def _seed_request(tmp_path: Path) -> tuple[Path, dict]:
    bench = tmp_path / "benchmark"
    (questions_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump({"id": "q001", "snapshot": "snap", "text": "how does x work"}), encoding="utf-8")
    (references_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump({"id": "q001", "places": [{"path": "a.py", "symbol": "x"}]}), encoding="utf-8")
    questions.review_request(bench, "q001")
    request = yaml.safe_load(
        (review_dir(bench, "snap") / "q001.request.yaml").read_text(encoding="utf-8"))
    return bench, request


def test_check_review_refuses_a_missing_review(tmp_path: Path):
    bench, _ = _seed_request(tmp_path)
    assert questions.check_review(bench, "q001") == \
        "question q001 has no review; a question runs only after review"


def test_check_review_passes_a_review_matching_the_request(tmp_path: Path):
    bench, request = _seed_request(tmp_path)
    review = {"reference_is_right": True, "question_is_ambiguous": False, "note": None,
              "reviewer_model": "opus", "question_sha256": request["question_sha256"],
              "reference_sha256": request["reference_sha256"]}
    (review_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    assert questions.check_review(bench, "q001") is None


def test_check_review_refuses_a_tampered_question_file_after_a_passing_review(tmp_path: Path):
    # inv: a request's hashes are frozen at request time, so the gate must also recheck them
    # against the live files, or an edit made after a passing review would slip through
    bench, request = _seed_request(tmp_path)
    review = {"reference_is_right": True, "question_is_ambiguous": False, "note": None,
              "reviewer_model": "opus", "question_sha256": request["question_sha256"],
              "reference_sha256": request["reference_sha256"]}
    (review_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    assert questions.check_review(bench, "q001") is None
    (questions_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump({"id": "q001", "snapshot": "snap", "text": "how does x work now"}), encoding="utf-8")
    assert questions.check_review(bench, "q001") == \
        "question q001 review hashes do not match the question or reference"


def test_check_review_refuses_a_tampered_reference_file_after_a_passing_review(tmp_path: Path):
    bench, request = _seed_request(tmp_path)
    review = {"reference_is_right": True, "question_is_ambiguous": False, "note": None,
              "reviewer_model": "opus", "question_sha256": request["question_sha256"],
              "reference_sha256": request["reference_sha256"]}
    (review_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    assert questions.check_review(bench, "q001") is None
    (references_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump({"id": "q001", "places": [{"path": "b.py", "symbol": "y"}]}), encoding="utf-8")
    assert questions.check_review(bench, "q001") == \
        "question q001 review hashes do not match the question or reference"


def test_check_review_refuses_a_hash_mismatch(tmp_path: Path):
    bench, request = _seed_request(tmp_path)
    review = {"reference_is_right": True, "question_is_ambiguous": False, "note": None,
              "reviewer_model": "opus", "question_sha256": "not-the-real-hash",
              "reference_sha256": request["reference_sha256"]}
    (review_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    assert questions.check_review(bench, "q001") == \
        "question q001 review hashes do not match the question or reference"


def test_check_review_refuses_a_missing_reviewer_model(tmp_path: Path):
    bench, request = _seed_request(tmp_path)
    review = {"reference_is_right": True, "question_is_ambiguous": False, "note": None,
              "reviewer_model": "", "question_sha256": request["question_sha256"],
              "reference_sha256": request["reference_sha256"]}
    (review_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    assert questions.check_review(bench, "q001") == "question q001 review lacks reviewer_model"


def test_check_review_grandfathered_case_has_no_request_file(tmp_path: Path):
    bench = tmp_path / "benchmark"
    (questions_dir(bench, "snap") / "q050.yaml").write_text("id: q050\n", encoding="utf-8")
    (references_dir(bench, "snap") / "q050.yaml").write_text("id: q050\n", encoding="utf-8")
    write_review(bench, "q050")
    assert not (review_dir(bench, "snap") / "q050.request.yaml").exists()
    assert questions.check_review(bench, "q050") is None


def test_check_review_grandfathered_case_refuses_a_tampered_question_file(tmp_path: Path):
    bench = tmp_path / "benchmark"
    (questions_dir(bench, "snap") / "q050.yaml").write_text("id: q050\n", encoding="utf-8")
    (references_dir(bench, "snap") / "q050.yaml").write_text("id: q050\n", encoding="utf-8")
    write_review(bench, "q050")
    assert questions.check_review(bench, "q050") is None
    (questions_dir(bench, "snap") / "q050.yaml").write_text(
        "id: q050\ntext: edited\n", encoding="utf-8")
    assert questions.check_review(bench, "q050") == \
        "question q050 review hashes do not match the question or reference"


def test_check_review_grandfathered_case_refuses_a_tampered_reference_file(tmp_path: Path):
    bench = tmp_path / "benchmark"
    (questions_dir(bench, "snap") / "q050.yaml").write_text("id: q050\n", encoding="utf-8")
    (references_dir(bench, "snap") / "q050.yaml").write_text("id: q050\n", encoding="utf-8")
    write_review(bench, "q050")
    assert questions.check_review(bench, "q050") is None
    (references_dir(bench, "snap") / "q050.yaml").write_text(
        "id: q050\nplaces: []\n", encoding="utf-8")
    assert questions.check_review(bench, "q050") == \
        "question q050 review hashes do not match the question or reference"


def _seed_review_gate(bench: Path, sealed_bench: Path) -> tuple[str, Path]:
    """Seed `sealed_bench` with a runnable question and reference, without any review."""
    qid, snapshot, index = _a_runnable_question(bench)
    shutil.copytree(bench / "systems" / "graphify", sealed_bench / "systems" / "graphify")
    seed_question(bench, sealed_bench, qid, snapshot, index)
    _reseal(sealed_bench)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "commit", "-qm", "fixtures"], check=True)
    return qid, sealed_bench.parent / "tmp"


def test_prepare_refuses_a_question_with_no_review(bench: Path, sealed_bench: Path):
    qid, tmp_root = _seed_review_gate(bench, sealed_bench)
    with pytest.raises(SystemExit, match="has no review"):
        prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=False)
    assert list(tmp_root.rglob("run.yaml")) == []
    assert prepare.ledger.rows(sealed_bench) == []


def test_prepare_refuses_a_review_that_withdraws_the_question(bench: Path, sealed_bench: Path):
    qid, tmp_root = _seed_review_gate(bench, sealed_bench)
    write_review(sealed_bench, qid)
    review_path = config.review_path(sealed_bench, qid)
    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    review["reference_is_right"] = False
    review_path.write_text(yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    _reseal(sealed_bench)
    with pytest.raises(SystemExit, match="withdrawn"):
        prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=False)
    assert list(tmp_root.rglob("run.yaml")) == []
    assert prepare.ledger.rows(sealed_bench) == []


def test_prepare_runs_a_question_whose_review_passes(bench: Path, sealed_bench: Path):
    qid, tmp_root = _seed_review_gate(bench, sealed_bench)
    write_review(sealed_bench, qid)
    _reseal(sealed_bench)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(sealed_bench.parent), "commit", "-qm", "review"], check=True)
    try:
        run_dir = prepare.prepare(sealed_bench, "graphify", qid, None, tmp_root, record=True)
    finally:
        prepare.release_lock(tmp_root)
    assert (run_dir / "journal.jsonl").is_file()


def test_review_check_cli_exits_zero_for_a_reviewed_question(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bench, request = _seed_request(tmp_path)
    review = {"reference_is_right": True, "question_is_ambiguous": False, "note": None,
              "reviewer_model": "claude-opus-5", "question_sha256": request["question_sha256"],
              "reference_sha256": request["reference_sha256"]}
    (review_dir(bench, "snap") / "q001.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(rules, "_ROOT", bench)
    assert questions.main(["review-check", "q001"]) == 0


def test_review_check_cli_exits_one_for_an_unreviewed_question(
        tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    bench, _ = _seed_request(tmp_path)
    monkeypatch.setattr(rules, "_ROOT", bench)
    assert questions.main(["review-check", "q001"]) == 1
    assert "has no review" in capsys.readouterr().err


def test_author_cli_exits_one_for_a_qid_two_snapshots_hold(
        tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    # why: the verb is a command line, so a duplicated id leaves as one named line rather than as
    # a traceback out of question_ids
    authored = [{"n": 1, "verdict": "keep", "reason": None,
                "question": "how is the score calculated", "why": "it sums the weighted terms"}]
    candidates = [{"n": 1, "path": "pkg/mod.py", "fqname": "pkg.mod.compute_score",
                  "bare": "compute_score", "start": 1, "end": 5}]
    bench = _seed_candidates(tmp_path, authored, candidates)
    for snapshot in ("old-a", "old-b"):
        write_question(bench, "q001",
                       {"id": "q001", "snapshot": snapshot, "text": "authored under both"})
    monkeypatch.setattr(rules, "_ROOT", bench)
    assert questions.main(["author", "snap"]) == 1
    err = capsys.readouterr().err
    assert "question q001 found in more than one snapshot" in err
    assert "Traceback" not in err


def test_review_check_cli_exits_one_for_an_id_no_snapshot_holds(
        tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    # inv: the gate answers one question -- may this id run -- so an id that names no question
    # leaves as a refusal with an exit code, not as a traceback out of the resolver
    bench, _ = _seed_request(tmp_path)
    monkeypatch.setattr(rules, "_ROOT", bench)
    assert questions.main(["review-check", "q999"]) == 1
    assert "no snapshot holds question q999" in capsys.readouterr().err
