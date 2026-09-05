import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchmark.harness import config, expand

PY_EXE = sys.executable


def test_content_words_drop_service_words_and_keep_question_order():
    assert expand.content_words("How is the billing score calculated?") == [
        "billing", "score", "calculated",
    ]


def test_content_words_drop_a_word_too_short_to_carry_a_subject():
    assert expand.content_words("score by ml") == ["score"]


def test_content_words_drop_a_word_seen_twice():
    assert expand.content_words("score and score") == ["score"]


def test_code_review_graph_gets_the_question_words_unchanged():
    got = expand.expand("how is the billing score calculated", set(), 12)
    assert got["code-review-graph"] == ["billing", "score", "calculated"]


def test_graphify_keeps_only_tokens_of_the_vocabulary():
    got = expand.expand("score invoice", {"score"}, 12)
    assert got["graphify"] == ["score"]


def test_graphify_pulls_the_same_stem_out_of_the_vocabulary():
    got = expand.expand("calculate", {"calculated", "calculation"}, 12)
    assert got["graphify"] == ["calculated", "calculation"]


def test_the_exact_word_precedes_a_relative_of_the_same_length():
    # why: length alone would put "scored" first, so only the exact-match rule can order these
    got = expand.expand("scorer", {"scored", "scorer"}, 12)
    assert got["graphify"][0] == "scorer"


def test_a_token_of_another_stem_stays_out():
    got = expand.expand("score", {"score", "unrelated"}, 12)
    assert got["graphify"] == ["score"]


def test_three_shared_letters_are_not_enough_for_a_relative():
    got = expand.expand("score", {"score", "scanner"}, 12)
    assert got["graphify"] == ["score"]


def test_slots_go_round_the_question_words():
    vocab = {"score", "scorer", "scoring", "scoreboard", "billing"}
    got = expand.expand("score billing", vocab, 2)
    assert got["graphify"] == ["score", "billing"]


def test_expansion_stops_at_the_cap():
    vocab = {"score", "scorer", "scoring", "scoreboard"}
    assert expand.expand("score", vocab, 3)["graphify"] == ["score", "scorer", "scoring"]


def test_relatives_of_equal_length_are_ordered_alphabetically():
    # why: eight equal-length tokens tie on every other key, so only the alphabet can order
    # them; a result read out of the set instead would land in this order once in 8! draws
    vocab = {f"score{letter}" for letter in "hbgcfdea"}
    got = expand.expand("score", vocab, 12)
    assert got["graphify"] == sorted(vocab)


def test_expand_answers_for_both_systems_and_no_others():
    assert set(expand.expand("score", set(), 12)) == {"code-review-graph", "graphify"}


def test_mismatches_reports_a_system_the_rule_does_not_cover():
    question = {"id": "q999", "text": "score", "rule": "mechanical",
                "expansion": {"invented": {"tokens": ["score"]}}}
    found = expand.mismatches(question, {"score"}, 12)
    assert len(found) == 1
    assert "invented" in found[0]


def test_vocabulary_matches_the_pinned_extract_script(tmp_path: Path, bench: Path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    labels = ["renderInvoice", "HTTPServerConfig", "a_v2_scope", "ab", "score",
              "Ünicode", "v3", "a" * 31]
    (out / "graph.json").write_text(json.dumps({"nodes": [{"label": lbl} for lbl in labels]}),
                                    encoding="utf-8")
    script = bench / "systems" / "graphify" / "docs" / "scripts" / "vocab_extract.py"
    proc = subprocess.run([PY_EXE, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    pinned = set((out / ".vocab.txt").read_text(encoding="utf-8").split("\n"))
    assert expand.vocabulary(out / "graph.json") == pinned


def test_mismatches_names_the_system_whose_tokens_were_edited():
    question = {"id": "q999", "text": "score", "rule": "mechanical",
                "expansion": {"graphify": {"tokens": ["scorer"]},
                              "code-review-graph": {"tokens": ["score"]}}}
    found = expand.mismatches(question, {"score", "scorer"}, 12)
    assert len(found) == 1
    assert "graphify" in found[0]


def test_mismatches_stays_silent_on_a_question_the_rule_produced():
    question = {"id": "q999", "text": "score", "rule": "mechanical",
                "expansion": {"graphify": {"tokens": ["score", "scorer"]},
                              "code-review-graph": {"tokens": ["score"]}}}
    assert expand.mismatches(question, {"score", "scorer"}, 12) == []


def test_a_question_without_the_rule_is_not_recomputed():
    question = {"id": "q001", "text": "score", "expansion": {"graphify": {"tokens": ["nothing"]}}}
    assert expand.mismatches(question, {"score"}, 12) == []


def test_committed_mechanical_questions_still_match_the_rule(bench: Path):
    # inv: the vocabulary is a property of one snapshot, so each question is checked against the
    # graph of the snapshot it names -- one shared vocabulary would judge the others by the wrong tree
    vocabularies: dict[Path, set[str]] = {}
    checked = 0
    for qid in config.question_ids(bench):
        path = config.question_path(bench, qid)
        question = yaml.safe_load(path.read_text(encoding="utf-8"))
        graph = (config.snapshot_dir(bench, config.question_snapshot(question, qid))
                 / "indexes" / "graphify" / "graph.json")
        if not graph.exists():
            continue
        if graph not in vocabularies:
            vocabularies[graph] = expand.vocabulary(graph)
        assert expand.mismatches(question, vocabularies[graph], 12) == [], path.name
        checked += 1
    if not checked:
        pytest.skip("no question with a built graphify index beside it")


def test_expand_main_takes_argv(tmp_path):
    import json

    from benchmark.harness import expand as one_module

    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"nodes": [{"id": "score", "label": "score"}]}), encoding="utf-8")
    assert one_module.MECHANICAL == "mechanical"
    assert one_module.main(["--graph", str(graph), "--max-tokens", "3", "how is the score computed"]) == 0
