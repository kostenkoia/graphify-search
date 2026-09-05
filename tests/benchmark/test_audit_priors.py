import json

from benchmark.harness import ledger
from benchmark.harness.audit import priors
from tests.benchmark.conftest import write_question


def _seed(bench, run_id: str, *, asked, reference_symbol: str, runner: bool = True) -> None:
    row = {"run_id": run_id, "question": "q1", "system": "s", "configuration": "c",
           "attempt": 1, "outcome": "completed", "stop": "answer_met", "runner": runner}
    ledger.path(bench).write_text(json.dumps(row) + "\n", encoding="utf-8")
    write_question(bench, "q1", {"id": "q1", "snapshot": "snap", "text": "t"},
                   {"places": [{"symbol": reference_symbol, "path": "x", "start": 1}]})
    run = bench / "record" / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    entries = [
        {"n": 1, "kind": "call", "by": "harness", "name": "version", "action": True},
        {"n": 2, "kind": "call", "by": "runner", "argv": asked},
    ]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    (run / "01_version.out").write_text("nothing about it here", encoding="utf-8")


def test_priors_flags_a_query_token_the_fixed_steps_never_printed(git_bench):
    _seed(git_bench, "q1-s-c-a01", asked=["grep", "secret_symbol"], reference_symbol="secret_symbol")

    problems = priors.check(git_bench)
    assert len(problems) == 1
    assert "secret_symbol" in problems[0]
    assert "q1-s-c-a01" in problems[0]

    _seed(git_bench, "q1-s-c-a01", asked=["grep", "unrelated"], reference_symbol="secret_symbol")
    assert priors.check(git_bench) == []


def test_priors_reads_the_runner_key_rather_than_guessing_from_the_stop(git_bench):
    # inv: `runner` is on every recorded row and says who drove the attempt; a stop other than
    # "harness" is a property of how the run ended, not of whether a model was in it
    _seed(git_bench, "q1-s-c-a01", asked=["grep", "secret_symbol"],
          reference_symbol="secret_symbol", runner=False)
    assert priors.check(git_bench) == []
