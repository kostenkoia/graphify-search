import json
from pathlib import Path

from benchmark.harness.audit import stops


def _run(bench: Path, run_id: str, out_text: str, place: dict) -> Path:
    run = bench / "record" / "runs" / run_id
    run.mkdir(parents=True)
    entries = [{"n": 1, "kind": "call", "by": "harness", "name": "query"},
               {"n": 2, "kind": "stop", "reason": "answer_met", "place": place}]
    (run / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    (run / "02_query.out").write_text(out_text, encoding="utf-8")
    return run


def test_stops_grounds_a_named_place_in_the_vendor_output(git_bench):
    place = {"path": "src/app.py", "symbol": "render_invoice", "start": 12}
    run = _run(git_bench, "q1-s-c-a01", "src/app.py:12 def render_invoice(): ...", place)

    assert stops.check(git_bench) == []

    (run / "02_query.out").write_text("nothing relevant here", encoding="utf-8")

    problems = stops.check(git_bench)
    assert len(problems) == 1
    assert "q1-s-c-a01" in problems[0]
