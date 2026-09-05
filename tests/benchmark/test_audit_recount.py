import json

from benchmark.harness import ledger
from benchmark.harness.audit import recount as recount_mod
from tests.benchmark.conftest import write_question


def test_recount_drops_placeless_records_and_reranks(git_bench):
    bench = git_bench
    row = {"run_id": "q001-s-c-a01", "question": "q001", "system": "s", "configuration": "c",
           "attempt": 1, "outcome": "completed", "stop": "harness", "hit_rank": 6}
    ledger.path(bench).write_text(json.dumps(row) + "\n", encoding="utf-8")
    write_question(bench, "q001", {"id": "q001", "snapshot": "snap", "text": "t"},
                   {"places": [{"path": "src/app.py", "start": 12, "end": 12,
                                "symbol": "render_invoice"}]})
    run = bench / "record" / "runs" / "q001-s-c-a01"
    run.mkdir(parents=True)
    # ranks 1-5 are placeless -- the shipped adapter still counts them -- and rank 6 is the
    # only real place, which is also the reference's hit
    records = [{"kind": "place", "n": n} for n in range(1, 6)] + [
        {"kind": "place", "n": 6, "path": "src/app.py", "start": 12, "symbol": "render_invoice"}]
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    report = recount_mod.recount(bench)

    assert report["s/c"] == {"n": 1, "hit": 1, "at5": 1, "dropped": 5, "kept": 1}
