"""Ask, for every driven run, whether the runner's own call already carried the answer.

A runner is meant to work from what the vendor printed. This reads each driven run's journal,
takes everything the runner could see before its first own call -- the fixed steps' output, which
is what prompt.md carries -- and asks whether the reference symbol occurs there. When it does not,
but the runner's own call text carries that symbol, the name came from somewhere other than the
run: the runner recognised the corpus.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from benchmark.harness import config, ledger

if TYPE_CHECKING:
    from pathlib import Path


def check(benchmark: Path) -> list[str]:
    """Return one message per driven run whose runner call named the reference before any fixed step printed it.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/attempts.jsonl`, `record/runs/` and `record/snapshots/`.

    Returns
    -------
    list of str
        One message per completed, driven run whose runner call text carries the question's
        reference symbol while no fixed step's own output did.
    """
    problems: list[str] = []
    rows_ = ledger.rows(benchmark)
    # inv: `runner` is the key that says a model drove the attempt; the stop reason says how the
    # run ended, which is a different fact
    driven = [r for r in rows_ if r.get("outcome") == "completed" and r.get("runner")]
    for row in driven:
        run = benchmark / ledger.RECORD / "runs" / row["run_id"]
        journal_path = run / "journal.jsonl"
        if not journal_path.is_file():
            continue
        reference = yaml.safe_load(config.reference_path(benchmark, row["question"])
                                   .read_text(encoding="utf-8"))["places"][0]
        symbol = str(reference["symbol"])
        entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        harness_calls = [e for e in entries if e.get("kind") == "call" and e.get("by") == "harness"
                         and e.get("action") is not False]
        runner_calls = [e for e in entries if e.get("kind") == "call" and e.get("by") == "runner"]
        if not runner_calls:
            continue
        prior = ""
        for e in harness_calls:
            p = run / f"{e['n']:02d}_{e['name']}.out"
            if p.is_file():
                prior += p.read_text(encoding="utf-8", errors="replace")
        asked = json.dumps([e.get("argv") or e.get("args") for e in runner_calls], ensure_ascii=False)
        if symbol in asked and symbol not in prior:
            problems.append(f"{row['run_id']}: runner named {symbol!r} before any fixed step printed it")
    return problems
