"""Recount hit and hit@5 from the run evidence, ranking only records that are places.

The methodology says a place is a path with a line; a record carrying neither is not one. The
shipped adapter still gives such a record a rank, so this recount re-reads each run's
records.jsonl, drops those entries, re-ranks what is left, and scores again. It publishes
nothing: it is a sensitivity check beside the ledger's own figures.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from benchmark.harness import config, ledger

if TYPE_CHECKING:
    from pathlib import Path


def _is_hit(rec: dict, ref: dict) -> bool:
    if rec.get("path") != ref["path"] or rec.get("start") is None:
        return False
    if not int(ref["start"]) <= int(rec["start"]) <= int(ref.get("end") or ref["start"]):
        return False
    return ref.get("symbol") is None or rec.get("symbol") == ref["symbol"]


def recount(benchmark: Path) -> dict:
    """Re-score hit and hit@5 over every completed (system, configuration) cell, places only.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/attempts.jsonl`, `record/runs/` and `record/snapshots/`.

    Returns
    -------
    dict
        `{"system/configuration": {"n", "hit", "at5", "dropped", "kept"}}`, one entry per
        cell the ledger's completed rows name; a report, not a pass/fail.
    """
    rows_ = ledger.rows(benchmark)
    cells = sorted({(r["system"], r["configuration"]) for r in rows_ if r.get("outcome") == "completed"})
    out: dict[str, dict[str, int]] = {}
    for system, configuration in cells:
        counts = {"n": 0, "hit": 0, "at5": 0, "dropped": 0, "kept": 0}
        questions = sorted({r["question"] for r in rows_
                             if r["system"] == system and r["configuration"] == configuration})
        for qid in questions:
            got = [r for r in rows_ if r["question"] == qid and r["system"] == system
                   and r["configuration"] == configuration
                   and r.get("outcome") == "completed" and r.get("stop") == "harness"]
            if not got:
                continue
            run_id = got[-1]["run_id"]
            path = benchmark / ledger.RECORD / "runs" / run_id / "records.jsonl"
            if not path.is_file():
                continue
            ref = yaml.safe_load(config.reference_path(benchmark, qid)
                                 .read_text(encoding="utf-8"))["places"][0]
            recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            places = [r for r in recs if r.get("kind") == "place"]
            real = [r for r in places if r.get("path") and r.get("start") is not None]
            counts["n"] += 1
            counts["dropped"] += len(places) - len(real)
            counts["kept"] += len(real)
            rank = None
            for n, rec in enumerate(real, start=1):
                if _is_hit(rec, ref):
                    rank = n
                    break
            if rank is not None:
                counts["hit"] += 1
                counts["at5"] += int(rank <= 5)
        out[f"{system}/{configuration}"] = counts
    return out
