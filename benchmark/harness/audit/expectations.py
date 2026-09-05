"""Check every collected run against the expectation now on disk.

The ledger audit reads prepared_outputs.yaml's git history to show that no expectation was ever
rewritten. Four of the five expectation files were committed once, in bulk, so that history does
not exist for them. This is the substitute: it recomputes, from each collected run's own journal,
the pair the harness would have compared, and asks whether the file on disk still says exactly
that. An expectation bent toward any one attempt would disagree with the others.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from benchmark.harness import config, ledger, prepare

if TYPE_CHECKING:
    from pathlib import Path


def check(benchmark: Path) -> list[str]:
    """Return one message per step whose collected journal disagrees with the expectation on disk.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/attempts.jsonl`, `record/runs/`, `record/snapshots/` and `systems/`.

    Returns
    -------
    list of str
        One message per fixed step of a completed run whose expectation is missing from
        `prepared_outputs.yaml`, or whose recorded value there disagrees with the run's journal.
    """
    problems: list[str] = []
    rows_ = {r["run_id"]: r for r in ledger.rows(benchmark)}
    runs_dir = benchmark / ledger.RECORD / "runs"
    if not runs_dir.is_dir():
        return problems
    for run in sorted(runs_dir.iterdir()):
        meta_path = run / "run.yaml"
        if not meta_path.is_file():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        row = rows_.get(meta["run_id"])
        if row is None or row.get("outcome") != "completed":
            continue
        harness = config.load_harness(benchmark, meta["system"])
        index_dir = (config.snapshot_dir(benchmark, meta["snapshot"])
                     / harness.configurations[meta["configuration"]]["index"])
        build = config.load_build(index_dir)
        prepared = prepare.load_prepared(index_dir)
        recipe = row.get("harness_sha256")
        steps = (((prepared.get(meta["configuration"]) or {}).get(recipe) or {}).get(meta["question"]) or {})
        excluded = list(build.get("excluded") or [])
        mutable = list(build.get("mutable") or [])
        entries = [json.loads(line) for line in (run / "journal.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        for entry in entries:
            if entry.get("kind") != "call" or entry.get("by") != "harness":
                continue
            name = entry["name"]
            observed = {"out": entry.get("canonical_sha256"),
                        "files": {f["path"]: f["sha256"] for f in entry.get("files") or []
                                  if prepare.keep_for_prepared(f["path"], excluded, mutable)}}
            expected = steps.get(name)
            if expected is None:
                problems.append(f"{meta['run_id']}/{name}: no expectation recorded under recipe {str(recipe)[:10]}")
            elif expected != observed:
                problems.append(f"{meta['run_id']}/{name}: expectation on disk differs from this run's journal")
    return problems
