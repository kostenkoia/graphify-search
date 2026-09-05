"""Check that every place a driven run named was one the vendor actually printed to it."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmark.harness import ledger

if TYPE_CHECKING:
    from pathlib import Path


def check(benchmark: Path) -> list[str]:
    """Return one message per run whose stop names a place its own vendor output never printed.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/runs/`.

    Returns
    -------
    list of str
        Empty when every run's stop carrying a place finds that place's `path`, `symbol` and
        `start` printed, verbatim, somewhere in that run's own `*.out` files; a key the place
        does not carry is compared as the empty string, which every file contains.
    """
    problems: list[str] = []
    runs_dir = benchmark / ledger.RECORD / "runs"
    if not runs_dir.is_dir():
        return problems
    for run in sorted(runs_dir.iterdir()):
        journal_path = run / "journal.jsonl"
        if not journal_path.is_file():
            continue
        entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        stop = next((e for e in reversed(entries) if e.get("kind") == "stop"), None)
        place = (stop or {}).get("place")
        if not place:
            continue
        text = "".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(run.glob("*.out")))
        path_seen = str(place.get("path", "")) in text
        symbol_seen = str(place.get("symbol", "")) in text
        start_seen = str(place.get("start", "")) in text
        if not (path_seen and symbol_seen and start_seen):
            problems.append(f"{run.name}: place {place} not grounded in the run's own vendor output "
                             f"(path_seen={path_seen} symbol_seen={symbol_seen} start_seen={start_seen})")
    return problems
