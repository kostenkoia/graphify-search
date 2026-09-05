"""Replay the blind check over one driven run's recorded request."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import yaml

from benchmark.harness import blind, config, prompt

if TYPE_CHECKING:
    from pathlib import Path


def check_blind(benchmark: Path, run_dir: Path) -> list[str]:
    """Return every way the request that was sent told the runner the answer.

    Parameters
    ----------
    benchmark : Path
        Root holding the snapshot that carries `references/<qid>.yaml`.
    run_dir : Path
        The run's `run/` directory, whose `request.json` records what was sent.

    Returns
    -------
    list of str
        Empty when the runner was blind or no model drove the run, in which case the run is
        named on stderr; one message per reference place that was not blind.
    """
    path = run_dir / "request.json"
    journal = run_dir / "journal.jsonl"
    driven = journal.is_file() and any(
        json.loads(line).get("by") == "runner"
        for line in journal.read_text(encoding="utf-8").splitlines() if line.strip())
    # inv: a run no model drove has nobody to have been told the answer; there is nothing to
    # show either way, and calling it a violation would fail every baseline ever collected
    if not driven:
        # inv: an empty list reads as "blind" everywhere else, so the one case where it means
        # "never asked" says so, or a recipe run is quoted as a passed blindness check
        print(f"no driven attempt in {run_dir}: nothing to check", file=sys.stderr)
        return []
    # inv: the audit reads the request, not prompt.md -- the file on disk can be rewritten
    # after a run, and only the recorded request says what the model was actually given
    if not path.is_file():
        return [f"{path} is absent, so what the runner was sent cannot be checked"]
    request = json.loads(path.read_text(encoding="utf-8"))
    meta = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load(
        config.reference_path(benchmark, meta["question"]).read_text(encoding="utf-8"))
    sent = "".join(str(m.get("content")) for m in request.get("messages") or [])
    above, _ = prompt.split(sent)
    return blind.violations(above, list(request.get("tools") or []), reference)
