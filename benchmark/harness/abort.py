"""Recover from a run the operating system killed: mark its row aborted, release its lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmark.harness import ledger, prepare, rules


def _lock_holder(tmp_root: Path) -> dict | None:
    lock = tmp_root / "sandbox" / "lock"
    if not lock.exists():
        return None
    try:
        held = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # inv: a lock written by a crash mid-write is unreadable rather than absent, and the
        # operator still needs it gone; an unreadable lock names no run and so blocks nothing here
        return {}
    return held if isinstance(held, dict) else {}


def _alive(pid: object) -> bool:
    if not isinstance(pid, int | str):
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def mark_aborted(benchmark: Path, run_id: str, tmp_root: Path) -> None:
    """Mark `run_id` aborted and release its lock; make no commit.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/`.
    run_id : str
        The row to mark aborted; not yet carrying an outcome.
    tmp_root : Path
        The tmp tree holding the sandbox lock this row's attempt took.

    Raises
    ------
    LedgerError
        When no row named `run_id` exists to mark.
    """
    # inv: bench marks, kia commits -- bench never writes inside .git/, so this writes the
    # outcome to the ledger file only and leaves the commit to `run`; release_lock sits in a
    # finally, so a write that fails still gives up the sandbox rather than leaving it held
    # by a process that is about to exit on the same failure
    try:
        ledger.complete_row(benchmark, run_id, {"outcome": "aborted"})
    finally:
        prepare.release_lock(tmp_root, run_id)


def abort(benchmark: Path, run_id: str, tmp_root: Path) -> list[str]:
    """Mark a killed run's ledger row aborted and release the sandbox lock it holds.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `systems/`; its parent is the git repository.
    run_id : str
        The run to abort, as `prepare` printed it.
    tmp_root : Path
        The tmp tree holding `sandbox/lock`.

    Returns
    -------
    list of str
        One line per action taken, in the order taken.

    Raises
    ------
    SystemExit
        When the row already carries an outcome, or the lock is held by a live
        process, or nothing about this run is in a state to recover.
    """
    rows = {r["run_id"]: r for r in ledger.rows(benchmark)}
    row = rows.get(run_id)
    ended = row is not None and "outcome" in row
    ended_as = row.get("outcome") if row is not None else None
    held = _lock_holder(tmp_root)
    # inv: a lock naming this run and held by a live process is a run still going, not wreckage
    if held and held.get("run_id") == run_id and _alive(held.get("pid")):
        raise SystemExit(f"{run_id} is still running (pid {held['pid']}); kill it before aborting")
    # inv: run.yaml with records.jsonl beside it is what tells a run collect can close from a
    # kill, so the pair decides before anything is written -- but only while collect can still
    # act on it: a row that ended is collected, and evidence already under benchmark/record/runs
    # makes collect refuse in turn, which would leave the operator sent between two commands
    # that each point at the other. A run killed inside the drive holds run.yaml and no
    # records.jsonl, and nothing but abort can close it
    collected = (benchmark / ledger.RECORD / "runs" / run_id).exists()
    if not ended and not collected and prepare.scored(tmp_root, run_id):
        raise SystemExit(f"{run_id} finished preparing and is waiting for collect; "
                          f"run benchmark.harness.collect on {tmp_root / run_id / 'run'} instead")
    # inv: marking the row and releasing the lock are two writes, so an ended row whose lock is
    # still on disk is a half-finished abort to complete, never a reason to refuse
    if ended and held is None:
        raise SystemExit(f"{run_id} already ended as {ended_as}, and no lock is at "
                          f"{tmp_root / 'sandbox' / 'lock'}; there is nothing to abort")
    # inv: `held == {}` is a lock that exists but names nobody, which is itself wreckage to clear;
    # only an absent lock together with an absent row means there is nothing here to recover
    if row is None and held is None:
        raise SystemExit(f"no ledger row and no lock for {run_id}; nothing to recover")
    if row is None and held and held.get("run_id") != run_id:
        raise SystemExit(f"no ledger row for {run_id}, and the lock names {held.get('run_id')!r}; "
                          f"nothing here belongs to {run_id}")
    done: list[str] = []
    # inv: captured before any branch below acts, because the marking branch releases the lock
    # itself (inside mark_aborted, to survive a failed write) rather than after every branch
    before = _lock_holder(tmp_root)
    if ended:
        done.append(f"ledger row {run_id} already ended as {ended_as}; left as it stands")
        prepare.release_lock(tmp_root, run_id)
    elif row is not None:
        mark_aborted(benchmark, run_id, tmp_root)
        # why: bench marks; `run` reads this row back and makes the one commit that closes it
        done.append(f"ledger row {run_id} marked aborted, not yet committed")
    else:
        done.append(f"no ledger row for {run_id}; the run was killed before it was recorded")
        prepare.release_lock(tmp_root, run_id)
    if before is not None and _lock_holder(tmp_root) is None:
        done.append(f"sandbox lock released: {tmp_root / 'sandbox' / 'lock'}")
    elif before is not None:
        done.append(f"sandbox lock left in place; it names {before.get('run_id')!r}, not {run_id}")
    else:
        # why: a run prepared under another --tmp-root leaves its lock where this command never
        # looked, and the operator sees no other sign of it
        done.append(f"no sandbox lock at {tmp_root}; if the run used another --tmp-root, "
                     f"run this again naming it")
    # why: evidence is never removed by a recovery command, so a stranded copy is reported and
    # left for the operator, who is the only one who can tell a truncated tree from a finished one
    # inv: a copy beside a row that ended is the evidence that row names, not wreckage
    stranded = benchmark / ledger.RECORD / "runs" / run_id
    if stranded.exists() and not ended:
        done.append(f"note: {stranded} exists and is unaccounted for; inspect it, then remove it by hand")
    return done


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark.harness abort",
                                 description="Recover from a run that was killed before it finished.")
    ap.add_argument("run_id")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Abort one killed run from the command line and print what was done.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        Always 0; a refusal raises `SystemExit` instead.
    """
    args = _parser().parse_args(argv)
    # inv: abort is a record verb -- it takes no path arguments, the same machine facts
    # attempt reads decide the benchmark root and the tmp root it recovers a lock under
    machine = rules.machine_facts()
    for line in abort(rules._ROOT, args.run_id, Path(machine["tmp_root"])):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
