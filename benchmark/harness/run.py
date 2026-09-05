"""Run one attempt as the operator: refusals, the sudo boundary, the attempt's one commit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark.harness import config, ledger, questions, rules

# why: the protocol's repeatability rule wants two independent baselines per cell -- not more; a
# third under the same recipe is refused here, never invited
BASELINE_ATTEMPTS = 2
# why: a cell tried this many times past its target depth is failing, not pending, and retrying it
# forever would spend the whole budget
BASELINE_RETRIES = 2

# NOT DERIVED: a bound on one unattended `run missing` pass, so a failing cell cannot hold the shell forever
DEFAULT_SECONDS = 480.0


def _machine_facts(benchmark: Path) -> dict[str, Any]:
    """Check the instrument is sealed and return `machine.yaml`, without the bench-uid gate.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory this harness lives in.

    Returns
    -------
    dict
        The parsed `machine.yaml`.

    Raises
    ------
    SystemExit
        When `benchmark` is not this harness's own root, the instrument is unlocked or
        unsealed, or the lock is not installed on this machine.
    """
    # inv: run is invoked by the operator, not bench, and elevates through sudo instead of
    # running as bench itself; rules.check_instrument is require_sealed's shared prologue,
    # the part that does not depend on who is calling -- the uid gate is attempt's own,
    # checked again for real once sudo lands there
    return rules.check_instrument(benchmark)


def _interpreter(machine: dict[str, Any]) -> str:
    """Return the sealed machine's own harness interpreter, named in `machine.yaml`'s `repo_root`."""
    return str(Path(machine["repo_root"]) / "benchmark" / "envs" / "harness" / "bin" / "python")


def _head_rows(benchmark: Path) -> list[dict]:
    """Return the ledger rows as HEAD has them, ignoring anything the working tree adds."""
    # inv: read through ledger._git, the one hardened git call in the package -- it pins
    # commit.gpgsign and core.excludesFile and passes --no-optional-locks
    try:
        out = ledger._git(benchmark, "show", f"HEAD:{ledger.rel(benchmark)}")
    except RuntimeError:
        return []
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _stranded_rows(benchmark: Path) -> list[dict]:
    """Return every ledger row the working tree holds that HEAD does not, oldest first.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/`.

    Returns
    -------
    list of dict
        Empty when the ledger already matches HEAD.
    """
    if not ledger.changed(benchmark, ledger.rel(benchmark)):
        return []
    head_ids = {r["run_id"] for r in _head_rows(benchmark)}
    # inv: the ledger is append-only, so the rows HEAD does not hold are its tail in the order
    # they were appended; a row reaches it through `run`, which commits at once, or through a
    # bare `attempt` over sudo, which commits nothing, so more than one row can be stranded
    return [r for r in ledger.rows(benchmark) if r["run_id"] not in head_ids]


def _prepared_rel(benchmark: Path, row: dict) -> str | None:
    """Return the row's cell's `prepared_outputs.yaml` path, derived from the row alone.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/` and `record/`.
    row : dict
        A ledger row, read for its `system`, `configuration` and `question`.

    Returns
    -------
    str or None
        None when the row's system or question can no longer be resolved on this tree --
        the fold this buys is best-effort, and never blocks the commit it would join.
    """
    from benchmark.harness import collect

    try:
        question = config.load_question(benchmark, row["question"])
        snapshot = config.question_snapshot(question, row["question"])
        meta = {"system": row["system"], "configuration": row["configuration"], "snapshot": snapshot}
        return collect.prepared_outputs_rel(benchmark, meta)
    except (config.ConfigError, KeyError, OSError) as exc:
        # why: the fold is best-effort and never blocks the commit, so the reason is printed --
        # silence would leave the expectations file to surface one attempt later as generic dirt
        print(f"expectations not folded into {row['run_id']}'s commit: {exc}", file=sys.stderr)
        return None


def _commit_row(benchmark: Path, row: dict, content: str | None = None) -> None:
    """Commit one ledger row, folding its cell's expectations file when that file is dirty.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/`.
    row : dict
        The ledger row to commit, already carrying an `outcome`.
    content : str or None, optional
        The ledger's content for this commit, staged as a blob; None commits the working file.
    """
    rel = _prepared_rel(benchmark, row)
    extra = (rel,) if rel is not None and ledger.changed(benchmark, rel) else ()
    message = f"chore(benchmark): attempt {row['run_id']} {row['outcome']}"
    if content is None:
        ledger.commit_rows(benchmark, message, extra_paths=extra)
        return
    ledger.commit_content(benchmark, message, content, extra_paths=extra)


def _repair_stranded(benchmark: Path, machine: dict[str, Any]) -> list[str]:
    """Commit every ledger row a prior call left written but uncommitted, oldest first.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/`.
    machine : dict
        `machine.yaml`'s facts, naming the interpreter a refusal points the operator at.

    Returns
    -------
    list of str
        The run_ids repaired, oldest first; empty when the ledger already matches HEAD.

    Raises
    ------
    SystemExit
        Naming every stranded row when any of them carries no outcome yet -- bench never
        wrote one, and `run` never invents one; the operator marks it with `abort` first.
    """
    stranded = _stranded_rows(benchmark)
    if not stranded:
        return []
    missing = [str(r["run_id"]) for r in stranded if "outcome" not in r]
    if missing:
        raise SystemExit(
            "stranded rows HEAD does not hold: "
            + ", ".join(str(r["run_id"]) for r in stranded) + "; "
            + "; ".join(f"{run_id} has no outcome; sudo -u bench {_interpreter(machine)} "
                        f"-m benchmark.harness abort {run_id} marks it" for run_id in missing)
            + ", then run again")
    all_rows = ledger.rows(benchmark)
    committed = len(all_rows) - len(stranded)
    # why: the audit refuses a commit that touches more than one row, so each row is committed
    # from a blob of the ledger as far as that row; the working file, which holds every row, is
    # never rewritten, and a kill between two commits leaves the rest stranded for the next `run`
    for i, row in enumerate(stranded, start=1):
        _commit_row(benchmark, row, ledger.as_text(all_rows[: committed + i]))
    return [str(r["run_id"]) for r in stranded]


def _round_key(driver: dict[str, Any] | None) -> tuple:
    if driver is None:
        return ()
    return (driver.get("model"), driver.get("effort"), driver.get("max_actions"), driver.get("max_tokens"))


def _refusal(
    benchmark: Path, kind: str, system: str, configuration: str, qid: str,
    driver: dict[str, Any] | None, repeat: bool,
) -> str | None:
    """Return why this attempt is refused, or None when it may proceed.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `systems/`.
    kind : str
        `"baseline"` or `"driven"`.
    system, configuration, qid : str
        The cell and the question asked for.
    driver : dict or None
        The driven settings, when `kind` is `"driven"`.
    repeat : bool
        Whether a second driven attempt of the same round was explicitly asked for.

    Returns
    -------
    str or None
        A refusal message, or None.
    """
    harness_path = benchmark / "systems" / system / "harness.yaml"
    if not harness_path.is_file():
        return f"no such system: {system}"
    recipe = rules.sha256_file(harness_path)
    # inv: scoped to the current recipe, so a cell's rows from before a harness.yaml change --
    # the grandfathered triples -- count toward nothing here; only a fresh attempt under today's
    # recipe can trip either refusal below
    cell = [r for r in ledger.rows(benchmark)
            if (r["question"], r["system"], r["configuration"]) == (qid, system, configuration)
            and r.get("harness_sha256") == recipe]
    if kind == "baseline":
        baseline_rows = [r for r in cell if not r.get("runner")]
        completed = sum(1 for r in baseline_rows if r.get("outcome") == "completed")
        if completed >= BASELINE_ATTEMPTS:
            return (f"{system}/{configuration}/{qid} already has {completed} completed baselines "
                    f"under this recipe")
        if len(baseline_rows) >= BASELINE_ATTEMPTS + BASELINE_RETRIES:
            return (f"{system}/{configuration}/{qid} has been tried {len(baseline_rows)} times as a "
                    f"baseline under this recipe; it is failing, not pending")
        return None
    round_key = _round_key(driver)
    driven_rows = [r for r in cell if r.get("runner")]
    round_rows = [r for r in driven_rows if _round_key(r) == round_key]
    completed = sum(1 for r in round_rows if r.get("outcome") == "completed")
    if completed >= 1 and not repeat:
        return (f"{system}/{configuration}/{qid} already has a completed driven attempt for this "
                f"round; pass --repeat to run another")
    if len(round_rows) >= BASELINE_ATTEMPTS + BASELINE_RETRIES:
        return (f"{system}/{configuration}/{qid} has been tried {len(round_rows)} times for this "
                f"round; it is failing, not pending")
    return None


def _invoke_attempt(machine: dict[str, Any], argv: list[str]) -> tuple[int, str, str]:
    """Run `attempt` as bench, over sudo, and return its exit status and output.

    Parameters
    ----------
    machine : dict
        `machine.yaml`'s facts, naming the harness interpreter's `repo_root`.
    argv : list of str
        `attempt`'s own argv, unchanged from what `run` was given.

    Returns
    -------
    tuple of (int, str, str)
        Exit status, stdout, stderr.
    """
    # why: sudo is resolved from PATH deliberately, as every lock/*.sh git/sed/visudo call already does
    cmd = ["sudo", "-u", "bench", _interpreter(machine), "-m", "benchmark.harness", "attempt", *argv]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603 — fixed argv
    return proc.returncode, proc.stdout, proc.stderr


_ROW_FIELDS = ("run_id", "outcome", "stop", "hit", "hit_rank", "tokens", "system_calls",
              "ceiling_calls", "runner_actions", "refused")


def _finish_success(benchmark: Path, run_id: str) -> dict:
    """Make the attempt's one commit and return the row it committed.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/`.
    run_id : str
        The attempt that just completed.

    Returns
    -------
    dict
        The committed ledger row.

    Raises
    ------
    SystemExit
        Naming `run_id` when the ledger holds no row for it.
    """
    row = next((r for r in reversed(ledger.rows(benchmark)) if r["run_id"] == run_id), None)
    if row is None:
        raise SystemExit(f"no row for {run_id}; the attempt printed it, the ledger does not hold it")
    _commit_row(benchmark, row)
    return row


def _run_one(
    kind: str, system: str, configuration: str, qid: str, *,
    model: str | None = None, effort: str | None = None, max_actions: int | None = None,
    max_tokens: int | None = None, repeat: bool = False,
) -> dict:
    """Refuse, dispatch to `attempt` over sudo, and make the attempt's one commit.

    Parameters
    ----------
    kind : str
        `"baseline"` or `"driven"`.
    system, configuration, qid : str
        The cell and the question to run.
    model, effort, max_actions, max_tokens : optional
        The driven settings; required together when `kind` is `"driven"`.
    repeat : bool, optional
        Admit a second driven attempt of the same round.

    Returns
    -------
    dict
        `_ROW_FIELDS` plus `seconds`, as printed.

    Raises
    ------
    SystemExit
        A refusal, `attempt`'s own failure text, or `_repair_stranded`'s no-outcome refusal.
    """
    started = time.time()
    benchmark = rules._ROOT
    machine = _machine_facts(benchmark)
    _repair_stranded(benchmark, machine)
    driver = ({"model": model, "effort": effort, "max_actions": max_actions, "max_tokens": max_tokens}
             if kind == "driven" else None)
    refusal = _refusal(benchmark, kind, system, configuration, qid, driver, repeat)
    if refusal is not None:
        raise SystemExit(refusal)
    argv = [kind, system, configuration, qid]
    if driver is not None:
        argv += ["--model", str(model), "--effort", str(effort),
                "--max-actions", str(max_actions), "--max-tokens", str(max_tokens)]
        if repeat:
            argv.append("--repeat")
    returncode, out, err = _invoke_attempt(machine, argv)
    if returncode != 0:
        # inv: bench only marks a failed attempt's row aborted (abort.mark_aborted) -- kia
        # commits it here, exactly the repair a killed `run` would owe on its next invocation
        _repair_stranded(benchmark, machine)
        raise SystemExit((err or out).strip())
    run_id = json.loads(out.strip().splitlines()[-1])["run_id"]
    row = _finish_success(benchmark, run_id)
    result = {k: row.get(k) for k in _ROW_FIELDS}
    result["seconds"] = round(time.time() - started, 1)
    return result


def _all_qids(benchmark: Path) -> list[str]:
    return config.question_ids(benchmark)


def _cells_for(benchmark: Path, qid: str) -> list[tuple[str, str]]:
    """Return every `(system, configuration)` this question can run, in `systems/` order.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/` and `record/snapshots/`.
    qid : str
        The question, whose snapshot decides which configurations have an index.

    Returns
    -------
    list of tuple of str
        Skips a system marked `reference`, a configuration marked `declared`, and any
        configuration whose index does not exist for this question's snapshot.
    """
    question = config.load_question(benchmark, qid)
    snapshot = config.question_snapshot(question, qid)
    out: list[tuple[str, str]] = []
    for sysdir in sorted((benchmark / "systems").iterdir()):
        if not (sysdir / "harness.yaml").is_file():
            continue
        h = config.load_harness(benchmark, sysdir.name)
        if h.status == "reference":
            continue
        for cfg_name, cfg in h.configurations.items():
            if (cfg or {}).get("status") == "declared":
                continue
            if (config.snapshot_dir(benchmark, snapshot) / cfg["index"]).is_dir():
                out.append((sysdir.name, cfg_name))
    return out


def _todo(benchmark: Path, kind: str, driver: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """Return every `(system, configuration, qid)` still missing an attempt of this kind."""
    out: list[tuple[str, str, str]] = []
    for qid in _all_qids(benchmark):
        # why: prepare refuses a question its review withdrew or never admitted, so listing it
        # here would spend the budget on attempts that can only abort
        if not questions.admitted(benchmark, qid):
            continue
        try:
            cells = _cells_for(benchmark, qid)
        except (config.ConfigError, KeyError):
            continue
        for system, configuration in cells:
            if _refusal(benchmark, kind, system, configuration, qid, driver, repeat=False) is None:
                out.append((system, configuration, qid))
    return out


def _run_missing(
    kind: str, *, model: str | None, effort: str | None, max_actions: int | None,
    max_tokens: int | None, seconds: float,
) -> dict:
    """Run whatever attempts of `kind` the tree is still missing, within a time budget.

    Parameters
    ----------
    kind : str
        `"baseline"` or `"driven"`.
    model, effort, max_actions, max_tokens : optional
        The driven settings; required together when `kind` is `"driven"`.
    seconds : float
        The wall-clock budget for this call.

    Returns
    -------
    dict
        `ran`, `completed`, `remaining_attempts`, `failures`, `failure_count`, and `halted`: True
        when the loop stopped early because a failure added no ledger row -- `_todo` can never
        advance past a cell that never reached bench, so retrying it would only spend the
        budget on the same machine-level problem.
    """
    benchmark = rules._ROOT
    driver = ({"model": model, "effort": effort, "max_actions": max_actions, "max_tokens": max_tokens}
             if kind == "driven" else None)
    deadline = time.time() + seconds
    ran = ok = 0
    failures: list[dict] = []
    halted = False
    while time.time() < deadline:
        pending = _todo(benchmark, kind, driver)
        if not pending:
            break
        system, configuration, qid = pending[0]
        ran += 1
        before = len(ledger.rows(benchmark))
        try:
            _run_one(kind, system, configuration, qid, model=model, effort=effort,
                    max_actions=max_actions, max_tokens=max_tokens, repeat=False)
            ok += 1
        except SystemExit as exc:
            failures.append({"cell": f"{system}/{configuration}", "question": qid,
                             "error": str(exc)[-300:]})
            if len(ledger.rows(benchmark)) == before:
                halted = True
                break
    left = len(_todo(benchmark, kind, driver))
    return {"ran": ran, "completed": ok, "remaining_attempts": left,
           "failures": failures[:5], "failure_count": len(failures), "halted": halted}


def _add_driver_args(p: argparse.ArgumentParser, *, required: bool) -> None:
    p.add_argument("--model", required=required)
    p.add_argument("--effort", required=required)
    p.add_argument("--max-actions", type=int, required=required)
    p.add_argument("--max-tokens", type=int, required=required)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark.harness run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("system")
    baseline.add_argument("configuration")
    baseline.add_argument("qid")
    driven = sub.add_parser("driven")
    driven.add_argument("system")
    driven.add_argument("configuration")
    driven.add_argument("qid")
    _add_driver_args(driven, required=True)
    driven.add_argument("--repeat", action="store_true")
    missing = sub.add_parser("missing")
    missing.add_argument("kind", choices=["baseline", "driven"])
    _add_driver_args(missing, required=False)
    missing.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run one attempt, or every missing one within a budget.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        0, or 1 when `missing` halted early on a failure that reached no ledger row; a
        refusal or a single attempt's failure raises `SystemExit` instead.
    """
    args = _parser().parse_args(argv)
    if args.cmd == "missing":
        if args.kind == "driven" and None in (args.model, args.effort, args.max_actions, args.max_tokens):
            raise SystemExit("missing driven needs --model, --effort, --max-actions and --max-tokens")
        result = _run_missing(args.kind, model=args.model, effort=args.effort,
                              max_actions=args.max_actions, max_tokens=args.max_tokens,
                              seconds=args.seconds)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result["halted"] else 0
    result = _run_one(args.cmd, args.system, args.configuration, args.qid,
                      model=getattr(args, "model", None), effort=getattr(args, "effort", None),
                      max_actions=getattr(args, "max_actions", None),
                      max_tokens=getattr(args, "max_tokens", None),
                      repeat=bool(getattr(args, "repeat", False)))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
