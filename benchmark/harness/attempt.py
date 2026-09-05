"""Run one attempt end to end, in one process, as bench -- `run` is the operator's front door."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from benchmark.harness import abort, audit, config, ledger, prepare, rules


def needs_record(benchmark: Path, h: config.Harness, configuration: str, qid: str) -> bool:
    """Tell whether this cell has no recorded fixed-step expectation for this question yet.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/snapshots/`.
    h : config.Harness
        The system under test.
    configuration : str
        The system's configuration.
    qid : str
        The question to run.

    Returns
    -------
    bool
        True when `--record-prepared` is owed because no expectation exists yet.
    """
    recipe = rules.sha256_file(benchmark / "systems" / h.system / "harness.yaml")
    question = config.load_question(benchmark, qid)
    snapshot = config.question_snapshot(question, qid)
    index_dir = config.snapshot_dir(benchmark, snapshot) / h.configurations[configuration]["index"]
    prepared = prepare.load_prepared(index_dir)
    return not (((prepared.get(configuration) or {}).get(recipe) or {}).get(qid))


def _merge_run_yaml(run_dir: Path, update: dict) -> None:
    path = run_dir / "run.yaml"
    meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meta.update(update)
    path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def _model_paths(h: config.Harness) -> set[str]:
    return {f".cache/huggingface/hub/{name}/{rel}" for name, m in h.models.items()
            for rel in list((m or {}).get("files") or {}) + list((m or {}).get("links") or {})}


def attempt(
    benchmark: Path, system: str, configuration: str, qid: str, *,
    tmp_root: Path, base_url: str, driver: dict[str, Any] | None = None, repeat: bool = False,
) -> dict:
    """Prepare, drive when asked, audit, score and collect one attempt; abort it on any failure.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `systems/`.
    system, configuration, qid : str
        The cell and the question to run.
    tmp_root : Path
        The tmp tree to prepare the run under.
    base_url : str
        The local chat-completions server a driven attempt is sent to.
    driver : dict or None, optional
        `model`, `effort`, `max_actions`, `max_tokens`, `backend` and `base_url`, for a driven
        attempt; None for baseline.
    repeat : bool, optional
        Written into the row; `run` reads it back to admit a second driven attempt of a round.

    Returns
    -------
    dict
        The ledger row this attempt completed, written but not committed -- bench never
        writes inside `.git/`; `run` reads this row back and makes the attempt's one commit.

    Raises
    ------
    SystemExit
        `failed: <step>: <what the step said>`, once any row this call appended has been
        marked aborted (`abort.mark_aborted`) and its lock released -- still uncommitted,
        for `run` to commit once this process exits nonzero.
    """
    # why: drive, collect and score each import tiktoken (score's own dependency) at module
    # load; deferring all three past main()'s env-var setup is what keeps the encoding cache
    # under the sealed machine's own TIKTOKEN_CACHE_DIR rather than whichever HOME started this
    from benchmark.harness import collect, drive, score

    h = config.load_harness(benchmark, system)
    configuration = configuration or h.default_configuration
    row_driver = {**driver, "repeat": repeat} if driver else None
    try:
        run_dir = prepare.prepare(benchmark, system, qid, configuration, tmp_root,
                                  needs_record(benchmark, h, configuration, qid),
                                  runner=driver is not None, driver=row_driver)
    except BaseException as exc:
        # inv: prepare's own failure handling already marks any row it appended
        # (abort.mark_aborted), or appended none at all -- bench never commits either way;
        # this call only reports the refusal, for `run` to read the mark and commit it
        raise SystemExit(f"failed: prepare: {str(exc).strip()[-600:]}") from exc
    run_id = run_dir.parent.name
    step = "drive"
    try:
        if driver is not None:
            from benchmark.harness.backends.lmstudio import LocalBackend

            outcome = drive.drive_run(benchmark, run_id, tmp_root, LocalBackend(base_url),
                                      model=driver["model"], effort=driver["effort"],
                                      max_actions=driver["max_actions"], max_tokens=driver["max_tokens"])
            served = outcome.get("model_served")
            # inv: run.yaml gains model_served here, the one update this attempt makes to it,
            # before collect reads the same file to write it into the row
            _merge_run_yaml(run_dir, {"model_served": served})
            if served is None or served != driver["model"]:
                step = "model_served"
                raise RuntimeError(f"served {served!r}, asked for {driver['model']!r}")
            step = "audit blind"
            blind_violations = audit.check_blind(benchmark, run_dir)
            if blind_violations:
                raise RuntimeError("; ".join(blind_violations))
        step = "audit run"
        result = audit.check_run(run_dir, {**h.invocation, "allowed_scripts": h.allowed_scripts},
                                 model_paths=_model_paths(h), benchmark=benchmark)
        if result["violations"]:
            raise RuntimeError("; ".join(result["violations"]))
        step = "score"
        score.records(benchmark, run_dir)
        score.cost(run_dir)
        hits = score.hits(benchmark, run_dir)
        step = "collect"
        collect.collect(benchmark, run_dir, hits)
        # inv: the lookup sits inside this try, so a ledger that no longer holds the row reaches
        # the marking path, whose own refusal names the missing row
        row = next((r for r in reversed(ledger.rows(benchmark)) if r["run_id"] == run_id), None)
        if row is None:
            raise ledger.LedgerError(f"no row for {run_id}")
    except BaseException as exc:
        tail = str(exc).strip()[-600:]
        # inv: bench marks; kia commits -- mark_aborted writes the outcome and releases the
        # lock, but makes no commit, so `run` reads this row back and commits it once this
        # process has exited nonzero
        abort.mark_aborted(benchmark, run_id, tmp_root)
        raise SystemExit(f"failed: {step}: {tail}") from exc
    return row


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark.harness attempt")
    sub = ap.add_subparsers(dest="kind", required=True)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("system")
    baseline.add_argument("configuration")
    baseline.add_argument("qid")
    driven = sub.add_parser("driven")
    driven.add_argument("system")
    driven.add_argument("configuration")
    driven.add_argument("qid")
    driven.add_argument("--model", required=True)
    driven.add_argument("--effort", required=True)
    driven.add_argument("--max-actions", type=int, required=True)
    driven.add_argument("--max-tokens", type=int, required=True)
    driven.add_argument("--repeat", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, validate the machine, run one attempt, print its row as one JSON line.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        Always 0; a refusal or a step failure raises `SystemExit` instead.
    """
    args = _parser().parse_args(argv)
    # inv: require_sealed runs before system/configuration/qid are ever loaded, so a refusal
    # here is never masked by a later, less clear configuration error
    machine = rules.machine_facts()
    os.environ["HOME"] = str(machine["home"])
    os.environ["TMPDIR"] = str(machine["tmpdir"])
    os.environ["TIKTOKEN_CACHE_DIR"] = str(machine["tiktoken_cache"])
    driver = None
    if args.kind == "driven":
        # inv: base_url is the value LocalBackend is constructed with below, so an aborted
        # driven row names the server the attempt was actually pointed at
        driver = {"model": args.model, "effort": args.effort,
                  "max_actions": args.max_actions, "max_tokens": args.max_tokens,
                  "backend": "local", "base_url": machine["base_url"]}
    row = attempt(rules._ROOT, args.system, args.configuration, args.qid,
                 tmp_root=Path(machine["tmp_root"]), base_url=machine["base_url"],
                 driver=driver, repeat=bool(getattr(args, "repeat", False)))
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
