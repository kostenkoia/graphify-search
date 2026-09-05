"""Close a run: clean the sandbox, complete the ledger, copy the evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from benchmark.harness import config, ledger, prepare, rules, score

_EVIDENCE = ("journal.jsonl", "build.yaml")


def clean_sandbox(sandbox: Path, artifacts: dict[str, str]) -> list[str]:
    """Remove every file under `sandbox` that is not a listed artifact; refuse evidence.

    Parameters
    ----------
    sandbox : Path
        The vendor's working directory to strip back to its artifacts.
    artifacts : dict of str to str
        Sandbox-relative artifact paths to keep, mapped to their expected sha256.

    Returns
    -------
    list of str
        Sandbox-relative paths removed.

    Raises
    ------
    SystemExit
        When any file under `sandbox` is named like run evidence, meaning this
        call is pointed at a directory it must not clean.
    """
    listing = rules.listing(sandbox)
    for rel in listing:
        name = Path(rel).name
        if name in _EVIDENCE or name.endswith((".out", ".cmd")):
            raise SystemExit(f"refusing to clean a directory holding evidence: {rel}")
    removed: list[str] = []
    for rel in sorted(listing):
        if rel not in artifacts:
            (sandbox / rel).unlink()
            removed.append(rel)
    for d in sorted((p for p in sandbox.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return removed


def _sha_or_none(path: Path) -> str | None:
    return rules.sha256_file(path) if path.exists() else None


def _recorded_removals(record: Path) -> list[str]:
    """Read an earlier `removed.json`, treating an unreadable one as empty.

    Parameters
    ----------
    record : Path
        The `removed.json` an earlier attempt left behind.

    Returns
    -------
    list of str
        The paths it names, or an empty list when it cannot be read as one.
    """
    # why: this is the retry path, so a record a kill left half-written must not stop the retry
    # that would replace it
    try:
        earlier = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(x) for x in earlier] if isinstance(earlier, list) else []


def _executed(entries: list[dict], run_dir: Path) -> list[dict]:
    # inv: the exemption keys on the full failed shape execute.py writes, as score._executed
    # and audit.check_run do
    return [e for e in entries
            if e.get("kind") == "call" and e.get("action") is not False
            and not ("error" in e and e.get("exit") is None
                     and not (run_dir / f"{e['n']:02d}_{e['name']}.out").exists())]


def collect(benchmark: Path, run_dir: Path, hits: dict | None = None) -> str:
    """Complete the attempt of `run_dir`: clean, verify, copy, write the row; return its outcome.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `systems/`.
    run_dir : Path
        The run's `run/` directory, as returned by `prepare.prepare`.
    hits : dict or None, optional
        The mapping `score.hits` would return for this run; computed by calling
        `score.hits` when omitted, so `collect` still runs standalone.

    Returns
    -------
    str
        `"completed"` when the audit is valid and `records.jsonl` exists, else `"void"`.

    Raises
    ------
    SystemExit
        When `benchmark/record/runs/<run-id>` already exists, `run.yaml` has no
        `artifacts` key, the sandbox holds evidence, or the question has no review file.
    """
    meta = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    rules.require_sealed(benchmark, Path(meta["tmp_root"]))
    run_id = meta["run_id"]
    # inv: the hit was scored against the question, system and configuration run.yaml names, and
    # the row publishes it under the ones the row names; no hash covers run.yaml, so a
    # disagreement here would publish one question's verdict under another's name
    row = next((r for r in ledger.rows(benchmark) if r["run_id"] == run_id), None)
    if row is not None:
        named = {k: (row.get(k), meta.get(k)) for k in ("question", "system", "configuration")}
        differ = {k: v for k, v in named.items() if v[0] != v[1]}
        if differ:
            raise SystemExit(f"run.yaml and the ledger row disagree for {run_id}: "
                              f"{ {k: {'row': a, 'run.yaml': b} for k, (a, b) in differ.items()} }")
    # inv: checked before any destructive step below, so a retry against an already-collected
    # run refuses immediately instead of re-cleaning the sandbox or rewriting audit.json first
    dest = benchmark / ledger.RECORD / "runs" / run_id
    if dest.exists():
        raise SystemExit(f"{dest} exists; evidence is never overwritten")
    tmp_root = Path(meta["tmp_root"])
    sandbox = tmp_root / "sandbox" / "index"
    # inv: a run killed before score wrote records.jsonl carries no verdict to read; it closes as
    # void below, with no hit, rather than failing inside score on the file's absence
    if hits is not None:
        hit = hits
    elif (run_dir / "records.jsonl").exists():
        hit = score.hits(benchmark, run_dir)
    else:
        hit = {}
    audit_path = run_dir / "audit.json"
    audit: dict = (json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists()
                   else {"valid": False, "violations": ["no audit.json"], "stop": None})
    cost_path = run_dir / "cost.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else {}
    if "artifacts" not in meta:
        raise SystemExit(f"{run_dir}/run.yaml has no artifacts key; refusing to guess what the sandbox may keep")
    # inv: the sandbox is shared, so cleaning it against this run's artifact list is only correct
    # while this run still holds it; a lock naming another run means a later one is using it
    lock = tmp_root / "sandbox" / "lock"
    if lock.exists():
        try:
            holder = json.loads(lock.read_text(encoding="utf-8")).get("run_id")
        except (OSError, ValueError):
            # inv: take_lock stops for a human on a lock it cannot read, so cleaning a sandbox
            # whose owner cannot be determined stops here too rather than proceeding silently
            raise SystemExit(f"unreadable sandbox lock at {lock}; no run can be identified from "
                              f"it, so inspect it and remove it") from None
        if holder is not None and holder != run_id:
            raise SystemExit(f"the sandbox is held by {holder}, not {run_id}; collecting now would "
                              f"clean a live run's sandbox against this run's artifacts")
    removed = clean_sandbox(sandbox, meta["artifacts"]) if sandbox.exists() else []
    # inv: a retry cleans a sandbox the first attempt already emptied, so the record accumulates
    # instead of being overwritten by the empty list the second clean returns
    record = run_dir / "removed.json"
    if record.exists():
        removed = sorted(set(_recorded_removals(record)) | set(removed))
    # inv: written through a rename, because collect is the retry command and a record truncated
    # by a kill mid-write would meet the read above on every attempt after this one
    tmp_record = record.with_suffix(".json.part")
    tmp_record.write_text(json.dumps(removed, indent=1), encoding="utf-8")
    tmp_record.replace(record)
    h = config.load_harness(benchmark, meta["system"])
    index_dir = (config.snapshot_dir(benchmark, meta["snapshot"])
                 / h.configurations[meta["configuration"]]["index"])
    try:
        prepare.verify_master(index_dir, config.load_build(index_dir))
    except SystemExit as exc:
        audit.update({"master_index_changed": True, "valid": False})
        audit["violations"].append(f"master index: {exc}")
        audit_path.write_text(json.dumps(audit, indent=1), encoding="utf-8")
    entries = [json.loads(line) for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    canonical = [e["canonical_sha256"] for e in _executed(entries, run_dir)]
    outcome = "completed" if audit.get("valid") and (run_dir / "records.jsonl").exists() else "void"
    update = {
        "outcome": outcome, "stop": audit.get("stop"), "hit": hit.get("hit"), "hit_rank": hit.get("hit_rank"),
        "hit_entry": hit.get("hit_entry"), "tokens": cost.get("tokens"), "system_calls": cost.get("system_calls"),
        "ceiling_calls": cost.get("ceiling_calls"),
        "canonical": canonical,
        "journal_sha256": rules.sha256_file(run_dir / "journal.jsonl"),
        "records_sha256": _sha_or_none(run_dir / "records.jsonl"),
        "cost_sha256": _sha_or_none(cost_path),
        "audit_sha256": _sha_or_none(audit_path),
        # inv: every row this call writes carries model_served; None means no server answered (a baseline)
        "model_served": meta.get("model_served"),
    }
    request_path = run_dir / "request.json"
    if request_path.is_file():
        update["request_sha256"] = rules.sha256_file(request_path)
    # inv: what a runner did belongs in the ledger, which is the durable record; results/ is
    # gitignored, so a figure computed only from it could not be rechecked from what git keeps
    # inv: a run no model drove grows none of these keys, so a baseline row keeps the shape
    # every published figure was written from
    update.update({k: hit[k] for k in ("stop_hit", "hit_by", "runner_actions", "refused", "model_usage")
                   if k in hit})
    try:
        # inv: prepare refuses a question with no review, so a missing file here is a defect in
        # the pipeline, not evidence to collect; raised inside this block, so the finally below
        # still gives up the sandbox lock instead of leaving it held for every later run
        review_path = config.review_path(benchmark, meta["question"])
        if not review_path.is_file():
            raise SystemExit(f"question {meta['question']} has no review at {review_path}; "
                              f"refusing to collect a run prepare should never have allowed")
        update["review_sha256"] = rules.sha256_file(review_path)
        try:
            # inv: copytree never creates dest's parent, and record/runs/ is fresh on a clean checkout
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, dest)
        except FileExistsError:
            # inv: `dest` was published between this run's own check and here, so it holds another
            # invocation's evidence -- refusing without a rollback is what keeps that evidence
            raise SystemExit(f"{dest} exists; evidence is never overwritten") from None
        except BaseException:
            # inv: any other failure is this copy dying partway, and the truncated tree it leaves
            # would meet the refusal above on every retry
            shutil.rmtree(dest, ignore_errors=True)
            raise
        # inv: the row is written here, but not committed -- bench never writes inside .git/;
        # once every step, this one included, has finished without raising, `run._finish_success`
        # reads this row back and makes the attempt's one commit as kia
        try:
            ledger.complete_row(benchmark, run_id, update)
        except BaseException:
            # why: this copy is one this invocation made, and outliving a failed write it would
            # wedge every retry on the dest-exists refusal above
            shutil.rmtree(dest, ignore_errors=True)
            raise
    finally:
        prepare.release_lock(tmp_root, run_id)
    return outcome


def prepared_outputs_rel(benchmark: Path, meta: dict) -> str:
    """Return the repo-relative path of a run's cell's `prepared_outputs.yaml`.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/` and `record/snapshots/`.
    meta : dict
        A run's `run.yaml`, carrying `system`, `configuration` and `snapshot`.

    Returns
    -------
    str
        The path, repo-relative, POSIX style; the caller decides whether it changed.
    """
    h = config.load_harness(benchmark, meta["system"])
    index_dir = (config.snapshot_dir(benchmark, meta["snapshot"])
                 / h.configurations[meta["configuration"]]["index"])
    # why: config.snapshot_dir resolves symlinks, so the repository root it is made relative to
    # is resolved too -- an unresolved root would not be a prefix of the resolved index path
    return (index_dir / prepare.PREPARED).relative_to(benchmark.resolve().parent).as_posix()


def main(argv: list[str] | None = None) -> int:
    """Collect one run from the command line and print its outcome.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        Always 0; a failed collection raises `SystemExit` instead.
    """
    ap = argparse.ArgumentParser(prog="benchmark.harness collect")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)
    print(collect(args.benchmark, args.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
