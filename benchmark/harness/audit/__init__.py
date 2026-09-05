"""Dispatch the audit package's checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from benchmark.harness import config, rules
from benchmark.harness.audit import expectations, priors, recount, stops
from benchmark.harness.audit.attempts import (
    RECORD_CLEARED,
    _prepared_leaves,
    check_attempts,
    check_rebaseline,
)
from benchmark.harness.audit.blind import check_blind
from benchmark.harness.audit.quotes import check_quotes, fold
from benchmark.harness.audit.run import check_run

__all__ = [
    "RECORD_CLEARED",
    "_prepared_leaves",
    "check_attempts",
    "check_blind",
    "check_quotes",
    "check_rebaseline",
    "check_run",
    "fold",
    "main",
]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark.harness audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quotes")
    q.add_argument("--system", required=True)
    r = sub.add_parser("run")
    r.add_argument("run_dir", type=Path)
    b = sub.add_parser("blind")
    b.add_argument("run_dir", type=Path)
    rb = sub.add_parser("rebaseline")
    rb.add_argument("system")
    for name in ("attempts", "stops", "priors", "expectations", "recount"):
        sub.add_parser(name)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Dispatch the subcommand and return the process exit code.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        1 when any problem was found, 0 otherwise.
    """
    args = _parser().parse_args(argv)
    # inv: audit is a record verb -- the tree it reads is the one this package lives in, so a
    # check can never be pointed at a benchmark someone assembled to pass it
    benchmark = rules._ROOT
    problems: list[str] = []
    if args.cmd == "quotes":
        problems = check_quotes(benchmark, args.system)
    elif args.cmd == "attempts":
        problems = check_attempts(benchmark)
    elif args.cmd == "blind":
        problems = check_blind(benchmark, args.run_dir)
    elif args.cmd == "rebaseline":
        problems = check_rebaseline(benchmark, args.system)
    elif args.cmd == "run":
        meta = yaml.safe_load((args.run_dir / "run.yaml").read_text(encoding="utf-8"))
        h = config.load_harness(benchmark, meta["system"])
        model_paths = {f".cache/huggingface/hub/{name}/{rel}" for name, m in h.models.items()
                       for rel in list((m or {}).get("files") or {}) + list((m or {}).get("links") or {})}
        result = check_run(args.run_dir, {**h.invocation, "allowed_scripts": h.allowed_scripts},
                            model_paths=model_paths, benchmark=benchmark)
        problems = result["violations"]
    elif args.cmd == "stops":
        problems = stops.check(benchmark)
    elif args.cmd == "priors":
        problems = priors.check(benchmark)
    elif args.cmd == "expectations":
        problems = expectations.check(benchmark)
    elif args.cmd == "recount":
        # why: recount is a report, not a pass/fail, so it prints a table and always exits clean
        report = recount.recount(benchmark)
        print(f"{'cell':32} {'hit':>8} {'hit@5':>8} {'places/q':>9} {'dropped/q':>10}")
        for key in sorted(report):
            c = report[key]
            if not c["n"]:
                continue
            print(f"{key:32} {c['hit']:>4}/{c['n']:<3} {c['at5']:>4}/{c['n']:<3} "
                  f"{c['kept'] / c['n']:>9.1f} {c['dropped'] / c['n']:>10.1f}")
        return 0
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
