"""Run one verb of the harness: `python -m benchmark.harness <verb> [args]`."""

from __future__ import annotations

import sys
from collections.abc import Callable

from benchmark.harness import abort, attempt, audit, expand, freeze_model, questions, report, run, seal, summary
from benchmark.harness.scoring import build_symbols

Verb = Callable[[list[str] | None], int]

# inv: a verb's name is the module's name with underscores as hyphens, so the command line and
# the code disagree on nothing but the character a shell can type without quoting
VERBS: dict[str, Verb] = {
    "run": run.main,
    "attempt": attempt.main,
    "abort": abort.main,
    "audit": audit.main,
    "expand": expand.main,
    "freeze-model": freeze_model.main,
    "build-symbols": build_symbols.main,
    "seal": seal.main,
    "questions": questions.main,
    "summary": summary.main,
    "report": report.main,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch `argv[0]` to its verb and return that verb's exit status.

    Parameters
    ----------
    argv : list of str or None
        The verb followed by its own arguments; None reads `sys.argv[1:]`.

    Returns
    -------
    int
        The verb's exit status.

    Raises
    ------
    SystemExit
        With status 2 when no verb or an unknown verb is named.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(f"benchmark.harness: one of: {', '.join(sorted(VERBS))}")
        return 0
    if not args or args[0] not in VERBS:
        named = args[0] if args else "(none)"
        print(f"benchmark.harness: unknown verb {named}; one of: {', '.join(sorted(VERBS))}",
              file=sys.stderr)
        raise SystemExit(2)
    return VERBS[args[0]](args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
