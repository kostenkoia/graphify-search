"""Run one call, write `.cmd/.out/.err`, append the journal entry."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from benchmark.harness import rules

if TYPE_CHECKING:
    from pathlib import Path

# NOT DERIVED: 40 keeps `NN_name.*` short enough for one terminal line
_NAME = re.compile(r"^[a-z0-9_]{1,40}$")

# NOT DERIVED: the bound separates a hung vendor from a slow one by two orders of magnitude,
# measured against a cold vendor call on a private corpus this repository does not publish; raise
# it for a vendor that is genuinely slower rather than reading 300 as a property of any vendor
VENDOR_TIMEOUT_S = 300


class ArtifactsChangedError(Exception):
    """A listed index artifact no longer matches its recorded hash."""


@dataclass
class Context:
    """Everything `execute` needs about the current run."""

    run_dir: Path
    sandbox: Path
    home: Path
    environment: dict[str, str]
    invocation: dict
    volatile: list[str]
    artifacts: dict[str, str]
    watched: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.watched:
            self.watched = {"sandbox": self.sandbox, "home": self.home}


def _journal(ctx: Context) -> Path:
    return ctx.run_dir / "journal.jsonl"


def append(ctx: Context, entry: dict) -> None:
    """Append one journal line."""
    with _journal(ctx).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def next_n(ctx: Context) -> int:
    """Return the next journal sequence number."""
    path = _journal(ctx)
    if not path.exists():
        return 1
    last = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = max(last, int(json.loads(line)["n"]))
    return last + 1


def watched_listing(ctx: Context) -> dict[str, tuple[int, int]]:
    """List every file under each watched root, keyed `<label>/<relative path>`."""
    return rules.labelled_listing(ctx.watched)


def changed(ctx: Context, before: dict, after: dict) -> list[dict]:
    """Diff two watched listings; changed files are hashed, removed ones carry `None`."""
    return rules.changed_files(before, after, ctx.watched)


def _check_artifacts(ctx: Context) -> None:
    for rel, expected in ctx.artifacts.items():
        if rules.sha256_file(ctx.sandbox / rel) != expected:
            raise ArtifactsChangedError(rel)


def _cmd_text(call: dict, quote: str | None) -> str:
    body = json.dumps({k: v for k, v in call.items() if k != "kind"}, ensure_ascii=False, indent=2)
    return f"kind: {call['kind']}\nquote: {json.dumps(quote, ensure_ascii=False)}\ncall: {body}\n"


def execute(
    ctx: Context,
    *,
    name: str,
    call: dict,
    quote: str | None,
    by: str,
    system_call: bool,
    ceiling_call: bool = False,
    tool_text: str | None = None,
    before: dict[str, tuple[int, int]] | None = None,
    provenance: list[dict] | None = None,
) -> dict:
    """Execute or refuse one call, write its files, append and return its journal entry.

    Parameters
    ----------
    ctx : Context
        The run's paths, environment and artifact hashes.
    name : str
        Short name used in `NN_name.*`.
    call : dict
        `{"kind": "act", "argv": [...]}` or `{"kind": "tool", "tool": ..., "args": {...}}`.
    quote : str or None
        The documentation sentence that authorises the call; None for fixed steps without one.
    by : str
        `harness` or `runner`.
    system_call : bool
        Whether this entry is an execution of the vendor package or an MCP tool call,
        as opposed to a `script:` step.
    ceiling_call : bool
        Whether this entry is an MCP tool call counted under a declared ceiling.
    tool_text : str or None
        For `kind: tool`, the reply text already fetched; no subprocess runs here.
    before : dict of str to tuple of int, or None
        A watched listing taken earlier than this call, for a `kind: tool` call whose
        side effects may start before this function runs; None takes the listing now.
    provenance : list of dict, or None
        Where each argument of a runner's call came from; None for a harness call,
        whose arguments the harness itself wrote.

    Returns
    -------
    dict
        The journal entry, also appended to `journal.jsonl`.

    Raises
    ------
    ValueError
        When `name` does not match the allowed pattern.
    ArtifactsChangedError
        When a listed index artifact differs from its recorded hash before the call.
    """
    if not _NAME.match(name):
        raise ValueError(f"name does not match {_NAME.pattern}: {name!r}")
    n = next_n(ctx)
    stem = f"{n:02d}_{name}"  # inv: fixed steps, server entries and the stop stay far below 100 entries
    entry: dict = {"n": n, "kind": "call", "name": name, "by": by, "quote": quote}
    # inv: the origins are set before the grammar is consulted, so a refused call keeps the
    # record of what its arguments were and where they came from
    if provenance is not None:
        entry["provenance"] = provenance
    if call["kind"] == "act":
        entry["argv"] = rules.resolve_launcher(ctx.invocation, list(call["argv"]))
        checked = {"kind": "act", "argv": entry["argv"]}
    else:
        entry["tool"], entry["args"] = call["tool"], dict(call.get("args") or {})
        checked = call
    refusal = rules.check_call(ctx.invocation, checked)
    if refusal is not None:
        entry.update({"action": False, "refused": refusal, "system_call": False})
        append(ctx, entry)
        return entry
    _check_artifacts(ctx)
    (ctx.run_dir / f"{stem}.cmd").write_text(_cmd_text(checked, quote), encoding="utf-8")
    before = watched_listing(ctx) if before is None else before
    out_path, err_path = ctx.run_dir / f"{stem}.out", ctx.run_dir / f"{stem}.err"
    if call["kind"] == "act":
        env = {**ctx.environment, "HOME": str(ctx.home)}
        try:
            # inv: the child writes to files, never to a pipe the harness drains, so a vendor
            # stuck printing fills a file the operator can delete instead of memory
            with out_path.open("wb") as out_fh, err_path.open("wb") as err_fh:
                proc = subprocess.run(  # noqa: S603 — argv is a vetted list, shell=False
                    entry["argv"], cwd=ctx.sandbox, env=env, stdout=out_fh, stderr=err_fh,
                    shell=False, check=False, timeout=VENDOR_TIMEOUT_S,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            # why: a call already vetted by check_call can still fail to exec or hang; the attempt
            # must land in the journal so next_n never overwrites this stem's orphaned .cmd
            # inv: score._executed, audit.check_run and collect all key their exemption on the
            # failed shape -- error, exit None, no .out -- so no part-written file may survive
            out_path.unlink(missing_ok=True)
            err_path.unlink(missing_ok=True)
            entry.update({"exit": None, "error": str(exc), "system_call": system_call, "ceiling_call": ceiling_call})
            append(ctx, entry)
            raise
        code = proc.returncode
    else:
        out_path.write_bytes((tool_text or "").encode("utf-8"))
        err_path.write_bytes(b"")
        code = 0
    after = watched_listing(ctx)
    entry.update({
        "exit": code,
        "out_sha256": rules.sha256_file(out_path),
        "canonical_sha256": rules.canonical_hash_file(out_path, ctx.volatile),
        "system_call": system_call,
        "ceiling_call": ceiling_call,
        "files": changed(ctx, before, after),
    })
    append(ctx, entry)
    return entry
