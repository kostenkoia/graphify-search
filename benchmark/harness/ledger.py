"""The committed record of every attempt."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

# inv: git is resolved from the harness's own PATH, never from the sandbox environment
_GIT = shutil.which("git") or "git"


class LedgerError(Exception):
    """The ledger is not in a state that allows a new attempt."""


RECORD = Path("record")
# inv: the ledger's earlier names are what git history holds it under, and the provenance walk
# reads every commit by the name that commit used
HISTORICAL_RELS: tuple[str, ...] = ("benchmark/attempts.jsonl",)


def path(benchmark: Path) -> Path:
    """Return the ledger file under `benchmark/record/`."""
    return benchmark / RECORD / "attempts.jsonl"


def rel(benchmark: Path) -> str:
    """Return the ledger's path relative to the git repository root, POSIX style."""
    return path(benchmark).relative_to(benchmark.parent).as_posix()


def _git(benchmark: Path, *args: str, stdin: str | None = None) -> str:
    # why: a global commit.gpgsign or core.excludesFile can make the harness's own commit
    # fail or silently stage nothing; pinning both keeps this path reproducible off this machine
    # why: --no-optional-locks keeps a read (status, diff) from writing the index's refresh
    # into .git/ -- bench holds no write access there beyond the commits this module makes itself
    proc = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
        [_GIT, "--no-optional-locks", "-c", "commit.gpgsign=false", "-c", "core.excludesFile=/dev/null",
         "-C", str(benchmark.parent), *args],
        capture_output=True, text=True, check=False, input=stdin,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {list(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _on_branch(benchmark: Path) -> str | None:
    """Return the branch HEAD points at, or None when HEAD is detached."""
    proc = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
        [_GIT, "-C", str(benchmark.parent), "symbolic-ref", "--quiet", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout.strip() or None


def rows(benchmark: Path) -> list[dict]:
    """Return every ledger row in order."""
    p = path(benchmark)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def next_attempt(benchmark: Path, question: str, system: str, configuration: str) -> int:
    """Return one more than the highest attempt number of the triple."""
    same = [r["attempt"] for r in rows(benchmark)
            if (r["question"], r["system"], r["configuration"]) == (question, system, configuration)]
    return (max(same) if same else 0) + 1


def require_clean(benchmark: Path) -> None:
    """Raise unless the tree is clean enough for an attempt to start.

    Raises
    ------
    LedgerError
        When a tracked file carries an uncommitted change, when an unignored untracked file
        sits under `benchmark/`, or when a started attempt still has no outcome.
    """
    if _git(benchmark, "status", "--porcelain", "--untracked-files=no"):
        raise LedgerError("the repository has uncommitted changes to tracked files")
    # inv: an untracked question or expectation is what the ledger cannot prove anyone did not
    # write after the fact, so it is dirt; an ignored run directory is evidence and is not
    under = benchmark.relative_to(benchmark.parent).as_posix()
    stray = [line for line in _git(benchmark, "ls-files", "--others", "--exclude-standard", "--", under).splitlines()
             if line.strip()]
    if stray:
        raise LedgerError(f"untracked file under {under}/: {stray[0]}" +
                          (f" (+{len(stray) - 1} more)" if len(stray) > 1 else ""))
    for r in rows(benchmark):
        if "outcome" not in r:
            raise LedgerError(f"attempt {r['run_id']} has no outcome; collect it, or if it was "
                              f"killed, run python -m benchmark.harness abort {r['run_id']}")


def commit_rows(benchmark: Path, message: str, extra_paths: tuple[str, ...] = ()) -> str:
    """Commit the ledger, plus any `extra_paths`, and return the new commit's sha.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/attempts.jsonl`.
    message : str
        The commit message.
    extra_paths : tuple of str, optional
        Further repo-relative paths to stage alongside the ledger; a path with no
        working-tree change contributes nothing to the commit.

    Returns
    -------
    str
        The new commit's sha.

    Raises
    ------
    LedgerError
        When HEAD is detached, so the commit would be reachable from nothing.
    """
    # inv: a detached-HEAD commit is reachable from nothing, so the refusal comes before the write
    if _on_branch(benchmark) is None:
        raise LedgerError("HEAD is detached; a ledger commit here would be unreachable")
    paths = (rel(benchmark), *extra_paths)
    _git(benchmark, "add", "--", *paths)
    _git(benchmark, "commit", "-q", "-m", message, "--only", "--", *paths)
    return _git(benchmark, "rev-parse", "HEAD")


def commit_content(benchmark: Path, message: str, content: str,
                   extra_paths: tuple[str, ...] = ()) -> str:
    """Commit `content` as the ledger's blob, plus any `extra_paths`, without touching the working tree.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/attempts.jsonl`.
    message : str
        The commit message.
    content : str
        What the ledger holds in this commit; the working file is neither read nor written.
    extra_paths : tuple of str, optional
        Further repo-relative paths to stage from the working tree; a path with no
        working-tree change contributes nothing to the commit.

    Returns
    -------
    str
        The new commit's sha.

    Raises
    ------
    LedgerError
        When HEAD is detached, so the commit would be reachable from nothing, or when the index
        already carries a staged change this commit of the whole index would sweep in.
    """
    # inv: a detached-HEAD commit is reachable from nothing, so the refusal comes before the write
    if _on_branch(benchmark) is None:
        raise LedgerError("HEAD is detached; a ledger commit here would be unreachable")
    # inv: the whole index is committed, so it must hold nothing but what this call stages; a
    # caller holding something else staged is refused rather than having it folded into this row
    already = [line for line in
               _git(benchmark, "diff-index", "--cached", "--name-only", "HEAD").splitlines()
               if line.strip()]
    if already:
        raise LedgerError(f"the index already holds a staged change ({already[0]}); "
                          "a ledger commit here would carry it")
    ledger_rel = rel(benchmark)
    blob = _git(benchmark, "hash-object", "-w", "--stdin", stdin=content)
    staged: tuple[str, ...] = ()
    try:
        _git(benchmark, "update-index", "--add", "--cacheinfo", f"100644,{blob},{ledger_rel}")
        staged = (ledger_rel,)
        if extra_paths:
            _git(benchmark, "add", "--", *extra_paths)
            staged = (ledger_rel, *extra_paths)
        # why: no pathspec and no --only, because what this commit records for the ledger is the
        # blob just staged and not what the working file holds
        _git(benchmark, "commit", "-q", "-m", message)
        staged = ()
    finally:
        # inv: a failed commit leaves nothing staged, so the next call's index check still passes
        if staged:
            _git(benchmark, "reset", "-q", "HEAD", "--", *staged)
    return _git(benchmark, "rev-parse", "HEAD")


def changed(benchmark: Path, rel_path: str) -> bool:
    """Return whether `rel_path` (repo-relative) carries an uncommitted change, tracked or not."""
    return bool(_git(benchmark, "status", "--porcelain", "--", rel_path))


def append_row(benchmark: Path, row: dict) -> None:
    """Append `row` to the ledger file; the caller commits it."""
    with path(benchmark).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_text(all_rows: list[dict]) -> str:
    """Return the ledger file's exact content for `all_rows`, one JSON object per line, in order."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows)


def rewrite(benchmark: Path, all_rows: list[dict]) -> None:
    """Replace the ledger's content with `all_rows`, one JSON object per line, in order."""
    path(benchmark).write_text(as_text(all_rows), encoding="utf-8")


def complete_row(benchmark: Path, run_id: str, update: dict) -> None:
    """Merge `update` into the row of `run_id` and rewrite the file; the caller commits it."""
    all_rows = rows(benchmark)
    for r in all_rows:
        if r["run_id"] == run_id:
            r.update(update)
            break
    else:
        raise LedgerError(f"no row for {run_id}")
    rewrite(benchmark, all_rows)
