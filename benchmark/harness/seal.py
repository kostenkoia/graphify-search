"""What is locked, and the seal that records it: `benchmark/INSTRUMENT.yaml`."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import yaml

from benchmark.harness import config, rules

_GIT = shutil.which("git") or "git"

SEAL = "INSTRUMENT.yaml"
FIRST_SEAL_REASON = "first seal of the checkout"
UNLOCKED = Path("lock") / "UNLOCKED"
MACHINE = Path("lock") / "machine.yaml"
HARNESS_ENV = Path("envs") / "harness"

# inv: a glob ending in `/**` names the directory, every directory under it and every file under it;
# `*` matches one path segment; every glob is relative to the repository root
INSTRUMENT: tuple[str, ...] = (
    "benchmark",
    "benchmark/harness/**",
    "benchmark/systems/**",
    "benchmark/record/snapshots",
    "benchmark/record/snapshots/*",
    "benchmark/record/snapshots/known_transitions.yaml",
    "benchmark/record/snapshots/*/meta.yaml",
    "benchmark/record/snapshots/*/fileset.sha256",
    "benchmark/record/snapshots/*/symbols.sha256",
    "benchmark/record/snapshots/*/indexes",
    "benchmark/record/snapshots/*/indexes/*/build.yaml",
    "benchmark/record/snapshots/*/questions/**",
    "benchmark/record/snapshots/*/references/**",
    "benchmark/INSTRUMENT.yaml",
    "benchmark/PROTOCOL.md",
    "benchmark/README.md",
    "benchmark/__init__.py",
    "tests/benchmark/**",
    "tests/test_comment_convention.py",
    "pyproject.toml",
    ".gitignore",
    ".github/**",
)
# inv: the lock's own files are root-owned in every state; `unlock` never touches them
LOCK: tuple[str, ...] = ("benchmark/lock/**",)
SYSTEM_UNDER_TEST: tuple[str, ...] = (
    "benchmark/envs/**",
    "benchmark/systems/*/models/**",
    "benchmark/record/snapshots/*/source/**",
    "benchmark/record/snapshots/*/symbols.jsonl",
    "benchmark/record/snapshots/*/indexes/*",
    "benchmark/record/snapshots/*/indexes/*/**",
)
_PREPARED_OUTPUTS = "benchmark/record/snapshots/*/indexes/*/prepared_outputs.yaml"
RECORD: tuple[str, ...] = (
    "benchmark/record/**",
    _PREPARED_OUTPUTS,
)
CLASSES: dict[str, tuple[str, ...]] = {
    "instrument": INSTRUMENT, "lock": LOCK, "record": RECORD, "sut": SYSTEM_UNDER_TEST,
}
# inv: weights sit under systems/**, an instrument glob; classify() checks _MODELS before
# INSTRUMENT, so this narrower glob wins there. build.yaml sits under indexes/*/**, a
# system-under-test glob; it wins the opposite way, by classify() checking INSTRUMENT (whose
# indexes/*/build.yaml glob is more specific) before SYSTEM_UNDER_TEST. prepared_outputs.yaml
# is the same shape as build.yaml — it too sits under indexes/*/**, a system-under-test glob —
# so classify() checks its own narrower record glob before SYSTEM_UNDER_TEST, the same
# narrow-glob-first mechanism _MODELS uses. record is checked last, as the catch-all
# `benchmark/record/**`: snapshots, questions and references all sit inside record/, so
# checking record first would classify every sealed path under it as record
_MODELS = "benchmark/systems/*/models/**"

_ROOT = Path(__file__).resolve().parents[1]


def _euid() -> int:
    return os.geteuid()


def _owner(path: Path) -> int:
    return path.stat().st_uid


def _mode_flags(path: Path) -> tuple[int, int]:
    # why: BSD and macOS carry per-file flags and Linux carries none, and the lock that sets
    # uchg runs on macOS alone, so a platform whose stat has no st_flags reports no flags
    st = path.stat()
    return st.st_mode & 0o777, getattr(st, "st_flags", 0)


def _matches(rel: str, glob: str) -> bool:
    if glob.endswith("/**"):
        base = glob[:-3]
        return _segments_equal(rel, base) or _under(rel, base)
    return _segments_equal(rel, glob)


def _segments_equal(rel: str, glob: str) -> bool:
    # why: fnmatch's `*` matches `.*` in the underlying regex and so crosses `/`; comparing
    # segment-wise, one path component at a time, is what confines `*` to one path segment
    parts, gparts = rel.split("/"), glob.split("/")
    return len(parts) == len(gparts) and all(
        fnmatch.fnmatchcase(p, g) for p, g in zip(parts, gparts, strict=True))


def _under(rel: str, base: str) -> bool:
    # why: fnmatch's `*` crosses `/`, so a base with `*` segments is matched segment-wise, one
    # path component at a time, the same way _segments_equal does for equal-length paths
    parts, bparts = rel.split("/"), base.split("/")
    # why: rel is deliberately longer than bparts here (it names something under base); zip
    # truncates to bparts on purpose, so strict=True would be wrong, not merely unspecified
    return len(parts) > len(bparts) and all(
        fnmatch.fnmatchcase(p, g) for p, g in zip(parts, bparts, strict=False))


def classify(rel: str) -> str | None:
    """Return the class of a repository-relative path, or None when no glob names it."""
    if any(_matches(rel, g) for g in LOCK):
        return "lock"
    if _matches(rel, _MODELS):
        return "sut"
    if any(_matches(rel, g) for g in INSTRUMENT):
        return "instrument"
    if _matches(rel, _PREPARED_OUTPUTS):
        return "record"
    if any(_matches(rel, g) for g in SYSTEM_UNDER_TEST):
        return "sut"
    if any(_matches(rel, g) for g in RECORD):
        return "record"
    return None


def _git_lines(repo: Path, *args: str) -> list[str]:
    # why: lock runs the seal as root inside the owner's repository, which git refuses as dubious
    # ownership unless the directory is named safe for this one call
    # why: a global core.excludesFile would make root's, bench's and kia's candidate sets
    # disagree; pinning it to /dev/null here is what ledger._git already pins for its own commit
    proc = subprocess.run(  # noqa: S603 — fixed git argv, shell=False
        [_GIT, "-c", f"safe.directory={repo}", "-c", "core.excludesFile=/dev/null",
         "-C", str(repo), "--no-optional-locks", *args],
        capture_output=True, text=True, check=True)
    return [line for line in proc.stdout.splitlines() if line]


def _candidates(repo: Path) -> list[str]:
    """Every tracked file, plus every untracked file that is not ignored, repository-relative."""
    tracked = _git_lines(repo, "ls-files")
    untracked = _git_lines(repo, "ls-files", "--others", "--exclude-standard")
    return sorted(set(tracked) | set(untracked))


def paths(benchmark: Path, cls: str) -> list[Path]:
    """Return the existing paths of one class, files and their directories, repository-relative.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory; its parent is the repository.
    cls : str
        One of `instrument`, `lock`, `record`, `sut`.

    Returns
    -------
    list of Path
        Sorted, relative to the repository root. Instrument and lock paths come from git
        (tracked, or untracked and not ignored); record and system-under-test paths from the
        file system, since most of them are ignored.
    """
    repo = benchmark.parent
    found: set[str] = set()
    if cls in ("instrument", "lock"):
        for rel in _candidates(repo):
            if classify(rel) == cls:
                found.add(rel)
                parent = Path(rel).parent
                while parent != Path("."):
                    if classify(parent.as_posix()) == cls:
                        found.add(parent.as_posix())
                    parent = parent.parent
    else:
        # why: a class the harness writes into is owned as a tree, so the directory and its
        # immediate children are enough for chown -R and for the ownership check; listing the
        # twelve thousand files under record/runs/ would only slow both down
        for glob in CLASSES[cls]:
            base = glob[:-3] if glob.endswith("/**") else glob
            for hit in repo.glob(base):
                rel = hit.relative_to(repo).as_posix()
                if classify(rel) != cls:
                    continue
                found.add(rel)
                if glob.endswith("/**") and hit.is_dir():
                    for child in hit.iterdir():
                        child_rel = child.relative_to(repo).as_posix()
                        if classify(child_rel) == cls:
                            found.add(child_rel)
    return [Path(p) for p in sorted(found)]


def launchers(benchmark: Path) -> list[Path]:
    """Return every launcher, interpreter and site of every system not marked reference-only."""
    out: list[Path] = []
    for hpath in sorted((benchmark / "systems").glob("*/harness.yaml")):
        h = config.load_harness(benchmark, hpath.parent.name)
        if h.status == "reference":
            continue
        pkg = (h.invocation.get("package") or {})
        out.extend(Path(pkg[k]) for k in ("launcher", "interpreter", "site") if k in pkg)
    return out


def _listing_sha256(root: Path) -> str | None:
    if not root.is_dir():
        return None
    import hashlib
    h = hashlib.sha256()
    # inv: bytecode is written under __pycache__ by the interpreter on the first import of a
    # module, as whichever user imported it, so a listing carrying it changes with what an
    # attempt happened to import rather than with what is installed
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(f"{path.relative_to(root).as_posix()} {rules.sha256_file(path)}\n".encode())
    return h.hexdigest()


def compute(benchmark: Path, reason: str) -> dict:
    """Return the seal mapping for the tree as it stands.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    reason : str
        The reason the instrument was unlocked, copied verbatim.

    Returns
    -------
    dict
        `reason`, `sealed_at_commit`, `files` (repository-relative path → sha256 of every
        instrument file), `interpreter` (path and listing hash of `envs/harness`, or None) and
        `machine_sha256` (of `lock/machine.yaml`, or None).
    """
    repo = benchmark.parent
    files = {p.as_posix(): rules.sha256_file(repo / p)
             for p in paths(benchmark, "instrument")
             if (repo / p).is_file() and p.as_posix() != f"benchmark/{SEAL}"}
    machine = benchmark / MACHINE
    env = benchmark / HARNESS_ENV
    return {
        "reason": reason,
        "sealed_at_commit": _git_lines(repo, "rev-parse", "HEAD")[0],
        "files": files,
        "interpreter": None if not env.is_dir() else {
            "path": (HARNESS_ENV / "bin" / "python").as_posix(), "listing_sha256": _listing_sha256(env)},
        "machine_sha256": rules.sha256_file(machine) if machine.is_file() else None,
    }


def load(benchmark: Path) -> dict | None:
    """Return the seal mapping, or None when the tree is not sealed."""
    path = benchmark / SEAL
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check(benchmark: Path) -> list[str]:
    """Return every way the tree differs from its seal; empty when sealed.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.

    Returns
    -------
    list of str
        `instrument is not sealed`, or one `instrument differs from its seal: <path>` line per
        changed, missing or unlisted file, then the interpreter and machine lines when the
        machine facts exist on this tree; while `lock/UNLOCKED` is present, the ownership pass
        is replaced by one line naming the open state instead of its expected violations.
    """
    doc = load(benchmark)
    if doc is None:
        return ["instrument is not sealed"]
    now = compute(benchmark, doc.get("reason", ""))
    problems = []
    for rel, digest in doc.get("files", {}).items():
        if rel not in now["files"]:
            problems.append(f"instrument differs from its seal: {rel} (missing)")
        elif now["files"][rel] != digest:
            problems.append(f"instrument differs from its seal: {rel}")
    problems.extend(f"instrument differs from its seal: {rel} (unlisted)"
                    for rel in now["files"] if rel not in doc.get("files", {}))
    # inv: a tree without machine.yaml is a checkout, not a machine, and has no interpreter to check
    if (benchmark / MACHINE).is_file():
        if now["interpreter"] != doc.get("interpreter"):
            problems.append("harness environment differs from its seal")
        if now["machine_sha256"] != doc.get("machine_sha256"):
            problems.append("machine facts differ from their seal")
        # why: mid-unlock every class is owner-writable by design, so the ownership pass would
        # report the state unlock creates as a wall of violations instead of naming the state
        if (benchmark / UNLOCKED).exists():
            problems.append("instrument is unlocked; ownership is not checked while open")
        else:
            problems.extend(_check_ownership(benchmark))
    return problems


def _check_ownership(benchmark: Path) -> list[str]:
    """Return every owner, mode and flags problem across the four classes.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory; `lock/machine.yaml` must exist.

    Returns
    -------
    list of str
        `machine.yaml is missing bench_uid` alone when the key is absent; otherwise
        `instrument not root-owned: <path>`, `instrument writable: <path>` or `instrument not
        immutable: <path>` for the instrument and lock classes, `record not owned by bench:
        <path>` or `record writable by others: <path>` for the record and system-under-test
        classes; empty when every path is owned, moded and flagged as its class requires.
    """
    repo = benchmark.parent
    machine = yaml.safe_load((benchmark / MACHINE).read_text(encoding="utf-8")) or {}
    if "bench_uid" not in machine:
        return ["machine.yaml is missing bench_uid"]
    bench_uid = int(machine["bench_uid"])
    problems: list[str] = []
    for cls in ("instrument", "lock"):
        for rel in paths(benchmark, cls):
            p = repo / rel
            if not p.exists():
                continue
            if _owner(p) != 0:
                problems.append(f"instrument not root-owned: {rel.as_posix()}")
            mode, flags = _mode_flags(p)
            if mode & 0o222:
                problems.append(f"instrument writable: {rel.as_posix()}")
            # why: install leaves benchmark/lock's own directory without uchg (a directory's
            # UF_IMMUTABLE would refuse creating or removing entries even for root, bricking
            # unlock); every instrument directory still carries it
            if (cls == "instrument" or p.is_file()) and not flags & stat.UF_IMMUTABLE:
                problems.append(f"instrument not immutable: {rel.as_posix()}")
    for cls in ("record", "sut"):
        for rel in paths(benchmark, cls):
            p = repo / rel
            if not p.exists():
                continue
            if _owner(p) != bench_uid:
                problems.append(f"record not owned by bench: {rel.as_posix()}")
            mode, _flags = _mode_flags(p)
            if mode & 0o022:
                problems.append(f"record writable by others: {rel.as_posix()}")
    return problems


def write(benchmark: Path) -> Path:
    """Write the seal: as root while unlocked, or without root with no seal and no machine facts.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.

    Returns
    -------
    Path
        The written `INSTRUMENT.yaml`. The reason is `lock/UNLOCKED`'s, or
        `FIRST_SEAL_REASON` on a checkout with no seal and no `lock/machine.yaml`.

    Raises
    ------
    SystemExit
        When a caller who is not root writes a seal on a tree that already carries one or
        carries machine facts, or when root writes one while `lock/UNLOCKED` is absent.
    """
    if _euid() != 0:
        # why: the file-hash half of the seal is a function of the git tree alone, so a checkout
        # holding neither a seal nor machine facts produces it once without root; either file
        # present closes this branch, and on a machine lock/machine.yaml, root-owned inside the
        # root-owned 555 lock/ directory, keeps it closed, so there only lock rewrites the seal
        if (benchmark / SEAL).exists() or (benchmark / MACHINE).exists():
            raise SystemExit("the seal is written by lock, as root")
        return _write(benchmark, FIRST_SEAL_REASON)
    unlocked = benchmark / UNLOCKED
    if not unlocked.is_file():
        raise SystemExit("no UNLOCKED file: the seal is written while the instrument is open")
    reason = str((yaml.safe_load(unlocked.read_text(encoding="utf-8")) or {}).get("reason", ""))
    return _write(benchmark, reason)


def _write(benchmark: Path, reason: str) -> Path:
    path = benchmark / SEAL
    doc = compute(benchmark, reason)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark.harness seal")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true")
    group.add_argument("--paths", choices=sorted(CLASSES))
    group.add_argument("--launchers", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Check the seal, list a class's paths or the launchers, or write the seal.

    Parameters
    ----------
    argv : list of str or None
        `--check`, `--paths <class>`, `--launchers`, or nothing to write; None reads `sys.argv`.

    Returns
    -------
    int
        Zero, or one when `--check` finds a difference.
    """
    args = _parser().parse_args(argv)
    if args.paths:
        for p in paths(_ROOT, args.paths):
            print(p.as_posix())
        return 0
    if args.launchers:
        for p in launchers(_ROOT):
            print(p)
        return 0
    if args.check:
        problems = check(_ROOT)
        for line in problems:
            print(line)
        return 1 if problems else 0
    print(write(_ROOT), file=sys.stderr)
    return 0
