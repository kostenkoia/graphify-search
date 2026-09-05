import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmark"

# why: a frozen corpus and its built indexes cost hours to rebuild and cannot be fabricated
# under tmp_path, so the few tests that run a vendor read the tree this repository ships; test_mcp
# globs the built code-review-graph indexes under this one name, while every other test reaches a
# shipped question, reference or index through the config helpers
SHIPPED_SNAPSHOTS = "record/snapshots"


@pytest.fixture
def repo() -> Path:
    return REPO


@pytest.fixture
def bench() -> Path:
    return BENCH


def _git(root: Path, *args: str) -> None:
    # why: a global commit.gpgsign or core.excludesFile can make a fixture commit fail or
    # silently stage nothing; pinning both keeps this fixture reproducible outside this machine
    proc = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.excludesFile=/dev/null", "-C", str(root), *args],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {args!r} failed: {proc.stderr.decode(errors='replace')}")


@pytest.fixture
def git_bench(tmp_path: Path) -> Path:
    """A throwaway git repository with an empty committed ledger; returns its `benchmark/`."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    bench = tmp_path / "benchmark"
    bench.mkdir()
    (bench / "record").mkdir()
    (bench / "record" / "attempts.jsonl").write_text("", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return bench


def _reseal(b: Path) -> None:
    """Rewrite `INSTRUMENT.yaml` for the tree as it stands, as `lock` would after an unlock/reseal."""
    from benchmark.harness import seal

    (b / "lock" / "UNLOCKED").write_text("reason: test\n", encoding="utf-8")
    seal.write(b)
    (b / "lock" / "UNLOCKED").unlink()


@pytest.fixture
def sealed_bench(git_bench: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`git_bench` sealed as if `lock` had run, with the machine checks fed this process's facts."""
    from benchmark.harness import rules, seal

    b = git_bench
    (b / "lock").mkdir(exist_ok=True)
    tmp_root = b.parent / "tmp"
    tmp_root.mkdir(exist_ok=True)
    (b / "lock" / "machine.yaml").write_text(yaml.safe_dump({
        "bench_uid": os.geteuid(), "repo_root": str(b.parent), "tmp_root": str(tmp_root),
        "base_url": "http://localhost:1234/v1", "home": str(tmp_root / "home"),
        "tmpdir": str(tmp_root / "tmp"), "tiktoken_cache": str(tmp_root / "tiktoken")}), encoding="utf-8")
    monkeypatch.setattr(seal, "_euid", lambda: 0)
    _reseal(b)
    monkeypatch.setattr(rules, "_ROOT", b)
    # why: a test tree is owned by the user running pytest, which stands in for bench
    monkeypatch.setattr(rules, "_owner", lambda p: os.geteuid())
    monkeypatch.setattr(rules, "_mode", lambda p: 0o755)
    # why: seal.check's ownership pass wants instrument/lock root-owned and record/sut
    # bench-owned; a test tree has neither, so both of seal's own stat helpers are faked the
    # same way rules' are, above
    monkeypatch.setattr(seal, "_owner", lambda p: 0 if seal.classify(
        Path(p).relative_to(b.parent).as_posix()) in ("instrument", "lock") else os.geteuid())
    monkeypatch.setattr(seal, "_mode_flags", lambda p: (0o444, stat.UF_IMMUTABLE))
    _git(b.parent, "add", "-A")
    _git(b.parent, "commit", "-qm", "seal the fixture")
    return b


def snapshot_dir(bench: Path, snapshot: str) -> Path:
    """Return `record/snapshots/<snapshot>` under `bench`, creating it.

    Parameters
    ----------
    bench : Path
        A test tree's `benchmark/` directory.
    snapshot : str
        The snapshot id.

    Returns
    -------
    Path
        The snapshot directory.
    """
    path = bench / "record" / "snapshots" / snapshot
    path.mkdir(parents=True, exist_ok=True)
    return path


def questions_dir(bench: Path, snapshot: str) -> Path:
    """Return the snapshot's `questions/` directory, creating it.

    Parameters
    ----------
    bench : Path
        A test tree's `benchmark/` directory.
    snapshot : str
        The snapshot id.

    Returns
    -------
    Path
        `record/snapshots/<snapshot>/questions`.
    """
    path = snapshot_dir(bench, snapshot) / "questions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def references_dir(bench: Path, snapshot: str) -> Path:
    """Return the snapshot's `references/` directory, creating it.

    Parameters
    ----------
    bench : Path
        A test tree's `benchmark/` directory.
    snapshot : str
        The snapshot id.

    Returns
    -------
    Path
        `record/snapshots/<snapshot>/references`.
    """
    path = snapshot_dir(bench, snapshot) / "references"
    path.mkdir(parents=True, exist_ok=True)
    return path


def review_dir(bench: Path, snapshot: str) -> Path:
    """Return the snapshot's `questions/review/` directory, creating it.

    Parameters
    ----------
    bench : Path
        A test tree's `benchmark/` directory.
    snapshot : str
        The snapshot id.

    Returns
    -------
    Path
        `record/snapshots/<snapshot>/questions/review`.
    """
    path = questions_dir(bench, snapshot) / "review"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_question(bench: Path, qid: str, question: dict, reference: dict | None = None) -> Path:
    """Write `<qid>.yaml` under the snapshot the question names; return the question file.

    Parameters
    ----------
    bench : Path
        A test tree's `benchmark/` directory.
    qid : str
        The question id.
    question : dict
        The question body; its `snapshot` key decides which snapshot holds the file.
    reference : dict or None, optional
        A reference body to write beside it, or None to write no reference.

    Returns
    -------
    Path
        The question file.
    """
    snapshot = question["snapshot"]
    path = questions_dir(bench, snapshot) / f"{qid}.yaml"
    path.write_text(yaml.safe_dump(question, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if reference is not None:
        (references_dir(bench, snapshot) / f"{qid}.yaml").write_text(
            yaml.safe_dump(reference, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_review(bench: Path, qid: str) -> None:
    """Write a passing review for `qid`, hashed against its committed question and reference."""
    from benchmark.harness import config, rules

    path = config.review_path(bench, qid)
    path.parent.mkdir(parents=True, exist_ok=True)
    review = {
        "reference_is_right": True,
        "question_is_ambiguous": False,
        "note": None,
        "reviewer_model": "test",
        "question_sha256": rules.sha256_file(config.question_path(bench, qid)),
        "reference_sha256": rules.sha256_file(config.reference_path(bench, qid)),
    }
    path.write_text(
        yaml.safe_dump(review, sort_keys=False, allow_unicode=True), encoding="utf-8")


def freeze_source(snapshot: Path) -> None:
    """Write `<snapshot>/fileset.sha256` for its `source/`, in the README's `shasum` format.

    Parameters
    ----------
    snapshot : Path
        A snapshot directory holding `source/`.
    """
    from benchmark.harness import rules

    source = snapshot / "source"
    files = sorted((p for p in source.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(source).as_posix())
    lines = [f"{rules.sha256_file(p)}  ./{p.relative_to(source).as_posix()}\n" for p in files]
    (snapshot / "fileset.sha256").write_text("".join(lines), encoding="utf-8")
