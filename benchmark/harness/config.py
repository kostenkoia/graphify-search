"""Load and validate the harness's own configuration files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(Exception):
    """A configuration file is missing a key the harness depends on."""


REQUIRED_HARNESS_KEYS = (
    "adapter", "version", "invocation", "fixed_steps", "default_configuration",
    "configurations", "sandbox_layout", "environment", "docs",
)


# inv: a system marked `reference` is kept as a record and never installed or run; a
# configuration marked `declared` has an index and no attempt, and publishes no figure
SYSTEM_STATUSES = frozenset({"reference"})
CONFIGURATION_STATUSES = frozenset({"declared"})


@dataclass
class Harness:
    """The harness-facing description of one system."""

    system: str
    adapter: str
    invocation: dict[str, Any]
    fixed_steps: list[dict[str, Any]]
    configurations: dict[str, dict[str, Any]]
    default_configuration: str
    environment: dict[str, str]
    sandbox_layout: dict[str, str]
    docs: dict[str, Any]
    models: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    volatile: list[str] = field(default_factory=list)
    allowed_scripts: dict[str, str] = field(default_factory=dict)
    ceiling: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping from `path`."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    return data


def harness_path(benchmark: Path, system: str) -> Path:
    """Return `systems/<system>/harness.yaml` under `benchmark`."""
    return benchmark / "systems" / system / "harness.yaml"


def load_harness(benchmark: Path, system: str) -> Harness:
    """Load `systems/<system>/harness.yaml` into a `Harness`.

    Raises
    ------
    ConfigError
        When a required key is absent, or a `status` value is not recognized.
    """
    path = harness_path(benchmark, system)
    data = load_yaml(path)
    missing = [k for k in REQUIRED_HARNESS_KEYS if k not in data]
    if missing:
        raise ConfigError(f"{path}: missing {', '.join(missing)}")
    status = data.get("status")
    if status is not None and status not in SYSTEM_STATUSES:
        raise ConfigError(f"{path}: status {status!r} is not one of {sorted(SYSTEM_STATUSES)}")
    for name, body in data["configurations"].items():
        cstatus = (body or {}).get("status")
        if cstatus is not None and cstatus not in CONFIGURATION_STATUSES:
            raise ConfigError(f"{path}: configuration {name}: status {cstatus!r} is not one of "
                              f"{sorted(CONFIGURATION_STATUSES)}")
    return Harness(
        system=system,
        adapter=str(data["adapter"]),
        invocation=data["invocation"],
        fixed_steps=list(data["fixed_steps"]),
        configurations=data["configurations"],
        default_configuration=str(data["default_configuration"]),
        environment={k: str(v) for k, v in data["environment"].items()},
        sandbox_layout=data["sandbox_layout"],
        docs=data["docs"],
        models=data.get("models") or {},
        status=status,
        volatile=list(data.get("volatile") or []),
        allowed_scripts=data.get("allowed_scripts") or {},
        ceiling=data.get("ceiling"),
        raw=data,
    )


def load_build(index_dir: Path) -> dict[str, Any]:
    """Read `build.yaml` of one index directory."""
    return load_yaml(index_dir / "build.yaml")


def snapshot_dir(benchmark: Path, snapshot: str) -> Path:
    """Return `record/snapshots/<snapshot>` under `benchmark`.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    snapshot : str
        Snapshot id, which must name one path segment directly inside `record/snapshots/`.

    Returns
    -------
    Path
        `record/snapshots/<snapshot>`.

    Raises
    ------
    ConfigError
        When `snapshot` is empty or resolves outside `record/snapshots/`.
    """
    root = (benchmark / "record" / "snapshots").resolve()
    path = (root / snapshot).resolve()
    # inv: a snapshot id is an identifier, not a path; without this a `../` or `a/b` walks out
    # of record/snapshots/ and any directory on the machine could be named by the id
    if not snapshot or path.parent != root:
        raise ConfigError(f"snapshot id is not a bare identifier: {snapshot!r}")
    return path


def _ambiguous(qid: str, paths: list[Path]) -> ConfigError:
    """Return the `ConfigError` for a `qid` held by more than one snapshot, naming every match."""
    named = ", ".join(str(p) for p in paths)
    return ConfigError(f"question {qid} found in more than one snapshot: {named}")


def question_path(benchmark: Path, qid: str) -> Path:
    """Return the one `record/snapshots/*/questions/<qid>.yaml` a qid names.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    qid : str
        Question id, which must name a file directly inside some snapshot's `questions/`.

    Returns
    -------
    Path
        The `questions/<qid>.yaml` file inside the one snapshot that holds it.

    Raises
    ------
    ConfigError
        When `qid` is not a bare identifier, when no snapshot holds it, or when more than one
        does (naming every match).
    """
    snapshots_root = (benchmark / "record" / "snapshots").resolve()
    probe_dir = (snapshots_root / "_" / "questions").resolve()
    # inv: a qid is an identifier, not a path; without this a `../` walks out of questions/ and
    # any YAML on the machine becomes a question, run under an id that names none of it. The
    # probe runs against a synthetic snapshot at the same depth as a real one, so it fires
    # before any filesystem read, even when record/snapshots/ is missing, empty, or holds no
    # directories
    if not qid or (probe_dir / f"{qid}.yaml").resolve().parent != probe_dir:
        raise ConfigError(f"question id is not a bare identifier: {qid!r}")
    snapshot_dirs = sorted(
        p for p in snapshots_root.iterdir() if p.is_dir()
    ) if snapshots_root.is_dir() else []
    matches: list[Path] = []
    for snapshot in snapshot_dirs:
        # inv: existence is checked literally, never via glob, so a wildcard-shaped id
        # (`q?01`, `q*`) never expands to match a different question
        candidate = (snapshot / "questions" / f"{qid}.yaml").resolve()
        if candidate.is_file():
            matches.append(candidate)
    if not matches:
        raise ConfigError(f"no snapshot holds question {qid}")
    if len(matches) > 1:
        raise _ambiguous(qid, matches)
    return matches[0]


def reference_path(benchmark: Path, qid: str) -> Path:
    """Return `references/<qid>.yaml` beside the question `question_path` resolves.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    qid : str
        Question id, resolved the same way `question_path` resolves it.

    Returns
    -------
    Path
        `references/<qid>.yaml` inside the snapshot that holds the question.

    Raises
    ------
    ConfigError
        Whatever `question_path` raises for this `qid`.
    """
    return question_path(benchmark, qid).parent.parent / "references" / f"{qid}.yaml"


def review_path(benchmark: Path, qid: str) -> Path:
    """Return `questions/review/<qid>.yaml` beside the question `question_path` resolves.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    qid : str
        Question id, resolved the same way `question_path` resolves it.

    Returns
    -------
    Path
        `questions/review/<qid>.yaml` inside the snapshot that holds the question.

    Raises
    ------
    ConfigError
        Whatever `question_path` raises for this `qid`.
    """
    return question_path(benchmark, qid).parent / "review" / f"{qid}.yaml"


def question_ids(benchmark: Path) -> list[str]:
    """Return every question id under `record/snapshots/*/questions/`, sorted.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.

    Returns
    -------
    list of str
        Question ids, one per file matching `record/snapshots/*/questions/q*.yaml`, sorted.

    Raises
    ------
    ConfigError
        When the same id is held by more than one snapshot (naming every match).
    """
    root = benchmark / "record" / "snapshots"
    by_id: dict[str, list[Path]] = {}
    # why: question_path resolves an id only to a file, so a directory named q*.yaml must not
    # become an id here that no resolver can then place
    for path in sorted(p for p in root.glob("*/questions/q*.yaml") if p.is_file()):
        by_id.setdefault(path.stem, []).append(path)
    duplicated = sorted(qid for qid, paths in by_id.items() if len(paths) > 1)
    if duplicated:
        raise _ambiguous(duplicated[0], by_id[duplicated[0]])
    return sorted(by_id)


def load_question(benchmark: Path, qid: str) -> dict[str, Any]:
    """Read the `questions/<qid>.yaml` that `question_path` resolves.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    qid : str
        Question id, resolved the same way `question_path` resolves it.

    Returns
    -------
    dict
        The parsed question.

    Raises
    ------
    ConfigError
        Whatever `question_path` raises for this `qid`.
    """
    return load_yaml(question_path(benchmark, qid))


def question_snapshot(question: dict[str, Any], qid: str) -> str:
    """Return the snapshot id a question names.

    Parameters
    ----------
    question : dict
        A parsed question, as `load_question` returns it.
    qid : str
        The question's id, named in the error when the key is missing.

    Returns
    -------
    str
        The snapshot id.

    Raises
    ------
    ConfigError
        When the question names no snapshot.
    """
    # inv: there is no fallback snapshot -- a question silently run against a corpus it was not
    # written from produces a verdict about the wrong tree, and the reference would name places
    # no index holds
    snapshot = question.get("snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        raise ConfigError(f"question names no snapshot: {qid}")
    return snapshot
