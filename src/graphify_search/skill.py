"""Install, remove and detect the packaged SKILL.md for Claude Code."""

from __future__ import annotations

import re
import shutil
from importlib import resources
from pathlib import Path

from graphify_search.errors import InputError
from graphify_search.utils import atomic_write_text

SKILL_NAME = "graphify-search"
_SKILLS_SUBPATH = Path(".claude") / "skills"
# inv: the name is read from the frontmatter block alone, so prose that merely mentions this
# package is not mistaken for the skill and overwritten
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)^---[ \t]*\r?$", re.DOTALL | re.MULTILINE)
_NAME_LINE = re.compile(r"^name:[ \t]*[\"']?graphify-search[\"']?[ \t]*$", re.MULTILINE)


def packaged_skill_md() -> Path:
    """Locate the SKILL.md shipped with this package.

    Returns
    -------
    Path
        Path to the packaged file.

    Raises
    ------
    InputError
        When neither the package resource nor the source-tree copy is readable.
    """
    try:
        packaged = resources.files("graphify_search") / "SKILL.md"
        if packaged.is_file():
            return Path(str(packaged))
    except (ModuleNotFoundError, TypeError, OSError):
        pass
    source_tree = Path(__file__).resolve().parents[2] / "SKILL.md"
    if source_tree.is_file():
        return source_tree
    raise InputError("packaged SKILL.md not found", hint="reinstall graphify-search")


def skill_dir(root: Path, *, global_install: bool = False) -> Path:
    """Return the directory SKILL.md belongs in for one scope.

    Parameters
    ----------
    root : Path
        Project root, used for the project scope.
    global_install : bool
        Target the user's home directory instead of the project.

    Returns
    -------
    Path
        The skill's own directory, which may not exist yet.
    """
    base = Path.home() if global_install else root
    return base / _SKILLS_SUBPATH / SKILL_NAME


def _reject_symlink(path: Path) -> None:
    # why: a symlinked target would redirect the write outside the scope the
    # caller asked for, so the write is refused rather than followed.
    if path.is_symlink():
        raise InputError(f"refusing to write through a symlink: {path}",
                         hint="remove the symlink, then re-run")


def _reject_symlink_chain(base: Path, target: Path) -> None:
    # inv: every component from `base` down to `target` is checked, because a symlink anywhere
    # on the way -- `.claude` most of all -- redirects the whole write or removal out of scope;
    # `base` itself is the caller's own argument and is followed as given
    current = base
    for part in target.relative_to(base).parts:
        current = current / part
        _reject_symlink(current)


def _is_this_skill(text: str) -> bool:
    """Tell whether `text` is this package's skill file, by its frontmatter name.

    Parameters
    ----------
    text : str
        Content of the file found at the install target.

    Returns
    -------
    bool
        True when a leading frontmatter block names this skill.
    """
    block = _FRONTMATTER.match(text)
    return bool(block and _NAME_LINE.search(block.group(1)))


def install(root: Path, *, global_install: bool = False, force: bool = False) -> dict:
    """Write the packaged SKILL.md into the Claude Code skills directory.

    Parameters
    ----------
    root : Path
        Project root.
    global_install : bool
        Install under the user's home directory instead of the project.
    force : bool
        Overwrite an existing file whose content is not this skill.

    Returns
    -------
    dict
        Record with ``scope``, ``path`` and ``written``.

    Raises
    ------
    InputError
        When the target is a symlink, holds foreign content without ``force``,
        or cannot be written.
    """
    base = Path.home() if global_install else root
    target_dir = skill_dir(root, global_install=global_install)
    target = target_dir / "SKILL.md"
    _reject_symlink_chain(base, target)
    if target.exists() and not force:
        existing = target.read_text(encoding="utf-8", errors="replace")
        if not _is_this_skill(existing):
            raise InputError(f"{target} holds different content",
                             hint="pass force=True to overwrite it")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, packaged_skill_md().read_text(encoding="utf-8"))
    except OSError as e:
        raise InputError(f"could not write {target}: {e}", hint="check permissions") from e
    return {"scope": "global" if global_install else "project",
            "path": str(target), "written": True}


def uninstall(root: Path, *, global_install: bool = False) -> dict:
    """Remove the skill directory for one scope.

    Parameters
    ----------
    root : Path
        Project root.
    global_install : bool
        Act under the user's home directory instead of the project.

    Returns
    -------
    dict
        Record with ``scope``, ``path``, ``removed`` and a ``reason`` when
        nothing was removed.
    """
    base = Path.home() if global_install else root
    target_dir = skill_dir(root, global_install=global_install)
    scope = "global" if global_install else "project"
    if not target_dir.exists():
        return {"scope": scope, "path": str(target_dir), "removed": False,
                "reason": "not installed"}
    _reject_symlink_chain(base, target_dir)
    try:
        shutil.rmtree(target_dir)
    except OSError as e:
        raise InputError(f"could not remove {target_dir}: {e}", hint="check permissions") from e
    return {"scope": scope, "path": str(target_dir), "removed": True}


def detect(root: Path) -> dict:
    """Report where the skill is installed and whether Claude Code is present.

    Parameters
    ----------
    root : Path
        Project root.

    Returns
    -------
    dict
        Record with ``cli_installed``, ``config_dir_exists``,
        ``installed_project`` and ``installed_global``.
    """
    return {
        "cli_installed": shutil.which("claude") is not None,
        "config_dir_exists": (Path.home() / ".claude").is_dir(),
        "installed_project": (skill_dir(root) / "SKILL.md").is_file(),
        "installed_global": (skill_dir(root, global_install=True) / "SKILL.md").is_file(),
    }
