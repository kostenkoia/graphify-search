"""Safe filesystem writes."""

from __future__ import annotations

import os
import stat
import tempfile
import time
from pathlib import Path

from graphify_search.errors import InputError

# inv: every knob below is a plain default; there is no config module to consult.
# NOT DERIVED: five tries 20 ms apart outlast the moment a Windows scanner holds a just-written file
REPLACE_RETRIES = 5
REPLACE_RETRY_DELAY = 0.02


def _default_new_file_mode() -> int:
    """Return the mode a brand-new file gets under the process umask."""
    # why: 0o666 is the base open() starts from before umask is subtracted
    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def _resolve_write_mode(path: Path) -> int:
    """Return the mode to apply to the replacement file.

    Returns
    -------
    int
        The target's own mode, or the umask default when the target is absent
        or carries 0o600.
    """
    # why: 0o600 is NamedTemporaryFile's own umask-blind mode, so it heals to the umask default
    # instead of being carried forward as drift
    try:
        existing = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return _default_new_file_mode()
    if existing == 0o600:
        return _default_new_file_mode()
    return existing


def _replace_with_retry(tmp_name: str, path: Path) -> None:
    """Rename `tmp_name` onto `path`, retrying a Windows PermissionError."""
    # why: on Windows a reader or a scanner can hold the destination open, while POSIX os.replace
    # never raises PermissionError, so the loop is a no-op there
    attempts = REPLACE_RETRIES
    for attempt in range(attempts):
        try:
            os.replace(tmp_name, str(path))
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise InputError(
                    f"could not replace {path}", hint="retry in a moment",
                ) from None
            time.sleep(REPLACE_RETRY_DELAY)


def _fsync_parent_dir(parent: Path) -> None:
    """Fsync the parent directory, best-effort."""
    # why: the directory entry is what makes a completed rename durable across a crash
    # why: Windows cannot os.open a directory, so its OSError leaves the fsync best-effort
    try:
        fd = os.open(str(parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, *, mode: str, data: str | bytes, encoding: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = _resolve_write_mode(path)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False,
                                         encoding=encoding) as tmp:
            tmp_name = tmp.name
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(tmp.fileno(), file_mode)
        if not hasattr(os, "fchmod"):
            os.chmod(tmp_name, file_mode)
        _replace_with_retry(tmp_name, path)
    except BaseException:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    _fsync_parent_dir(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file and a rename.

    Parameters
    ----------
    path : Path
        Destination file.
    text : str
        UTF-8 content to write.
    """
    _atomic_write(Path(path), mode="w", data=text, encoding="utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` through a temporary file and a rename.

    Parameters
    ----------
    path : Path
        Destination file.
    data : bytes
        Content to write.
    """
    _atomic_write(Path(path), mode="wb", data=data, encoding=None)
