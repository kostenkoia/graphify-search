"""Copy the embedding model directory and print its hashes for `harness.yaml`."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from benchmark.harness import rules


def _validate_link(path: Path, dst: Path) -> str:
    """Return the raw target of a symlink after checking it resolves inside `dst`.

    Parameters
    ----------
    path : Path
        Symlink to inspect.
    dst : Path
        Root of the frozen copy; every symlink must resolve to a real file or
        directory inside it.

    Returns
    -------
    str
        The raw `os.readlink` value, for reconstructing the link elsewhere.

    Raises
    ------
    SystemExit
        When the target does not resolve, or resolves outside `dst`.
    """
    raw = os.readlink(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"{path}: symlink does not resolve: {raw}") from exc
    if not resolved.is_relative_to(dst.resolve()):
        raise SystemExit(f"{path}: symlink target escapes {dst}: {raw}")
    return raw


def freeze(src: Path, dst: Path) -> dict[str, dict[str, str]]:
    """Copy `src` to `dst` with symlinks preserved and hash everything underneath.

    Parameters
    ----------
    src : Path
        Directory to copy.
    dst : Path
        Destination; must not already exist.

    Returns
    -------
    dict of str to dict of str to str
        `{"files": {relative_path: sha256}, "links": {relative_path: raw_target}}`.

    Raises
    ------
    SystemExit
        When `dst` already exists, or a symlink under it is dangling or
        escapes `dst` (see `_validate_link`).
    """
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    shutil.copytree(src, dst, symlinks=True)
    files: dict[str, str] = {}
    links: dict[str, str] = {}
    # inv: a symlinked directory is yielded in `dirs`, not `names` -- both are walked
    # the same way so a link to a directory is never left unattributed
    for dirpath, dirs, names in os.walk(dst, followlinks=False):
        for name in (*dirs, *names):
            p = Path(dirpath) / name
            rel = p.relative_to(dst).as_posix()
            if p.is_symlink():
                links[rel] = _validate_link(p, dst)
            elif p.is_file():
                files[rel] = rules.sha256_file(p)
    return {"files": files, "links": links}


def main(argv: list[str] | None = None) -> int:
    """Freeze a model directory and print its `models` block.

    Parameters
    ----------
    argv : list of str or None
        `--src` and `--dst`, or None to use `sys.argv`.

    Returns
    -------
    int
        Zero.
    """
    ap = argparse.ArgumentParser(prog="benchmark.harness freeze-model")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args(argv)
    result = freeze(args.src, args.dst)
    print(yaml.safe_dump({args.dst.name: result}, sort_keys=True, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
