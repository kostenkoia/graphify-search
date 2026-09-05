"""Adapters that turn one system's raw output into records."""

from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load(name: str) -> ModuleType:
    """Import the adapter module registered under `name`.

    Parameters
    ----------
    name : str
        Value of `adapter` in the system's `harness.yaml`.

    Returns
    -------
    ModuleType
        A module exposing `VERSION` and
        `parse(call, text, *, search_modes=None, path_prefix=None)`.
    """
    return importlib.import_module(f"benchmark.harness.scoring.adapters.{name}")


def normalize_paths(records: list[dict], prefix: str | None) -> list[dict]:
    """Strip the index root from every record `path`, or reject the first that lacks it.

    Parameters
    ----------
    records : list of dict
        Records as the adapter built them; entries with no `path` pass through.
    prefix : str or None
        The index root vendor paths carry, from `build.yaml`; `None` when the
        index stores paths already relative to the corpus.

    Returns
    -------
    list of dict
        `records` with each `path` made corpus-relative, or a single `unparsed`
        record when one `path` does not sit under `prefix`.
    """
    if prefix is None:
        return records
    # inv: a path outside the declared index root means the index is not what its build.yaml
    # says; scoring compares paths literally, so silently keeping it would read as a clean miss
    for r in records:
        p = r.get("path")
        if not isinstance(p, str):
            continue
        rest = p[len(prefix) + 1:] if p.startswith(prefix + "/") else ""
        # inv: an empty remainder names the root itself and a leading slash names a doubled
        # separator; both strip to something no reference can match, so they fail here loudly
        if not rest or rest.startswith("/"):
            return [{"kind": "unparsed", "text": f"path {p!r} is not a file under index root {prefix!r}"}]
    # inv: `qualified_name` spells the same location as `path`, so it is stripped with it; nothing
    # scores it, so a value outside the root passes through as the vendor wrote it
    return [{k: v[len(prefix) + 1:] if k in ("path", "qualified_name") and isinstance(v, str)
             and v.startswith(prefix + "/") else v for k, v in r.items()} for r in records]


def symbol_of(label: str) -> str | None:
    """Return the label as a symbol, with a vendor's `.` prefix and `()` suffix removed."""
    core = label[:-2] if label.endswith("()") else label
    # inv: a method is decorated with a leading dot by one vendor and written bare by the other,
    # so the decoration is stripped here or the same place scores a hit for one and not the other
    core = core[1:] if core.startswith(".") else core
    return core if _IDENT.match(core) else None
