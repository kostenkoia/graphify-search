"""Turn a run's raw outputs into records, cost and hits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import tiktoken
import yaml

from benchmark.harness import config
from benchmark.harness.scoring import adapters

if TYPE_CHECKING:
    from collections.abc import Callable


class UnparsedError(Exception):
    """An adapter met a shape it was not written for."""


def _journal(run_dir: Path) -> list[dict]:
    lines = (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _run_yaml(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))


def _out_path(run_dir: Path, entry: dict) -> Path:
    return run_dir / f"{entry['n']:02d}_{entry['name']}.out"


def _executed(entries: list[dict], run_dir: Path) -> list[dict]:
    # inv: execute.py pairs "error" only with exit: None and no .out; the exemption here
    # matches audit.check_run's, keyed on that full shape rather than the "error" tag alone,
    # so a tampered journal cannot tag a real .out into being skipped
    return [e for e in entries
            if e.get("kind") == "call" and e.get("action") is not False
            and not ("error" in e and e.get("exit") is None and not _out_path(run_dir, e).exists())]


def _path_prefix(benchmark: Path, h: config.Harness, meta: dict, configuration: str) -> str | None:
    """Return the index root the configuration's vendor paths carry, or None when relative."""
    if "snapshot" not in meta:
        return None
    build = config.load_build(config.snapshot_dir(benchmark, meta["snapshot"])
                              / h.configurations[configuration]["index"])
    if (build.get("properties") or {}).get("paths_in_index") != "absolute":
        return None
    # inv: `build_cwd` is where an index records the root its stored paths carry, so an index
    # declaring absolute paths without it cannot be made comparable to the reference at all
    cwd = build.get("build_cwd")
    if not isinstance(cwd, str) or not cwd:
        raise config.ConfigError(f"build.yaml declares paths_in_index: absolute without build_cwd: {meta['snapshot']}")
    return cwd.rstrip("/")


def records(benchmark: Path, run_dir: Path) -> list[dict]:
    """Write `records.jsonl` from every system call's `.out`.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/<system>/harness.yaml`.
    run_dir : Path
        The run's `run/` directory.

    Returns
    -------
    list of dict
        Every record parsed from the run, in journal order.

    Raises
    ------
    UnparsedError
        When the version line carries neither the system's `version.cli` nor,
        failing that, the adapter's `VERSION`, or any record comes back
        `unparsed`.
    KeyError
        When `run.yaml`'s configuration is not one of the system's
        `harness.yaml` `configurations`.
    ConfigError
        When the index declares `paths_in_index: absolute` with no `build_cwd`.
    """
    meta = _run_yaml(run_dir)
    h = config.load_harness(benchmark, meta["system"])
    adapter = adapters.load(h.adapter)
    # inv: an unknown configuration must raise, not fall back to an empty mapping that
    # silently drops the search-mode check the named configuration would have required
    configuration = meta.get("configuration", h.default_configuration)
    modes = h.configurations[configuration].get("search_mode")
    prefix = _path_prefix(benchmark, h, meta, configuration)
    # why: one adapter serves several cells and each names the wheel its own harness.yaml froze,
    # so the system's own `version.cli` is what its recorded output must carry; the adapter's
    # VERSION stands only for a system that declares none
    expected = str((h.raw.get("version") or {}).get("cli") or adapter.VERSION)
    out: list[dict] = []
    for e in _executed(_journal(run_dir), run_dir):
        if not e.get("system_call"):
            continue
        text = _out_path(run_dir, e).read_text(encoding="utf-8", errors="replace")
        if e.get("name") == "version" and expected not in text:
            raise UnparsedError(f"version output {text.strip()!r} does not carry version {expected}")
        for r in adapter.parse(e, text, search_modes=modes if isinstance(modes, list) else None,
                               path_prefix=prefix):
            out.append({"run": meta["run_id"], "n": e["n"], "by": e.get("by"), **r})
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    (run_dir / "records.jsonl").write_text(payload, encoding="utf-8")
    # inv: a fixed step's output is the recipe's own and must parse, or the recipe is wrong; a
    # call the runner chose may print anything the grammar admits, and an output no place can
    # be read from is that call answering nothing, kept as an unparsed record and scored as none
    bad = [r for r in out if r["kind"] == "unparsed" and r.get("by") != "runner"]
    if bad:
        raise UnparsedError(f"{len(bad)} unparsed record(s), first: {bad[0].get('text', '')!r}")
    return out


def parser_for(benchmark: Path, meta: dict) -> Callable[[dict, str], list[dict]]:
    """Return the callable that turns one journal entry's output into records.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/` and `record/snapshots/`.
    meta : dict
        A run's `run.yaml`, or any mapping carrying its `system`, `configuration`
        and `snapshot`.

    Returns
    -------
    callable
        `parse(entry, text) -> list of dict`, bound to the system's adapter, its
        configuration's search modes and its index's path prefix.
    """
    h = config.load_harness(benchmark, meta["system"])
    adapter = adapters.load(h.adapter)
    configuration = meta.get("configuration", h.default_configuration)
    modes = h.configurations[configuration].get("search_mode")
    prefix = _path_prefix(benchmark, h, meta, configuration)

    def parse(entry: dict, text: str) -> list[dict]:
        return adapter.parse(entry, text, search_modes=modes if isinstance(modes, list) else None,
                             path_prefix=prefix)

    return parse


def cost(run_dir: Path) -> dict:
    """Write `cost.json`: cl100k_base tokens over every executed call's `.out`, plus counts.

    Parameters
    ----------
    run_dir : Path
        The run's `run/` directory.

    Returns
    -------
    dict
        `{"run", "tokens", "system_calls", "ceiling_calls", "actions"}`.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = system_calls = ceiling_calls = actions = 0
    for e in _executed(_journal(run_dir), run_dir):
        actions += 1
        system_calls += int(bool(e.get("system_call")))
        ceiling_calls += int(bool(e.get("ceiling_call")))
        text = _out_path(run_dir, e).read_text(encoding="utf-8", errors="replace")
        tokens += len(enc.encode(text, disallowed_special=()))
    result = {"run": _run_yaml(run_dir)["run_id"], "tokens": tokens, "system_calls": system_calls,
              "ceiling_calls": ceiling_calls, "actions": actions}
    (run_dir / "cost.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _is_hit(rec: dict, ref: dict) -> bool:
    # inv: a reference without a symbol matches any symbol; one naming a symbol needs equality
    if rec.get("kind") != "place" or rec.get("path") != ref["path"] or rec.get("start") is None:
        return False
    if not int(ref["start"]) <= int(rec["start"]) <= int(ref.get("end") or ref["start"]):
        return False
    return ref.get("symbol") is None or rec.get("symbol") == ref["symbol"]


def first_hit(run_dir: Path, reference: Path) -> dict:
    """Return the first record matching any reference place, and where it sat.

    Parameters
    ----------
    run_dir : Path
        The run's `run/` directory; `records.jsonl` must already exist.
    reference : Path
        A `references/<qid>.yaml` file carrying a `places` list.

    Returns
    -------
    dict
        `{"hit", "hit_rank", "hit_entry"}`.
    """
    refs = yaml.safe_load(reference.read_text(encoding="utf-8"))["places"]
    for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if any(_is_hit(rec, ref) for ref in refs):
            return {"hit": True, "hit_rank": rec.get("rank"), "hit_entry": rec["n"]}
    return {"hit": False, "hit_rank": None, "hit_entry": None}


def stop_reason(run_dir: Path) -> str | None:
    """Return the reason the journal's stop entry names, or None when it has none."""
    for entry in reversed(_journal(run_dir)):
        if entry.get("kind") == "stop":
            return entry.get("reason")
    return None


def _stop_place(run_dir: Path) -> dict | None:
    for entry in reversed(_journal(run_dir)):
        if entry.get("kind") == "stop":
            return entry.get("place")
    return None


def runner_verdict(run_dir: Path, reference: Path, path_prefix: str | None = None) -> dict:
    """Return what the runner answered and who the hit is credited to.

    Parameters
    ----------
    run_dir : Path
        The run's `run/` directory; `records.jsonl` must already exist.
    reference : Path
        A `references/<qid>.yaml` file carrying a `places` list.
    path_prefix : str or None
        The index root this system's paths carry, when it stores them absolute.

    Returns
    -------
    dict
        `{"stop", "stop_hit", "hit_by"}`; `hit_by` is `runner` when the runner named a
        reference place, `harness` when only the vendor's own output carried one, and
        None when neither did.
    """
    refs = yaml.safe_load(reference.read_text(encoding="utf-8"))["places"]
    place = _stop_place(run_dir)
    # inv: the runner's answer is compared exactly as a record is, so a place it names is
    # neither easier nor harder to score than one the vendor printed
    # inv: the runner's symbol is read exactly as a vendor's label is -- the harness strips the
    # same decorations from both, or one place scores two ways depending on who named it
    symbol = adapters.symbol_of(str(place["symbol"])) if place and place.get("symbol") else None
    named = {**place, "kind": "place", "symbol": symbol} if place else None
    if named is not None:
        # inv: the runner's path is made corpus-relative exactly as a vendor record's is; a path
        # outside the index root comes back unparsed, which is a miss, as it should be
        named = adapters.normalize_paths([named], path_prefix)[0]
    stop_hit = bool(named and any(_is_hit(named, ref) for ref in refs))
    # why: the run's own answer is its stop, so a runner that named the place is credited even
    # when the vendor's output carried it too; the record hit stays reported on its own
    hit_by = None
    if stop_hit:
        hit_by = "runner"
    elif first_hit(run_dir, reference)["hit"]:
        hit_by = "harness"
    return {"stop": stop_reason(run_dir), "stop_hit": stop_hit, "hit_by": hit_by}


def runner_actions(run_dir: Path) -> dict:
    """Return how many calls the runner spent and how many of them were refused."""
    entries = [e for e in _journal(run_dir) if e.get("kind") == "call" and e.get("by") == "runner"]
    return {"runner_actions": len(entries),
            "refused": sum(1 for e in entries if e.get("action") is False)}


def model_usage(run_dir: Path) -> dict:
    """Sum the usage every recorded exchange reports.

    Parameters
    ----------
    run_dir : Path
        The run's `run/` directory, whose `api/` holds one file per exchange.

    Returns
    -------
    dict
        Every usage key the responses carried, summed; empty when no model drove the run.
    """
    # inv: the exchanges are what the model was actually billed for, so usage is summed from
    # them and never from the journal, which counts the vendor's output instead
    totals: dict[str, int] = {}
    for path in sorted((run_dir / "api").glob("*.json")) if (run_dir / "api").is_dir() else []:
        usage = (json.loads(path.read_text(encoding="utf-8")).get("response") or {}).get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _driven(run_dir: Path) -> bool:
    return any(e.get("by") == "runner" for e in _journal(run_dir))


def hits(benchmark: Path, run_dir: Path) -> dict:
    """Score a run against its question's reference.

    Parameters
    ----------
    benchmark : Path
        Root holding the snapshot that carries `references/<qid>.yaml`, and `systems/`.
    run_dir : Path
        The run's `run/` directory.

    Returns
    -------
    dict
        `hit`, `hit_rank`, `hit_entry`; when a model drove the run, also the runner's
        verdict, action count and usage.
    """
    meta = _run_yaml(run_dir)
    reference = config.reference_path(benchmark, meta["question"])
    result = first_hit(run_dir, reference)
    # inv: a run no model drove keeps the shape every published baseline was written in
    if _driven(run_dir):
        h = config.load_harness(benchmark, meta["system"])
        prefix = _path_prefix(benchmark, h, meta, meta.get("configuration", h.default_configuration))
        result = {**result, **runner_verdict(run_dir, reference, prefix), **runner_actions(run_dir),
                  "model_usage": model_usage(run_dir)}
    return result


def main(argv: list[str] | None = None) -> int:
    """Score one run from the command line.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        1 when `records` raises `UnparsedError`, 0 otherwise.
    """
    ap = argparse.ArgumentParser(prog="benchmark.harness score")
    ap.add_argument("run_dir", type=Path, nargs="?")
    ap.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)
    try:
        records(args.benchmark, args.run_dir)
    except UnparsedError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(cost(args.run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
