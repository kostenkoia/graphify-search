"""Replay the harness's own rules over one finished run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from benchmark.harness import config, ledger, prepare, rules

# inv: any absolute path of two or more segments, so the allowed-roots filter below decides
# what may appear; naming the roots here instead would let a leak under an unlisted one pass
_ABS = re.compile(r"(?<![\w/])/[A-Za-z0-9_.\-]+(?:/[^\s'\"]+)+")
_STEM = "{n:02d}_{name}"


def _names_something_here(candidate: str) -> bool:
    """Report whether an absolute path found in output actually names a file on this machine.

    Parameters
    ----------
    candidate : str
        An absolute path as it appeared in a `.out` or `.err`.

    Returns
    -------
    bool
        True when the path, or any parent of it, exists here.
    """
    # why: a node label can carry a route inside ordinary prose -- "tests for GET /invoices/totals."
    # -- which is slash-led and scans as absolute without naming a file anywhere. Existence is what
    # separates it from the leak this check exists to catch, a path of this machine's own.
    # inv: parents are walked too, so a leaked directory with a fabricated tail still counts
    path = Path(candidate)
    for probe in (path, *path.parents):
        if probe == Path(probe.root):
            return False
        try:
            if probe.exists():
                return True
        except OSError:
            return False
    return False


def _artifact_bytes(index_dir: Path, build: dict) -> bytes:
    """Concatenate the bytes of every artifact one index's `build.yaml` declares.

    Parameters
    ----------
    index_dir : Path
        The master index directory the run's sandbox was laid out from.
    build : dict
        That index's parsed `build.yaml`.

    Returns
    -------
    bytes
        The declared artifacts' bytes, joined by a NUL that no path can span.
    """
    blobs = [path.read_bytes() for rel in build.get("artifacts") or {}
             if (path := index_dir / rel).is_file()]
    return b"\0".join(blobs)


def _under_any(candidate: str, roots: tuple[Path, ...]) -> bool:
    # inv: comparing resolved Paths, not prefix strings, so a symlinked spelling of a root still
    # matches it and a sibling whose name merely starts with a root's name never does
    resolved = Path(candidate).resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def _prepared_output_violations(build: dict, index_dir: Path, meta: dict, calls: list[dict]) -> list[str]:
    """Replay prepare's `prepared_outputs` comparison over a finished run's journal.

    Parameters
    ----------
    build : dict
        The index's parsed `build.yaml`.
    index_dir : Path
        The master index directory holding the recorded expectations.
    meta : dict
        The run's parsed `run.yaml`.
    calls : list of dict
        The journal's `call` entries.

    Returns
    -------
    list of str
        One message per step whose recorded output no longer matches the
        expectation `build.yaml` holds for this run's recipe.
    """
    recipe = meta.get("harness_sha256")
    # inv: expectations are keyed by recipe and question, so a run.yaml naming neither matches
    # nothing and the replay is skipped rather than reported clean
    if not isinstance(recipe, str):
        return []
    by_recipe = prepare.load_prepared(index_dir).get(meta.get("configuration")) or {}
    expectations = (by_recipe.get(recipe) or {}).get(meta.get("question")) or {}
    excluded, mutable = list(build.get("excluded") or []), list(build.get("mutable") or [])
    problems = []
    for e in calls:
        expected = expectations.get(e.get("name"))
        if expected is None:
            continue
        observed = {"out": e.get("canonical_sha256"),
                    "files": {f["path"]: f["sha256"] for f in e.get("files") or []
                              if prepare.keep_for_prepared(f["path"], excluded, mutable)}}
        if observed != expected:
            problems.append(f"entry {e['n']}: {e.get('name')} differs from prepared_outputs")
    return problems


def check_run(
    run_dir: Path,
    invocation: dict,
    *,
    model_paths: frozenset[str] | set[str] = frozenset(),
    skip_environment: bool = False,
    benchmark: Path | None = None,
) -> dict:
    """Replay the harness's own rules over a finished run and write `audit.json`.

    Parameters
    ----------
    run_dir : Path
        The run's `run/` directory.
    invocation : dict
        The system's `invocation` mapping, plus `allowed_scripts`.
    model_paths : set of str, optional
        `home/`-relative paths the frozen model owns; exempt from attribution.
    skip_environment : bool, optional
        Skip the vendor-environment and launcher/interpreter hash checks.
    benchmark : Path or None, optional
        Root holding `systems/` and `record/snapshots/`; when given, the system's
        `harness.yaml` is the authority for the expected fixed-step count
        instead of `run.yaml`'s own copy, and the master index is reverified
        so `master_index_changed` reflects it; when absent,
        `master_index_changed` stays `False`.

    Returns
    -------
    dict
        `{"run", "valid", "stop", "violations", "master_index_changed"}`.
    """
    # inv: collect.py has already cleaned the live sandbox a run under benchmark/record/runs/ was
    # judged against, so re-auditing there would overwrite the audit.json the ledger's hash already
    # names
    if benchmark is not None and _under_any(str(run_dir), ((benchmark / ledger.RECORD / "runs").resolve(),)):
        raise SystemExit(
            f"refusing to re-audit collected evidence under {benchmark / ledger.RECORD / 'runs'}: {run_dir}")
    meta = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    tmp_root = Path(meta["tmp_root"])
    allowed_roots = (Path(tmp_root / "sandbox").resolve(), Path(tmp_root / meta["run_id"]).resolve())
    sandbox = tmp_root / "sandbox" / "index"
    violations: list[str] = []
    entries = [json.loads(line) for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    stops = [i for i, e in enumerate(entries) if e.get("kind") == "stop"]
    if not entries or len(stops) != 1 or stops[0] != len(entries) - 1:
        violations.append("journal must carry exactly one stop entry, last")
    calls = [e for e in entries if e.get("kind") == "call"]
    # inv: fixed_steps counts the harness's own calls; a runner's are bounded by its action
    # limit, which harness.yaml does not record, so counting them here would fail every driven run
    harness_calls = [e for e in calls if e.get("by") != "runner"]
    h = config.load_harness(benchmark, meta["system"]) if benchmark is not None else None
    # why: harness.yaml is git-tracked and run.yaml is not, so its step count is the authority
    # whenever a benchmark root is given
    expected_steps, source = (len(h.fixed_steps), "harness.yaml") if h is not None \
        else (meta.get("fixed_steps"), "run.yaml")
    if expected_steps is not None and len(harness_calls) != expected_steps:
        violations.append(f"journal has {len(harness_calls)} harness call entries, "
                          f"{source} records fixed_steps={expected_steps}")
    index_dir: Path | None = None
    build: dict = {}
    # inv: `h` exists only when `benchmark` does, so the second test never changes the branch taken
    if h is not None and benchmark is not None and "snapshot" in meta and "configuration" in meta:
        index_dir = (config.snapshot_dir(benchmark, meta["snapshot"])
                     / h.configurations[meta["configuration"]]["index"])
        build = config.load_build(index_dir)
    # why: existence on this machine separates a real path from a fabricated one; it cannot
    # separate a path leaked from the sandbox from one the corpus itself quotes. A string the
    # frozen artifact already holds is corpus content -- the sandbox is laid out from that
    # artifact, prepare.verify_master hashed it before the run, and the tool under test cannot
    # edit it -- so it names nothing about this machine that the artifact did not already carry.
    corpus = _artifact_bytes(index_dir, build) if index_dir is not None else b""
    seen_files: set[str] = set()
    claimed: dict[str, dict] = {}
    for e in entries:
        seen_files.update(f["path"] for f in e.get("files") or [])
        if e.get("kind") == "server" and e.get("event") == "start":
            argv = list(e.get("argv") or [])
            if "--http" in argv:
                violations.append(f"entry {e['n']}: server argv uses --http")
            idx = argv.index("--repo") if "--repo" in argv else -1
            if idx == -1 or idx + 1 >= len(argv) or argv[idx + 1] != str(sandbox):
                violations.append(f"entry {e['n']}: server argv does not name --repo {sandbox}")
        if e.get("kind") != "call":
            continue
        claimed[_STEM.format(n=e["n"], name=e["name"])] = e
        if e.get("action") is False:
            continue
        stem = run_dir / _STEM.format(n=e["n"], name=e["name"])
        out = stem.with_suffix(".out")
        # inv: the exemption keys on the full failed shape execute.py writes -- error, exit None
        # and no .out -- so a planted "error" key alone waves nothing through
        if "error" in e and e.get("exit") is None and not out.exists():
            continue
        # why: out_sha256 hashes raw .out bytes; canonical_sha256 hashes those bytes decoded with
        # errors="replace" (execute.execute), so reproducing it needs that same decode, not this one
        if not out.exists() or rules.sha256_file(out) != e.get("out_sha256"):
            violations.append(f"entry {e['n']}: out_sha256 mismatch")
        call = ({"kind": "act", "argv": e["argv"]} if "argv" in e
                else {"kind": "tool", "tool": e["tool"], "args": e.get("args", {})})
        if (reason := rules.check_call(invocation, call)) is not None:
            violations.append(f"entry {e['n']}: {reason}")
        if e.get("exit") != 0:
            violations.append(f"entry {e['n']}: exit {e.get('exit')}")
    # inv: every *.out/*.err in the run directory is scanned, whether or not its entry ran to
    # completion, so a refused entry's planted output and an orphan file both surface
    for p in sorted(run_dir.glob("*.out")) + sorted(run_dir.glob("*.err")):
        if p.stem not in claimed:
            violations.append(f"{p.name}: not claimed by any journal entry")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        leak = next((m for m in _ABS.findall(text)
                     if _names_something_here(m) and not _under_any(m, allowed_roots)
                     and m.encode("utf-8") not in corpus), None)
        if leak is not None:
            violations.append(f"{p.name}: absolute path outside the run: {leak}")
        if "Embedding search failed" in text:
            violations.append(f"{p.name}: Embedding search failed")
    home = tmp_root / meta["run_id"] / "home"
    artifacts = meta.get("artifacts") or {}
    master_index_changed = False
    if index_dir is not None:
        violations.extend(_prepared_output_violations(build, index_dir, meta, calls))
        try:
            prepare.verify_master(index_dir, build)
        except SystemExit as exc:
            master_index_changed = True
            violations.append(f"master index: {exc}")
    # inv: every sandbox file that is not a declared artifact must be named by some entry's
    # `files`, with no exemption for `build.yaml`'s `mutable` list
    for root, label, exempt in ((sandbox, "sandbox", set(artifacts)), (home, "home", model_paths)):
        # inv: prepare.make_sandbox creates both roots, so an absent one means the audit could
        # not look, never that nothing is there
        if not root.exists():
            violations.append(f"{label} root absent, so nothing could be attributed: {root}")
            continue
        try:
            listing = rules.listing(root)
        except NotADirectoryError:
            violations.append(f"{label} root is not a directory: {root}")
            continue
        for rel in listing:
            if rel not in exempt and f"{label}/{rel}" not in seen_files:
                violations.append(f"{label} file not attributed: {rel}")
    if not skip_environment:
        site = Path(invocation["package"]["site"])
        if rules.environment_hash(site) != meta.get("environment_sha256"):
            violations.append("environment_sha256 differs")
        for key in ("launcher", "interpreter"):
            if rules.sha256_file(Path(invocation["package"][key])) != meta.get(f"{key}_sha256"):
                violations.append(f"{key}_sha256 differs")
    result = {"run": meta["run_id"], "valid": not violations, "stop": entries[-1].get("reason") if entries else None,
              "violations": violations, "master_index_changed": master_index_changed}
    (run_dir / "audit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
