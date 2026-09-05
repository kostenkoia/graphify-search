"""Prepare one run: record the attempt, verify, copy, execute the fixed steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from benchmark.harness import blind, config, execute, expand, ledger, prompt, questions, rules, seal

if TYPE_CHECKING:
    from benchmark.harness import mcp

# why: the sandbox is rebuilt per run and holds nothing worth keeping, so it belongs under
# the platform's own temp root rather than a path only one machine has
DEFAULT_TMP_ROOT = Path(tempfile.gettempdir()) / "graphify-bench"
# inv: the harness's own record of what each step printed, kept beside the index but out of
# build.yaml -- build.yaml is a hand-authored freeze record and must not change after the freeze
PREPARED = "prepared_outputs.yaml"
# inv: the characters a directory name may carry here; anything else could leave tmp_root
_RUN_ID = re.compile(r"[A-Za-z0-9_.-]+")
# why: graphify's vocabulary step selects at most 12 tokens (its query.md)
MAX_EXPANSION_TOKENS = 12


def take_lock(tmp_root: Path, run_id: str) -> None:
    """Create `<tmp root>/sandbox/lock`; exit when another live run holds it."""
    lock = tmp_root / "sandbox" / "lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "run_id": run_id, "started": time.time()})
    try:
        # inv: exclusive creation, not exists()-then-write, so two concurrent callers cannot both pass
        with lock.open("x", encoding="utf-8") as fh:
            fh.write(payload)
        return
    except FileExistsError:
        pass
    try:
        held = json.loads(lock.read_text(encoding="utf-8"))
        pid, holder = held["pid"], held["run_id"]
    except (OSError, ValueError, KeyError, TypeError):
        # inv: a lock written by a crash mid-write names nobody, so no run can be advised on;
        # the operator is told the path rather than meeting a parse error from inside the harness
        raise SystemExit(f"unreadable sandbox lock at {lock}; no run can be identified from it, "
                          f"so inspect it and remove it") from None
    try:
        os.kill(int(pid), 0)
        alive = True
    except (OSError, TypeError, ValueError):
        alive = False
    if alive:
        raise SystemExit(f"sandbox locked by live run {holder} (pid {pid})")
    # inv: run.yaml tells a finished prepare from a run killed inside it, and records.jsonl tells
    # a finished score from a run killed inside the drive; only a run that reached both has
    # anything collect can close, so the pair decides which advice is printed
    if scored(tmp_root, holder):
        raise SystemExit(f"run {holder} is prepared and waiting at {lock}; collect it")
    # why: abort looks for the lock under the tmp root it is given, so the advice names this one
    # rather than leaving the operator to default elsewhere
    advice = f"run benchmark.harness.abort {holder}"
    if tmp_root != DEFAULT_TMP_ROOT:
        advice += f" --tmp-root {tmp_root}"
    raise SystemExit(f"run {holder} was killed before it finished preparing, and its lock is at "
                      f"{lock}; {advice}")


def scored(tmp_root: Path, run_id: str) -> bool:
    """Tell whether `run_id` under `tmp_root` finished preparing and scoring, so collect can close it."""
    run = tmp_root / run_id / "run"
    return (run / "run.yaml").exists() and (run / "records.jsonl").exists()


def release_lock(tmp_root: Path, run_id: str | None = None) -> None:
    """Remove the sandbox lock, when it belongs to `run_id`.

    Parameters
    ----------
    tmp_root : Path
        The tmp tree holding `sandbox/lock`.
    run_id : str or None, optional
        Release only a lock this run took; `None` releases whichever lock is there.
    """
    lock = tmp_root / "sandbox" / "lock"
    # inv: a late collect of an old run must not unlock the sandbox a live run is using, which
    # would let a third prepare rmtree it mid-flight; the lock names its holder, so ask
    if run_id is not None and lock.exists():
        try:
            held = json.loads(lock.read_text(encoding="utf-8")).get("run_id")
        except (OSError, ValueError):
            held = None
        if held is not None and held != run_id:
            return
    lock.unlink(missing_ok=True)


def verify_packages(h: config.Harness) -> dict[str, str]:
    """Check the vendor environment against its RECORD files; return its hashes."""
    pkg = h.invocation["package"]
    for key in ("launcher", "interpreter"):
        if not Path(pkg[key]).exists():
            raise SystemExit(f"{key} missing: {pkg[key]}")
    bad = rules.verify_records(Path(pkg["site"]))
    if bad:
        # NOT DERIVED: a handful of names keeps the message readable
        raise SystemExit("package files differ from RECORD: " + ", ".join(bad[:5]))
    return {
        "environment_sha256": rules.environment_hash(Path(pkg["site"])),
        "launcher_sha256": rules.sha256_file(Path(pkg["launcher"])),
        "interpreter_sha256": rules.sha256_file(Path(pkg["interpreter"])),
    }


def _under(rel: str, prefixes: list[str]) -> bool:
    return any(rel == p.rstrip("/") or rel.startswith(p if p.endswith("/") else p + "/") for p in prefixes)


def verify_master(index_dir: Path, build: dict) -> None:
    """Check listed artifacts by hash and refuse any unlisted, unexcluded, non-mutable file."""
    artifacts = build.get("artifacts") or {}
    for rel, expected in artifacts.items():
        path = index_dir / rel
        if not path.is_file():
            raise SystemExit(f"artifact missing: {rel}")
        if rules.sha256_file(path) != expected:
            raise SystemExit(f"artifact differs: {rel}")
    allowed = set(artifacts) | {"build.yaml", PREPARED} | set(build.get("mutable") or [])
    for rel in rules.listing(index_dir):
        if rel in allowed or _under(rel, list(build.get("excluded") or [])):
            continue
        raise SystemExit(f"unlisted file in master index: {rel}")


def make_sandbox(
    h: config.Harness, build: dict, index_dir: Path, sandbox: Path, home: Path, benchmark: Path,
) -> dict[str, str]:
    """Lay out the sandbox and `home/`; return sandbox-relative artifact hashes."""
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    home.mkdir(parents=True, exist_ok=True)
    if (home / ".code-review-graph" / "registry.json").exists():
        raise SystemExit("home/.code-review-graph/registry.json present")
    (layout_key,) = h.sandbox_layout.keys()
    target = sandbox / layout_key
    target.mkdir()
    artifacts: dict[str, str] = {}
    for rel, expected in (build.get("artifacts") or {}).items():
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(index_dir / rel, dst)
        artifacts[f"{layout_key}/{rel}"] = expected
    for name in h.models:
        src = benchmark / "systems" / h.system / "models" / name
        dst = home / ".cache" / "huggingface" / "hub" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, symlinks=True)
    return artifacts


def ctx_artifacts(build: dict, layout_key: str, artifacts: dict[str, str]) -> dict[str, str]:
    """Drop entries the vendor itself rewrites on connect.

    Parameters
    ----------
    build : dict
        The index's `build.yaml` mapping.
    layout_key : str
        The `sandbox_layout` prefix the artifacts were copied under.
    artifacts : dict of str to str
        Sandbox-relative artifact paths mapped to their expected sha256, as
        `make_sandbox` returns.

    Returns
    -------
    dict of str to str
        `artifacts` with every path named in `vendor_writes` removed.
    """
    # inv: a vendor_writes entry cannot be hash-verified during a run, only at the master
    # index before the copy and again after the run; run.yaml keeps the unfiltered mapping
    vendor_writes = {f"{layout_key}/{name}" for name in (build.get("vendor_writes") or [])}
    return {rel: sha for rel, sha in artifacts.items() if rel not in vendor_writes}


def substitute(value: object, question: dict, system: str) -> object:
    """Replace `<expansion>` and `<question>` inside strings, lists and dict values."""
    def expansion() -> str:
        # inv: a system whose recipe never asks for an expansion needs no expansion block, so the
        # tokens are looked up only when a string actually carries the placeholder
        return " ".join(question["expansion"][system]["tokens"])
    if isinstance(value, str):
        if "<expansion>" in value:
            value = value.replace("<expansion>", expansion())
        return value.replace("<question>", question["text"])
    if isinstance(value, list):
        return [substitute(v, question, system) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, question, system) for k, v in value.items()}
    return value


def keep_for_prepared(path: str, excluded: list[str], mutable: list[str]) -> bool:
    """Tell whether a changed file counts towards `prepared_outputs`."""
    label, _, rest = path.partition("/")
    if label != "sandbox":
        return True
    inner = rest.split("/", 1)[1] if "/" in rest else rest
    return not _under(inner, excluded) and inner not in set(mutable)


def _check_expansion(h: config.Harness, question: dict, sandbox: Path) -> None:
    tokens = question["expansion"][h.system]["tokens"]
    if len(tokens) > MAX_EXPANSION_TOKENS:
        raise SystemExit(f"more than {MAX_EXPANSION_TOKENS} expansion tokens")
    vocab = sandbox / "graphify-out" / ".vocab.txt"
    words = set(vocab.read_text(encoding="utf-8").split("\n")) if vocab.exists() else None
    if words is not None:
        missing = [t for t in tokens if t not in words]
        if missing:
            raise SystemExit(f"expansion tokens not in vocabulary: {missing}")
    if question.get("rule") != expand.MECHANICAL:
        return
    # inv: a question claiming the mechanical rule is rechecked against the vocabulary the run
    # itself wrote, so a hand-edited token list cannot reach a vendor
    if words is None:
        raise SystemExit("a mechanical expansion needs the vocabulary of the run to be rechecked")
    found = expand.mismatches(question, words, MAX_EXPANSION_TOKENS)
    if found:
        raise SystemExit("; ".join(found))


def halt_reason(h: config.Harness, question: dict, step: dict) -> str | None:
    """Return why a fixed step is journaled as halted instead of run, or None when it runs.

    Parameters
    ----------
    h : config.Harness
        The system under test.
    question : dict
        The question, holding the per-system `expansion` token lists.
    step : dict
        One entry of `fixed_steps`.

    Returns
    -------
    str or None
        The reason, for a step carrying `<expansion>` under a system whose recipe stops on an
        empty selection when the question's list for it is empty; None otherwise.
    """
    if h.system not in expand.HALTS_ON_EMPTY:
        return None
    if "<expansion>" not in json.dumps(step.get("argv") or step.get("args") or []):
        return None
    if question["expansion"][h.system]["tokens"]:
        return None
    return "empty expansion: no vocabulary token matches the question, and the vendor's step 0 stops here"


def _halt_step(ctx: execute.Context, h: config.Harness, step: dict, question: dict, reason: str) -> dict:
    # inv: the entry keeps the shape of a refused call -- `action: False`, no `.out` -- so the audit
    # counts it toward fixed_steps and every reader of the journal skips it as never executed
    entry: dict = {"n": execute.next_n(ctx), "kind": "call", "name": step["name"], "by": "harness",
                   "quote": step.get("quote"), "action": False, "halted": reason,
                   "system_call": False, "ceiling_call": False, "files": []}
    if "argv" in step:
        entry["argv"] = substitute(step["argv"], question, h.system)
    else:
        entry["tool"], entry["args"] = step.get("tool"), substitute(step.get("args") or {}, question, h.system)
    execute.append(ctx, entry)
    return entry


def _run_step(
    ctx: execute.Context, h: config.Harness, step: dict, question: dict, benchmark: Path, server: mcp.Server | None,
) -> dict:
    name = step["name"]
    if "argv" in step:
        call = {"kind": "act", "argv": substitute(step["argv"], question, h.system)}
        # inv: a command-line step counts against the vendor's ceiling on the same terms as a tool
        # call, so `drive.ceiling_left` can subtract what the fixed steps already spent
        return execute.execute(ctx, name=name, call=call, quote=step.get("quote"), by="harness",
                               system_call=bool(step.get("system_call", True)),
                               ceiling_call=bool(step.get("ceiling_call", False)))
    if "script" in step:
        script = str(benchmark / "systems" / h.system / step["script"])
        call = {"kind": "act", "argv": [h.invocation["package"]["interpreter"], script]}
        return execute.execute(ctx, name=name, call=call, quote=step.get("quote"), by="harness", system_call=False)
    if server is None:
        raise SystemExit(f"step {name} needs the MCP server")
    args = substitute(step["args"], question, h.system)
    if not isinstance(args, dict):
        raise SystemExit(f"step {name}: args must be a mapping, got {type(args).__name__}")
    # why: this vendor writes files inside the tool call itself, so the diff window opens before
    # server.call rather than after it
    before = execute.watched_listing(ctx)
    try:
        text = server.call(step["tool"], args)
    except Exception as exc:
        # why: a tool call that raises never reaches execute.execute, so this entry is the only
        # journal line naming the step that failed
        execute.append(ctx, {"n": execute.next_n(ctx), "kind": "call", "name": name, "by": "harness",
                             "tool": step["tool"], "args": args, "exit": None, "error": str(exc),
                             "system_call": True, "ceiling_call": bool(step.get("ceiling_call", False)),
                             "files": []})
        raise
    return execute.execute(ctx, name=name, call={"kind": "tool", "tool": step["tool"], "args": args},
                           quote=step.get("quote"), by="harness", system_call=True,
                           ceiling_call=bool(step.get("ceiling_call", False)), tool_text=text, before=before)


def _sha_or_absent(path: Path) -> str | None:
    """Return the file's sha256, or None when a first run has not created it yet."""
    return rules.sha256_file(path) if path.is_file() else None


def load_prepared(index_dir: Path) -> dict:
    """Read an index's recorded step expectations, or an empty mapping when it has none.

    Parameters
    ----------
    index_dir : Path
        The master index directory.

    Returns
    -------
    dict
        `{configuration: {recipe: {question: {step: expectation}}}}`.
    """
    path = index_dir / PREPARED
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def write_prepared(index_dir: Path, prepared: dict) -> None:
    """Write an index's recorded step expectations.

    Parameters
    ----------
    index_dir : Path
        The master index directory.
    prepared : dict
        The full mapping to serialize.
    """
    (index_dir / PREPARED).write_text(
        yaml.safe_dump(prepared, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_fixed_steps(
    ctx: execute.Context, h: config.Harness, question: dict, benchmark: Path, build: dict, index_dir: Path,
    configuration: str, record: bool, server: mcp.Server | None,
) -> None:
    """Execute every fixed step and compare its outputs with the recorded expectations."""
    # inv: an expectation is keyed by recipe and question, everything its step's output depends on
    recipe = rules.sha256_file(benchmark / "systems" / h.system / "harness.yaml")
    qid = question["id"]
    prepared = load_prepared(index_dir)
    # inv: `or {}` at each level, not setdefault alone -- a key present with a null value is
    # legal YAML and would make the chain operate on None
    by_recipe = prepared[configuration] = prepared.get(configuration) or {}
    by_question = by_recipe[recipe] = by_recipe.get(recipe) or {}
    steps = by_question[qid] = by_question.get(qid) or {}
    excluded = list(build.get("excluded") or [])
    mutable = list(build.get("mutable") or [])
    for step in h.fixed_steps:
        name = step["name"]
        reason = halt_reason(h, question, step)
        entry = (_halt_step(ctx, h, step, question, reason) if reason
                 else _run_step(ctx, h, step, question, benchmark, server))
        if "refused" in entry:
            raise SystemExit(f"fixed step {name} refused: {entry['refused']}")
        if "expansion" in (step.get("checks") or []):
            _check_expansion(h, question, ctx.sandbox)
        # inv: a halted step has no output, and its expectation is that absence, so a later attempt
        # under the same recipe meets it only by halting again
        observed = {"out": entry.get("canonical_sha256"),
                    "files": {f["path"]: f["sha256"] for f in entry["files"]
                             if keep_for_prepared(f["path"], excluded, mutable)}}
        expected = steps.get(name)
        if expected is None:
            if not record:
                raise SystemExit(f"no expectation for {configuration}/{qid}/{name}; "
                                  f"rerun with --record-prepared")
            steps[name] = observed
            write_prepared(index_dir, prepared)
        elif expected != observed:
            raise SystemExit(f"fixed step {name} differs from the expectation recorded for {qid}")


def check_blind(benchmark: Path, run_dir: Path, qid: str, tool_definitions: list[dict] | None = None) -> None:
    """Refuse a run whose prompt or tools told the runner the answer.

    Parameters
    ----------
    benchmark : Path
        Root holding the snapshot that carries `references/<qid>.yaml`.
    run_dir : Path
        The run directory holding `prompt.md`.
    qid : str
        The question being asked.
    tool_definitions : list of dict, or None
        The tools the runner will be offered, when they are known yet.

    Raises
    ------
    SystemExit
        When any reference place reached the part of the prompt the owner wrote,
        or a tool definition.
    """
    reference = config.load_yaml(config.reference_path(benchmark, qid))
    above, _ = prompt.split((run_dir / "prompt.md").read_text(encoding="utf-8"))
    # inv: only the part above the heading is checked; below it sits the output of a journaled
    # action, which is the one way the reference is allowed to reach a runner
    found = blind.violations(above, tool_definitions or [], reference)
    if found:
        raise SystemExit("the runner would not be blind: " + "; ".join(found))


def hand_over(ctx: execute.Context, *, runner: bool) -> None:
    """Close the journal, unless a runner is to take the run on.

    Parameters
    ----------
    ctx : execute.Context
        The run's paths.
    runner : bool
        Whether a model will drive the rest of the run.
    """
    # inv: the last entry of a journal is its stop, and its reason names who ended the run;
    # a run handed to a runner is ended by the driver, so preparation must not end it here
    if runner:
        return
    execute.append(ctx, {"n": execute.next_n(ctx), "kind": "stop", "by": "harness", "reason": "harness"})


def prompt_steps(run_dir: Path, entries: list[dict]) -> list[dict]:
    """Return the executed calls of a run, as the prompt shows them.

    Parameters
    ----------
    run_dir : Path
        The run directory holding `NN_name.cmd` and `NN_name.out`.
    entries : list of dict
        The journal, in order.

    Returns
    -------
    list of dict
        `{"name", "cmd", "out"}` per executed call, in journal order.
    """
    steps = []
    for entry in entries:
        if entry.get("kind") != "call" or entry.get("action") is False:
            continue
        stem = f"{entry['n']:02d}_{entry['name']}"
        out_path = run_dir / f"{stem}.out"
        # inv: a call whose output is absent never ran to completion, and a prompt that showed
        # it as a step would tell the runner an action happened that did not
        if not out_path.is_file():
            continue
        steps.append({
            "name": entry["name"],
            "cmd": (run_dir / f"{stem}.cmd").read_text(encoding="utf-8", errors="replace"),
            "out": out_path.read_text(encoding="utf-8", errors="replace"),
        })
    return steps


def write_prompt(benchmark: Path, ctx: execute.Context, h: config.Harness, question: dict) -> Path:
    """Write the prompt a runner is given and return its path.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/`.
    ctx : execute.Context
        The run whose executed steps the prompt shows.
    h : config.Harness
        The system being run.
    question : dict
        The question, whose text the prompt carries, and whose expansion it carries when
        this system has one.

    Returns
    -------
    Path
        The written `prompt.md`.
    """
    # inv: the prompt quotes the vendor's own prescribed workflow; the benchmark's methodology
    # states the reference answer and is never a source here
    manifest = config.load_yaml(benchmark / "systems" / h.system / "manifest.yaml")
    entries = [json.loads(line) for line in
               (ctx.run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # inv: a system whose recipe has no expansion step has no expansion block to read, and the
    # prompt then carries no expansion section rather than an empty one
    tokens = list(((question.get("expansion") or {}).get(h.system) or {}).get("tokens") or [])
    text = prompt.build(manifest, question, tokens, prompt_steps(ctx.run_dir, entries), h.invocation)
    path = ctx.run_dir / "prompt.md"
    path.write_text(text, encoding="utf-8")
    return path


def prepare(
    benchmark: Path, system: str, qid: str, configuration: str | None, tmp_root: Path, record: bool,
    runner: bool = False, *, driver: dict | None = None,
) -> Path:
    """Prepare one run and return its `run/` directory.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `systems/`.
    system : str
        The system under test.
    qid : str
        The question to run.
    configuration : str or None
        The system's configuration, or None for its default.
    tmp_root : Path
        The tmp tree to prepare the run under.
    record : bool
        Whether an unmet fixed-step expectation records itself instead of raising.
    runner : bool, optional
        Whether a model will drive the rest of the run.
    driver : dict or None, optional
        The runner's resolved settings (`model`, `effort`, `max_actions`, `max_tokens`,
        `backend`, `base_url`), written into the row and `run.yaml` verbatim when given.
    """
    rules.require_sealed(benchmark, tmp_root)
    # inv: a refused question leaves no ledger row, so the review gate sits before the lock and
    # every write that follows it
    review_message = questions.check_review(benchmark, qid)
    if review_message is not None:
        raise SystemExit(review_message)
    if not questions.admitted(benchmark, qid):
        raise SystemExit(f"question {qid} was withdrawn by its review")
    h = config.load_harness(benchmark, system)
    configuration = configuration or h.default_configuration
    question = config.load_question(benchmark, qid)
    snapshot = config.question_snapshot(question, qid)
    # inv: the recipe's frozen weights and the corpus are checked on every attempt, so a run never
    # measures a model or a tree the freeze record does not describe
    problems = rules.check_models(benchmark, system, h.models)
    snapshot_dir = config.snapshot_dir(benchmark, snapshot)
    if (snapshot_dir / "source").is_dir():
        problems += rules.check_corpus(snapshot_dir)
    if problems:
        raise SystemExit("\n".join(problems))
    index_dir = snapshot_dir / h.configurations[configuration]["index"]
    build_path = index_dir / "build.yaml"
    qpath = config.question_path(benchmark, qid)
    ledger.require_clean(benchmark)
    attempt = ledger.next_attempt(benchmark, qid, system, configuration)
    run_id = f"{qid}-{system}-{configuration}-a{attempt:02d}"  # inv: attempts per triple stay far below 100
    # inv: a separator or a `..` in run_id would place the run outside tmp_root, so its shape is
    # checked even though every part already names an existing config entry
    if not _RUN_ID.fullmatch(run_id):
        raise SystemExit(f"run id is not a plain name: {run_id!r}")
    take_lock(tmp_root, run_id)
    try:
        hashes = verify_packages(h)
        # inv: the recipe is an input like the question and the index, so two attempts whose
        # canonical outputs differ are only evidence of nondeterminism when this hash agrees
        row = {"run_id": run_id, "question": qid, "system": system, "configuration": configuration,
               "attempt": attempt, "question_sha256": rules.sha256_file(qpath),
               "harness_sha256": rules.sha256_file(benchmark / "systems" / system / "harness.yaml"),
               "reference_sha256": rules.sha256_file(config.reference_path(benchmark, qid)),
               "build_yaml_sha256": rules.sha256_file(build_path),
               "prepared_sha256": _sha_or_absent(index_dir / PREPARED),
               "instrument_sha256": rules.sha256_file(benchmark / seal.SEAL), **hashes,
               "runner": runner, **(driver or {})}
        ledger.append_row(benchmark, row)
    except BaseException:
        # inv: bench never commits -- kia's `run` makes the attempt's one commit on success, and
        # reads a mark_aborted row back into a commit on failure past this point; a failure here
        # means append_row itself never landed, so there is nothing yet for either to mark
        release_lock(tmp_root)
        raise
    try:
        build = config.load_build(index_dir)
        verify_master(index_dir, build)
        run_dir = tmp_root / run_id / "run"
        home = tmp_root / run_id / "home"
        sandbox = tmp_root / "sandbox" / "index"
        run_dir.mkdir(parents=True)
        artifacts = make_sandbox(h, build, index_dir, sandbox, home, benchmark)
        env = {k: v.replace("<sandbox>", str(sandbox)) for k, v in h.environment.items()}
        invocation = {**h.invocation, "allowed_scripts": h.allowed_scripts}
        (layout_key,) = h.sandbox_layout.keys()
        ctx = execute.Context(run_dir=run_dir, sandbox=sandbox, home=home, environment=env,
                              invocation=invocation, volatile=h.volatile,
                              artifacts=ctx_artifacts(build, layout_key, artifacts))
        execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
        baseline = hashlib.sha256(json.dumps(execute.watched_listing(ctx), sort_keys=True).encode()).hexdigest()
        tools = [s["tool"] for s in h.fixed_steps if "tool" in s]
        if tools:
            # why: fastmcp is a `bench`-extra dependency; importing it here keeps every
            # non-MCP path in this module collectable without that extra installed
            from benchmark.harness import mcp as mcp_module

            with mcp_module.Server(ctx, Path(h.invocation["package"]["launcher"]), tools) as server:
                run_fixed_steps(ctx, h, question, benchmark, build, index_dir, configuration, record, server)
        else:
            run_fixed_steps(ctx, h, question, benchmark, build, index_dir, configuration, record, None)
        if runner:
            write_prompt(benchmark, ctx, h, question)
            check_blind(benchmark, run_dir, qid)
        hand_over(ctx, runner=runner)
        run_yaml = {**row, "snapshot": snapshot, "tmp_root": str(tmp_root),
                    "fixed_steps": len(h.fixed_steps), "baseline_listing": baseline, "artifacts": artifacts,
                    "outcome": "prepared"}
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run_yaml, sort_keys=False), encoding="utf-8")
    except BaseException:
        # why: deferred -- abort imports this module for release_lock, so importing it at load
        # time would cycle; this run is still live, so its own mark_aborted call bypasses the
        # liveness refusal abort.abort's CLI path raises for a process aborting itself
        from benchmark.harness import abort

        abort.mark_aborted(benchmark, run_id, tmp_root)
        raise
    # why: collect.py still needs the sandbox this lock guards, so a run that finished
    # preparing leaves the lock in place for collect.py to release
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, prepare, print the run directory."""
    ap = argparse.ArgumentParser(prog="benchmark.harness prepare")
    ap.add_argument("--system", required=True)
    ap.add_argument("--question-id", required=True)
    ap.add_argument("--configuration")
    ap.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    ap.add_argument("--record-prepared", action="store_true")
    ap.add_argument("--runner", action="store_true",
                    help="write prompt.md and leave the journal open for drive.py to close")
    ap.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)
    print(prepare(args.benchmark, args.system, args.question_id, args.configuration, args.tmp_root,
                  args.record_prepared, args.runner))
    return 0


if __name__ == "__main__":
    sys.exit(main())
