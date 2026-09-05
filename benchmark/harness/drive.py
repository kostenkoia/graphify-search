"""Drive a model through one run: one call per turn, every call executed by the harness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from benchmark.harness import config, execute, prepare, rules, score, tools

if TYPE_CHECKING:
    from collections.abc import Callable

ANSWER_MET = "answer_met"
NO_FURTHER_ACTION = "no_further_action"
CEILING_REACHED = "ceiling_reached"
ERROR = "error"
HARNESS_LIMIT = "harness_limit"

STOP = "stop"

ONE_CALL_PER_TURN = "one call per turn: this call was not executed"
REASK = "Answer by calling a tool. A reply written in prose is not read."


class DriveError(Exception):
    """Raised when a run cannot go on."""


class ToolServer(Protocol):
    """A running vendor server the harness can call a tool on."""

    def call(self, tool: str, args: dict) -> str:
        """Return the reply text of one tool call."""
        ...


class Backend(Protocol):
    """A source of assistant turns.

    A backend that reshapes a request before sending it also offers `as_sent`, so the
    record of an exchange can carry what actually went on the wire.
    """

    def send(self, request: dict) -> dict:
        """Return the next assistant turn for `request`."""
        ...


def _result(uid: str, text: str, is_error: bool) -> dict:
    return {"type": "tool_result", "tool_use_id": uid, "content": text, "is_error": is_error}


def _outcome(reason: str, actions: int, ceiling_calls: int, **extra: object) -> dict:
    return {"reason": reason, "actions": actions, "ceiling_calls": ceiling_calls, **extra}


def loop(
    *,
    backend: Backend,
    request: dict,
    run_call: Callable[[str, dict], dict],
    max_actions: int,
    ceiling: int | None = None,
    record: Callable[[str, dict], None] | None = None,
) -> dict:
    """Run one model through its turns and return why the run ended.

    Parameters
    ----------
    backend : Backend
        Source of assistant turns.
    request : dict
        The first request, whose `messages` the loop grows.
    run_call : callable
        Executes one runner call and returns `{"text", "is_error", "ceiling"}`.
    max_actions : int
        Ceiling on runner calls executed or refused.
    ceiling : int or None
        The vendor's own ceiling on its counted calls, when it declares one.
    record : callable or None
        Given `("request", payload)` before the first send and `("exchange", payload)`
        after every turn.

    Returns
    -------
    dict
        `reason`, `actions`, `ceiling_calls`, and `stop` when the runner named a place.
    """
    messages = [dict(message) for message in request["messages"]]
    actions = 0
    ceiling_calls = 0
    reasked = False
    first = True
    while True:
        # inv: the payload takes a snapshot of the turns so far, because a later turn appending
        # to a shared list would rewrite what the record says was sent
        payload = {**request, "messages": list(messages)}
        # inv: the first request is handed over before it is sent, so the record of what was
        # asked cannot be written after seeing the answer
        if first and record is not None:
            record("request", payload)
        first = False
        response = backend.send(payload)
        if record is not None:
            exchange = {"request": payload, "response": response}
            # inv: a backend that reshapes the request before sending it says so here, or the
            # record would claim settings the far side never received
            render = getattr(backend, "as_sent", None)
            if callable(render):
                exchange["sent"] = render(payload)
            record("exchange", exchange)
        # why: a refusal carries no turn to act on, and a turn cut off at the token ceiling may
        # carry half a call; either way the run cannot go on honestly
        if response.get("stop_reason") in ("refusal", "max_tokens"):
            return _outcome(ERROR, actions, ceiling_calls)
        content = list(response.get("content") or [])
        # inv: the turn is echoed back whole, because a reasoning block dropped here would
        # change what the model is continuing from
        messages.append({"role": "assistant", "content": content})
        uses = [block for block in content if block.get("type") == "tool_use"]
        if not uses:
            if reasked:
                return _outcome(NO_FURTHER_ACTION, actions, ceiling_calls)
            reasked = True
            messages.append({"role": "user", "content": REASK})
            continue
        reasked = False
        results: list[dict] = []
        stop_args: dict | None = None
        for index, block in enumerate(uses):
            arguments = dict(block.get("input") or {})
            if index > 0:
                # why: a call the harness declines to run is still a call the runner spent, so
                # it counts against the same limit an executed one does
                actions += 1
                results.append(_result(block["id"], ONE_CALL_PER_TURN, is_error=True))
                continue
            if block["name"] == STOP:
                stop_args = arguments
                results.append(_result(block["id"], "the place is recorded", is_error=False))
                continue
            actions += 1
            answer = run_call(block["name"], arguments)
            if answer.get("ceiling"):
                ceiling_calls += 1
            results.append(_result(block["id"], answer["text"], bool(answer.get("is_error"))))
        messages.append({"role": "user", "content": results})
        if stop_args is not None:
            return _outcome(ANSWER_MET, actions, ceiling_calls, stop=stop_args)
        # inv: the vendor's own ceiling is answered before the harness's, so a run that reaches
        # both is reported as the vendor's boundary rather than ours
        if ceiling is not None and ceiling_calls >= ceiling:
            return _outcome(CEILING_REACHED, actions, ceiling_calls)
        if actions >= max_actions:
            return _outcome(HARNESS_LIMIT, actions, ceiling_calls)


# inv: the same shape `execute` accepts for `NN_name.*`, so a vendor's own tool name can never
# make a filename the harness refuses to write
_UNFIT = re.compile(r"[^a-z0-9_]+")
NAME_LIMIT = 40  # NOT DERIVED: 40 keeps a journal stem `NN_name` short enough for one terminal line


def entry_name(name: str) -> str:
    """Return a journal entry name a tool name can always be written as.

    Parameters
    ----------
    name : str
        The tool name as the runner called it.

    Returns
    -------
    str
        Lowercase letters, digits and underscores, at most `NAME_LIMIT` long.
    """
    fitted = _UNFIT.sub("_", name.lower()).strip("_")
    return (fitted or "call")[:NAME_LIMIT]


def to_call(name: str, args: dict) -> dict:
    """Return the harness call one runner tool call stands for.

    Parameters
    ----------
    name : str
        The tool the runner called.
    args : dict
        Its arguments, including the `quote` that authorises it.

    Returns
    -------
    dict
        `{"kind": "act", "argv": [...]}` or `{"kind": "tool", "tool": ..., "args": {...}}`.
    """
    if name == tools.ACT:
        return {"kind": "act", "argv": list(args.get("argv") or [])}
    # inv: the citation is not sent to the vendor; it is checked against the frozen document
    return {"kind": "tool", "tool": name, "args": {k: v for k, v in args.items() if k != "quote"}}


class Recorder:
    """Write the first request and every exchange of a run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.api = run_dir / "api"
        self._exchanges = 0

    def __call__(self, kind: str, payload: dict) -> None:
        """Write one request or one exchange.

        Parameters
        ----------
        kind : str
            `request` for the first request, `exchange` for a request and its response.
        payload : dict
            What to write.
        """
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if kind == "request":
            (self.run_dir / "request.json").write_text(body, encoding="utf-8")
            return
        self._exchanges += 1
        self.api.mkdir(exist_ok=True)
        (self.api / f"{self._exchanges:02d}.json").write_text(body, encoding="utf-8")


class CallRunner:
    """Execute one runner call through the harness and answer the loop with its result."""

    def __init__(
        self,
        ctx: execute.Context,
        *,
        prompt_text: str,
        server: ToolServer | None = None,
        parse: Callable[[dict, str], list[dict]] | None = None,
        ceiling_tools: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.ctx = ctx
        self.prompt_text = prompt_text
        self.server = server
        self.parse = parse
        self.ceiling_tools = set(ceiling_tools)
        self.records: list[dict] = []

    def __call__(self, name: str, args: dict) -> dict:
        """Execute one call and return `{"text", "is_error", "ceiling"}`.

        Parameters
        ----------
        name : str
            The tool the runner called.
        args : dict
            Its arguments, including its `quote`.

        Returns
        -------
        dict
            The reply text the runner is given, whether it is an error, and whether the
            call counted under the vendor's own ceiling.

        Raises
        ------
        DriveError
            When the runner calls a server tool and the run has no server.
        """
        quote = args.get("quote")
        call = to_call(name, args)
        # inv: the origins are read before this call's own results join them, so a call can
        # never be its own source
        origins = rules.call_provenance(self.ctx.invocation, call, self.prompt_text, self.records)
        stem = entry_name(name)
        ceiling = name in self.ceiling_tools
        if call["kind"] == "tool":
            return self._tool(stem, call, quote, origins, ceiling=ceiling)
        entry = execute.execute(self.ctx, name=stem, call=call, quote=quote, by="runner",
                                system_call=True, ceiling_call=ceiling, provenance=origins)
        return self._answer(entry, ceiling=ceiling)

    def _tool(self, stem: str, call: dict, quote: str | None, origins: list[dict], *, ceiling: bool) -> dict:
        if self.server is None:
            raise DriveError(f"the run has no server to call {call['tool']} on")
        # why: this vendor writes files inside the tool call itself, so the diff window opens
        # before the call rather than after it
        before = execute.watched_listing(self.ctx)
        try:
            text = self.server.call(call["tool"], call["args"])
        except Exception as exc:
            # why: a tool call that raises never reaches execute, so this entry is the only
            # journal line naming what the runner tried
            execute.append(self.ctx, {
                "n": execute.next_n(self.ctx), "kind": "call", "name": stem, "by": "runner",
                "quote": quote, "provenance": origins, "tool": call["tool"], "args": call["args"],
                "exit": None, "error": str(exc), "system_call": True, "ceiling_call": ceiling,
                "files": [],
            })
            # inv: a call that never reached the vendor does not spend the vendor's own ceiling
            return {"text": str(exc), "is_error": True, "ceiling": False}
        entry = execute.execute(self.ctx, name=stem, call=call, quote=quote, by="runner",
                                system_call=True, ceiling_call=ceiling, tool_text=text,
                                before=before, provenance=origins)
        return self._answer(entry, ceiling=ceiling)

    def _answer(self, entry: dict, *, ceiling: bool) -> dict:
        if entry.get("action") is False:
            return {"text": entry["refused"], "is_error": True, "ceiling": False}
        stem = f"{entry['n']:02d}_{entry['name']}"
        out = (self.ctx.run_dir / f"{stem}.out").read_text(encoding="utf-8", errors="replace")
        if self.parse is not None:
            # inv: the parser is given the journal entry, not a call rebuilt from it, because the
            # adapter branches on the entry's own shape and scoring reads it the same way
            for record in self.parse(entry, out):
                self.records.append({**record, "n": entry["n"]})
        if entry.get("exit"):
            err = (self.ctx.run_dir / f"{stem}.err").read_text(encoding="utf-8", errors="replace")
            # why: a vendor that failed may have said why only on its error stream, and a runner
            # shown an empty result would read the failure as an empty corpus
            return {"text": f"{out}{err}", "is_error": True, "ceiling": ceiling}
        return {"text": out, "is_error": False, "ceiling": ceiling}


def finish(ctx: execute.Context, outcome: dict) -> dict:
    """Close a run's journal with the stop entry the loop earned.

    Parameters
    ----------
    ctx : execute.Context
        The run's paths.
    outcome : dict
        What `loop` returned.

    Returns
    -------
    dict
        The stop entry, also appended to the journal.
    """
    entry = {"n": execute.next_n(ctx), "kind": "stop", "by": "runner", "reason": outcome["reason"]}
    if outcome.get("stop"):
        entry["place"] = outcome["stop"]
    execute.append(ctx, entry)
    return entry


# NOT DERIVED: keeps one reply inside the SDK's own request timeout without streaming
DEFAULT_MAX_TOKENS = 16000

RUNNER_DEFAULTS = "runner_defaults"


def request_from(
    prompt_text: str, tool_definitions: list[dict], *, model: str, effort: str, max_tokens: int,
) -> dict:
    """Return the first request of a run.

    Parameters
    ----------
    prompt_text : str
        The whole prompt, sent as the only message.
    tool_definitions : list of dict
        Every tool the runner may call.
    model : str
        The model to run.
    effort : str
        Its reasoning effort.
    max_tokens : int
        Ceiling on one reply.

    Returns
    -------
    dict
        A request the backends send as it stands.
    """
    return {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        "tools": list(tool_definitions),
        # inv: one call per turn is asked of the model here and enforced by the loop as well,
        # because a model is asked and a harness decides
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [{"role": "user", "content": prompt_text}],
    }


def runner_settings(
    harness: config.Harness, *, model: str | None, effort: str | None, max_actions: int | None,
) -> dict:
    """Return the runner's model, effort and action limit.

    Parameters
    ----------
    harness : config.Harness
        The system, whose `runner_defaults` supply what the caller left out.
    model, effort : str or None
        Values given on the command line, which win over the harness.
    max_actions : int or None
        The same, for the action limit.

    Returns
    -------
    dict
        `model`, `effort` and `max_actions`.

    Raises
    ------
    DriveError
        When a setting is stated neither on the command line nor in the harness.
    """
    stated = harness.raw.get(RUNNER_DEFAULTS) or {}
    chosen = {"model": model or stated.get("model"),
              "effort": effort or stated.get("effort"),
              "max_actions": max_actions or stated.get("max_actions")}
    # why: a measurement whose model or limit was guessed is a measurement of the guess
    missing = sorted(key for key, value in chosen.items() if value is None)
    if missing:
        raise DriveError(f"the runner has no {', '.join(missing)}: state it in harness.yaml "
                         f"{RUNNER_DEFAULTS} or on the command line")
    return chosen


def ceiling_tools(harness: config.Harness) -> set[str]:
    """Return the calls that count under the vendor's declared ceiling.

    Parameters
    ----------
    harness : config.Harness
        The system, whose `ceiling` may be declared and whose `tools` the runner may call.

    Returns
    -------
    set of str
        Every name a runner can call, when a ceiling is declared; empty otherwise.
    """
    if not harness.ceiling:
        return set()
    # inv: one vendor is reached through server tools and the other by running its command, so
    # both count; counting only the tools would leave a command-line vendor a ceiling that can
    # never be reached
    return {tools.ACT, *(harness.invocation.get("tools") or {})}


def ceiling_limit(harness: config.Harness) -> int | None:
    """Return the number of calls the vendor's ceiling allows, when it states one.

    Parameters
    ----------
    harness : config.Harness
        The system, whose `ceiling` may carry a `calls` count.

    Returns
    -------
    int or None
        None when the ceiling is prose alone; a count is never read out of the quote.
    """
    # inv: the count is taken from a field, never parsed out of the quoted sentence, or the
    # limit would be the harness's reading of the vendor rather than the vendor's own number
    return (harness.ceiling or {}).get("calls")


def ceiling_left(harness: config.Harness, ctx: execute.Context) -> int | None:
    """Return how much of the vendor's declared ceiling the runner still has.

    Parameters
    ----------
    harness : config.Harness
        The system, whose `ceiling` may state a call count.
    ctx : execute.Context
        The run, whose journal already holds the fixed steps' own counted calls.

    Returns
    -------
    int or None
        The declared count less what the fixed steps already spent, never below zero;
        None when the vendor declares no number.
    """
    declared = ceiling_limit(harness)
    if declared is None:
        return None
    # inv: the vendor's ceiling counts a whole run, and the prescribed steps are part of that
    # run; leaving them out would hand the runner a budget the vendor never offered
    journal = ctx.run_dir / "journal.jsonl"
    spent = 0
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("ceiling_call"):
                spent += 1
    return max(0, int(declared) - spent)


def context_from_run(
    benchmark: Path, harness: config.Harness, run_yaml: dict, tmp_root: Path,
) -> execute.Context:
    """Rebuild the context of a run that was already prepared.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/` and `record/snapshots/`.
    harness : config.Harness
        The system being run.
    run_yaml : dict
        The run's `run.yaml`, written by preparation.
    tmp_root : Path
        The root the run was prepared under.

    Returns
    -------
    execute.Context
        The same paths, environment and artifact hashes preparation used.
    """
    run_id = run_yaml["run_id"]
    sandbox = tmp_root / "sandbox" / "index"
    index_dir = (config.snapshot_dir(benchmark, run_yaml["snapshot"])
                 / harness.configurations[run_yaml["configuration"]]["index"])
    build = config.load_build(index_dir)
    (layout_key,) = harness.sandbox_layout.keys()
    environment = {k: v.replace("<sandbox>", str(sandbox)) for k, v in harness.environment.items()}
    invocation = {**harness.invocation, "allowed_scripts": harness.allowed_scripts}
    return execute.Context(
        run_dir=tmp_root / run_id / "run", sandbox=sandbox, home=tmp_root / run_id / "home",
        environment=environment, invocation=invocation, volatile=harness.volatile,
        artifacts=prepare.ctx_artifacts(build, layout_key, run_yaml["artifacts"]),
    )


def drive_run(
    benchmark: Path, run_id: str, tmp_root: Path, backend: Backend, *,
    model: str | None = None, effort: str | None = None, max_actions: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Run one prepared attempt through a model and close its journal.

    Parameters
    ----------
    benchmark : Path
        Root holding `systems/` and `record/snapshots/`.
    run_id : str
        The attempt to drive, already prepared.
    tmp_root : Path
        The root the attempt was prepared under.
    backend : Backend
        Source of assistant turns.
    model, effort : str or None
        Overrides for the harness's own runner settings.
    max_actions : int or None
        The same, for the action limit.
    max_tokens : int
        Ceiling on one reply.

    Returns
    -------
    dict
        What `loop` returned, after the stop entry was written, plus `model_served`:
        the `model` the first exchange's response carried, or None when it carried none.

    Raises
    ------
    SystemExit
        When the attempt was prepared without a prompt for a runner.
    """
    run_dir = tmp_root / run_id / "run"
    run_yaml = config.load_yaml(run_dir / "run.yaml")
    harness = config.load_harness(benchmark, run_yaml["system"])
    settings = runner_settings(harness, model=model, effort=effort, max_actions=max_actions)
    prompt_path = run_dir / "prompt.md"
    if not prompt_path.is_file():
        raise SystemExit(f"{prompt_path} is absent: prepare the attempt for a runner first")
    prompt_text = prompt_path.read_text(encoding="utf-8")
    qid = str(run_yaml.get("question") or "")
    ctx = context_from_run(benchmark, harness, run_yaml, tmp_root)
    parse = score.parser_for(benchmark, run_yaml)
    named = set(harness.invocation.get("tools") or {})
    if not named:
        outcome = _drive(ctx, harness, prompt_text, {}, None, backend, settings, max_tokens,
                         run_dir, benchmark, qid, parse=parse)
    else:
        # why: fastmcp is a `bench`-extra dependency; importing it here keeps every non-server
        # path in this module collectable without that extra installed
        from benchmark.harness import mcp as mcp_module

        launcher = Path(harness.invocation["package"]["launcher"])
        with mcp_module.Server(ctx, launcher, sorted(named)) as server:
            schemas = {t["name"]: t for t in server.list_tools()}
            outcome = _drive(ctx, harness, prompt_text, schemas, server, backend, settings, max_tokens,
                             run_dir, benchmark, qid, parse=parse)
    # inv: outside the server's `with` block, so the run's stop is the journal's last entry --
    # a server still running when the journal closes could still append after it
    finish(ctx, outcome)
    return {**outcome, "model_served": _model_served(run_dir)}


def _model_served(run_dir: Path) -> str | None:
    """Return the model the first recorded exchange's response named, or None."""
    first_exchange = run_dir / "api" / "01.json"
    if not first_exchange.is_file():
        return None
    response = json.loads(first_exchange.read_text(encoding="utf-8")).get("response") or {}
    model = response.get("model")
    return model if isinstance(model, str) else None


def _drive(
    ctx: execute.Context, harness: config.Harness, prompt_text: str, schemas: dict,
    server: ToolServer | None, backend: Backend, settings: dict, max_tokens: int, run_dir: Path,
    benchmark: Path, qid: str, parse: Callable[[dict, str], list[dict]] | None = None,
) -> dict:
    definitions = tools.offered(harness.invocation, schemas)
    # inv: blindness is shown before the first request leaves, because a leak found afterwards
    # is a run already spent on measuring the leak
    prepare.check_blind(benchmark, run_dir, qid, definitions)
    request = request_from(prompt_text, definitions, model=settings["model"],
                           effort=settings["effort"], max_tokens=max_tokens)
    runner = CallRunner(ctx, prompt_text=prompt_text, server=server, parse=parse,
                        ceiling_tools=ceiling_tools(harness))
    # inv: the journal is not closed here -- the caller closes it once the server it started
    # is gone, or the server's own stop entry would land after the run's
    return loop(backend=backend, request=request, run_call=runner,
                max_actions=int(settings["max_actions"]), ceiling=ceiling_left(harness, ctx),
                record=Recorder(run_dir))
