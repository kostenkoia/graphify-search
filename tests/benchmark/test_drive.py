import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmark.harness import config, drive, execute, prompt, rules, score
from benchmark.harness.backends.journal import JournalBackend
from tests.benchmark.conftest import references_dir, snapshot_dir, write_question


class Backend:
    """A backend that replays prepared responses and remembers what it was sent."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.sent: list[dict] = []

    def send(self, request: dict) -> dict:
        self.sent.append(request)
        if not self.responses:
            raise AssertionError("the loop asked for one turn more than the test prepared")
        return self.responses.pop(0)


class Executor:
    """A call executor that answers from a script and remembers every call."""

    def __init__(self, answers: list[dict] | None = None) -> None:
        self.answers = list(answers or [])
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if self.answers:
            return self.answers.pop(0)
        return {"text": "ok", "is_error": False, "ceiling": False}


def use(uid: str, name: str, **args: object) -> dict:
    return {"type": "tool_use", "id": uid, "name": name, "input": dict(args)}


def turn(*blocks: dict) -> dict:
    return {"content": list(blocks), "stop_reason": "tool_use", "usage": {"output_tokens": 1}}


def text(body: str) -> dict:
    return {"content": [{"type": "text", "text": body}], "stop_reason": "end_turn", "usage": {}}


REQUEST = {"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "prompt"}], "tools": []}
STOP_CALL = use("u9", "stop", path="a.py", symbol="f", start=1)


def test_a_call_is_executed_and_answered_by_its_own_identifier():
    ex = Executor([{"text": "NODE thing", "is_error": False, "ceiling": False}])
    backend = Backend([turn(use("u1", "act", argv=["graphify", "query", "x"], quote="q")), turn(STOP_CALL)])
    drive.loop(backend=backend, request=REQUEST, run_call=ex, max_actions=5)
    assert ex.calls[0][0] == "act"
    result = backend.sent[1]["messages"][-1]["content"][0]
    assert result == {"type": "tool_result", "tool_use_id": "u1", "content": "NODE thing", "is_error": False}


def test_a_refused_call_comes_back_to_the_runner_as_an_error():
    ex = Executor([{"text": "flag is rejected: --graph", "is_error": True, "ceiling": False}])
    backend = Backend([turn(use("u1", "act", argv=["graphify", "query", "--graph"], quote="q")), turn(STOP_CALL)])
    drive.loop(backend=backend, request=REQUEST, run_call=ex, max_actions=5)
    result = backend.sent[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert result["content"] == "flag is rejected: --graph"


def test_only_the_first_call_of_a_turn_is_executed():
    ex = Executor()
    backend = Backend([turn(use("u1", "act", argv=["a"], quote="q"), use("u2", "act", argv=["b"], quote="q")),
                       turn(STOP_CALL)])
    drive.loop(backend=backend, request=REQUEST, run_call=ex, max_actions=5)
    assert [name for name, _ in ex.calls] == ["act"]
    second = backend.sent[1]["messages"][-1]["content"][1]
    assert second["tool_use_id"] == "u2"
    assert second["is_error"] is True
    assert "one call per turn" in second["content"]


def test_a_reply_with_no_call_is_asked_again():
    ex = Executor()
    backend = Backend([text("I think the answer is obvious"), turn(STOP_CALL)])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=ex, max_actions=5)
    assert outcome["reason"] == "answer_met"
    assert backend.sent[1]["messages"][-1]["role"] == "user"


def test_two_replies_with_no_call_end_the_run():
    backend = Backend([text("no"), text("still no")])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)
    assert outcome["reason"] == "no_further_action"


def test_a_call_between_two_wordy_replies_earns_a_fresh_second_chance():
    backend = Backend([text("one"), turn(use("u1", "act", argv=["a"], quote="q")), text("two"), turn(STOP_CALL)])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)
    assert outcome["reason"] == "answer_met"


def test_the_stop_call_ends_the_run_and_keeps_the_place_it_names():
    backend = Backend([turn(STOP_CALL)])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)
    assert outcome["reason"] == "answer_met"
    assert outcome["stop"] == {"path": "a.py", "symbol": "f", "start": 1}


def test_the_stop_call_is_not_an_action():
    backend = Backend([turn(STOP_CALL)])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)
    assert outcome["actions"] == 0


def test_the_run_ends_when_the_harness_limit_is_reached():
    backend = Backend([turn(use("u1", "act", argv=["a"], quote="q")),
                       turn(use("u2", "act", argv=["b"], quote="q"))])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=2)
    assert outcome["reason"] == "harness_limit"
    assert outcome["actions"] == 2


def test_a_refused_call_counts_towards_the_harness_limit():
    ex = Executor([{"text": "refused", "is_error": True, "ceiling": False}])
    backend = Backend([turn(use("u1", "act", argv=["--graph"], quote="q"))])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=ex, max_actions=1)
    assert outcome["reason"] == "harness_limit"
    assert outcome["actions"] == 1


def test_a_call_dropped_by_the_one_per_turn_rule_counts_too():
    backend = Backend([turn(use("u1", "act", argv=["a"], quote="q"), use("u2", "act", argv=["b"], quote="q"))])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=2)
    assert outcome["actions"] == 2
    assert outcome["reason"] == "harness_limit"


def test_the_run_ends_when_the_vendor_ceiling_is_reached():
    ex = Executor([{"text": "a", "is_error": False, "ceiling": True},
                   {"text": "b", "is_error": False, "ceiling": True}])
    backend = Backend([turn(use("u1", "search", query="x", quote="q")),
                       turn(use("u2", "search", query="y", quote="q"))])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=ex, max_actions=9, ceiling=2)
    assert outcome["reason"] == "ceiling_reached"
    assert outcome["ceiling_calls"] == 2


def test_the_assistant_turn_is_echoed_back_whole():
    thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
    backend = Backend([{"content": [thinking, use("u1", "act", argv=["a"], quote="q")],
                        "stop_reason": "tool_use", "usage": {}}, turn(STOP_CALL)])
    drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)
    echoed = backend.sent[1]["messages"][1]
    assert echoed["role"] == "assistant"
    assert echoed["content"][0] == thinking


def test_every_exchange_is_handed_to_the_recorder_the_first_one_before_it_is_sent():
    seen: list[str] = []

    class Watched(Backend):
        def send(self, request: dict) -> dict:
            seen.append("send")
            return super().send(request)

    def record(kind: str, _payload: dict) -> None:
        seen.append(kind)

    drive.loop(backend=Watched([turn(STOP_CALL)]), request=REQUEST, run_call=Executor(),
               max_actions=5, record=record)
    assert seen == ["request", "send", "exchange"]


def test_the_vendor_ceiling_is_answered_before_the_harness_limit():
    ex = Executor([{"text": "a", "is_error": False, "ceiling": True},
                   {"text": "b", "is_error": False, "ceiling": True}])
    backend = Backend([turn(use("u1", "search", query="x", quote="q")),
                       turn(use("u2", "search", query="y", quote="q"))])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=ex, max_actions=2, ceiling=2)
    assert outcome["reason"] == "ceiling_reached"


def test_a_refusal_from_the_backend_ends_the_run_with_an_error():
    backend = Backend([{"content": [], "stop_reason": "refusal", "usage": {}}])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)
    assert outcome["reason"] == "error"


def test_the_journal_backend_replays_what_was_recorded():
    backend = JournalBackend([turn(STOP_CALL)])
    outcome = drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)
    assert outcome["reason"] == "answer_met"


def test_the_journal_backend_refuses_to_invent_a_turn_it_never_recorded():
    backend = JournalBackend([])
    with pytest.raises(drive.DriveError, match="no recorded"):
        drive.loop(backend=backend, request=REQUEST, run_call=Executor(), max_actions=5)


PY_EXE = sys.executable


def make_ctx(tmp_path: Path) -> execute.Context:
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for directory in (run, sandbox, home):
        directory.mkdir()
    (sandbox / "graph.json").write_text("{}")
    invocation = {
        "package": {"launcher": PY_EXE, "interpreter": "/nonexistent/python", "site": str(tmp_path)},
        "subcommands": {"-c": {"positional": 1, "flags": {}}},
        "rejected_subcommands": [],
        "tools": {"search_tool": {"keys": {"q": {}}}},
    }
    ctx = execute.Context(run_dir=run, sandbox=sandbox, home=home,
                          environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                          invocation=invocation, volatile=[],
                          artifacts={"graph.json": rules.sha256_file(sandbox / "graph.json")})
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    return ctx


def journal(ctx: execute.Context) -> list[dict]:
    return [json.loads(line) for line in (ctx.run_dir / "journal.jsonl").read_text().splitlines()]


class Server:
    """A stand-in MCP server that answers from a script."""

    def __init__(self, answers: list[str | Exception]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool: str, args: dict) -> str:
        self.calls.append((tool, args))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_the_entry_name_of_an_awkward_tool_is_made_fit_for_a_filename():
    assert drive.entry_name("Semantic-Search/Nodes") == "semantic_search_nodes"
    assert len(drive.entry_name("x" * 80)) == 40


def test_a_runner_call_maps_onto_the_harness_call_shape():
    assert drive.to_call("act", {"argv": ["g", "query", "x"], "quote": "q"}) == {
        "kind": "act", "argv": ["g", "query", "x"]}
    assert drive.to_call("search_tool", {"q": "x", "quote": "q"}) == {
        "kind": "tool", "tool": "search_tool", "args": {"q": "x"}}


def test_the_runner_call_is_executed_and_its_output_comes_back(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="")
    answer = runner("act", {"argv": [PY_EXE, "-c", "print('hello')"], "quote": "the docs say so"})
    assert answer == {"text": "hello\n", "is_error": False, "ceiling": False}
    entry = journal(ctx)[-1]
    assert entry["by"] == "runner"
    assert entry["quote"] == "the docs say so"


def test_a_refused_call_comes_back_with_the_reason_and_is_journaled(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="")
    answer = runner("act", {"argv": [PY_EXE, "--forbidden"], "quote": "q"})
    assert answer["is_error"] is True
    assert "subcommand" in answer["text"]
    assert journal(ctx)[-1]["refused"] == answer["text"]


def test_the_origin_of_every_argument_is_journaled(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="print('hello')")
    runner("act", {"argv": [PY_EXE, "-c", "print('hello')"], "quote": "q"})
    origins = journal(ctx)[-1]["provenance"]
    assert [row["at"] for row in origins] == ["argv[1]", "argv[2]"]
    assert [row["kind"] for row in origins] == ["literal", "prompt"]


def test_a_place_returned_earlier_becomes_the_origin_of_a_later_argument(tmp_path: Path):
    ctx = make_ctx(tmp_path)

    def parse(call: dict, text: str) -> list[dict]:
        del call
        return [{"kind": "place", "path": text.strip(), "symbol": None,
                 "label": None, "qualified_name": None}]

    runner = drive.CallRunner(ctx, prompt_text="", parse=parse)
    runner("act", {"argv": [PY_EXE, "-c", "print('app/found.py')"], "quote": "q"})
    runner("act", {"argv": [PY_EXE, "-c", "app/found.py"], "quote": "q"})
    origins = journal(ctx)[-1]["provenance"]
    assert origins[-1]["kind"] == "record"
    assert origins[-1]["n"] == 1


def test_a_tool_call_takes_its_text_from_the_server(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    server = Server(["{\"results\": []}"])
    runner = drive.CallRunner(ctx, prompt_text="", server=server)
    answer = runner("search_tool", {"q": "x", "quote": "q"})
    assert answer["text"] == "{\"results\": []}"
    assert server.calls == [("search_tool", {"q": "x"})]
    assert journal(ctx)[-1]["tool"] == "search_tool"


def test_a_tool_call_that_raises_is_journaled_and_comes_back_as_an_error(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="", server=Server([RuntimeError("server gone")]))
    answer = runner("search_tool", {"q": "x", "quote": "q"})
    assert answer["is_error"] is True
    assert "server gone" in answer["text"]
    assert journal(ctx)[-1]["error"] == "server gone"


def test_a_tool_call_with_no_server_is_a_harness_fault_not_a_runner_mistake(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="")
    with pytest.raises(drive.DriveError, match="server"):
        runner("search_tool", {"q": "x", "quote": "q"})


def test_a_call_that_exits_non_zero_comes_back_as_an_error_carrying_both_streams(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="")
    code = "import sys; print('out'); sys.stderr.write('boom'); sys.exit(3)"
    answer = runner("act", {"argv": [PY_EXE, "-c", code], "quote": "q"})
    assert answer["is_error"] is True
    assert "out" in answer["text"]
    assert "boom" in answer["text"]


def test_a_tool_under_the_vendor_ceiling_is_counted_as_one(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="", server=Server(["{}"]),
                              ceiling_tools={"search_tool"})
    assert runner("search_tool", {"q": "x", "quote": "q"})["ceiling"] is True


def test_the_recorder_writes_the_request_first_then_one_file_per_exchange(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    recorder = drive.Recorder(run_dir)
    recorder("request", {"messages": []})
    recorder("exchange", {"request": {}, "response": {"a": 1}})
    recorder("exchange", {"request": {}, "response": {"a": 2}})
    assert json.loads((run_dir / "request.json").read_text()) == {"messages": []}
    assert json.loads((run_dir / "api" / "01.json").read_text())["response"] == {"a": 1}
    assert json.loads((run_dir / "api" / "02.json").read_text())["response"] == {"a": 2}


def test_the_run_is_closed_by_a_stop_entry_naming_the_reason_and_the_place(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    outcome = {"reason": "answer_met", "actions": 2, "ceiling_calls": 0,
               "stop": {"path": "a.py", "symbol": "f", "start": 1}}
    drive.finish(ctx, outcome)
    entry = journal(ctx)[-1]
    assert entry["kind"] == "stop"
    assert entry["by"] == "runner"
    assert entry["reason"] == "answer_met"
    assert entry["place"] == {"path": "a.py", "symbol": "f", "start": 1}


def test_a_run_that_named_no_place_is_closed_without_one(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    drive.finish(ctx, {"reason": "no_further_action", "actions": 1, "ceiling_calls": 0})
    entry = journal(ctx)[-1]
    assert entry["reason"] == "no_further_action"
    assert "place" not in entry


def test_a_tool_call_that_never_reached_the_vendor_does_not_spend_its_ceiling(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    runner = drive.CallRunner(ctx, prompt_text="", server=Server([RuntimeError("gone")]),
                              ceiling_tools={"search_tool"})
    assert runner("search_tool", {"q": "x", "quote": "q"})["ceiling"] is False


def test_a_refused_call_does_not_spend_the_vendor_ceiling(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    # why: the tool is configured and the server answers, so the grammar is what refuses this
    # call -- otherwise the test would exercise the raising path instead
    runner = drive.CallRunner(ctx, prompt_text="", server=Server(["{}"]),
                              ceiling_tools={"search_tool"})
    answer = runner("search_tool", {"unknown_key": "x", "quote": "q"})
    assert answer["is_error"] is True
    assert "unknown key" in answer["text"]
    assert answer["ceiling"] is False


def a_harness(**extra: Any) -> config.Harness:
    return config.Harness(
        system="graphify", adapter="graphify", invocation={"package": {}, "subcommands": {}},
        fixed_steps=[], configurations={"default": {"index": "indexes/graphify"}},
        default_configuration="default", environment={"PATH": "/usr/bin:/bin", "D": "<sandbox>/x"},
        sandbox_layout={"graphify-out": "<artifacts>"}, docs={}, volatile=["a"],
        allowed_scripts={"s.py": "abc"}, **extra)


def test_the_request_carries_the_prompt_as_the_only_message():
    request = drive.request_from("the prompt", [], model="claude-sonnet-5",
                                 effort="high", max_tokens=16000)
    assert request["messages"] == [{"role": "user", "content": "the prompt"}]


def test_the_request_names_the_model_and_the_effort_it_was_given():
    request = drive.request_from("p", [], model="claude-sonnet-5",
                                 effort="high", max_tokens=16000)
    assert request["model"] == "claude-sonnet-5"
    assert request["output_config"]["effort"] == "high"


def test_the_request_thinks_adaptively_and_asks_for_one_call_at_a_time():
    request = drive.request_from("p", [], model="m", effort="high", max_tokens=16000)
    assert request["thinking"] == {"type": "adaptive"}
    assert request["tool_choice"]["disable_parallel_tool_use"] is True


def test_the_request_offers_exactly_the_tools_it_was_given():
    definitions = [{"name": "act", "description": "d", "input_schema": {}}]
    request = drive.request_from("p", definitions, model="m", effort="high",
                                 max_tokens=16000)
    assert request["tools"] == definitions


def test_the_runner_settings_come_from_the_harness_when_it_states_them():
    harness = a_harness()
    harness.raw["runner_defaults"] = {"model": "claude-sonnet-5", "effort": "high", "max_actions": 8}
    assert drive.runner_settings(harness, model=None, effort=None, max_actions=None) == {
        "model": "claude-sonnet-5", "effort": "high", "max_actions": 8}


def test_a_setting_given_on_the_command_line_wins_over_the_harness():
    harness = a_harness()
    harness.raw["runner_defaults"] = {"model": "claude-sonnet-5", "effort": "high", "max_actions": 8}
    assert drive.runner_settings(harness, model="other", effort=None, max_actions=None)["model"] == "other"


def test_a_setting_stated_nowhere_is_refused_rather_than_invented():
    with pytest.raises(drive.DriveError, match="model"):
        drive.runner_settings(a_harness(), model=None, effort="high", max_actions=3)


def test_the_context_is_rebuilt_from_the_run_that_was_prepared(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    index = snapshot_dir(benchmark, "snap") / "indexes" / "graphify"
    index.mkdir(parents=True)
    (index / "build.yaml").write_text("vendor_writes: [graph.db]\n", encoding="utf-8")
    tmp_root = tmp_path / "root"
    (tmp_root / "r1" / "run").mkdir(parents=True)
    run_yaml = {"run_id": "r1", "snapshot": "snap", "configuration": "default",
                "artifacts": {"graphify-out/graph.json": "aa", "graphify-out/graph.db": "bb"}}
    ctx = drive.context_from_run(benchmark, a_harness(), run_yaml, tmp_root)
    assert ctx.sandbox == tmp_root / "sandbox" / "index"
    assert ctx.home == tmp_root / "r1" / "home"
    assert ctx.environment["D"] == f"{ctx.sandbox}/x"
    assert ctx.invocation["allowed_scripts"] == {"s.py": "abc"}
    assert ctx.volatile == ["a"]
    # inv: a file the vendor rewrites cannot be hash-checked before every call
    assert ctx.artifacts == {"graphify-out/graph.json": "aa"}


def test_every_call_a_runner_makes_counts_under_a_declared_ceiling():
    # why: the ceiling counts the vendor's own calls, and one vendor is reached through server
    # tools while the other is reached by running its command; counting only the first would
    # leave a command-line vendor with a ceiling that can never be reached
    harness = a_harness(ceiling={"quote": "at most five", "kind": "target"})
    harness.invocation = {"tools": {"a_tool": {"keys": {}}, "b_tool": {"keys": {}}}}
    assert drive.ceiling_tools(harness) == {"act", "a_tool", "b_tool"}


def test_a_command_line_vendor_still_has_its_calls_counted():
    harness = a_harness(ceiling={"quote": "owner's number", "kind": "owner", "calls": 5})
    harness.invocation = {"subcommands": {"query": {"positional": 1}}}
    assert drive.ceiling_tools(harness) == {"act"}


def test_a_system_declaring_no_ceiling_counts_no_tool_under_one():
    harness = a_harness()
    harness.invocation = {"tools": {"a_tool": {"keys": {}}}}
    assert drive.ceiling_tools(harness) == set()


def test_the_ceiling_limit_is_taken_only_from_a_number_the_vendor_states():
    assert drive.ceiling_limit(a_harness()) is None
    # why: the vendor's own quote carries digits, so a number found here would be the harness
    # reading the sentence rather than the vendor stating a field
    prose = {"quote": "\u22645 tool calls and \u2264800 total output tokens", "kind": "target"}
    assert drive.ceiling_limit(a_harness(ceiling=prose)) is None
    assert drive.ceiling_limit(a_harness(ceiling={"quote": "q", "kind": "target", "calls": 5})) == 5


def test_the_parser_the_driver_uses_is_the_one_scoring_uses(bench: Path, tmp_path: Path):
    tree = tmp_path / "bench"
    (tree / "systems").mkdir(parents=True)
    shutil.copytree(bench / "systems" / "graphify", tree / "systems" / "graphify")
    index = snapshot_dir(tree, "snap") / "indexes" / "graphify"
    index.mkdir(parents=True)
    (index / "build.yaml").write_text(yaml.safe_dump({"properties": {}}), encoding="utf-8")
    parse = score.parser_for(tree, {"system": "graphify", "configuration": "default",
                                    "snapshot": "snap"})
    entry = {"n": 1, "name": "query", "argv": ["/abs/graphify", "query", "x"], "system_call": True}
    records = parse(entry, "NODE render_invoice() [src=pkg/billing.py loc=L63 community=1]\n")
    assert [r["kind"] for r in records] == ["place"]
    assert records[0]["symbol"] == "render_invoice"


def prepared_attempt(tmp_path: Path, output: str = "found") -> tuple[Path, Path]:
    """Build the smallest tree `drive_run` accepts: a system, an index and a prepared run."""
    benchmark = tmp_path / "benchmark"
    system = benchmark / "systems" / "toy"
    system.mkdir(parents=True)
    # why: the adapter reads argv[1] to choose a shape, so the toy vendor is a real launcher
    # answering a real subcommand -- output alone would come back unparsed, and rightly
    launcher = tmp_path / "toy-vendor"
    launcher.write_text("#!/bin/sh\nprintf '%s\\n' \"$TOY_OUTPUT\"\n", encoding="utf-8")
    launcher.chmod(0o755)
    index = snapshot_dir(benchmark, "snap") / "indexes" / "toy"
    index.mkdir(parents=True)
    (index / "build.yaml").write_text("properties: {}\n", encoding="utf-8")
    (system / "harness.yaml").write_text(f"""
adapter: graphify
version: {{cli: "0.9.27"}}
invocation:
  package: {{launcher: {launcher}, interpreter: /nonexistent/python, site: {tmp_path}}}
  subcommands: {{-c: {{positional: 1, flags: {{}}}}, query: {{positional: 1, flags: {{}}}}}}
  rejected_subcommands: []
fixed_steps: []
default_configuration: default
configurations: {{default: {{index: indexes/toy}}}}
sandbox_layout: {{out: "<artifacts>"}}
environment: {{PATH: "/usr/bin:/bin", LANG: "C.UTF-8", TOY_OUTPUT: "{output}"}}
docs: {{}}
runner_defaults: {{model: claude-sonnet-5, effort: high, max_actions: 4}}
""", encoding="utf-8")
    tmp_root = tmp_path / "root"
    run_dir = tmp_root / "r1" / "run"
    run_dir.mkdir(parents=True)
    (tmp_root / "r1" / "home").mkdir()
    sandbox = tmp_root / "sandbox" / "index"
    sandbox.mkdir(parents=True)
    (run_dir / "run.yaml").write_text(
        "run_id: r1\nsystem: toy\nconfiguration: default\nsnapshot: snap\nquestion: q001\n"
        "artifacts: {}\n", encoding="utf-8")
    # inv: a real attempt always names a question and a reference, and blindness is shown
    # against that reference before the first request leaves
    write_question(benchmark, "q001", {"id": "q001", "snapshot": "snap", "text": "t"})
    (references_dir(benchmark, "snap") / "q001.yaml").write_text(
        "places:\n  - {path: pkg/elsewhere.py, symbol: unrelated_symbol, start: 1}\n",
        encoding="utf-8")
    (run_dir / "prompt.md").write_text(
        f"find the place\n{prompt.HEADING}\n\nnothing ran yet\n", encoding="utf-8")
    execute.append(execute.Context(run_dir=run_dir, sandbox=sandbox, home=tmp_root / "r1" / "home",
                                   environment={}, invocation={}, volatile=[], artifacts={}),
                   {"n": 0, "kind": "header", "rules_version": 1})
    return benchmark, tmp_root


def test_a_prepared_attempt_is_driven_and_its_journal_closed(tmp_path: Path):
    benchmark, tmp_root = prepared_attempt(tmp_path)
    backend = JournalBackend([
        turn(use("u1", "act", argv=[str(tmp_path / "toy-vendor"), "query", "x"], quote="the docs say so")),
        turn(STOP_CALL)])
    outcome = drive.drive_run(benchmark, "r1", tmp_root, backend)
    assert outcome["reason"] == "answer_met"
    assert outcome["actions"] == 1
    entries = [json.loads(line) for line in
               (tmp_root / "r1" / "run" / "journal.jsonl").read_text().splitlines()]
    assert entries[-1] == {"n": 2, "kind": "stop", "by": "runner", "reason": "answer_met",
                           "place": {"path": "a.py", "symbol": "f", "start": 1}}
    assert entries[-2]["by"] == "runner"
    assert entries[-2]["provenance"][0]["kind"] == "literal"
    assert (tmp_root / "r1" / "run" / "request.json").is_file()
    assert (tmp_root / "r1" / "run" / "api" / "01.json").is_file()


def test_drive_run_names_the_model_the_first_exchange_served(tmp_path: Path):
    benchmark, tmp_root = prepared_attempt(tmp_path)
    backend = JournalBackend([{**turn(STOP_CALL), "model": "qwen3-8b"}])
    outcome = drive.drive_run(benchmark, "r1", tmp_root, backend)
    assert outcome["model_served"] == "qwen3-8b"


def test_drive_run_names_no_model_when_the_response_carries_none(tmp_path: Path):
    benchmark, tmp_root = prepared_attempt(tmp_path)
    outcome = drive.drive_run(benchmark, "r1", tmp_root, JournalBackend([turn(STOP_CALL)]))
    assert outcome["model_served"] is None


def test_drive_run_names_the_first_exchanges_model_even_when_a_later_one_answers(tmp_path: Path):
    node = "NODE thing() [src=app/x.py loc=L7 community=1]"
    benchmark, tmp_root = prepared_attempt(tmp_path, output=node)
    vendor = str(tmp_path / "toy-vendor")
    backend = JournalBackend([
        {**turn(use("u1", "act", argv=[vendor, "query", "x"], quote="q")), "model": "first-model"},
        {**turn(STOP_CALL), "model": "second-model"}])
    outcome = drive.drive_run(benchmark, "r1", tmp_root, backend)
    assert outcome["model_served"] == "first-model"


def test_an_attempt_prepared_without_a_prompt_is_refused(tmp_path: Path):
    benchmark, tmp_root = prepared_attempt(tmp_path)
    (tmp_root / "r1" / "run" / "prompt.md").unlink()
    with pytest.raises(SystemExit, match="prompt.md"):
        drive.drive_run(benchmark, "r1", tmp_root, JournalBackend([]))


def test_a_driven_attempt_builds_records_even_with_no_server(tmp_path: Path):
    node = "NODE thing() [src=app/x.py loc=L7 community=1]"
    benchmark, tmp_root = prepared_attempt(tmp_path, output=node)
    vendor = str(tmp_path / "toy-vendor")
    backend = JournalBackend([
        turn(use("u1", "act", argv=[vendor, "query", "x"], quote="q")),
        turn(use("u2", "act", argv=[vendor, "query", "app/x.py"], quote="q")),
        turn(STOP_CALL)])
    drive.drive_run(benchmark, "r1", tmp_root, backend)
    entries = [json.loads(line) for line in
               (tmp_root / "r1" / "run" / "journal.jsonl").read_text().splitlines()]
    second = next(e for e in entries if e.get("n") == 2)
    assert second["provenance"][-1] == {"at": "argv[2]", "value": "app/x.py", "kind": "record", "n": 1}


def test_a_leaking_tool_definition_stops_the_run_before_anything_is_sent(tmp_path: Path):
    benchmark, tmp_root = prepared_attempt(tmp_path)
    (references_dir(benchmark, "snap") / "q001.yaml").write_text(
        "places:\n  - {path: pkg/logic.py, symbol: find_the_place, start: 1}\n", encoding="utf-8")
    # why: the prompt itself carries the symbol, which is what the check must catch
    (tmp_root / "r1" / "run" / "prompt.md").write_text(
        f"find_the_place\n{prompt.HEADING}\n\nnothing\n", encoding="utf-8")
    backend = Backend([turn(STOP_CALL)])
    with pytest.raises(SystemExit, match="find_the_place"):
        drive.drive_run(benchmark, "r1", tmp_root, backend)
    assert backend.sent == []


def test_the_record_of_an_exchange_says_what_went_on_the_wire():
    class Rendering(Backend):
        def as_sent(self, request: dict) -> dict:
            return {"only": "what this server speaks", "model": request["model"]}

    seen: list[dict] = []
    drive.loop(backend=Rendering([turn(STOP_CALL)]), request=REQUEST, run_call=Executor(),
               max_actions=5, record=lambda kind, payload: seen.append({kind: payload}))
    exchange = next(entry["exchange"] for entry in seen if "exchange" in entry)
    assert exchange["sent"] == {"only": "what this server speaks", "model": "claude-sonnet-5"}
    # inv: both are kept -- what the driver built and what the far side actually received
    assert exchange["request"]["messages"] == REQUEST["messages"]
    assert exchange["request"] != exchange["sent"]


def test_a_backend_that_sends_the_request_as_it_stands_records_it_once():
    seen: list[dict] = []
    drive.loop(backend=Backend([turn(STOP_CALL)]), request=REQUEST, run_call=Executor(),
               max_actions=5, record=lambda kind, payload: seen.append({kind: payload}))
    exchange = next(entry["exchange"] for entry in seen if "exchange" in entry)
    assert "sent" not in exchange


def test_the_turns_are_driven_without_closing_the_journal(tmp_path: Path):
    # inv: only drive_run closes the journal, after the server it started is gone -- a stop
    # written while the server still runs would not be the journal's last entry
    benchmark, tmp_root = prepared_attempt(tmp_path)
    run_dir = tmp_root / "r1" / "run"
    harness = config.load_harness(benchmark, "toy")
    ctx = drive.context_from_run(benchmark, harness, config.load_yaml(run_dir / "run.yaml"), tmp_root)
    drive._drive(ctx, harness, "find the place", {}, None, JournalBackend([turn(STOP_CALL)]),
                 {"model": "m", "effort": "high", "max_actions": 4}, 4096, run_dir,
                 benchmark, "q001")
    entries = [json.loads(line) for line in (run_dir / "journal.jsonl").read_text().splitlines()]
    assert [e for e in entries if e["kind"] == "stop"] == []


def _ceiling_ctx(tmp_path: Path, spent: int) -> execute.Context:
    ctx = make_ctx(tmp_path)
    for i in range(spent):
        execute.append(ctx, {"n": i + 1, "kind": "call", "name": "search", "by": "harness",
                             "ceiling_call": True, "exit": 0})
    return ctx


def test_what_the_fixed_steps_already_spent_is_taken_off_the_vendor_ceiling(tmp_path: Path):
    harness = a_harness(ceiling={"quote": "q", "kind": "target", "calls": 5})
    assert drive.ceiling_left(harness, _ceiling_ctx(tmp_path, spent=2)) == 3


def test_a_vendor_that_declares_no_number_bounds_nothing(tmp_path: Path):
    assert drive.ceiling_left(a_harness(), _ceiling_ctx(tmp_path, spent=2)) is None


def test_a_ceiling_already_spent_leaves_the_runner_none_of_it(tmp_path: Path):
    harness = a_harness(ceiling={"quote": "q", "kind": "target", "calls": 2})
    assert drive.ceiling_left(harness, _ceiling_ctx(tmp_path, spent=3)) == 0
