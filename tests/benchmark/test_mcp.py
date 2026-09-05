import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

from mcp.shared.exceptions import McpError

from benchmark.harness import execute, mcp, rules
from tests.benchmark.conftest import BENCH, SHIPPED_SNAPSHOTS

CRG = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "code-review-graph"
# inv: whichever snapshot in this tree holds a built code-review-graph index; the tests below read
# its graph.db and nothing else about it, so any one of them serves
_BUILT = sorted((BENCH / SHIPPED_SNAPSHOTS).glob("*/indexes/code-review-graph/graph.db"))
INDEX = _BUILT[0].parent if _BUILT else None

pytestmark = pytest.mark.slow

_VENDOR_ABSENT = not (CRG.exists() and INDEX is not None)


def _server_gone(sandbox: Path) -> bool:
    """Tell whether no vendor server is serving this test's own sandbox."""
    # inv: matched on the sandbox path, not the launcher path -- the launcher is shared, so a
    # server any other process runs would otherwise read as this block failing to shut its own down
    return subprocess.run(["pgrep", "-f", str(sandbox)], capture_output=True, check=False).returncode != 0

def _make_ctx(tmp_path: Path) -> tuple[execute.Context, Path, Path]:
    # inv: every caller is skipped when _VENDOR_ABSENT, so INDEX is a directory by the time
    # this runs; the assert is what says so to a reader and to the checker
    assert INDEX is not None
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for d in (run, sandbox / ".code-review-graph", home):
        d.mkdir(parents=True)
    (sandbox / ".code-review-graph" / "graph.db").write_bytes((INDEX / "graph.db").read_bytes())
    ctx = execute.Context(
        run_dir=run, sandbox=sandbox, home=home,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HF_HUB_OFFLINE": "1",
                     "CRG_DATA_DIR": str(sandbox / ".code-review-graph")},
        invocation={"package": {"launcher": str(CRG), "interpreter": "/nonexistent", "site": ""}, "subcommands": {},
                    "tools": {"list_graph_stats_tool": {"keys": {}}}},
        volatile=["_graph.age_seconds", "_graph.updated_at"],
        artifacts={".code-review-graph/graph.db": rules.sha256_file(sandbox / ".code-review-graph" / "graph.db")},
    )
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    return ctx, run, sandbox


@pytest.mark.skipif(_VENDOR_ABSENT, reason="vendor package or index absent")
def test_server_lists_and_calls(tmp_path: Path):
    ctx, run, sandbox = _make_ctx(tmp_path)
    with mcp.Server(ctx, CRG, ["list_graph_stats_tool"]) as s:
        names = [t["name"] for t in s.list_tools()]
        assert "list_graph_stats_tool" in names
        text = s.call("list_graph_stats_tool", {})
        assert json.loads(text)["status"] == "ok"
        # why: repo_root must always be pinned to the sandbox, even when a caller passes another path
        overridden = s.call("list_graph_stats_tool", {"repo_root": "/nonexistent/elsewhere"})
        assert json.loads(overridden)["status"] == "ok"
        with pytest.raises(mcp.ToolError):
            s.call("does_not_exist_tool", {})
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in entries] == ["header", "server", "server"]
    assert entries[1]["event"] == "start"
    assert entries[1]["argv"] == [str(CRG), "serve", "--repo", str(sandbox), "--tools", "list_graph_stats_tool"]
    # why: this vendor writes .gitignore and the wal/shm pair inside a tool call, not at connect
    # or disconnect (see benchmark/harness/mcp.py); a non-empty diff here would mean it changed
    assert entries[1]["files"] == []
    assert entries[2]["event"] == "stop"
    assert entries[2]["files"] == []


@pytest.mark.skipif(_VENDOR_ABSENT, reason="vendor package or index absent")
def test_exit_reraises_a_listing_failure_when_the_body_did_not_raise(tmp_path: Path):
    ctx, run, sandbox = _make_ctx(tmp_path)
    server = mcp.Server(ctx, CRG, ["list_graph_stats_tool"])
    server.__enter__()
    text = server.call("list_graph_stats_tool", {})
    assert json.loads(text)["status"] == "ok"
    shutil.rmtree(sandbox)
    sandbox.write_bytes(b"clobbered")  # a root that exists but is not a directory
    with pytest.raises(NotADirectoryError):
        server.__exit__(None, None, None)
    assert _server_gone(sandbox)
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    assert entries[-1]["event"] == "stop"
    assert "error" in entries[-1]


@pytest.mark.skipif(_VENDOR_ABSENT, reason="vendor package or index absent")
def test_exit_does_not_displace_a_body_exception_with_its_own_listing_failure(tmp_path: Path):
    ctx, run, sandbox = _make_ctx(tmp_path)

    class BodyError(Exception):
        pass

    def _run_and_fail() -> None:
        with mcp.Server(ctx, CRG, ["list_graph_stats_tool"]) as s:
            text = s.call("list_graph_stats_tool", {})
            assert json.loads(text)["status"] == "ok"
            shutil.rmtree(sandbox)
            sandbox.write_bytes(b"clobbered")  # a root that exists but is not a directory
            raise BodyError("body failed for its own reason")

    with pytest.raises(BodyError):
        _run_and_fail()
    assert _server_gone(sandbox)
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
    assert entries[-1]["event"] == "stop"
    assert "error" in entries[-1]


@pytest.mark.skipif(_VENDOR_ABSENT, reason="vendor package or index absent")
def test_call_times_out_instead_of_waiting_forever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ctx, _run, sandbox = _make_ctx(tmp_path)
    # inv: the timeout is shortened after the handshake, which the same constant bounds -- set
    # before, it would time out the connection and this test would never reach a call
    with mcp.Server(ctx, CRG, ["list_graph_stats_tool"]) as s:
        monkeypatch.setattr(execute, "VENDOR_TIMEOUT_S", 0.001)
        with pytest.raises(McpError) as excinfo:
            s.call("list_graph_stats_tool", {})
    # inv: the ceiling is enforced by the client, so a vendor that stops replying still returns
    # control to the harness rather than blocking the run with no journal line
    assert "Timed out" in str(excinfo.value)
    assert _server_gone(sandbox)


def test_handshake_gives_up_on_a_server_that_starts_and_never_speaks(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # inv: a vendor that holds its pipes open and never replies is bounded by the same timeout as
    # a tool call; unbounded, it would hold this thread and the sandbox lock with no journal line
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for d in (run, sandbox, home):
        d.mkdir(parents=True)
    mute = tmp_path / "mute-server.sh"
    mute.write_text("#!/bin/sh\nexec sleep 100000\n", encoding="utf-8")
    mute.chmod(0o755)
    ctx = execute.Context(
        run_dir=run, sandbox=sandbox, home=home,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        invocation={"package": {"launcher": str(mute), "interpreter": "/nonexistent", "site": ""},
                    "subcommands": {}, "tools": {}},
        volatile=[], artifacts={},
    )
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    monkeypatch.setattr(execute, "VENDOR_TIMEOUT_S", 1.0)
    # inv: the handshake runs off the main thread, so an unbounded one fails this test on the join
    # instead of hanging the suite -- which is exactly what the bound removed does
    outcome: list[BaseException | None] = []

    def enter() -> None:
        try:
            mcp.Server(ctx, mute, []).__enter__()
            outcome.append(None)
        except BaseException as exc:  # fastmcp wraps the timeout in a type of its own
            outcome.append(exc)

    worker = threading.Thread(target=enter, daemon=True)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "the handshake never returned; it is not bounded"
    assert outcome, "the handshake thread recorded no outcome"
    assert outcome[0] is not None, "the handshake succeeded against a server that never spoke"
    assert "initialize" in str(outcome[0]).lower() or "timed out" in str(outcome[0]).lower()
    # inv: the failed handshake still lands in the journal, so the attempt is not silent
    entries = [json.loads(x) for x in (run / "journal.jsonl").read_text().splitlines()]
    assert entries[-1]["kind"] == "server"
    assert "error" in entries[-1]
