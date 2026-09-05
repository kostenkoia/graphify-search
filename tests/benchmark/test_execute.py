import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.harness import execute, rules

PY = sys.executable


def make_ctx(tmp_path: Path) -> execute.Context:
    run, sandbox, home = tmp_path / "run", tmp_path / "sandbox", tmp_path / "home"
    for d in (run, sandbox, home):
        d.mkdir()
    (sandbox / "graph.json").write_text("{}")
    inv = {
        "package": {"launcher": PY, "interpreter": "/nonexistent/python", "site": str(tmp_path)},
        "subcommands": {"-c": {"positional": 1, "flags": {}}},
        "rejected_subcommands": [],
        "tools": {"t": {"keys": {"q": {}}}},
    }
    ctx = execute.Context(run_dir=run, sandbox=sandbox, home=home,
                          environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                          invocation=inv, volatile=[],
                          artifacts={"graph.json": rules.sha256_file(sandbox / "graph.json")})
    execute.append(ctx, {"n": 0, "kind": "header", "rules_version": 1})
    return ctx


def test_execute_writes_pair_and_journal(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    code = "import os, pathlib; print('hi'); pathlib.Path('state.txt').write_text('s'); print(os.environ.get('HOME'))"
    entry = execute.execute(ctx, name="probe", call={"kind": "act", "argv": [PY, "-c", code]},
                            quote=None, by="harness", system_call=True)
    out = (ctx.run_dir / "01_probe.out").read_text()
    assert out.splitlines()[0] == "hi"
    assert out.splitlines()[1] == str(ctx.home)
    assert entry["n"] == 1
    assert entry["exit"] == 0
    assert entry["out_sha256"] == hashlib.sha256(out.encode()).hexdigest()
    assert {f["path"] for f in entry["files"]} == {"sandbox/state.txt"}
    lines = (ctx.run_dir / "journal.jsonl").read_text().splitlines()
    assert json.loads(lines[-1]) == entry
    assert (ctx.run_dir / "01_probe.cmd").exists()
    assert (ctx.run_dir / "01_probe.err").exists()


def test_execute_uses_fixed_environment_and_no_shell(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    code = "import os, json; print(json.dumps(sorted(os.environ))); print(os.getcwd())"
    execute.execute(ctx, name="env", call={"kind": "act", "argv": [PY, "-c", code]},
                    quote=None, by="harness", system_call=False)
    out = (ctx.run_dir / "01_env.out").read_text().splitlines()
    # inv: macOS injects __CF_USER_TEXT_ENCODING into every child; only dunder keys may appear beyond ours
    keys = {k for k in json.loads(out[0]) if not k.startswith("__")}
    assert keys == {"PATH", "LANG", "HOME"}
    assert out[1] == str(ctx.sandbox)
    # why: the shared "-c" row allows one positional; widen it locally so the second
    # argv element below survives check_call as data instead of being refused as unconsumed
    ctx.invocation = {**ctx.invocation, "subcommands": {"-c": {"positional": 2, "flags": {}}}}
    shell_argv = [PY, "-c", "import sys; print(sys.argv)", "$(echo x)"]
    execute.execute(ctx, name="shell", call={"kind": "act", "argv": shell_argv},
                    quote=None, by="harness", system_call=False)
    # inv: shell=False, so "$(echo x)" is argv data, never a command
    assert "$(echo x)" in (ctx.run_dir / "02_shell.out").read_text()


def test_execute_refuses_when_artifact_changed(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    (ctx.sandbox / "graph.json").write_text('{"x": 1}')
    with pytest.raises(execute.ArtifactsChangedError):
        execute.execute(ctx, name="x", call={"kind": "act", "argv": [PY, "-c", "print(1)"]},
                        quote=None, by="harness", system_call=True)
    assert not (ctx.run_dir / "01_x.cmd").exists()
    lines = (ctx.run_dir / "journal.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["header"]


def test_execute_records_a_refusal_without_running(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    entry = execute.execute(ctx, name="bad", call={"kind": "act", "argv": [PY, "nope", "x"]},
                            quote=None, by="harness", system_call=True)
    assert entry["action"] is False
    assert "unknown subcommand" in entry["refused"]
    assert not (ctx.run_dir / "01_bad.out").exists()
    lines = (ctx.run_dir / "journal.jsonl").read_text().splitlines()
    assert json.loads(lines[-1]) == entry


def test_execute_reraises_and_journals_an_exec_failure(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    # why: the call must pass check_call and still fail to exec, so it names an allowlisted script
    # under an interpreter that is not on disk
    script = tmp_path / "vetted.py"
    script.write_text("print(1)\n")
    ctx.invocation = {**ctx.invocation, "allowed_scripts": {str(script): rules.sha256_file(script)}}
    argv = ["/nonexistent/python", str(script)]
    with pytest.raises(FileNotFoundError):
        execute.execute(ctx, name="boom", call={"kind": "act", "argv": argv},
                        quote=None, by="harness", system_call=True)
    assert (ctx.run_dir / "01_boom.cmd").exists()
    assert not (ctx.run_dir / "01_boom.out").exists()
    # inv: both halves of the pair go, not just the .out -- an .err the child never wrote to is
    # still a file the readers would find beside a journalled failure
    assert not (ctx.run_dir / "01_boom.err").exists()
    lines = (ctx.run_dir / "journal.jsonl").read_text().splitlines()
    entry = json.loads(lines[-1])
    assert entry["n"] == 1
    assert entry["exit"] is None
    assert "error" in entry
    # why: the failed attempt is on the journal, so the next call gets its own n and .cmd stem
    assert execute.next_n(ctx) == 2


def test_execute_reraises_and_journals_a_hung_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(execute, "VENDOR_TIMEOUT_S", 0.5)
    # why: the child prints on both streams before it hangs, so each file is genuinely part-written
    # when the timeout fires -- a hang that wrote nothing cannot show that a partial file is removed
    code = ("import sys, time; sys.stdout.write('o' * 4096); sys.stderr.write('e' * 4096); "
            "sys.stdout.flush(); sys.stderr.flush(); time.sleep(30)")
    with pytest.raises(subprocess.TimeoutExpired):
        execute.execute(ctx, name="hang", call={"kind": "act", "argv": [PY, "-c", code]},
                        quote=None, by="harness", system_call=True)
    assert not (ctx.run_dir / "01_hang.out").exists()
    # inv: a flood that hangs is the case this guards -- the part-written .err must go too, or it
    # survives beside a journalled failure and is copied into the published record
    assert not (ctx.run_dir / "01_hang.err").exists()
    entry = json.loads((ctx.run_dir / "journal.jsonl").read_text().splitlines()[-1])
    # inv: a hang lands in the same shape as a failed exec -- exit None with an error and no
    # .out -- which is the shape audit.check_run and score._executed both exempt on
    assert entry["exit"] is None
    assert "timed out" in entry["error"]
    assert execute.next_n(ctx) == 2


def test_execute_tracks_home_root_and_a_removed_sandbox_file(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    setup_code = "import pathlib; pathlib.Path('drop_me.txt').write_text('d')"
    execute.execute(ctx, name="setup", call={"kind": "act", "argv": [PY, "-c", setup_code]},
                    quote=None, by="harness", system_call=True)
    code = (
        "import os, pathlib; "
        "pathlib.Path('drop_me.txt').unlink(); "
        "pathlib.Path(os.environ['HOME'], 'greeting.txt').write_text('hi')"
    )
    entry = execute.execute(ctx, name="move", call={"kind": "act", "argv": [PY, "-c", code]},
                            quote=None, by="harness", system_call=True)
    by_path = {f["path"]: f["sha256"] for f in entry["files"]}
    assert by_path["home/greeting.txt"] == hashlib.sha256(b"hi").hexdigest()
    assert by_path["sandbox/drop_me.txt"] is None


def test_execute_rejects_bad_name(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    with pytest.raises(ValueError, match="name"):
        execute.execute(ctx, name="../x", call={"kind": "act", "argv": [PY, "-c", "print(1)"]},
                        quote=None, by="harness", system_call=True)


def test_execute_records_tool_text(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    entry = execute.execute(ctx, name="t", call={"kind": "tool", "tool": "t", "args": {"q": "a"}},
                            quote="q", by="harness", system_call=True, ceiling_call=True,
                            tool_text='{"status": "ok"}')
    assert entry["tool"] == "t"
    assert entry["args"] == {"q": "a"}
    assert (ctx.run_dir / "01_t.out").read_text() == '{"status": "ok"}'


def test_execute_before_param_extends_the_diff_window(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    # why: simulates a tool call whose side effect happened before execute() itself was invoked
    early = execute.watched_listing(ctx)
    (ctx.sandbox / "written_during_call.txt").write_text("x")
    call = {"kind": "tool", "tool": "t", "args": {"q": "a"}}
    without_before = execute.execute(ctx, name="a", call=call, quote=None, by="harness",
                                     system_call=True, tool_text="ok")
    with_before = execute.execute(ctx, name="b", call=call, quote=None, by="harness",
                                  system_call=True, tool_text="ok", before=early)
    assert [f["path"] for f in without_before["files"]] == []
    assert [f["path"] for f in with_before["files"]] == ["sandbox/written_during_call.txt"]


def test_execute_journals_the_child_s_real_exit_code(tmp_path: Path):
    # inv: audit.check_run's non-zero-exit violation can only ever fire if this value is the
    # child's own; journaling a constant would make a wholly failed run look like a clean miss
    ctx = make_ctx(tmp_path)
    entry = execute.execute(ctx, name="fail", call={"kind": "act", "argv": [PY, "-c", "raise SystemExit(3)"]},
                            quote=None, by="harness", system_call=True)
    assert entry["exit"] == 3
    assert json.loads((ctx.run_dir / "journal.jsonl").read_text().splitlines()[-1])["exit"] == 3


def test_execute_records_the_origin_of_every_argument(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    origins = [{"at": "argv[1]", "value": "-c", "kind": "literal"}]
    entry = execute.execute(ctx, name="probe", call={"kind": "act", "argv": [PY, "-c", "print(1)"]},
                            quote=None, by="runner", system_call=True, provenance=origins)
    assert entry["provenance"] == origins


def test_a_refused_call_still_records_where_its_arguments_came_from(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    origins = [{"at": "argv[1]", "value": "--forbidden", "kind": "none"}]
    entry = execute.execute(ctx, name="probe", call={"kind": "act", "argv": [PY, "--forbidden"]},
                            quote=None, by="runner", system_call=True, provenance=origins)
    assert entry["action"] is False
    assert entry["provenance"] == origins
