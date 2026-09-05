import os
import pwd
import stat
import subprocess
from pathlib import Path

import pytest

from benchmark.harness import seal

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmark"
LOCK = BENCH / "lock"
# why: the plan names the owner common.sh derives -- SUDO_USER when sudo carried one, else the
# name of the uid the tests run under, which no environment variable can move
OWNER = os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name


def _skip_unless_the_launchers_are_this_checkouts():
    # why: install refuses a launcher outside this checkout's envs/ before it prints a line, so
    # the plan is computable only where the launchers are this checkout's own
    envs = BENCH / "envs"
    outside = [p for p in seal.launchers(BENCH) if envs not in Path(p).parents]
    if outside:
        pytest.skip(f"install refuses a launcher outside {envs}: {outside[0]} — the shipped "
                    "harness.yaml files point at another machine's envs/, and README §4 step 1 "
                    "rewrites them for this checkout")


def _run(script: str, *args: str, root: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/bash", str(LOCK / script), "--dry-run", "--root", str(root or REPO), *args],
                          capture_output=True, text=True, check=False)


@pytest.mark.parametrize("script", ["install", "uninstall", "unlock", "lock"])
def test_scripts_are_executable_and_refuse_without_root_outside_dry_run(script):
    assert LOCK.joinpath(script).stat().st_mode & stat.S_IXUSR
    proc = subprocess.run(["/bin/bash", str(LOCK / script), "--root", str(REPO)],
                          capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "must run as root" in proc.stderr


def test_install_plan_covers_every_class_the_checkout_holds_and_touches_git_never():
    _skip_unless_the_launchers_are_this_checkouts()
    proc = _run("install")
    assert proc.returncode == 0, proc.stderr
    plan = proc.stdout
    # inv: install creates the account only where it is absent, so the plan names it only on a
    # machine that does not already hold it
    assert "sysadminctl -addUser bench" in plan or _bench_exists()
    assert f"dscl . -append /Groups/bench GroupMembership {OWNER}" in plan or "dseditgroup" in plan
    # inv: the group is created with a gid, or chown cannot name it; the plan shows the create
    # only on a machine whose group does not already carry one
    assert "dscl . -create /Groups/bench PrimaryGroupID" in plan or _group_has_gid()
    assert "chown root:wheel benchmark/harness" in plan or "chown -h root:wheel benchmark/harness" in plan
    assert "chflags uchg benchmark/harness" in plan
    assert "chown -R bench:bench benchmark/record" in plan
    # why: the system under test is owned as a tree, one chown per path the seal lists, and a
    # checkout whose machine is not set up and whose snapshot is not built holds none of those
    # paths; naming one here would assert a machine's state instead of the plan's coverage
    # inv: seal.paths finds no sut path on such a checkout, so this loop covers the sut class only
    # on a machine whose snapshot is built
    for rel in seal.paths(BENCH, "sut"):
        assert f"chown -R bench:bench {rel.as_posix()}" in plan
    assert "benchmark/envs/harness" in plan
    assert "chown root:wheel benchmark/lock" in plan
    assert "/etc/sudoers.d/bench-harness" in plan
    assert "visudo -cf" in plan
    assert "benchmark/lock/machine.yaml" in plan
    # actual plan line: printf '[safe]\n\tdirectory = %s\n' "<root>" > /Users/bench/.gitconfig
    # — an ini-format safe.directory entry, not the literal token "safe.directory"
    assert "[safe]" in plan
    assert "directory = " in plan
    assert ".git" not in [line.split()[-1] for line in plan.splitlines() if line.startswith(("chown", "chflags"))]


def test_install_keeps_the_executable_bit_only_where_git_has_it():
    _skip_unless_the_launchers_are_this_checkouts()
    lines = _run("install").stdout.splitlines()
    # inv: git tracks the executable bit, so a 555 on a file committed 644 leaves the tree dirty
    # and require_clean refuses every attempt
    assert "chmod 444 benchmark/lock/common.sh" in lines
    assert "chmod 444 benchmark/lock/sudoers.template" in lines
    assert "chmod 555 benchmark/lock/install" in lines
    assert "chmod 555 benchmark/lock" in lines


def test_install_never_chflags_the_lock_directory_itself():
    # a directory's UF_IMMUTABLE refuses creating or removing entries even for root, which
    # would brick unlock's write into benchmark/lock/UNLOCKED; root-owned 555 already refuses
    # every non-root write, so only the lock class's files may carry the flag
    _skip_unless_the_launchers_are_this_checkouts()
    proc = _run("install")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert "chflags uchg benchmark/lock/install" in lines
    assert "chflags uchg benchmark/lock" not in lines


def test_scripts_refuse_a_root_without_a_benchmark(tmp_path):
    proc = _run("install", root=tmp_path)      # no benchmark/ here at all
    assert proc.returncode != 0
    assert "no benchmark/ under" in proc.stderr


def test_unlock_plan_opens_the_instrument_and_nothing_else():
    proc = _run("unlock", "because I say so")
    assert proc.returncode == 0, proc.stderr
    plan = proc.stdout
    assert "chflags nouchg benchmark/harness" in plan
    assert (f"chown -h {OWNER} benchmark/harness" in plan
            or f"chown {OWNER} benchmark/harness" in plan)
    assert "benchmark/lock/UNLOCKED" in plan
    assert "because I say so" in plan
    forbidden_paths = ("benchmark/lock/install", "benchmark/record", "benchmark/envs",
                       "benchmark/systems/graphify-search-minilm/models")
    touched = [line.split()[-1] for line in plan.splitlines() if line.startswith(("chown", "chflags"))]
    for forbidden in forbidden_paths:
        assert forbidden not in touched


def test_unlock_quotes_a_reason_as_a_single_quoted_yaml_scalar():
    # a `:` or `#` in an unquoted reason breaks the YAML lock later reloads; the YAML
    # `''`-doubling convention is what keeps an embedded `'` from ending the scalar early
    proc = _run("unlock", "fix q018: the 'query' step")
    assert proc.returncode == 0, proc.stderr
    assert "reason: 'fix q018: the ''query'' step'" in proc.stdout


def test_unlock_refuses_while_a_row_has_no_outcome(tmp_path):
    # a copy of the real tree is too big; a minimal repo with one open row suffices
    b = tmp_path / "benchmark"
    (b / "record").mkdir(parents=True)
    (b / "harness").mkdir()
    (b / "record" / "attempts.jsonl").write_text('{"run_id": "x", "attempt": 1}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    proc = _run("unlock", "reason", root=tmp_path)
    assert proc.returncode != 0
    assert "a row has no outcome" in proc.stderr


def test_lock_plan_seals_then_closes_and_refuses_a_dirty_tree(tmp_path):
    proc = _run("lock")
    assert proc.returncode == 0, proc.stderr
    plan = proc.stdout
    assert "benchmark.harness seal" in plan
    assert "chflags uchg" in plan
    # actual plan line: rm <root>/benchmark/lock/UNLOCKED — BENCH is always an absolute path
    # (common.sh resolves ROOT with pwd -P, or takes --root verbatim), so "rm " is never
    # immediately followed by the relative "benchmark/lock/UNLOCKED"
    assert any(line.startswith("rm ") and "benchmark/lock/UNLOCKED" in line for line in plan.splitlines())
    assert plan.index("benchmark.harness seal") < plan.index("chflags uchg")
    # actual plan line: sudo -u "<owner>" git -C "<root>" commit -q --only -m "..." -- benchmark/INSTRUMENT.yaml
    # — "-C \"<root>\"" separates "git" from "commit", so the two are checked apart
    assert "git -C" in plan
    assert " commit " in plan
    assert "chore(benchmark): seal" in plan


def test_harness_py_resolves_venv_then_the_interpreter_on_path(tmp_path):
    # a runner that pip-installed the package into the system interpreter has neither
    # envs/harness nor .venv, and common.sh must still find an interpreter there
    (tmp_path / "benchmark").mkdir()
    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    stub_python3 = stub_bin / "python3"
    stub_python3.write_text("#!/bin/bash\n", encoding="utf-8")
    stub_python3.chmod(0o755)
    script = f'source "{LOCK / "common.sh"}" --root "{tmp_path}"; echo "$HARNESS_PY"'
    env = {"PATH": f"{stub_bin}:/usr/bin:/bin"}
    proc = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True,
                          check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(stub_python3)

    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/bash\n", encoding="utf-8")
    venv_python.chmod(0o755)
    proc = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True,
                          check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(venv_python)


def test_common_names_the_three_places_it_looked_when_no_interpreter_exists(tmp_path):
    (tmp_path / "benchmark").mkdir()
    script = f'source "{LOCK / "common.sh"}" --root "{tmp_path}"; echo "$HARNESS_PY"'
    proc = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True,
                          check=False, env={"PATH": str(tmp_path / "no-bin")})
    assert proc.returncode == 2
    assert "no harness interpreter: envs/harness, .venv or python3 on PATH" in proc.stderr


def _group_has_gid() -> bool:
    proc = subprocess.run(["dscl", ".", "-read", "/Groups/bench", "PrimaryGroupID"],
                          capture_output=True, text=True, check=False)
    return "PrimaryGroupID:" in proc.stdout


def test_uninstall_plan_undoes_every_class_and_keeps_the_record():
    _skip_unless_the_launchers_are_this_checkouts()
    proc = _run("uninstall")
    assert proc.returncode == 0, proc.stderr
    plan = proc.stdout
    assert "chflags nouchg benchmark/harness" in plan
    assert f"chown -h {OWNER}:" in plan
    assert "chmod 755 benchmark/harness" in plan
    assert f"chown -R {OWNER}:" in plan
    assert "benchmark/record" in plan
    assert "sysadminctl -deleteUser bench" in plan or not _bench_exists()
    assert "dscl . -delete /Groups/bench" in plan or not _bench_group_exists()
    # inv: the record is handed back, never removed; only what install created is removed
    removed = [line.split()[-1] for line in plan.splitlines() if line.startswith("rm ")]
    for path in removed:
        assert "attempts.jsonl" not in path
        assert "snapshots" not in path
        assert "reports" not in path
    assert ".git" not in [line.split()[-1] for line in plan.splitlines() if line.startswith(("chown", "chflags", "rm"))]


def _bench_exists() -> bool:
    return subprocess.run(["id", "bench"], capture_output=True, check=False).returncode == 0


def _bench_group_exists() -> bool:
    return subprocess.run(["dscl", ".", "-read", "/Groups/bench"], capture_output=True, check=False).returncode == 0


def test_uninstall_refuses_while_a_row_has_no_outcome(tmp_path):
    bench = tmp_path / "benchmark"
    (bench / "record").mkdir(parents=True)
    (bench / "record" / "attempts.jsonl").write_text('{"run_id": "r1"}\n', encoding="utf-8")
    proc = _run("uninstall", root=tmp_path)
    assert proc.returncode != 0
    assert "no outcome" in proc.stderr
