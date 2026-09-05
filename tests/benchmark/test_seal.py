import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from benchmark.harness import rules, seal


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "-C", str(root), *args],
                          check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize(("rel", "expected"), [
    ("benchmark/record", "record"),
    ("benchmark/record/attempts.jsonl", "record"),
    ("benchmark/record/runs/x/y.out", "record"),
    ("benchmark/record/snapshots", "instrument"),
    ("benchmark/record/snapshots/s1", "instrument"),
    ("benchmark/record/snapshots/s1/questions/q001.yaml", "instrument"),
    ("benchmark/record/snapshots/s1/questions/review/q001.yaml", "instrument"),
    ("benchmark/record/snapshots/s1/questions/candidates/x.yaml", "instrument"),
    ("benchmark/record/snapshots/s1/references/q001.yaml", "instrument"),
    ("benchmark/record/snapshots/s1/meta.yaml", "instrument"),
    ("benchmark/record/snapshots/s1/fileset.sha256", "instrument"),
    ("benchmark/record/snapshots/s1/symbols.sha256", "instrument"),
    ("benchmark/record/snapshots/s1/indexes", "instrument"),
    ("benchmark/record/snapshots/s1/indexes/i1/build.yaml", "instrument"),
    ("benchmark/record/snapshots/s1/indexes/i1/prepared_outputs.yaml", "record"),
    ("benchmark/record/snapshots/s1/indexes/i1", "sut"),
    ("benchmark/record/snapshots/s1/indexes/i1/graph.db", "sut"),
    ("benchmark/record/snapshots/s1/source/a.py", "sut"),
    ("benchmark/record/snapshots/s1/symbols.jsonl", "sut"),
    ("benchmark/record/snapshots/known_transitions.yaml", "instrument"),
])
def test_classify_matches_the_record_layout_table(rel, expected):
    assert seal.classify(rel) == expected


class _StatWithoutFlags:
    st_mode = 0o100644
    st_uid = 0


def test_mode_flags_reports_no_flags_where_the_platform_carries_none(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "stat", lambda self, **kwargs: _StatWithoutFlags())
    assert seal._mode_flags(tmp_path) == (0o644, 0)


def _tree(tmp_path: Path) -> Path:
    """A committed repo with one file of every class the seal distinguishes."""
    root = tmp_path
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    b = root / "benchmark"
    for rel in ("harness/x.py", "systems/s/harness.yaml",
                "record/snapshots/snap/questions/q001.yaml", "record/snapshots/snap/questions/review/q001.yaml",
                "record/snapshots/snap/questions/candidates/x.yaml", "record/snapshots/snap/references/q001.yaml",
                "record/snapshots/snap/meta.yaml", "record/snapshots/snap/fileset.sha256",
                "record/snapshots/snap/symbols.sha256", "record/snapshots/snap/indexes/i/build.yaml",
                "record/snapshots/known_transitions.yaml", "PROTOCOL.md", "README.md",
                "lock/install", "record/attempts.jsonl",
                "record/snapshots/snap/indexes/i/prepared_outputs.yaml"):
        p = b / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel + "\n", encoding="utf-8")
    (root / "tests" / "benchmark").mkdir(parents=True)
    (root / "tests" / "benchmark" / "test_x.py").write_text("x\n", encoding="utf-8")
    (root / "tests" / "test_comment_convention.py").write_text("x\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
    (root / ".gitignore").write_text("benchmark/envs/\nbenchmark/record/runs/\n*.ignored\n", encoding="utf-8")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("x\n", encoding="utf-8")
    (b / "systems" / "s" / "harness.yaml").write_text(yaml.safe_dump({
        "adapter": "a", "version": {"cli": "1"}, "fixed_steps": [], "default_configuration": "c",
        "configurations": {"c": {"index": "indexes/i"}}, "sandbox_layout": {}, "environment": {}, "docs": {},
        "invocation": {"package": {"launcher": str(b / "envs/s/bin/s"), "interpreter": str(b / "envs/s/bin/python"),
                                   "site": str(b / "envs/s/lib/site-packages")}}}), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "tree")
    (b / "envs" / "s" / "bin").mkdir(parents=True)          # ignored, system under test
    (b / "record" / "runs").mkdir(parents=True)             # ignored, record
    return b


def test_instrument_paths_are_the_locked_list_and_nothing_else(tmp_path):
    b = _tree(tmp_path)
    got = {p.as_posix() for p in seal.paths(b, "instrument")}
    assert "benchmark/harness/x.py" in got
    assert "benchmark/systems/s/harness.yaml" in got
    assert "benchmark/record/snapshots/snap/questions/q001.yaml" in got
    assert "benchmark/record/snapshots/snap/questions/review/q001.yaml" in got
    assert "benchmark/record/snapshots/snap/questions/candidates/x.yaml" in got
    assert "benchmark/record/snapshots/snap/references/q001.yaml" in got
    assert "benchmark/record/snapshots/snap/indexes/i/build.yaml" in got
    assert "benchmark/record/snapshots/known_transitions.yaml" in got
    assert "tests/benchmark/test_x.py" in got
    assert "pyproject.toml" in got
    assert ".gitignore" in got
    assert ".github/workflows/ci.yml" in got
    assert "benchmark/record/attempts.jsonl" not in got
    assert "benchmark/record/snapshots/snap/indexes/i/prepared_outputs.yaml" not in got
    assert "benchmark/lock/install" not in got            # the lock is its own class, never unlocked
    assert "benchmark/envs/s/bin" not in got
    # the index directory is bench-owned; build.yaml inside it is not
    assert "benchmark/record/snapshots/snap/indexes/i" not in got
    assert seal.classify("benchmark/systems/s/models/m/w.bin") == "sut"
    assert seal.classify("benchmark/record/snapshots/snap/indexes/i/build.yaml") == "instrument"


def test_git_lines_pins_excludesfile_against_a_global_gitignore(tmp_path, monkeypatch):
    # a global core.excludesFile that ignores the same file lock/require_sealed/seal --check
    # scan for would let root, bench and kia disagree on the candidate set; the pin makes every
    # caller compute the same set regardless of a global gitignore on the machine that runs it
    b = _tree(tmp_path)
    global_ignore = tmp_path / "global-ignore-for-this-test"
    global_ignore.write_text("*.envtest\n", encoding="utf-8")
    global_config = tmp_path / "global-gitconfig-for-this-test"
    global_config.write_text(f"[core]\n\texcludesFile = {global_ignore}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    (b / "harness" / "new.envtest").write_text("x\n", encoding="utf-8")
    got = {p.as_posix() for p in seal.paths(b, "instrument")}
    assert "benchmark/harness/new.envtest" in got


def test_an_untracked_unignored_file_under_a_locked_glob_counts(tmp_path):
    b = _tree(tmp_path)
    (b / "harness" / "new.py").write_text("x\n", encoding="utf-8")
    (b / "harness" / "scratch.ignored").write_text("x\n", encoding="utf-8")
    got = {p.as_posix() for p in seal.paths(b, "instrument")}
    assert "benchmark/harness/new.py" in got
    assert "benchmark/harness/scratch.ignored" not in got


def test_record_and_sut_paths(tmp_path):
    b = _tree(tmp_path)
    assert {p.as_posix() for p in seal.paths(b, "record")} >= {
        "benchmark/record", "benchmark/record/attempts.jsonl",
        "benchmark/record/snapshots/snap/indexes/i/prepared_outputs.yaml"}
    assert "benchmark/envs" in {p.as_posix() for p in seal.paths(b, "sut")}


def test_classify_on_further_sut_shapes():
    """classify() on a models/source directory itself, and a file under source/."""
    assert seal.classify("benchmark/systems/s/models") == "sut"
    assert seal.classify("benchmark/record/snapshots/snap/source/a.py") == "sut"


def test_sut_paths_include_models_and_source_and_exclude_instrument_and_record(tmp_path):
    b = _tree(tmp_path)
    (b / "systems" / "s" / "models" / "m").mkdir(parents=True)
    (b / "systems" / "s" / "models" / "m" / "w.bin").write_text("w\n", encoding="utf-8")
    (b / "record" / "snapshots" / "snap" / "source").mkdir(parents=True)
    (b / "record" / "snapshots" / "snap" / "source" / "a.py").write_text("a\n", encoding="utf-8")
    got = {p.as_posix() for p in seal.paths(b, "sut")}
    assert "benchmark/systems/s/models" in got
    assert "benchmark/systems/s/models/m" in got
    assert "benchmark/record/snapshots/snap/source" in got
    assert "benchmark/record/snapshots/snap/source/a.py" in got
    # the models/source directories are sut, but build.yaml and prepared_outputs.yaml, both
    # reachable from the sut globs' iterdir step, are instrument and record respectively
    assert not any(p.endswith("build.yaml") for p in got)
    assert not any(p.endswith("prepared_outputs.yaml") for p in got)


def test_lock_paths_include_the_lock_files(tmp_path):
    b = _tree(tmp_path)
    got = {p.as_posix() for p in seal.paths(b, "lock")}
    assert "benchmark/lock" in got
    assert "benchmark/lock/install" in got


def test_launchers_skip_a_reference_system(tmp_path):
    b = _tree(tmp_path)
    got = seal.launchers(b)
    assert any(p.name == "s" for p in got)
    assert len(got) == 3
    h = yaml.safe_load((b / "systems/s/harness.yaml").read_text(encoding="utf-8"))
    (b / "systems/s/harness.yaml").write_text(yaml.safe_dump({**h, "status": "reference"}), encoding="utf-8")
    assert seal.launchers(b) == []


def test_compute_check_roundtrip_and_every_kind_of_difference(tmp_path, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_euid", lambda: 0)
    (b / "lock").mkdir(exist_ok=True)
    (b / "lock" / "UNLOCKED").write_text("reason: fix the prompt\n", encoding="utf-8")
    path = seal.write(b)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["reason"] == "fix the prompt"
    assert doc["sealed_at_commit"] == _git(b.parent, "rev-parse", "HEAD").strip()
    assert doc["files"]["benchmark/harness/x.py"] == rules.sha256_file(b / "harness/x.py")
    assert "benchmark/INSTRUMENT.yaml" not in doc["files"]
    # no machine.yaml in this tree
    assert doc["interpreter"] is None
    assert doc["machine_sha256"] is None
    assert seal.check(b) == []

    (b / "harness" / "x.py").write_text("changed\n", encoding="utf-8")
    assert seal.check(b) == ["instrument differs from its seal: benchmark/harness/x.py"]
    (b / "harness" / "x.py").write_text("harness/x.py\n", encoding="utf-8")
    (b / "harness" / "y.py").write_text("x\n", encoding="utf-8")
    assert seal.check(b) == ["instrument differs from its seal: benchmark/harness/y.py (unlisted)"]
    (b / "harness" / "y.py").unlink()
    (b / "record" / "snapshots" / "snap" / "questions" / "q001.yaml").unlink()
    assert seal.check(b) == [
        "instrument differs from its seal: benchmark/record/snapshots/snap/questions/q001.yaml (missing)"]


def test_check_flags_a_changed_harness_environment_when_machine_facts_exist(tmp_path, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_euid", lambda: 0)
    (b / "lock").mkdir(exist_ok=True)
    (b / "lock" / "UNLOCKED").write_text("reason: fix the prompt\n", encoding="utf-8")
    (b / "lock" / "machine.yaml").write_text("bench_uid: 0\n", encoding="utf-8")
    (b / "envs" / "harness" / "bin").mkdir(parents=True)
    (b / "envs" / "harness" / "bin" / "python").write_text("py\n", encoding="utf-8")
    monkeypatch.setattr(seal, "_owner", lambda p: 0)
    monkeypatch.setattr(seal, "_mode_flags", lambda p: (0o444, stat.UF_IMMUTABLE))
    seal.write(b)
    (b / "lock" / "UNLOCKED").unlink()
    assert seal.check(b) == []
    (b / "envs" / "harness" / "bin" / "python").write_text("changed\n", encoding="utf-8")
    assert seal.check(b) == ["harness environment differs from its seal"]


def _machine_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bench_uid: int = 0) -> Path:
    """A sealed tree with `lock/machine.yaml`, ready for `seal.check`'s ownership pass."""
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_euid", lambda: 0)
    (b / "lock").mkdir(exist_ok=True)
    (b / "lock" / "UNLOCKED").write_text("reason: fix the prompt\n", encoding="utf-8")
    (b / "lock" / "machine.yaml").write_text(yaml.safe_dump({"bench_uid": bench_uid}), encoding="utf-8")
    seal.write(b)
    # why: seal.write requires UNLOCKED present, but the ownership pass this fixture feeds
    # runs only once the tree is locked again
    (b / "lock" / "UNLOCKED").unlink()
    return b


def _clean_owner(b: Path, bench_uid: int):
    def fn(p: Path) -> int:
        rel = Path(p).relative_to(b.parent).as_posix()
        return 0 if seal.classify(rel) in ("instrument", "lock") else bench_uid
    return fn


_CLEAN_MODE_FLAGS = (0o444, stat.UF_IMMUTABLE)


def test_check_ownership_all_clean_when_machine_facts_exist(tmp_path, monkeypatch):
    b = _machine_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(seal, "_owner", _clean_owner(b, 0))
    monkeypatch.setattr(seal, "_mode_flags", lambda p: _CLEAN_MODE_FLAGS)
    assert seal.check(b) == []


def test_check_flags_instrument_not_root_owned(tmp_path, monkeypatch):
    b = _machine_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(seal, "_owner", lambda p: 501)
    monkeypatch.setattr(seal, "_mode_flags", lambda p: _CLEAN_MODE_FLAGS)
    problems = seal.check(b)
    assert any(line.startswith("instrument not root-owned: ") for line in problems)


def test_check_flags_instrument_writable(tmp_path, monkeypatch):
    b = _machine_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(seal, "_owner", _clean_owner(b, 0))

    def mode_flags(p: Path) -> tuple[int, int]:
        rel = Path(p).relative_to(b.parent).as_posix()
        return (0o644, stat.UF_IMMUTABLE) if seal.classify(rel) in ("instrument", "lock") else _CLEAN_MODE_FLAGS

    monkeypatch.setattr(seal, "_mode_flags", mode_flags)
    problems = seal.check(b)
    assert any(line.startswith("instrument writable: ") for line in problems)
    assert not any(line.startswith("record") for line in problems)


def test_check_flags_instrument_not_immutable(tmp_path, monkeypatch):
    b = _machine_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(seal, "_owner", _clean_owner(b, 0))

    def mode_flags(p: Path) -> tuple[int, int]:
        rel = Path(p).relative_to(b.parent).as_posix()
        return (0o444, 0) if seal.classify(rel) in ("instrument", "lock") else _CLEAN_MODE_FLAGS

    monkeypatch.setattr(seal, "_mode_flags", mode_flags)
    problems = seal.check(b)
    assert any(line.startswith("instrument not immutable: ") for line in problems)


def test_check_flags_record_not_owned_by_bench(tmp_path, monkeypatch):
    b = _machine_tree(tmp_path, monkeypatch, bench_uid=999)
    # owner "0" satisfies instrument+lock (root-owned) but mismatches the required bench_uid 999
    monkeypatch.setattr(seal, "_owner", _clean_owner(b, 0))
    monkeypatch.setattr(seal, "_mode_flags", lambda p: _CLEAN_MODE_FLAGS)
    problems = seal.check(b)
    assert any(line.startswith("record not owned by bench: ") for line in problems)


def test_check_flags_record_writable_by_others(tmp_path, monkeypatch):
    b = _machine_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(seal, "_owner", _clean_owner(b, 0))

    def mode_flags(p: Path) -> tuple[int, int]:
        rel = Path(p).relative_to(b.parent).as_posix()
        return _CLEAN_MODE_FLAGS if seal.classify(rel) in ("instrument", "lock") else (0o646, stat.UF_IMMUTABLE)

    monkeypatch.setattr(seal, "_mode_flags", mode_flags)
    problems = seal.check(b)
    assert any(line.startswith("record writable by others: ") for line in problems)


def test_write_as_root_refuses_without_unlocked(tmp_path, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_euid", lambda: 0)
    with pytest.raises(SystemExit, match="no UNLOCKED"):
        seal.write(b)


def test_first_seal_of_a_checkout_is_written_without_root(tmp_path, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_euid", lambda: 501)
    path = seal.write(b)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["reason"] == seal.FIRST_SEAL_REASON
    assert doc["files"]["benchmark/harness/x.py"] == rules.sha256_file(b / "harness/x.py")
    assert doc["files"]["benchmark/record/snapshots/snap/questions/q001.yaml"] == \
        rules.sha256_file(b / "record/snapshots/snap/questions/q001.yaml")
    assert "benchmark/INSTRUMENT.yaml" not in doc["files"]
    assert doc["interpreter"] is None
    assert doc["machine_sha256"] is None
    assert seal.check(b) == []


def test_a_sealed_checkout_is_never_sealed_again_without_root(tmp_path, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_euid", lambda: 501)
    seal.write(b)
    with pytest.raises(SystemExit, match="seal is written by lock, as root"):
        seal.write(b)


def test_a_tree_with_machine_facts_is_never_sealed_without_root(tmp_path, monkeypatch):
    b = _tree(tmp_path)
    (b / "lock").mkdir(exist_ok=True)
    (b / "lock" / "machine.yaml").write_text("bench_uid: 0\n", encoding="utf-8")
    monkeypatch.setattr(seal, "_euid", lambda: 501)
    with pytest.raises(SystemExit, match="seal is written by lock, as root"):
        seal.write(b)


def test_seal_verb_writes_the_first_seal_of_a_checkout(tmp_path, capsys, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_ROOT", b)
    monkeypatch.setattr(seal, "_euid", lambda: 501)
    assert seal.main([]) == 0
    assert str(b / seal.SEAL) in capsys.readouterr().err
    assert seal.check(b) == []


def test_check_without_a_seal_names_it(tmp_path):
    b = _tree(tmp_path)
    assert seal.check(b) == ["instrument is not sealed"]


def test_seal_verb_prints_paths_and_checks(tmp_path, capsys, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_ROOT", b)
    assert seal.main(["--paths", "instrument"]) == 0
    assert "benchmark/harness/x.py" in capsys.readouterr().out.splitlines()
    assert seal.main(["--check"]) == 1
    assert "instrument is not sealed" in capsys.readouterr().out


def test_main_launchers_prints_the_fixture_paths(tmp_path, capsys, monkeypatch):
    b = _tree(tmp_path)
    monkeypatch.setattr(seal, "_ROOT", b)
    assert seal.main(["--launchers"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert set(out) == {str(b / "envs/s/bin/s"), str(b / "envs/s/bin/python"), str(b / "envs/s/lib/site-packages")}


def test_the_interpreter_listing_ignores_bytecode_written_on_import(tmp_path: Path):
    env = tmp_path / "env"
    (env / "lib").mkdir(parents=True)
    (env / "lib" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    before = seal._listing_sha256(env)
    (env / "lib" / "__pycache__").mkdir()
    (env / "lib" / "__pycache__" / "mod.cpython-314.pyc").write_bytes(b"\x00bytecode")
    assert seal._listing_sha256(env) == before
    (env / "lib" / "other.py").write_text("y = 2\n", encoding="utf-8")
    assert seal._listing_sha256(env) != before
