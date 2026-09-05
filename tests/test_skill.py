import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from graphify_search import cli, skill
from graphify_search.errors import InputError

REPO = Path(__file__).resolve().parent.parent
FOREIGN = "a file that is not this skill\n"


def _target(root: Path, *, global_install: bool = False) -> Path:
    return skill.skill_dir(root, global_install=global_install) / "SKILL.md"


def test_install_writes_the_packaged_file_and_prints_its_record(tmp_path, capsys):
    assert cli.main(["install", "--root", str(tmp_path)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record == {"scope": "project", "path": str(_target(tmp_path)), "written": True}
    assert _target(tmp_path).read_text() == skill.packaged_skill_md().read_text()


def test_install_global_writes_under_the_home_directory(tmp_path, capsys, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    assert cli.main(["install", "--root", str(tmp_path), "--global"]) == 0
    assert json.loads(capsys.readouterr().out)["scope"] == "global"
    assert _target(tmp_path, global_install=True).is_file()
    # inv: the global scope is the home directory, so the project tree gains nothing
    assert not (tmp_path / ".claude").exists()


def test_install_defaults_the_root_to_the_working_directory(tmp_path, capsys, monkeypatch):
    # inv: the parser names no directory, so the default is the directory the command runs in
    monkeypatch.chdir(tmp_path)
    assert cli.main(["install"]) == 0
    assert json.loads(capsys.readouterr().out)["path"] == str(_target(tmp_path))


def test_install_refuses_foreign_content_and_says_so_on_stderr(tmp_path, capsys):
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(FOREIGN)
    assert cli.main(["install", "--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    # inv: a refusal leaves stdout empty, so a caller parsing this command's JSON never reads
    # an error message as a record
    assert captured.out == ""
    assert "holds different content" in captured.err
    assert "force" in captured.err
    assert target.read_text() == FOREIGN


def test_install_force_overwrites_foreign_content(tmp_path, capsys):
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text(FOREIGN)
    assert cli.main(["install", "--root", str(tmp_path), "--force"]) == 0
    capsys.readouterr()
    assert target.read_text() == skill.packaged_skill_md().read_text()


def test_installing_twice_is_not_refused_as_foreign(tmp_path, capsys):
    # inv: the marker install keys on is in the file it writes, so its own output is never foreign
    assert cli.main(["install", "--root", str(tmp_path)]) == 0
    assert cli.main(["install", "--root", str(tmp_path)]) == 0
    capsys.readouterr()


def test_uninstall_removes_the_tree_then_reports_nothing_to_remove(tmp_path, capsys):
    cli.main(["install", "--root", str(tmp_path)])
    capsys.readouterr()
    assert cli.main(["uninstall", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["removed"] is True
    assert not skill.skill_dir(tmp_path).exists()
    assert cli.main(["uninstall", "--root", str(tmp_path)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert (second["removed"], second["reason"]) == (False, "not installed")


def test_detect_follows_the_install_state(tmp_path, capsys, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    assert cli.main(["detect", "--root", str(tmp_path)]) == 0
    before = json.loads(capsys.readouterr().out)
    assert (before["installed_project"], before["installed_global"]) == (False, False)
    cli.main(["install", "--root", str(tmp_path)])
    capsys.readouterr()
    cli.main(["detect", "--root", str(tmp_path)])
    after = json.loads(capsys.readouterr().out)
    assert (after["installed_project"], after["installed_global"]) == (True, False)


def test_a_missing_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


def test_install_refuses_to_write_through_a_symlinked_file(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text(FOREIGN)
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)
    with pytest.raises(InputError, match="symlink"):
        skill.install(tmp_path)
    # inv: the write is refused, not followed, so the file the link pointed at is untouched
    assert outside.read_text() == FOREIGN


def test_install_refuses_to_write_through_a_symlinked_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target_dir = skill.skill_dir(tmp_path)
    target_dir.parent.mkdir(parents=True)
    target_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(InputError, match="symlink"):
        skill.install(tmp_path)
    assert list(outside.iterdir()) == []


def test_uninstall_refuses_to_follow_a_symlinked_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("not ours\n")
    target_dir = skill.skill_dir(tmp_path)
    target_dir.parent.mkdir(parents=True)
    target_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(InputError, match="symlink"):
        skill.uninstall(tmp_path)
    assert (outside / "keep.txt").is_file()


def test_packaged_skill_md_is_a_real_file_the_install_check_recognises():
    packaged = skill.packaged_skill_md()
    assert packaged.is_file()
    # inv: install reads the target's frontmatter to decide whether it is ours, so the file it
    # writes must satisfy that same test or a second install refuses the first one's output
    assert skill._is_this_skill(packaged.read_text())


def test_the_skill_is_recognised_by_its_frontmatter_name_not_by_prose():
    # inv: a file that merely mentions this package is not ours, and overwriting it without
    # force would destroy content the refusal exists to protect
    prose = ("# My notes\nThe package uses name: graphify-search as its marker.\n"
             "Do not delete.\n")
    assert not skill._is_this_skill(prose)
    # inv: the name is read as a frontmatter key, so ordinary YAML quoting is still ours
    for name_line in ('name: graphify-search', 'name: "graphify-search"', "name: 'graphify-search'",
                      "name:   graphify-search  "):
        assert skill._is_this_skill(f"---\n{name_line}\ndescription: x\n---\nbody\n"), name_line
    # inv: another skill's file is not ours, however close its name
    for foreign in ("---\nname: graphify-search-2\n---\n", "---\nname: other\n---\n",
                    "---\nnot_name: graphify-search\n---\n", "no frontmatter at all\n"):
        assert not skill._is_this_skill(foreign), foreign


def test_install_refuses_a_symlink_anywhere_between_the_root_and_the_target(tmp_path):
    # inv: only the leaf was checked once, so a symlinked `.claude` redirected the whole write
    # out of the scope the caller named while the record still said "project"
    for link_at in (".claude", ".claude/skills"):
        project, outside = tmp_path / link_at.replace("/", "_"), tmp_path / f"out_{link_at.count('/')}"
        outside.mkdir(parents=True)
        link = project / link_at
        link.parent.mkdir(parents=True)
        link.symlink_to(outside, target_is_directory=True)
        with pytest.raises(InputError, match="symlink"):
            skill.install(project)
        assert list(outside.rglob("*")) == [], link_at


def test_uninstall_refuses_a_symlink_anywhere_between_the_root_and_the_target(tmp_path):
    # inv: the removal walks the same chain, or an `uninstall` scoped to a project deletes a
    # directory that lives outside it
    project, precious = tmp_path / "project", tmp_path / "precious"
    (precious / "skills" / skill.SKILL_NAME).mkdir(parents=True)
    keep = precious / "skills" / skill.SKILL_NAME / "important.txt"
    keep.write_text("irreplaceable\n")
    project.mkdir()
    (project / ".claude").symlink_to(precious, target_is_directory=True)
    with pytest.raises(InputError, match="symlink"):
        skill.uninstall(project)
    assert keep.read_text() == "irreplaceable\n"


def test_module_entry_point_runs_the_same_command():
    # inv: `python -m graphify_search` is the path that survives a bin directory missing from PATH
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run([sys.executable, "-m", "graphify_search", "detect", "--root", str(REPO)],
                          capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert set(json.loads(proc.stdout)) == {
        "cli_installed", "config_dir_exists", "installed_project", "installed_global"}


def test_the_declared_console_script_points_at_something_that_exists():
    # inv: an entry point naming a module that was deleted installs a command that fails on every
    # run, and nothing but this check notices until someone runs it
    # inv: read as text, not with tomllib, which this package's own floor of Python 3.10 lacks
    block = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[|\Z)", (REPO / "pyproject.toml").read_text(),
                      re.MULTILINE | re.DOTALL)
    assert block, "pyproject declares no [project.scripts]"
    declared = dict(re.findall(r'^(\S+)\s*=\s*"([^"]+)"', block.group(1), re.MULTILINE))
    assert declared == {"graphify-search": "graphify_search.cli:main"}
    module_name, _, attribute = declared["graphify-search"].partition(":")
    target = getattr(importlib.import_module(module_name), attribute)
    assert callable(target)
