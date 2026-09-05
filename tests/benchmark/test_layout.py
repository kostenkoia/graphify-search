import argparse
import ast
import importlib
import pathlib
import re
import subprocess

from benchmark.harness import __main__ as cli

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmark"

ENTRIES = {"__init__.py", "PROTOCOL.md", "README.md", "INSTRUMENT.yaml", "harness", "systems",
           "record", "envs", "lock"}
# why: envs/ appears only after a machine is set up and INSTRUMENT.yaml only once the checkout is
# sealed, so a clean checkout carries neither
OPTIONAL = {"envs", "INSTRUMENT.yaml"}
LEFTOVER = {"__pycache__"}
# why: the seal is a tracked top-level file from the first seal on, and a clean checkout carries
# none, so the tracked set is bounded above by both names and below by the three that always exist
TRACKED_TOP = {"__init__.py", "PROTOCOL.md", "README.md", "INSTRUMENT.yaml"}
TRACKED_TOP_OPTIONAL = {"INSTRUMENT.yaml"}
# why: summary.json and SUMMARY.md are a function of the ledger, written by bench and
# committed by no verb, so the ledger is the one tracked file directly under record/
RECORD_TRACKED = {"attempts.jsonl"}
# why: a snapshot's freeze records, its questions and its references are what git keeps of a
# campaign, and they all sit under record/snapshots/
RECORD_TRACKED_DIRS = {"evidence", "reports", "snapshots"}
DOCUMENTS = ("PROTOCOL.md", "README.md")
# inv: the shell form the documents show -- an interpreter path, `-m benchmark.harness`, then the
# verb; the capture stops before an option so a flag is never read as a verb
INVOCATION = re.compile(r"-m benchmark\.harness ([a-z][a-z-]*)")
# inv: the verbs that touch the record, the instrument or a run; the three setup verbs -- expand,
# freeze-model and build-symbols -- take the file they operate on and are not among them
NO_PATH_VERBS = ("run", "attempt", "abort", "summary", "report", "seal", "audit", "questions")
# why: run_dir names the live sandbox run under audit, the object the verb inspects, not a root
# that would move the record, the instrument or the snapshot the verb reads
PATH_ARGUMENTS_ALLOWED = {"run_dir"}
# why: the journal backend replays recorded assistant turns, which is the agent backend the command
# line refuses to offer; it stays beside the local backend as the double the drive tests loop
# through, and no verb may reach it
TEST_DOUBLES = {"backends.journal"}


def _tracked(prefix: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", prefix],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def test_benchmark_root_holds_only_the_named_entries():
    present = {p.name for p in BENCH.iterdir()} - LEFTOVER
    assert present <= ENTRIES, present - ENTRIES
    assert present >= ENTRIES - OPTIONAL, (ENTRIES - OPTIONAL) - present


def test_nothing_about_runs_is_tracked_at_the_root():
    top_level_files = {t.split("/")[1] for t in _tracked("benchmark") if t.count("/") == 1}
    assert top_level_files <= TRACKED_TOP, top_level_files - TRACKED_TOP
    assert top_level_files >= TRACKED_TOP - TRACKED_TOP_OPTIONAL, \
        (TRACKED_TOP - TRACKED_TOP_OPTIONAL) - top_level_files


def test_every_verb_is_named_in_the_protocol():
    protocol = (BENCH / "PROTOCOL.md").read_text(encoding="utf-8")
    missing = {verb for verb in cli.VERBS if f"`{verb}`" not in protocol}
    assert missing == set(), missing


def test_every_command_the_documents_show_is_a_verb():
    invented = {(name, word)
                for name in DOCUMENTS
                for word in INVOCATION.findall((BENCH / name).read_text(encoding="utf-8"))
                if word not in cli.VERBS}
    assert invented == set(), invented


def test_record_tracks_only_the_ledger_snapshots_evidence_and_reports():
    for t in _tracked("benchmark/record"):
        parts = t.split("/")
        assert parts[2] in RECORD_TRACKED or parts[2] in RECORD_TRACKED_DIRS, t


def test_every_harness_module_is_imported_or_is_a_verb():
    modules = {p.relative_to(BENCH / "harness").with_suffix("").as_posix().replace("/", ".")
               for p in (BENCH / "harness").rglob("*.py")
               if p.name not in {"__init__.py", "__main__.py"}}
    imported: set[str] = set()
    # why: scoring/adapters/__init__.py resolves the adapter module by name at runtime via
    # importlib.import_module(f"benchmark.harness.scoring.adapters.{name}"), so no static
    # import references those modules; the f-string's literal prefix marks the whole package
    dynamic: set[str] = set()
    for p in (BENCH / "harness").rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("benchmark.harness"):
                base = node.module.removeprefix("benchmark.harness").lstrip(".")
                for alias in node.names:
                    imported.add(f"{base}.{alias.name}".strip("."))
                if base:
                    imported.add(base)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("benchmark.harness"):
                        base = alias.name.removeprefix("benchmark.harness").lstrip(".")
                        if base:
                            imported.add(base)
            elif (isinstance(node, ast.Call)
                  and ((isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                       or (isinstance(node.func, ast.Name) and node.func.id == "import_module"))
                  and node.args and isinstance(node.args[0], ast.JoinedStr)
                  and node.args[0].values
                  and isinstance(node.args[0].values[0], ast.Constant)
                  and isinstance(node.args[0].values[0].value, str)
                  and node.args[0].values[0].value.startswith("benchmark.harness.")):
                prefix = node.args[0].values[0].value.removeprefix("benchmark.harness.").removesuffix(".")
                dynamic.add(prefix)
    verbs = {fn.__module__.removeprefix("benchmark.harness.") for fn in cli.VERBS.values()}
    unreferenced = {m for m in modules - TEST_DOUBLES if m not in imported and m not in verbs
                    and not any(i.startswith(m + ".") for i in imported)
                    and not any(m.startswith(prefix + ".") for prefix in dynamic)}
    assert unreferenced == set(), unreferenced


def _all_actions(parser):
    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _all_actions(sub)


def _path_arguments(verb):
    parser = importlib.import_module(cli.VERBS[verb].__module__)._parser()
    return {action.option_strings[0] if action.option_strings else action.dest
            for action in _all_actions(parser) if action.type is pathlib.Path}


def test_no_verb_that_touches_the_record_takes_a_path():
    offenders = {verb: taken for verb in NO_PATH_VERBS
                 if (taken := _path_arguments(verb) - PATH_ARGUMENTS_ALLOWED)}
    assert offenders == {}, offenders
