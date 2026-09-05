import json
from pathlib import Path

import pytest
import yaml

from benchmark.harness.scoring import (
    build_symbols,
    symbols_docs,
    symbols_python,
    symbols_shell,
    symbols_sql,
)


def _by_name(records: list[dict]) -> dict[str, dict]:
    return {r["fqname"].rsplit(".", 1)[-1]: r for r in records}


PY_SRC = '''"""Module."""


def top(a):
    return a


class Holder:
    """A class."""

    def method(self):
        def inner():
            return 1
        return inner


async def waited():
    return None
'''


def test_a_module_level_definition_is_a_function():
    found = _by_name(symbols_python.extract("pkg/mod.py", PY_SRC))
    assert found["top"]["kind"] == "function"


def test_a_definition_inside_a_class_is_a_method():
    found = _by_name(symbols_python.extract("pkg/mod.py", PY_SRC))
    assert found["method"]["kind"] == "method"


def test_a_class_is_its_own_place():
    found = _by_name(symbols_python.extract("pkg/mod.py", PY_SRC))
    assert found["Holder"]["kind"] == "class"


def test_a_nested_definition_is_addressable_too():
    # why: the walk is documented to continue past module level, because a question can point
    # at a nested definition
    found = _by_name(symbols_python.extract("pkg/mod.py", PY_SRC))
    assert found["inner"]["kind"] == "function"
    assert found["inner"]["fqname"] == "pkg.mod.Holder.method.inner"


def test_an_awaited_definition_is_a_function_like_any_other():
    assert _by_name(symbols_python.extract("pkg/mod.py", PY_SRC))["waited"]["kind"] == "function"


def test_a_place_spans_the_lines_it_really_occupies():
    top = _by_name(symbols_python.extract("pkg/mod.py", PY_SRC))["top"]
    assert (top["start"], top["end"]) == (4, 5)


def test_the_qualified_name_is_built_from_the_path():
    assert _by_name(symbols_python.extract("a/b/c.py", PY_SRC))["top"]["fqname"] == "a.b.c.top"


def test_a_file_that_does_not_parse_raises_rather_than_returning_nothing():
    with pytest.raises(SyntaxError):
        symbols_python.extract("pkg/broken.py", "def (:\n")


def test_a_documentation_file_is_one_place():
    recs = symbols_docs.extract("docs/guide.md", "one\ntwo\nthree\n")
    assert recs == [{"path": "docs/guide.md", "fqname": "docs/guide.md", "kind": "doc",
                     "start": 1, "end": 3}]


def test_an_empty_documentation_file_is_no_place_at_all():
    assert symbols_docs.extract("docs/empty.md", "") == []


@pytest.mark.parametrize(("rel", "roots", "expected"), [
    ("docs/guide.md", ["docs"], True),
    ("backend/app/main.py", ["docs"], False),
    ("docs/deep/nested.md", ["docs"], True),
])
def test_only_a_file_under_a_declared_root_counts_as_documentation(rel, roots, expected):
    assert symbols_docs.is_doc(rel, roots) is expected


SH_SRC = """#!/bin/sh
plain() {
  echo hi
}

function worded {
  echo hi
}

echo "not_a_definition() {"
"""


def test_both_spellings_a_shell_accepts_are_found():
    found = _by_name(symbols_shell.extract("scripts/run.sh", SH_SRC))
    assert set(found) == {"plain", "worded"}
    assert all(r["kind"] == "shell_function" for r in found.values())


SQL_SRC = """-- CREATE TABLE commented_out (id int);
CREATE TABLE users (id int);
CREATE OR REPLACE FUNCTION bump() RETURNS void AS $$ BEGIN END; $$ LANGUAGE plpgsql;
CREATE MATERIALIZED VIEW summary AS SELECT 1;
"""


def test_every_declared_schema_object_is_a_place():
    found = _by_name(symbols_sql.extract("db/schema.sql", SQL_SRC))
    assert {"users", "bump", "summary"} <= set(found)


def test_a_create_in_the_middle_of_a_line_is_not_a_statement():
    found = _by_name(symbols_sql.extract("db/s.sql", "SELECT 1; CREATE TABLE inline (id int);\n"))
    assert "inline" not in found


def test_a_create_that_is_not_a_statement_is_not_a_place():
    # why: the pattern is anchored at line start, so a CREATE inside a comment body is not one
    assert "commented_out" not in _by_name(symbols_sql.extract("db/schema.sql", SQL_SRC))


TS_SRC = """export function declared(a: number) {
  return a;
}

export const arrow = (b: number) => b + 1;

class Widget {
  render() {
    return null;
  }
}

const notAFunction = 42;
"""


@pytest.fixture
def typescript():
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_typescript")
    from benchmark.harness.scoring import symbols_typescript
    return symbols_typescript


def test_a_declared_function_is_a_place(typescript):
    assert _by_name(typescript.extract("web/api.ts", TS_SRC))["declared"]["kind"] == "function"


def test_a_const_bound_arrow_is_a_function_too(typescript):
    # why: a component is written as a declaration, a const-bound arrow or a default export,
    # and all three are places a question can point at
    assert _by_name(typescript.extract("web/api.ts", TS_SRC))["arrow"]["kind"] == "function"


def test_a_class_and_its_method_are_told_apart(typescript):
    found = _by_name(typescript.extract("web/api.ts", TS_SRC))
    assert found["Widget"]["kind"] == "class"
    assert found["render"]["kind"] == "method"


def test_a_const_holding_a_number_is_no_place(typescript):
    assert "notAFunction" not in _by_name(typescript.extract("web/api.ts", TS_SRC))


def test_a_method_is_named_under_the_class_that_holds_it(typescript):
    assert _by_name(typescript.extract("web/api.ts", TS_SRC))["render"]["fqname"].endswith(
        "Widget.render")


UNIVERSE = {"extensions": [".py", ".sql"], "doc_roots": ["docs"], "skip_dirs": ["node_modules"]}


def _tree(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    (src / "pkg").mkdir(parents=True)
    (src / "docs").mkdir()
    (src / "node_modules" / "lib").mkdir(parents=True)
    (src / "pkg" / "mod.py").write_text("def one():\n    return 1\n", encoding="utf-8")
    (src / "pkg" / "notes.txt").write_text("nothing here\n", encoding="utf-8")
    # why: this extension has an extractor, so only the snapshot's own declaration keeps it out
    (src / "pkg" / "undeclared.sh").write_text("helper() {\n  echo hi\n}\n", encoding="utf-8")
    (src / "docs" / "guide.md").write_text("a\nb\n", encoding="utf-8")
    (src / "node_modules" / "lib" / "vendored.py").write_text("def skipped():\n    pass\n",
                                                              encoding="utf-8")
    return src


def test_a_snapshot_that_declares_no_universe_is_refused(tmp_path: Path):
    (tmp_path / "meta.yaml").write_text(yaml.safe_dump({"tree": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="universe"):
        build_symbols.load_universe(tmp_path)


def test_the_universe_is_read_from_the_snapshot_that_declares_it(tmp_path: Path):
    (tmp_path / "meta.yaml").write_text(yaml.safe_dump({"universe": UNIVERSE}), encoding="utf-8")
    assert build_symbols.load_universe(tmp_path)["extensions"] == [".py", ".sql"]


def test_only_a_declared_extension_or_a_doc_root_is_indexed(tmp_path: Path):
    records, failed, _ = build_symbols.build(_tree(tmp_path), UNIVERSE)
    assert failed == []
    paths = {r["path"] for r in records}
    assert "pkg/mod.py" in paths
    assert "docs/guide.md" in paths
    assert "pkg/notes.txt" not in paths
    assert "pkg/undeclared.sh" not in paths


def test_a_directory_the_snapshot_skips_contributes_nothing(tmp_path: Path):
    records, _, _ = build_symbols.build(_tree(tmp_path), UNIVERSE)
    assert not any("node_modules" in r["path"] for r in records)


def test_a_file_that_cannot_be_parsed_is_reported_not_dropped(tmp_path: Path):
    src = _tree(tmp_path)
    (src / "pkg" / "broken.py").write_text("def (:\n", encoding="utf-8")
    records, failed, _ = build_symbols.build(src, UNIVERSE)
    # inv: a place that silently vanishes is a place no question can ever be scored against
    assert any("pkg/broken.py" in line for line in failed)
    assert not any(r["path"] == "pkg/broken.py" for r in records)


def test_each_place_is_hashed_over_its_own_lines(tmp_path: Path):
    import hashlib

    records, _, _ = build_symbols.build(_tree(tmp_path), UNIVERSE)
    one = next(r for r in records if r["fqname"].endswith(".one"))
    assert one["sha256"] == hashlib.sha256(b"def one():\n    return 1").hexdigest()


def test_the_ids_follow_the_sorted_order_and_do_not_move_between_runs(tmp_path: Path):
    src = _tree(tmp_path)
    first, _, _ = build_symbols.build(src, UNIVERSE)
    second, _, _ = build_symbols.build(src, UNIVERSE)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert [r["id"] for r in first] == [f"sym_{i:05d}" for i in range(len(first))]
    assert [(r["path"], r["start"]) for r in first] == sorted(
        (r["path"], r["start"]) for r in first)


UNTERMINATED = """CREATE TABLE first (id int)
CREATE TABLE second (id int)
"""


def test_a_statement_without_its_semicolon_stops_where_the_next_one_starts():
    # why: without this two objects would share a span, and a hit on one would read as a hit
    # on the other
    found = _by_name(symbols_sql.extract("db/s.sql", UNTERMINATED))
    assert (found["first"]["start"], found["first"]["end"]) == (1, 1)
    assert found["second"]["start"] == 2


def test_a_last_statement_without_its_semicolon_runs_to_the_end_of_the_file():
    found = _by_name(symbols_sql.extract("db/s.sql", "CREATE VIEW only AS SELECT 1\nmore\n"))
    assert found["only"]["end"] == 2


NESTED_SH = """outer() {
  if true; then
    echo hi
  }
}
after
"""


def test_a_shell_body_ends_at_the_brace_in_its_own_column():
    # why: a nested block is indented deeper, so its closing brace must not end the function
    found = _by_name(symbols_shell.extract("scripts/s.sh", NESTED_SH))
    assert found["outer"]["end"] == 5


def test_the_builder_writes_one_line_per_place_and_names_what_it_could_not_read(tmp_path, capsys):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    src = _tree(snapshot)
    (src / "pkg" / "broken.py").write_text("def (:\n", encoding="utf-8")
    (snapshot / "meta.yaml").write_text(yaml.safe_dump({"universe": UNIVERSE}), encoding="utf-8")
    build_symbols.main([str(snapshot)])
    written = [json.loads(x) for x in
               (snapshot / "symbols.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert {r["fqname"] for r in written} >= {"pkg.mod.one", "docs/guide.md"}
    captured = capsys.readouterr()
    assert f"{len(written)} symbols" in captured.out
    assert "pkg/broken.py" in captured.err
    assert "1 file(s) could not be parsed" in captured.err


def test_a_shell_function_whose_brace_never_closes_runs_to_the_end():
    found = _by_name(symbols_shell.extract("scripts/s.sh", "broken() {\n  echo hi\n"))
    assert found["broken"]["end"] == 2


def test_a_snapshot_the_builder_could_read_whole_reports_no_failure(tmp_path, capsys):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    _tree(snapshot)
    (snapshot / "meta.yaml").write_text(yaml.safe_dump({"universe": UNIVERSE}), encoding="utf-8")
    build_symbols.main([str(snapshot)])
    captured = capsys.readouterr()
    assert "symbols ->" in captured.out
    assert captured.err == ""


def test_the_builder_sorts_whatever_order_an_extractor_hands_it(tmp_path, monkeypatch):
    # why: the extractors written today happen to emit in line order, so the sort looks idle;
    # what it defends is the next extractor, which need not
    def jumbled(rel: str, text: str) -> list[dict]:
        del text
        return [{"path": rel, "fqname": f"{rel}.late", "kind": "function", "start": 9, "end": 9},
                {"path": rel, "fqname": f"{rel}.early", "kind": "function", "start": 1, "end": 1}]

    monkeypatch.setitem(build_symbols.EXTRACTORS, ".py", jumbled)
    src = tmp_path / "source"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "mod.py").write_text("\n" * 10, encoding="utf-8")
    records, _, _ = build_symbols.build(src, {"extensions": [".py"], "doc_roots": [],
                                                     "skip_dirs": []})
    assert [r["start"] for r in records] == [1, 9]
    assert [r["id"] for r in records] == ["sym_00000", "sym_00001"]


def test_an_extension_the_snapshot_declares_with_no_extractor_is_returned(tmp_path, capsys):
    src = _tree(tmp_path)
    (src / "pkg" / "server.go").write_text("func main() {}\n", encoding="utf-8")
    records, failed, holes = build_symbols.build(src, {**UNIVERSE,
                                                       "extensions": [".py", ".sql", ".go"]})
    assert holes == ["no extractor for .go: pkg/server.go"]
    assert failed == []
    assert not any(r["path"] == "pkg/server.go" for r in records)
    # inv: build() reports through its return value alone, so a caller that wants the holes
    # elsewhere is not fighting a write to the error stream
    assert capsys.readouterr().err == ""


def test_a_hole_the_universe_declares_is_printed_once_per_file(tmp_path, capsys):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    src = _tree(snapshot)
    (src / "pkg" / "server.go").write_text("func main() {}\n", encoding="utf-8")
    (src / "pkg" / "other.go").write_text("func other() {}\n", encoding="utf-8")
    (snapshot / "meta.yaml").write_text(
        yaml.safe_dump({"universe": {**UNIVERSE, "extensions": [".py", ".sql", ".go"]}}),
        encoding="utf-8")
    build_symbols.main([str(snapshot)])
    err = capsys.readouterr().err
    assert "no extractor for .go: pkg/server.go" in err
    assert "no extractor for .go: pkg/other.go" in err
    assert err.count("pkg/server.go") == 1
    assert "2 file(s) have no extractor" in err


def test_an_extension_the_snapshot_never_declared_is_not_named(tmp_path):
    # inv: only a declared extension is part of the universe, so an undeclared suffix is
    # ordinary silence rather than a gap the reader has to know about
    _, _, holes = build_symbols.build(_tree(tmp_path), UNIVERSE)
    assert holes == []


def test_build_symbols_writes_a_hash_beside_the_symbols(tmp_path):
    from benchmark.harness import rules
    from benchmark.harness.scoring import build_symbols

    snap = tmp_path / "snap"
    (snap / "source").mkdir(parents=True)
    (snap / "source" / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (snap / "meta.yaml").write_text(
        "universe:\n  extensions: ['.py']\n  doc_roots: []\n  skip_dirs: []\n", encoding="utf-8")
    assert build_symbols.main([str(snap)]) == 0
    digest = (snap / "symbols.sha256").read_text(encoding="utf-8")
    assert digest == f"{rules.sha256_file(snap / 'symbols.jsonl')}  symbols.jsonl\n"


def test_build_symbols_hash_only_does_not_rebuild(tmp_path):
    from benchmark.harness.scoring import build_symbols

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "symbols.jsonl").write_text('{"id": "sym_00000"}\n', encoding="utf-8")
    assert build_symbols.main(["--hash-only", str(snap)]) == 0
    assert (snap / "symbols.sha256").read_text(encoding="utf-8").endswith("  symbols.jsonl\n")
    assert (snap / "symbols.jsonl").read_text(encoding="utf-8") == '{"id": "sym_00000"}\n'
