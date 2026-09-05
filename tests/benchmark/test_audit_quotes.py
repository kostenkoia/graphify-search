import hashlib
from pathlib import Path

import pytest
import yaml

from benchmark.harness import audit, rules


def _system(tmp_path: Path, doc: str, quote: str, doc_hash: str | None = None) -> Path:
    sysdir = tmp_path / "systems" / "s"
    (sysdir / "docs").mkdir(parents=True)
    (sysdir / "docs" / "d.md").write_text(doc, encoding="utf-8")
    h = doc_hash or hashlib.sha256(doc.encode()).hexdigest()
    (sysdir / "harness.yaml").write_text(
        "adapter: graphify\nversion: {cli: x}\ninvocation: {}\nfixed_steps:\n"
        f"  - {{name: q, argv: [a], quote: {quote!r}}}\n"
        "default_configuration: d\nconfigurations: {}\nsandbox_layout: {}\nenvironment: {}\n"
        f"docs: {{d.md: {{sha256: {h!r}}}}}\n",
        encoding="utf-8",
    )
    (sysdir / "manifest.yaml").write_text("steps:\n  - quote: 'Build the   string'\n", encoding="utf-8")
    return tmp_path


def _write_system(
    root: Path,
    docs: dict,
    doc_files: dict[str, str],
    fixed_steps: list[dict],
    manifest: dict | None = None,
) -> Path:
    sysdir = root / "systems" / "s"
    docdir = sysdir / "docs"
    docdir.mkdir(parents=True, exist_ok=True)
    for name, content in doc_files.items():
        (docdir / name).write_text(content, encoding="utf-8")
    harness = {
        "adapter": "graphify",
        "version": {"cli": "x"},
        "invocation": {},
        "fixed_steps": fixed_steps,
        "default_configuration": "d",
        "configurations": {},
        "sandbox_layout": {},
        "environment": {},
        "docs": docs,
    }
    (sysdir / "harness.yaml").write_text(yaml.safe_dump(harness), encoding="utf-8")
    (sysdir / "manifest.yaml").write_text(yaml.safe_dump(manifest or {"steps": []}), encoding="utf-8")
    return root


def test_fold():
    assert audit.fold("a  b\n\tc ") == "a b c"


def test_quotes_pass_when_substrings(tmp_path: Path):
    bench = _system(tmp_path, "Run it.\nBuild the string by joining.\n", "Build the string")
    assert audit.check_quotes(bench, "s") == []


def test_quotes_fail_when_absent_or_doc_changed(tmp_path: Path):
    bench = _system(tmp_path, "Nothing here.\n", "Build the string")
    assert any("Build the string" in m for m in audit.check_quotes(bench, "s"))
    bench2 = _system(tmp_path / "b", "Build the string\n", "Build the string", doc_hash="0" * 64)
    assert any("sha256" in m for m in audit.check_quotes(bench2, "s"))


def test_empty_or_whitespace_only_quote_is_a_problem(tmp_path: Path):
    doc = "Build the string by joining.\n"
    doc_hash = hashlib.sha256(doc.encode()).hexdigest()
    bench = _write_system(
        tmp_path,
        docs={"d.md": {"sha256": doc_hash}},
        doc_files={"d.md": doc},
        fixed_steps=[{"name": "q", "argv": ["a"], "quote": ""}],
    )
    assert any("quote is empty" in m for m in audit.check_quotes(bench, "s"))

    bench2 = _write_system(
        tmp_path / "b",
        docs={"d.md": {"sha256": doc_hash}},
        doc_files={"d.md": doc},
        fixed_steps=[{"name": "q", "argv": ["a"], "quote": "   \n\t  "}],
    )
    assert any("quote is empty" in m for m in audit.check_quotes(bench2, "s"))


def test_doc_missing_is_a_problem(tmp_path: Path):
    bench = _write_system(
        tmp_path,
        docs={"missing.md": {"sha256": "0" * 64}},
        doc_files={},
        fixed_steps=[{"name": "q", "argv": ["a"], "quote": None}],
    )
    msgs = audit.check_quotes(bench, "s")
    assert any("doc missing" in m and "missing.md" in m for m in msgs)
    # inv: a null quote is a placeholder, not a malformed value, and must not itself be reported
    assert not any("not a string" in m for m in msgs)


def test_symlinked_doc_is_a_problem(tmp_path: Path):
    doc = "Build the string by joining.\n"
    doc_hash = hashlib.sha256(doc.encode()).hexdigest()
    bench = _write_system(
        tmp_path,
        docs={"d.md": {"sha256": doc_hash}},
        doc_files={},
        fixed_steps=[{"name": "q", "argv": ["a"], "quote": "Build the string"}],
    )
    real = tmp_path / "real.md"
    real.write_text(doc, encoding="utf-8")
    (bench / "systems" / "s" / "docs" / "d.md").symlink_to(real)
    msgs = audit.check_quotes(bench, "s")
    assert any("symlink" in m and "d.md" in m for m in msgs)


def test_doc_path_escaping_docs_directory_is_a_problem(tmp_path: Path):
    sysdir = tmp_path / "systems" / "s"
    sysdir.mkdir(parents=True)
    outside = sysdir / "outside.md"
    outside.write_text("Build the string by joining.\n", encoding="utf-8")
    outside_hash = hashlib.sha256(outside.read_bytes()).hexdigest()
    bench = _write_system(
        tmp_path,
        docs={"../outside.md": {"sha256": outside_hash}},
        doc_files={},
        fixed_steps=[{"name": "q", "argv": ["a"], "quote": "Build the string"}],
    )
    msgs = audit.check_quotes(bench, "s")
    assert any("escapes docs directory" in m for m in msgs)


def test_non_string_quote_is_a_problem(tmp_path: Path):
    bench = _write_system(
        tmp_path,
        docs={},
        doc_files={},
        fixed_steps=[{"name": "q", "argv": ["a"], "quote": ["False A", "False B"]}],
    )
    assert any("not a string" in m for m in audit.check_quotes(bench, "s"))


def test_main_exits_1_and_reports_to_stderr_when_a_quote_fails(
        tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    bench = _system(tmp_path, "Nothing here.\n", "Build the string")
    monkeypatch.setattr(rules, "_ROOT", bench)
    rc = audit.main(["quotes", "--system", "s"])
    assert rc == 1
    out, err = capsys.readouterr()
    assert "Build the string" in err
    assert out == ""


def test_main_exits_0_when_all_quotes_and_hashes_check_out(
        tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch):
    bench = _system(tmp_path, "Run it.\nBuild the string by joining.\n", "Build the string")
    monkeypatch.setattr(rules, "_ROOT", bench)
    rc = audit.main(["quotes", "--system", "s"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
