import base64
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import yaml

from benchmark.harness import rules
from tests.benchmark.conftest import _reseal, freeze_source, snapshot_dir


def test_canonical_hash_drops_volatile_keys():
    a = json.dumps({"status": "ok", "_graph": {"age_seconds": 1, "updated_at": "x"}, "results": [1]})
    b = json.dumps({"results": [1], "_graph": {"updated_at": "y", "age_seconds": 2}, "status": "ok"})
    vol = ["_graph.age_seconds", "_graph.updated_at"]
    assert rules.canonical_hash(a, vol) == rules.canonical_hash(b, vol)
    assert rules.canonical_hash(a, []) != rules.canonical_hash(b, [])


def test_canonical_hash_of_plain_text_is_sha256_of_bytes():
    text = "graphify 0.9.27\n"
    assert rules.canonical_hash(text, []) == hashlib.sha256(text.encode()).hexdigest()


def test_listing_and_changed_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b.txt").write_text("2")
    assert set(rules.listing(tmp_path)) == {"a.txt", "d/b.txt"}
    roots = {"root": tmp_path}
    before = rules.labelled_listing(roots)
    assert set(before) == {"root/a.txt", "root/d/b.txt"}
    time.sleep(0.01)
    (tmp_path / "a.txt").write_text("11")
    (tmp_path / "c.txt").write_text("3")
    (tmp_path / "d" / "b.txt").unlink()
    after = rules.labelled_listing(roots)
    by_path = {c["path"]: c["sha256"] for c in rules.changed_files(before, after, roots)}
    assert by_path["root/a.txt"] == hashlib.sha256(b"11").hexdigest()
    assert by_path["root/c.txt"] == hashlib.sha256(b"3").hexdigest()
    assert by_path["root/d/b.txt"] is None


def test_listing_does_not_follow_symlinks(tmp_path: Path):
    target = tmp_path / "blob"
    target.write_text("x")
    (tmp_path / "link").symlink_to(target)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "hidden.txt").write_text("y")
    (tmp_path / "dirlink").symlink_to(real_dir)
    lst = rules.listing(tmp_path)
    assert "link" in lst
    assert "blob" in lst
    assert "dirlink" in lst
    assert "real/hidden.txt" in lst
    assert not any(rel.startswith("dirlink/") for rel in lst)
    assert rules.sha256_file(tmp_path / "link") == hashlib.sha256(str(target).encode()).hexdigest()
    assert rules.sha256_file(tmp_path / "dirlink") == hashlib.sha256(str(real_dir).encode()).hexdigest()


def test_listing_raises_for_missing_root(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        rules.listing(tmp_path / "nope")


def test_changed_files_hashes_symlink_target_string_not_contents(tmp_path: Path):
    target = tmp_path / "blob"
    target.write_text("x")
    link = tmp_path / "link"
    link.symlink_to(target)
    roots = {"root": tmp_path}
    before = rules.labelled_listing(roots)
    time.sleep(0.01)
    link.unlink()
    link.symlink_to(tmp_path)
    after = rules.labelled_listing(roots)
    by_path = {c["path"]: c["sha256"] for c in rules.changed_files(before, after, roots)}
    assert by_path["root/link"] == hashlib.sha256(str(tmp_path).encode()).hexdigest()


def test_labelled_listing_skips_a_missing_root_and_raises_for_a_file_root(tmp_path: Path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "f.txt").write_text("1")
    missing_root = tmp_path / "missing"
    out = rules.labelled_listing({"d": tmp_path / "dir", "m": missing_root})
    assert set(out) == {"d/f.txt"}
    file_root = tmp_path / "not_a_dir.txt"
    file_root.write_text("x")
    with pytest.raises(NotADirectoryError):
        rules.labelled_listing({"f": file_root})


def _record_line(rel: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{rel},sha256={digest},{len(data)}\n"


def test_verify_records_detects_edit_and_handles_escaping_rows(tmp_path: Path):
    site = tmp_path / "lib" / "site"
    pkg = site / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_bytes(b"print(1)\n")
    launcher = tmp_path / "bin" / "tool"
    launcher.parent.mkdir()
    launcher.write_bytes(b"#!/x\n")
    info = site / "pkg-1.0.dist-info"
    info.mkdir()
    (info / "RECORD").write_text(
        _record_line("pkg/mod.py", b"print(1)\n")
        + _record_line("../../bin/tool", b"#!/x\n")
        + "pkg-1.0.dist-info/RECORD,,\n",
    )
    assert rules.verify_records(site) == []
    h1 = rules.environment_hash(site)
    (pkg / "mod.py").write_bytes(b"print(2)\n")
    assert rules.verify_records(site) == ["pkg/mod.py"]
    assert rules.environment_hash(site) != h1
    launcher.write_bytes(b"#!/y\n")
    assert "../../bin/tool" in rules.verify_records(site)


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _inv(tmp_path: Path) -> dict:
    (tmp_path / "docs" / "scripts").mkdir(parents=True)
    script = _write(tmp_path, "docs/scripts/vocab_extract.py", b"print(1)\n")
    return {
        "package": {"launcher": "/x/bin/graphify", "interpreter": "/x/bin/python", "site": "/x/site"},
        "subcommands": {
            "query": {"positional": 1, "flags": {}, "rejected": ["--dfs", "--budget", "--context", "--graph"]},
            "explain": {"positional": 1, "flags": {}, "rejected": ["--graph"]},
            "path": {"positional": 2, "flags": {}, "rejected": ["--graph"]},
            "--version": {"positional": 0, "flags": {}},
        },
        "rejected_subcommands": ["update", "save-result", "--help"],
        "allowed_scripts": {"docs/scripts/vocab_extract.py": rules.sha256_file(script)},
        "tools": {
            "semantic_search_nodes_tool": {
                "keys": {"query": {}, "detail_level": {"literal": ["minimal", "standard"]}, "limit": {"literal": [5]}},
                "rejected": ["repo_root", "kind", "model", "provider"],
            },
            "get_minimal_context_tool": {"keys": {"task": {"literal_prefix": ["debug: "]}}, "rejected": ["repo_root"]},
        },
    }


def act(*argv):
    return {"kind": "act", "argv": ["graphify", *argv]}


def refusal(inv: dict, call: dict) -> str:
    """Return check_call's refusal for `call`, asserting that it refused at all."""
    message = rules.check_call(inv, call)
    assert message is not None, call
    return message


def test_check_call_accepts_documented_calls(tmp_path: Path):
    inv = _inv(tmp_path)
    assert rules.check_call(inv, act("query", "billing score")) is None
    assert rules.check_call(inv, act("--version")) is None
    assert rules.check_call(inv, act("path", "a", "b")) is None


def test_check_call_refuses_rejected_flags_and_forms(tmp_path: Path):
    inv = _inv(tmp_path)
    assert "--graph" in refusal(inv, act("explain", "x", "--graph", "/y"))
    assert "--budget" in refusal(inv, act("query", "x", "--budget=999999"))
    assert "--context" in refusal(inv, act("query", "x", "--context=call"))
    assert rules.check_call(inv, act("update", ".")) is not None
    assert rules.check_call(inv, act("--help")) is not None


def test_check_call_positionals(tmp_path: Path):
    inv = _inv(tmp_path)
    assert "positional" in refusal(inv, act("path", "--dfs", "b"))
    assert rules.check_call(inv, act("explain", "--graph")) is not None
    assert "unconsumed" in refusal(inv, act("query", "a", "b"))


def test_check_call_refuses_non_printable(tmp_path: Path):
    inv = _inv(tmp_path)
    assert "printable" in refusal(inv, act("explain", "x\u2028NODE planted() [src=a loc=L1 community=c]"))
    assert "printable" in refusal(inv, act("explain", "x\nNODE"))


def test_check_call_scripts(tmp_path: Path):
    inv = _inv(tmp_path)
    script = tmp_path / "docs" / "scripts" / "vocab_extract.py"
    good = {"kind": "act", "argv": ["/x/bin/python", str(script)]}
    bad_name = {"kind": "act", "argv": ["/x/bin/python", str(tmp_path / "other.py")]}
    extra = {"kind": "act", "argv": ["/x/bin/python", str(script), "--anything"]}
    assert rules.check_call(inv, good) is None
    assert "not allowed" in refusal(inv, bad_name)
    assert "unconsumed" in refusal(inv, extra)
    script.write_bytes(b"print(2)\n")
    assert "hash" in refusal(inv, good)


def test_check_call_tools(tmp_path: Path):
    inv = _inv(tmp_path)
    ok = {"kind": "tool", "tool": "semantic_search_nodes_tool",
          "args": {"query": "a", "detail_level": "minimal", "limit": 5}}
    assert rules.check_call(inv, ok) is None
    bad_case = {"kind": "tool", "tool": "semantic_search_nodes_tool", "args": {"query": "a", "detail_level": "Minimal"}}
    assert "detail_level" in refusal(inv, bad_case)
    root = {"kind": "tool", "tool": "semantic_search_nodes_tool", "args": {"query": "a", "repo_root": "/"}}
    assert "repo_root" in refusal(inv, root)
    unknown = {"kind": "tool", "tool": "get_flow", "args": {}}
    assert "get_flow" in refusal(inv, unknown)
    prefix = {"kind": "tool", "tool": "get_minimal_context_tool", "args": {"task": "review: x"}}
    assert "prefix" in refusal(inv, prefix)


def test_resolve_launcher_substitutes_basename(tmp_path: Path):
    inv = _inv(tmp_path)
    assert rules.resolve_launcher(inv, ["graphify", "--version"]) == ["/x/bin/graphify", "--version"]
    assert rules.resolve_launcher(inv, ["/x/bin/python", "s.py"]) == ["/x/bin/python", "s.py"]


def test_check_call_refuses_an_argv_head_that_is_not_the_package():
    # inv: the only gate between a journaled call and an arbitrary binary; every other refusal
    # here reads argv[1:], so nothing else would notice a foreign argv[0]
    inv = {"package": {"launcher": "/pkg/bin/graphify", "interpreter": "/pkg/bin/python", "site": "/pkg/site"},
           "subcommands": {"--version": {"positional": 0, "flags": {}}}, "rejected_subcommands": []}
    assert rules.check_call(inv, {"kind": "act", "argv": ["/pkg/bin/graphify", "--version"]}) is None
    assert rules.check_call(inv, {"kind": "act", "argv": ["graphify", "--version"]}) is None
    refusal = rules.check_call(inv, {"kind": "act", "argv": ["/somewhere/else/evil", "--version"]})
    assert refusal == "argv[0] is not the package: /somewhere/else/evil"


def test_canonical_hash_file_matches_the_in_memory_hash(tmp_path: Path):
    # inv: the streaming form must be identical to the loaded one for every shape, or every
    # expectation and every canonical list already recorded becomes wrong
    for name, text in (("plain", "graphify 0.9.27\n"),
                       ("json", '{"b": 1, "a": {"age": 3, "k": "v"}}'),
                       ("json_after_space", '\n\n  {"a": 1}\n'),
                       ("array", '[{"a": 1}]'),
                       ("brace_but_not_json", "{not json at all}\n"),
                       ("crlf", "one\r\ntwo\r\n"),
                       ("empty", "")):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        assert rules.canonical_hash_file(p, ["a.age"]) == rules.canonical_hash(text, ["a.age"]), name


def test_canonical_hash_file_finds_json_behind_non_ascii_whitespace(tmp_path: Path):
    # inv: str.strip()'s whitespace class is wider than bytes.lstrip()'s, so JSON behind such a
    # character must still be recognised on disk
    body = '{"stable": 1, "a": {"age": 3}}'
    for name, prefix in (("nel", "\x85"), ("nbsp", "\xa0"), ("ideographic", "　"),
                         ("file_separator", "\x1c"), ("line_separator", " ")):
        p = tmp_path / name
        p.write_text(prefix + body, encoding="utf-8")
        assert rules.canonical_hash_file(p, ["a.age"]) == rules.canonical_hash(prefix + body, ["a.age"]), name
        moved = tmp_path / f"{name}_moved"
        moved.write_text(prefix + '{"stable": 1, "a": {"age": 99}}', encoding="utf-8")
        assert rules.canonical_hash_file(p, ["a.age"]) == rules.canonical_hash_file(moved, ["a.age"]), name


def test_canonical_hash_file_survives_a_character_split_across_chunks(tmp_path: Path, monkeypatch):
    # inv: an incremental decoder, because a multi-byte character straddling two reads would
    # otherwise decode as two replacements and change the digest
    monkeypatch.setattr(rules, "_CHUNK", 3)
    text = "abé" * 40 + "\n"
    raw = text.encode("utf-8")
    chunks = [raw[i : i + rules._CHUNK] for i in range(0, len(raw), rules._CHUNK)]
    # inv: the case is only a case when a character really straddles a read; a text whose
    # characters align with every boundary would pass without the decoder this pins
    assert "".join(c.decode("utf-8", errors="replace") for c in chunks) != text
    p = tmp_path / "multibyte"
    p.write_text(text, encoding="utf-8")
    assert rules.canonical_hash_file(p, []) == rules.canonical_hash(text, [])


def test_canonical_hash_file_finds_json_behind_more_space_than_one_read(tmp_path: Path, monkeypatch):
    # inv: the scan reads on until a non-space character appears, so leading space spanning
    # several reads still finds the JSON behind it
    monkeypatch.setattr(rules, "_CHUNK", 3)
    text = " " * 20 + '{"b": 1, "a": {"age": 3}}'
    p = tmp_path / "spaced"
    p.write_text(text, encoding="utf-8")
    assert rules.canonical_hash_file(p, ["a.age"]) == rules.canonical_hash(text, ["a.age"])


def test_canonical_hash_file_matches_on_invalid_utf8(tmp_path: Path):
    # inv: decode-with-replace is not identity, so the streamed digest must be taken over the
    # replaced text, exactly as the loaded form does -- not over the raw bytes
    p = tmp_path / "broken"
    p.write_bytes(b"before \xff\xfe after\n")
    assert rules.canonical_hash_file(p, []) == rules.canonical_hash(
        p.read_bytes().decode("utf-8", errors="replace"), [])


def test_canonical_hash_keeps_non_ascii_text_as_itself():
    # inv: the canonical payload is dumped without escaping, so the digest is over the reply's own
    # characters -- escaping them would move every hash ever recorded over non-ASCII output
    text = '{"b": 1, "a": "é中"}'
    assert rules.canonical_hash(text, []) == hashlib.sha256(
        '{"a":"é中","b":1}'.encode()).hexdigest()


def test_canonical_hash_file_matches_on_non_ascii_json(tmp_path: Path):
    text = '{"b": 1, "a": {"age": 3, "k": "é中"}}'
    p = tmp_path / "reply"
    p.write_text(text, encoding="utf-8")
    assert rules.canonical_hash_file(p, ["a.age"]) == rules.canonical_hash(text, ["a.age"])


PROMPT = "how is the billing score calculated\nExpansion: billing score calculated\n"
RECORDS = [
    {"n": 3, "kind": "place", "path": "app/core/pricing.py", "label": "render_invoice()",
     "symbol": "render_invoice", "qualified_name": None},
    {"n": 3, "kind": "candidate", "path": "app/models.py", "label": "InvoiceResult",
     "symbol": "InvoiceResult", "qualified_name": "app/models.py::InvoiceResult"},
    {"n": 4, "kind": "unparsed", "text": "invented_by_nobody"},
    {"n": 4, "kind": "no_results", "vendor_status": "ambiguous", "vendor_message": "ambiguous_name"},
]
LITERALS = {"query", "standard", 5}


def test_a_value_the_grammar_pins_comes_from_the_grammar():
    assert rules.provenance("standard", PROMPT, RECORDS, LITERALS) == {"kind": "literal"}


def test_a_number_the_grammar_pins_comes_from_the_grammar():
    assert rules.provenance(5, PROMPT, RECORDS, LITERALS) == {"kind": "literal"}


def test_a_value_written_in_the_prompt_comes_from_the_prompt():
    assert rules.provenance("billing score", PROMPT, RECORDS, set()) == {"kind": "prompt"}


def test_a_value_equal_to_an_earlier_symbol_comes_from_that_entry():
    assert rules.provenance("render_invoice", "", RECORDS, set()) == {"kind": "record", "n": 3}


def test_a_value_equal_to_an_earlier_qualified_name_comes_from_that_entry():
    assert rules.provenance("app/models.py::InvoiceResult", "", RECORDS, set()) == {"kind": "record", "n": 3}


def test_a_value_equal_to_an_earlier_path_comes_from_that_entry():
    assert rules.provenance("app/core/pricing.py", "", RECORDS, set()) == {"kind": "record", "n": 3}


def test_a_value_equal_to_an_earlier_label_comes_from_that_entry():
    assert rules.provenance("render_invoice()", "", RECORDS, set()) == {"kind": "record", "n": 3}


def test_a_value_accounted_for_by_nothing_has_no_origin():
    assert rules.provenance("invented", "", RECORDS, set()) == {"kind": "none"}


def test_a_record_match_is_byte_exact_and_survives_no_normalisation():
    records = [{"n": 2, "kind": "place", "path": None, "label": None,
                "symbol": "caf\u00e9_score", "qualified_name": None}]
    assert rules.provenance("cafe\u0301_score", "", records, set()) == {"kind": "none"}


def test_an_unparsed_record_is_not_an_origin():
    assert rules.provenance("invented_by_nobody", "", RECORDS, set()) == {"kind": "none"}


def test_an_empty_reply_is_not_an_origin():
    assert rules.provenance("ambiguous_name", "", RECORDS, set()) == {"kind": "none"}


def test_the_grammar_answers_before_the_prompt_does():
    assert rules.provenance("billing score", PROMPT, RECORDS, {"billing score"}) == {"kind": "literal"}


def test_the_prompt_answers_before_an_earlier_entry_does():
    prompt = "render_invoice is mentioned here"
    assert rules.provenance("render_invoice", prompt, RECORDS, set()) == {"kind": "prompt"}


def test_an_act_call_accounts_for_every_token_after_the_launcher():
    invocation = {"package": {"launcher": "/abs/bin/graphify", "interpreter": "/abs/bin/python"},
                  "subcommands": {"query": {"positional": 1, "flags": {}}}}
    call = {"kind": "act", "argv": ["/abs/bin/graphify", "query", "billing score", "invented"]}
    got = rules.call_provenance(invocation, call, PROMPT, RECORDS)
    assert [row["at"] for row in got] == ["argv[1]", "argv[2]", "argv[3]"]
    assert [row["kind"] for row in got] == ["literal", "prompt", "none"]


def test_a_tool_call_accounts_for_every_key_but_the_quote():
    invocation = {"package": {"launcher": "/abs/bin/crg", "interpreter": "/abs/bin/python"},
                  "tools": {"search": {"keys": {"query": {}, "detail_level": {"literal": ["standard"]}}}}}
    call = {"kind": "tool", "tool": "search",
            "args": {"query": "billing score", "detail_level": "standard", "quote": "some sentence"}}
    got = rules.call_provenance(invocation, call, PROMPT, RECORDS)
    assert [row["at"] for row in got] == ["detail_level", "query"]
    assert [row["kind"] for row in got] == ["literal", "prompt"]


def test_a_record_of_a_kind_outside_the_allowlist_is_not_an_origin():
    # why: a traversal seed names a place the vendor did not return as a result, so a runner
    # that sends it back is inventing, not citing
    records = [{"n": 2, "kind": "seed", "path": "app/core/pricing.py", "symbol": "render_invoice"}]
    assert rules.provenance("render_invoice", "", records, set()) == {"kind": "none"}


def test_a_flag_value_the_grammar_pins_comes_from_the_grammar():
    invocation = {"package": {"launcher": "/abs/bin/graphify", "interpreter": "/abs/bin/python"},
                  "subcommands": {"query": {"positional": 1, "flags": {"--mode": {"literal": ["bfs"]}}}}}
    call = {"kind": "act", "argv": ["/abs/bin/graphify", "query", "billing score", "--mode", "bfs"]}
    got = rules.call_provenance(invocation, call, PROMPT, RECORDS)
    assert [row["kind"] for row in got] == ["literal", "prompt", "literal", "literal"]


def test_the_allowlist_of_origin_kinds_is_what_decides_not_the_field_that_matched():
    # why: no adapter emits an unparsed line carrying a path today, so this shape is invented
    # on purpose -- what it pins is the allowlist itself, which is the guarantee that survives
    # an adapter learning to fill more fields
    records = [{"n": 2, "kind": "unparsed", "text": "junk", "path": "app/core/pricing.py"}]
    assert rules.provenance("app/core/pricing.py", "", records, set()) == {"kind": "none"}


def test_require_sealed_passes_on_a_sealed_tree(sealed_bench):
    machine = rules.require_sealed(sealed_bench, Path(yaml.safe_load(
        (sealed_bench / "lock" / "machine.yaml").read_text(encoding="utf-8"))["tmp_root"]))
    assert machine["bench_uid"] == os.geteuid()


def test_require_sealed_refuses_a_foreign_root(sealed_bench, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(SystemExit, match="is not the benchmark this harness lives in"):
        rules.require_sealed(other, tmp_path)


def test_require_sealed_refuses_at_every_gate(sealed_bench, monkeypatch, tmp_path):
    b = sealed_bench
    tmp_root = Path(yaml.safe_load((b / "lock" / "machine.yaml").read_text(encoding="utf-8"))["tmp_root"])

    (b / "INSTRUMENT.yaml").rename(b / "INSTRUMENT.yaml.bak")
    with pytest.raises(SystemExit, match="instrument is not sealed"):
        rules.require_sealed(b, tmp_root)
    (b / "INSTRUMENT.yaml.bak").rename(b / "INSTRUMENT.yaml")

    (b / "harness").mkdir(exist_ok=True)
    (b / "harness" / "planted.py").write_text("x\n", encoding="utf-8")
    planted = r"instrument differs from its seal: benchmark/harness/planted.py \(unlisted\)"
    with pytest.raises(SystemExit, match=planted):
        rules.require_sealed(b, tmp_root)
    (b / "harness" / "planted.py").unlink()

    (b / "lock" / "UNLOCKED").write_text("reason: open\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="runs are refused while the instrument is unlocked"):
        rules.require_sealed(b, tmp_root)
    (b / "lock" / "UNLOCKED").unlink()

    monkeypatch.setattr(rules, "_euid", lambda: 12345)
    with pytest.raises(SystemExit, match="the harness runs as bench, not uid 12345"):
        rules.require_sealed(b, tmp_root)
    monkeypatch.setattr(rules, "_euid", os.geteuid)

    with pytest.raises(SystemExit, match="tmp root .* is not the one machine.yaml names"):
        rules.require_sealed(b, tmp_path / "other-tmp")

    monkeypatch.setattr(rules, "_owner", lambda p: 0)
    with pytest.raises(SystemExit, match="record not owned by bench"):
        rules.require_sealed(b, tmp_root)
    monkeypatch.setattr(rules, "_owner", lambda p: os.geteuid())

    monkeypatch.setattr(rules, "_mode", lambda p: 0o777 if Path(p) == tmp_root else 0o755)
    with pytest.raises(SystemExit, match="record writable by others"):
        rules.require_sealed(b, tmp_root)
    monkeypatch.setattr(rules, "_mode", lambda p: 0o755)

    env = b / "envs" / "harness" / "bin"
    env.mkdir(parents=True)
    (env / "python").write_text("x\n", encoding="utf-8")
    _reseal(b)
    (env / "python").write_text("y\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="harness environment differs from its seal"):
        rules.require_sealed(b, tmp_root)
    (env / "python").write_text("x\n", encoding="utf-8")


def test_unlocked_refusal_is_one_line_even_when_ownership_would_fail(sealed_bench, monkeypatch):
    # inv: an installed machine mid-unlock is owner-writable everywhere, and the refusal the
    # operator sees is still the one line about being unlocked, not an ownership wall
    from benchmark.harness import seal

    monkeypatch.setattr(seal, "_owner", lambda p: 501)
    monkeypatch.setattr(seal, "_mode_flags", lambda p: (0o755, 0))
    (sealed_bench / "lock" / "UNLOCKED").write_text("reason: 'open'\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        rules.require_sealed(sealed_bench, Path(yaml.safe_load(
            (sealed_bench / "lock" / "machine.yaml").read_text(encoding="utf-8"))["tmp_root"]))
    assert str(exc.value) == "runs are refused while the instrument is unlocked"
    (sealed_bench / "lock" / "UNLOCKED").unlink()


def test_seal_check_names_the_open_state_instead_of_the_ownership_wall(sealed_bench, monkeypatch):
    from benchmark.harness import seal

    monkeypatch.setattr(seal, "_owner", lambda p: 501)
    monkeypatch.setattr(seal, "_mode_flags", lambda p: (0o755, 0))
    (sealed_bench / "lock" / "UNLOCKED").write_text("reason: 'open'\n", encoding="utf-8")
    problems = seal.check(sealed_bench)
    marker = "instrument is unlocked; ownership is not checked while open"
    assert [p for p in problems if p.startswith(("instrument", "record"))] == [marker]
    (sealed_bench / "lock" / "UNLOCKED").unlink()


def test_machine_yaml_without_bench_uid_refuses_by_name(sealed_bench):
    machine = sealed_bench / "lock" / "machine.yaml"
    doc = yaml.safe_load(machine.read_text(encoding="utf-8"))
    del doc["bench_uid"]
    machine.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        rules.require_sealed(sealed_bench, Path(doc["tmp_root"]))
    assert str(exc.value).splitlines()[-1] == "machine.yaml is missing bench_uid"


def test_machine_yaml_without_tmp_root_refuses_by_name(sealed_bench):
    b = sealed_bench
    machine = b / "lock" / "machine.yaml"
    doc = yaml.safe_load(machine.read_text(encoding="utf-8"))
    tmp_root = doc["tmp_root"]
    del doc["tmp_root"]
    machine.write_text(yaml.safe_dump(doc), encoding="utf-8")
    # why: seal.check's own machine_sha256 check would otherwise catch the edited machine.yaml
    # first, masking the guard this test targets
    _reseal(b)
    with pytest.raises(SystemExit, match="machine.yaml is missing tmp_root"):
        rules.require_sealed(b, Path(tmp_root))


def test_require_sealed_checks_models_and_corpus(sealed_bench, monkeypatch):
    b = sealed_bench
    snap = snapshot_dir(b, "snap")
    (snap / "source").mkdir(parents=True)
    (snap / "source" / "a.py").write_text("a\n", encoding="utf-8")
    (snap / "fileset.sha256").write_text(
        f"{rules.sha256_file(snap / 'source' / 'a.py')}  ./a.py\n", encoding="utf-8")
    (snap / "symbols.jsonl").write_text("{}\n", encoding="utf-8")
    (snap / "symbols.sha256").write_text(
        f"{rules.sha256_file(snap / 'symbols.jsonl')}  symbols.jsonl\n", encoding="utf-8")
    sysdir = b / "systems" / "s" / "models" / "m"
    sysdir.mkdir(parents=True)
    (sysdir / "w.bin").write_text("w\n", encoding="utf-8")
    models = {"m": {"files": {"w.bin": rules.sha256_file(sysdir / "w.bin")}, "links": {}}}
    _reseal(b)
    assert rules.check_models(b, "s", models) == []
    assert rules.check_corpus(snap) == []

    (sysdir / "w.bin").write_text("W\n", encoding="utf-8")
    assert rules.check_models(b, "s", models) == ["model differs: m: w.bin"]
    (sysdir / "extra").write_text("e\n", encoding="utf-8")
    assert "model differs: m: extra (unlisted)" in rules.check_models(b, "s", models)
    (snap / "source" / "a.py").write_text("A\n", encoding="utf-8")
    assert rules.check_corpus(snap) == ["corpus differs: a.py"]
    (snap / "symbols.jsonl").write_text("{x}\n", encoding="utf-8")
    assert "corpus differs: symbols.jsonl" in rules.check_corpus(snap)


def test_require_sealed_refuses_when_lock_is_not_installed(sealed_bench):
    b = sealed_bench
    tmp_root = Path(yaml.safe_load((b / "lock" / "machine.yaml").read_text(encoding="utf-8"))["tmp_root"])
    (b / "lock" / "machine.yaml").unlink()
    with pytest.raises(SystemExit, match="lock is not installed on this machine; run benchmark/lock/install"):
        rules.require_sealed(b, tmp_root)


def test_check_corpus_refuses_an_unfrozen_source_then_passes_once_frozen(tmp_path: Path):
    snap = tmp_path / "snap"
    (snap / "source").mkdir(parents=True)
    (snap / "source" / "a.py").write_text("a\n", encoding="utf-8")
    assert rules.check_corpus(snap) == ["corpus is not frozen: no fileset.sha256"]
    freeze_source(snap)
    assert rules.check_corpus(snap) == []


def test_check_models_flags_a_symlink_with_the_wrong_target(tmp_path: Path):
    root = tmp_path / "systems" / "s" / "models" / "m"
    root.mkdir(parents=True)
    (root / "w.bin").symlink_to("/actual/target")
    models = {"m": {"files": {}, "links": {"w.bin": "/wrong/target"}}}
    assert rules.check_models(tmp_path, "s", models) == ["model differs: m: w.bin (link)"]
