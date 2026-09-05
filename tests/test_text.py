from graphify_search import text as t


def test_split_identifiers_camel_and_snake():
    assert t.split_identifiers("on_asset_archived") == ["on", "asset", "archived"]
    assert t.split_identifiers("AssetDetailPage") == ["asset", "detail", "page"]
    assert t.split_identifiers("HTTPServer2") == ["http", "server"]
    assert t.split_identifiers("m_3_assignments") == ["assignments"]


def test_tokens_over_a_path():
    assert t.tokens("backend/app/on_asset_archived.py") == ["backend", "app", "on", "asset", "archived", "py"]


def test_node_text_layout():
    out = t.node_text("alpha()", "app/main.py", ["Compute alpha."], "def alpha(x):\n    return 2")
    assert out == "alpha()\napp main py\nCompute alpha.\ndef alpha(x):\n    return 2"
    assert t.node_text("guide.md", "docs/guide.md", [], "") == "guide.md\ndocs guide md"


def test_body_from_takes_body_lines_from_start_and_stops_at_the_file_end():
    lines = [f"l{i}" for i in range(1, 100)]
    assert t.body_from(lines, 5).splitlines() == [f"l{i}" for i in range(5, 5 + t.BODY_LINES)]
    assert t.body_from(lines, 97) == "l97\nl98\nl99"
    assert t.body_from(lines, 200) == ""
    assert t.body_from(lines, 0) == ""


def test_snippet_and_hash():
    assert t.snippet_of("\n".join(str(i) for i in range(10))) == "0\n1\n2\n3\n4\n5"
    assert t.text_sha256("x") == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"


def test_tokens_are_unicode_aware():
    assert t.tokens("как работает поиск") == ["как", "работает", "поиск"]
    assert t.tokens("naïve_café") == ["naïve", "café"]
    assert t.tokens("КАК Работает") == ["как", "работает"]
