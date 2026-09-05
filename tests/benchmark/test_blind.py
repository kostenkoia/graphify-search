import pytest

from benchmark.harness import blind

REFERENCE = {"id": "q001", "places": [
    {"path": "pkg/billing/pricing.py",
     "symbol": "render_invoice", "start": 63, "end": 149,
     "why": "the function that computes the score"},
]}

CLEAN = "# Your question\n\nhow is the billing score calculated\n\nbilling score scores\n"
TOOLS = [{"name": "act", "description": "Run the vendor package once.", "input_schema": {}}]


def test_a_clean_prompt_leaks_nothing():
    assert blind.violations(CLEAN, TOOLS, REFERENCE) == []


def test_a_prompt_naming_the_reference_path_is_caught():
    found = blind.violations(CLEAN + "look in pkg/billing/pricing.py",
                             TOOLS, REFERENCE)
    assert len(found) == 1
    assert "pricing.py" in found[0]


def test_a_prompt_naming_only_the_file_is_caught():
    found = blind.violations(CLEAN + "the answer is in pricing.py", TOOLS, REFERENCE)
    assert len(found) == 1


def test_a_prompt_naming_the_reference_symbol_is_caught():
    found = blind.violations(CLEAN + "start from render_invoice", TOOLS, REFERENCE)
    assert any("render_invoice" in message for message in found)


def test_the_leak_is_caught_whatever_case_it_is_written_in():
    found = blind.violations(CLEAN + "start from Render_Invoice", TOOLS, REFERENCE)
    assert found != []


def test_a_tool_description_naming_the_symbol_is_caught():
    tools = [{"name": "act", "description": "Call render_invoice first.", "input_schema": {}}]
    found = blind.violations(CLEAN, tools, REFERENCE)
    assert any("act" in message for message in found)


def test_a_tool_schema_naming_the_symbol_is_caught():
    tools = [{"name": "act", "description": "d",
              "input_schema": {"properties": {"argv": {"description": "e.g. pricing.py"}}}}]
    assert blind.violations(CLEAN, tools, REFERENCE) != []


def test_a_qualified_name_of_the_reference_is_caught():
    reference = {"places": [{"path": "a.py", "symbol": "method_one", "start": 1,
                             "qualified_name": "a.py::Klass.method"}]}
    found = blind.violations("see a.py::Klass.method", TOOLS, reference)
    assert any("a.py::Klass.method" in message for message in found)


def test_the_reasoning_written_beside_the_reference_is_caught_verbatim():
    found = blind.violations(CLEAN + "the function that computes the score", TOOLS, REFERENCE)
    assert any("computes the score" in message for message in found)


def test_a_reference_place_with_no_symbol_is_still_checked_by_its_path():
    reference = {"places": [{"path": "pkg/thing.py", "symbol": None, "start": 3}]}
    assert blind.violations("open pkg/thing.py", TOOLS, reference) != []


def test_every_leaking_place_is_named_not_only_the_first():
    reference = {"places": [{"path": "a.py", "symbol": "one", "start": 1},
                            {"path": "b.py", "symbol": "two", "start": 2}]}
    found = blind.violations("one and two", TOOLS, reference)
    assert len(found) == 2


def test_a_symbol_too_short_to_be_evidence_is_refused_rather_than_ignored():
    reference = {"places": [{"path": "a.py", "symbol": "f", "start": 1}]}
    with pytest.raises(blind.BlindError, match="too short"):
        blind.violations(CLEAN, TOOLS, reference)
