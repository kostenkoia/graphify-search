import pytest

from benchmark.harness import tools

SERVER = {
    "semantic_search_nodes_tool": {
        "description": "Search nodes semantically.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "detail_level": {"type": "string"},
                "limit": {"type": "integer"},
                "repo_root": {"type": "string"},
            },
            "required": ["query", "repo_root"],
        },
    },
    "list_graph_stats_tool": {"description": "Stats.", "inputSchema": {"type": "object", "properties": {}}},
    "never_configured_tool": {"description": "Absent from the harness.", "inputSchema": {"type": "object"}},
}

INVOCATION = {
    "package": {"launcher": "/abs/bin/crg", "interpreter": "/abs/bin/python"},
    "subcommands": {"--version": {"positional": 0, "flags": {}}},
    "tools": {
        "semantic_search_nodes_tool": {
            "keys": {"query": {}, "detail_level": {"literal": ["minimal", "standard"]},
                     "limit": {"literal": [5]}},
            "rejected": ["repo_root"],
        },
        "list_graph_stats_tool": {"keys": {}},
    },
    "harness_only": ["list_graph_stats_tool"],
}


def _by_name(offered: list[dict]) -> dict[str, dict]:
    return {tool["name"]: tool for tool in offered}


def test_the_runner_is_offered_act_stop_and_the_derived_tools():
    names = set(_by_name(tools.offered(INVOCATION, SERVER)))
    assert names == {"act", "stop", "semantic_search_nodes_tool"}


def test_a_harness_only_tool_is_never_offered():
    assert "list_graph_stats_tool" not in _by_name(tools.offered(INVOCATION, SERVER))


def test_a_tool_the_harness_never_configured_is_never_offered():
    assert "never_configured_tool" not in _by_name(tools.offered(INVOCATION, SERVER))


def test_a_server_property_outside_the_configured_keys_is_dropped():
    schema = _by_name(tools.offered(INVOCATION, SERVER))["semantic_search_nodes_tool"]["input_schema"]
    assert "repo_root" not in schema["properties"]


def test_a_configured_key_the_server_does_not_offer_is_dropped():
    invocation = {**INVOCATION, "tools": {"semantic_search_nodes_tool": {
        "keys": {"query": {}, "invented": {}}, "rejected": []}}, "harness_only": []}
    schema = _by_name(tools.offered(invocation, SERVER))["semantic_search_nodes_tool"]["input_schema"]
    assert "invented" not in schema["properties"]


def test_required_is_the_server_requirement_narrowed_to_the_configured_keys():
    schema = _by_name(tools.offered(INVOCATION, SERVER))["semantic_search_nodes_tool"]["input_schema"]
    assert set(schema["required"]) == {"query", "quote"}


def test_every_derived_tool_asks_for_the_quote_that_authorises_it():
    schema = _by_name(tools.offered(INVOCATION, SERVER))["semantic_search_nodes_tool"]["input_schema"]
    assert schema["properties"]["quote"]["type"] == "string"
    assert "quote" in schema["required"]


def test_a_key_bound_to_literals_becomes_an_enum_of_exactly_those_values():
    schema = _by_name(tools.offered(INVOCATION, SERVER))["semantic_search_nodes_tool"]["input_schema"]
    assert schema["properties"]["detail_level"]["enum"] == ["minimal", "standard"]
    assert schema["properties"]["limit"]["enum"] == [5]


def test_a_derived_tool_accepts_no_key_beyond_its_schema():
    schema = _by_name(tools.offered(INVOCATION, SERVER))["semantic_search_nodes_tool"]["input_schema"]
    assert schema["additionalProperties"] is False


def test_the_description_is_the_servers_own_words():
    tool = _by_name(tools.offered(INVOCATION, SERVER))["semantic_search_nodes_tool"]
    assert tool["description"] == SERVER["semantic_search_nodes_tool"]["description"]


def test_act_takes_an_argument_vector_of_strings_and_a_quote():
    schema = _by_name(tools.offered(INVOCATION, SERVER))["act"]["input_schema"]
    assert schema["properties"]["argv"]["items"]["type"] == "string"
    assert set(schema["required"]) == {"argv", "quote"}


def test_stop_takes_the_place_that_answers_the_question():
    schema = _by_name(tools.offered(INVOCATION, SERVER))["stop"]["input_schema"]
    assert set(schema["required"]) == {"path", "symbol", "start"}
    assert schema["properties"]["start"]["type"] == "integer"


def test_a_system_with_no_tools_is_offered_act_and_stop_alone():
    invocation = {"package": {"launcher": "/abs/bin/graphify", "interpreter": "/abs/bin/python"},
                  "subcommands": {"query": {"positional": 1, "flags": {}}}}
    assert set(_by_name(tools.offered(invocation, {}))) == {"act", "stop"}


def test_the_tool_list_does_not_depend_on_the_order_the_grammar_names_them():
    server = {name: {"description": name, "inputSchema": {"type": "object", "properties": {}}}
              for name in ("c_tool", "a_tool", "b_tool")}
    forward = {"package": {}, "tools": {n: {"keys": {}} for n in ("a_tool", "b_tool", "c_tool")}}
    backward = {"package": {}, "tools": {n: {"keys": {}} for n in ("c_tool", "b_tool", "a_tool")}}
    order = [tool["name"] for tool in tools.offered(forward, server)]
    assert order == [tool["name"] for tool in tools.offered(backward, server)]
    assert order == ["act", "a_tool", "b_tool", "c_tool", "stop"]


def test_a_configured_tool_the_server_does_not_offer_is_refused_not_dropped():
    invocation = {"package": {}, "tools": {"semantic_search_nodes_tool": {"keys": {"query": {}}},
                                           "gone_from_the_server_tool": {"keys": {}}}}
    with pytest.raises(tools.ToolsError, match="gone_from_the_server_tool"):
        tools.offered(invocation, SERVER)


def test_a_withheld_tool_the_server_lacks_is_not_a_refusal():
    invocation = {"package": {}, "tools": {"gone_from_the_server_tool": {"keys": {}}},
                  "harness_only": ["gone_from_the_server_tool"]}
    assert [tool["name"] for tool in tools.offered(invocation, SERVER)] == ["act", "stop"]


def test_the_act_tool_names_the_command_the_runner_must_call():
    description = _by_name(tools.offered(INVOCATION, SERVER))["act"]["description"]
    assert "crg" in description


def test_the_act_tool_names_the_subcommands_the_grammar_allows():
    invocation = {**INVOCATION, "subcommands": {"query": {"positional": 1}, "--version": {}}}
    description = _by_name(tools.offered(invocation, SERVER))["act"]["description"]
    assert "query" in description
    assert "--version" in description


def test_the_act_tool_does_not_name_a_subcommand_the_grammar_rejects():
    invocation = {**INVOCATION, "subcommands": {"query": {"positional": 1}},
                  "rejected_subcommands": ["serve"]}
    assert "serve" not in _by_name(tools.offered(invocation, SERVER))["act"]["description"]
