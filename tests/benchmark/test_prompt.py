import pytest

from benchmark.harness import prompt

MANIFEST = {
    "id": "graphify",
    "prescribed_workflow": {
        "source": "SKILL.md",
        "steps": [
            {"id": 0, "name": "constrained query expansion", "quote": "no stemming, no synonyms"},
            {"id": 1, "name": "traversal", "quote": "Build the expanded query string",
             "command": 'graphify query "<expanded tokens>"'},
        ],
    },
}
QUESTION = {"id": "q001", "text": "how is the billing score calculated"}
TOKENS = ["billing", "score", "calculated"]
STEPS = [
    {"name": "version", "cmd": "kind: act\ncall: graphify --version", "out": "graphify 0.9.27"},
    {"name": "query", "cmd": "kind: act\ncall: graphify query ...", "out": "NODE thing src=a.py"},
]


def test_fence_is_three_backticks_when_the_text_has_none():
    assert prompt.fence("plain output") == "```"


def test_fence_is_one_longer_than_the_longest_backtick_run():
    assert prompt.fence("a ```` b ``` c") == "`````"


def test_a_step_output_cannot_break_out_of_its_block():
    steps = [{"name": "query", "cmd": "c", "out": "before\n```\n# heading\n```\nafter"}]
    block = prompt.executed(steps)
    opening = next(line for line in block.splitlines() if line.startswith("```"))
    assert len(opening) >= 4


def test_the_heading_appears_exactly_once():
    text = prompt.build(MANIFEST, QUESTION, TOKENS, STEPS)
    assert text.count(prompt.HEADING) == 1


def test_split_returns_the_authored_part_and_the_executed_part():
    text = prompt.build(MANIFEST, QUESTION, TOKENS, STEPS)
    above, below = prompt.split(text)
    assert QUESTION["text"] in above
    assert "graphify 0.9.27" in below


def test_split_refuses_a_prompt_carrying_the_heading_twice():
    text = prompt.build(MANIFEST, QUESTION, TOKENS, STEPS) + f"\n{prompt.HEADING}\n"
    with pytest.raises(prompt.PromptError, match="twice"):
        prompt.split(text)


def test_split_refuses_a_prompt_with_no_heading():
    with pytest.raises(prompt.PromptError, match="heading"):
        prompt.split("nothing here")


def test_the_authored_part_names_the_question():
    above, _ = prompt.split(prompt.build(MANIFEST, QUESTION, TOKENS, STEPS))
    assert QUESTION["text"] in above


def test_the_authored_part_names_every_expansion_token():
    above, _ = prompt.split(prompt.build(MANIFEST, QUESTION, TOKENS, STEPS))
    for token in TOKENS:
        assert token in above


def test_the_authored_part_quotes_each_prescribed_step_verbatim():
    above, _ = prompt.split(prompt.build(MANIFEST, QUESTION, TOKENS, STEPS))
    for step in MANIFEST["prescribed_workflow"]["steps"]:
        assert step["quote"] in above


def test_the_executed_part_keeps_one_block_per_step_in_order():
    _, below = prompt.split(prompt.build(MANIFEST, QUESTION, TOKENS, STEPS))
    assert below.index("graphify 0.9.27") < below.index("NODE thing src=a.py")


def test_the_executed_part_names_the_command_of_each_step():
    _, below = prompt.split(prompt.build(MANIFEST, QUESTION, TOKENS, STEPS))
    for step in STEPS:
        assert step["cmd"] in below


def test_no_output_of_a_fixed_step_reaches_the_authored_part():
    above, _ = prompt.split(prompt.build(MANIFEST, QUESTION, TOKENS, STEPS))
    for step in STEPS:
        assert step["out"] not in above


def test_the_authored_part_asks_for_the_answer_as_a_stop_call():
    above, _ = prompt.split(prompt.build(MANIFEST, QUESTION, TOKENS, STEPS))
    sentence = next(line for line in above.splitlines() if "stop" in line)
    for field in ("path", "symbol", "start"):
        assert field in sentence


def test_a_prescribed_step_with_no_quote_is_still_shown_by_name():
    manifest = {"id": "toy", "prescribed_workflow": {"source": "SKILL.md", "steps": [
        {"id": 3, "name": "targeted follow-ups", "commands": ['graphify explain "<node>"',
                                                              'graphify path "<A>" "<B>"']},
    ]}}
    above, _ = prompt.split(prompt.build(manifest, QUESTION, TOKENS, STEPS))
    assert "targeted follow-ups" in above


def test_a_step_naming_several_commands_shows_every_one():
    manifest = {"id": "toy", "prescribed_workflow": {"source": "SKILL.md", "steps": [
        {"id": 3, "name": "follow-ups", "commands": ['graphify explain "<node>"',
                                                     'graphify path "<A>" "<B>"']},
    ]}}
    above, _ = prompt.split(prompt.build(manifest, QUESTION, TOKENS, STEPS))
    assert 'graphify explain "<node>"' in above
    assert 'graphify path "<A>" "<B>"' in above


CALL_STEPS = {"id": "crg", "prescribed_workflow": {"source": "debug_prompt.md", "steps": [
    {"id": 2, "call": 'semantic_search_nodes(query=<keywords>, detail_level="standard")',
     "note": "keywords, not the sentence"},
]}}


def test_a_prescribed_step_with_no_name_is_shown_by_its_number():
    above, _ = prompt.split(prompt.build(CALL_STEPS, QUESTION, TOKENS, STEPS))
    assert "Step 2" in above


def test_a_prescribed_step_stated_as_a_call_shows_that_call():
    above, _ = prompt.split(prompt.build(CALL_STEPS, QUESTION, TOKENS, STEPS))
    assert 'semantic_search_nodes(query=<keywords>, detail_level="standard")' in above


def test_a_note_the_vendor_wrote_beside_a_step_is_kept():
    above, _ = prompt.split(prompt.build(CALL_STEPS, QUESTION, TOKENS, STEPS))
    assert "keywords, not the sentence" in above


WITHHELD = {"id": "toy", "prescribed_workflow": {"source": "SKILL.md", "steps": [
    {"id": 1, "name": "traversal", "quote": "Build the expanded query string",
     "command": 'graphify query "<tokens>"'},
    {"id": 3, "name": "targeted follow-ups",
     "commands": ['graphify explain "<node>"', 'graphify path "<A>" "<B>"']},
]}}
INVOCATION = {"subcommands": {"query": {"positional": 1}},
              "rejected_subcommands": ["explain", "path"]}


def test_a_command_the_grammar_rejects_is_not_offered_as_a_command():
    above, _ = prompt.split(prompt.build(WITHHELD, QUESTION, TOKENS, STEPS, INVOCATION))
    assert 'Its command: `graphify explain "<node>"`' not in above


def test_the_step_is_still_shown_and_says_it_is_not_available():
    above, _ = prompt.split(prompt.build(WITHHELD, QUESTION, TOKENS, STEPS, INVOCATION))
    assert "targeted follow-ups" in above
    assert prompt.WITHHELD_NOTE in above


def test_a_command_the_grammar_accepts_is_still_offered():
    above, _ = prompt.split(prompt.build(WITHHELD, QUESTION, TOKENS, STEPS, INVOCATION))
    assert 'Its command: `graphify query "<tokens>"`' in above


def test_a_prompt_built_without_a_grammar_withholds_nothing():
    above, _ = prompt.split(prompt.build(WITHHELD, QUESTION, TOKENS, STEPS))
    assert 'graphify explain "<node>"' in above
    assert prompt.WITHHELD_NOTE not in above


WITH_PROCEDURE = {"id": "toy", "prescribed_workflow": {"source": "SKILL.md", "steps": [
    {"id": 0, "name": "expansion", "quote": "no stemming, no synonyms",
     "procedure": "Select up to 12 tokens from that exact list. Inventing tokens is forbidden."},
    {"id": 1, "name": "traversal", "command": 'graphify query "<tokens>"',
     "modes": {"bfs": "default", "dfs": "--dfs, for tracing a specific chain"}},
]}}
GRAMMAR = {"subcommands": {"query": {"positional": 1, "rejected": ["--dfs", "--budget"]}},
           "rejected_subcommands": []}


def test_the_procedure_a_vendor_prescribes_reaches_the_runner():
    # why: dropping it measures a workflow the runner was never shown in full
    above, _ = prompt.split(prompt.build(WITH_PROCEDURE, QUESTION, TOKENS, STEPS, GRAMMAR))
    assert "Inventing tokens is forbidden" in above


def test_a_mode_the_grammar_accepts_is_offered():
    above, _ = prompt.split(prompt.build(WITH_PROCEDURE, QUESTION, TOKENS, STEPS, GRAMMAR))
    assert "bfs" in above


def test_a_mode_naming_a_flag_the_grammar_rejects_is_not_offered():
    above, _ = prompt.split(prompt.build(WITH_PROCEDURE, QUESTION, TOKENS, STEPS, GRAMMAR))
    assert "--dfs" not in above


def test_a_command_naming_a_rejected_flag_is_not_offered_either():
    manifest = {"id": "toy", "prescribed_workflow": {"steps": [
        {"id": 1, "name": "traversal", "command": 'graphify query "<t>" --budget 999'}]}}
    above, _ = prompt.split(prompt.build(manifest, QUESTION, TOKENS, STEPS, GRAMMAR))
    assert "--budget" not in above
    assert prompt.WITHHELD_NOTE in above


def test_a_system_with_no_expansion_gets_no_expansion_section():
    above, _ = prompt.split(prompt.build(MANIFEST, QUESTION, [], STEPS))
    assert "The expansion the harness prepared" not in above


def test_a_prompt_with_no_expansion_still_asks_the_question_and_the_answer_rule():
    above, _ = prompt.split(prompt.build(MANIFEST, QUESTION, [], STEPS))
    assert QUESTION["text"] in above
    assert prompt.ANSWER_RULE in above


RULED = {"id": "crg", "prescribed_workflow": {"source": "debug_prompt.md", "rules": [
    {"quote": "ALWAYS call `get_minimal_context` first with a task description."},
    {"quote": "Never request more than 3 tool calls per turn unless absolutely necessary."},
], "steps": [
    {"id": 1, "call": 'get_minimal_context(task="debug: <question>")'},
]}}


def test_every_rule_the_manifest_quotes_is_shown_verbatim():
    above, _ = prompt.split(prompt.build(RULED, QUESTION, TOKENS, STEPS))
    for rule in RULED["prescribed_workflow"]["rules"]:
        assert rule["quote"] in above


def test_the_rules_are_shown_before_the_first_step():
    above, _ = prompt.split(prompt.build(RULED, QUESTION, TOKENS, STEPS))
    assert above.index("ALWAYS call") < above.index("## Step 1")


def test_a_manifest_without_rules_shows_no_rules_heading():
    above, _ = prompt.split(prompt.build(CALL_STEPS, QUESTION, TOKENS, STEPS))
    assert "## Its rules" not in above
