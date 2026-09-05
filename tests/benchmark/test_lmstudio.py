import json

import pytest

from benchmark.harness.backends import lmstudio

TOOLS = [{"name": "act", "description": "Run once.",
          "input_schema": {"type": "object", "properties": {"argv": {"type": "array"}},
                           "required": ["argv"], "additionalProperties": False}}]

REQUEST = {
    "model": "qwen3.8-27b", "max_tokens": 4096, "thinking": {"type": "adaptive"},
    "output_config": {"effort": "high"}, "tools": TOOLS,
    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    "messages": [{"role": "user", "content": "the prompt"}],
}


def test_a_tool_keeps_its_schema_under_the_shape_this_server_speaks():
    sent = lmstudio.to_chat(REQUEST)
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "act"
    assert sent["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]


def test_the_first_message_passes_through_as_it_is():
    assert lmstudio.to_chat(REQUEST)["messages"] == [{"role": "user", "content": "the prompt"}]


def test_what_this_server_has_no_word_for_is_not_sent():
    sent = lmstudio.to_chat(REQUEST)
    assert "thinking" not in sent
    assert "output_config" not in sent


def test_an_assistant_turn_becomes_a_call_with_its_arguments_as_text():
    request = {**REQUEST, "messages": [
        {"role": "user", "content": "p"},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking aloud"},
                                          {"type": "tool_use", "id": "u1", "name": "act",
                                           "input": {"argv": ["g", "query"]}}]},
    ]}
    message = lmstudio.to_chat(request)["messages"][1]
    assert message["role"] == "assistant"
    assert message["content"] == "thinking aloud"
    assert message["tool_calls"][0]["id"] == "u1"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"argv": ["g", "query"]}


def test_a_result_becomes_its_own_message_keyed_by_the_call_it_answers():
    request = {**REQUEST, "messages": [
        {"role": "user", "content": "p"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1",
                                      "content": "NODE thing", "is_error": False}]},
    ]}
    message = lmstudio.to_chat(request)["messages"][1]
    assert message == {"role": "tool", "tool_call_id": "u1", "content": "NODE thing"}


def test_a_failed_call_says_so_in_words_since_this_shape_has_no_flag_for_it():
    request = {**REQUEST, "messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1",
                                      "content": "flag is rejected", "is_error": True}]},
    ]}
    assert lmstudio.to_chat(request)["messages"][0]["content"].startswith(lmstudio.ERROR_MARK)


def test_two_results_in_one_turn_become_two_messages():
    request = {**REQUEST, "messages": [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "u1", "content": "a", "is_error": False},
            {"type": "tool_result", "tool_use_id": "u2", "content": "b", "is_error": True}]},
    ]}
    assert [m["tool_call_id"] for m in lmstudio.to_chat(request)["messages"]] == ["u1", "u2"]


def test_a_call_comes_back_as_a_call_with_its_arguments_read():
    reply = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "act", "arguments": '{"argv": ["g"], "quote": "q"}'}}]}}],
        "usage": {"prompt_tokens": 430, "completion_tokens": 86}}
    turn = lmstudio.from_chat(reply)
    assert turn["stop_reason"] == "tool_use"
    assert turn["content"][0] == {"type": "tool_use", "id": "c1", "name": "act",
                                  "input": {"argv": ["g"], "quote": "q"}}


def test_a_reply_in_words_comes_back_as_words():
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": "I think so"}}], "usage": {}}
    turn = lmstudio.from_chat(reply)
    assert turn["stop_reason"] == "end_turn"
    assert turn["content"] == [{"type": "text", "text": "I think so"}]


def test_a_turn_cut_short_is_named_the_way_the_loop_knows_it():
    reply = {"choices": [{"finish_reason": "length", "message": {"content": "half a th"}}], "usage": {}}
    assert lmstudio.from_chat(reply)["stop_reason"] == "max_tokens"


def test_what_the_turn_cost_is_carried_over_under_the_names_scoring_sums():
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
             "usage": {"prompt_tokens": 430, "completion_tokens": 86}}
    assert lmstudio.from_chat(reply)["usage"] == {"input_tokens": 430, "output_tokens": 86}


def test_the_reply_names_the_model_that_actually_served_it():
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
             "usage": {}, "model": "qwen3-8b"}
    assert lmstudio.from_chat(reply)["model"] == "qwen3-8b"


def test_a_reply_naming_no_model_carries_none():
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}], "usage": {}}
    assert lmstudio.from_chat(reply)["model"] is None


def test_arguments_that_are_not_readable_json_stop_the_run(tmp_path):
    reply = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "act", "arguments": "{not json"}}]}}],
        "usage": {}}
    with pytest.raises(lmstudio.LocalBackendError, match="arguments"):
        lmstudio.from_chat(reply)


def test_the_backend_posts_to_the_servers_own_address_and_returns_the_turn():
    seen = {}

    def post(url: str, payload: dict) -> dict:
        seen["url"], seen["payload"] = url, payload
        return {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}], "usage": {}}

    backend = lmstudio.LocalBackend("http://localhost:1234/v1", post=post)
    turn = backend.send(REQUEST)
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert seen["payload"]["model"] == "qwen3.8-27b"
    assert turn["content"] == [{"type": "text", "text": "hi"}]


def test_the_local_backend_can_show_what_it_puts_on_the_wire():
    backend = lmstudio.LocalBackend()
    assert backend.as_sent(REQUEST) == lmstudio.to_chat(REQUEST)
