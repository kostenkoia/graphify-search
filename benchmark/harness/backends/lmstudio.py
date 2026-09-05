"""Take assistant turns from a local server that speaks the chat-completions shape."""

from __future__ import annotations

import json
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# why: this shape has no field for a failed tool result, so the failure is said in the text
# the runner reads; dropping it would let a refusal read as an ordinary answer
ERROR_MARK = "TOOL ERROR: "

# NOT DERIVED: the owner's LM Studio address, the value graphify_search.settings.DEFAULT_ENDPOINT carries
DEFAULT_BASE_URL = "http://localhost:1234/v1"

# NOT DERIVED: a local model on a laptop answers in tens of seconds, and a first load can take
# minutes; the same bound the harness gives a vendor call is far too short here
LOCAL_TIMEOUT_S = 900

_FINISH = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}


class LocalBackendError(Exception):
    """The local server answered in a shape the harness will not guess at."""


def _tool_calls_of(blocks: list[dict]) -> list[dict]:
    return [{"id": block["id"], "type": "function",
             "function": {"name": block["name"],
                          "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)}}
            for block in blocks if block.get("type") == "tool_use"]


def _text_of(blocks: list[dict]) -> str:
    return "".join(block.get("text") or "" for block in blocks if block.get("type") == "text")


def _messages_of(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        blocks = list(content or [])
        results = [b for b in blocks if b.get("type") == "tool_result"]
        if results:
            # inv: one message per result, because this shape answers one call per message and
            # a merged answer would leave a call unanswered
            out.extend({"role": "tool", "tool_call_id": r["tool_use_id"],
                        "content": (ERROR_MARK if r.get("is_error") else "") + str(r.get("content") or "")}
                       for r in results)
            continue
        turn: dict = {"role": message["role"], "content": _text_of(blocks)}
        calls = _tool_calls_of(blocks)
        if calls:
            turn["tool_calls"] = calls
        out.append(turn)
    return out


def to_chat(request: dict) -> dict:
    """Return the request in the shape a chat-completions server speaks.

    Parameters
    ----------
    request : dict
        The request as the driver built it.

    Returns
    -------
    dict
        The same run, with tools, turns and results renamed; anything this shape has
        no word for is left out rather than invented.
    """
    payload = {
        "model": request["model"],
        "max_tokens": request.get("max_tokens"),
        "messages": _messages_of(list(request.get("messages") or [])),
        "tools": [{"type": "function",
                   "function": {"name": tool["name"], "description": tool.get("description", ""),
                                "parameters": tool.get("input_schema") or {}}}
                  for tool in request.get("tools") or []],
    }
    # inv: reasoning depth and effort are not spoken here; sending them would be the harness
    # claiming a setting the server never applied
    return {k: v for k, v in payload.items() if v is not None}


def from_chat(reply: dict) -> dict:
    """Return one chat-completions reply in the shape the driver's loop reads.

    Parameters
    ----------
    reply : dict
        The server's own answer.

    Returns
    -------
    dict
        `content`, `stop_reason`, `usage` and `model`.

    Raises
    ------
    LocalBackendError
        When a tool call's arguments are not readable as JSON.
    """
    choice = (reply.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            # why: a half-written call is not a call; guessing at it would put the harness's
            # reading of the model into the journal as the model's own argument
            raise LocalBackendError(f"tool call arguments are not JSON: {raw!r}") from exc
        content.append({"type": "tool_use", "id": call.get("id"),
                        "name": function.get("name"), "input": arguments})
    usage = reply.get("usage") or {}
    translated = {"input_tokens": usage.get("prompt_tokens"),
                  "output_tokens": usage.get("completion_tokens")}
    return {"content": content,
            "stop_reason": _FINISH.get(str(choice.get("finish_reason")), "end_turn"),
            "usage": {k: v for k, v in translated.items() if v is not None},
            "model": reply.get("model")}


def _post(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 — the base url is the operator's own, not user input
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=LOCAL_TIMEOUT_S) as response:  # noqa: S310 — the base url is the operator's own, not user input
        return json.loads(response.read().decode("utf-8"))


class LocalBackend:
    """One turn per request, from a local chat-completions server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 post: Callable[[str, dict], dict] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.post = post or _post

    def as_sent(self, request: dict) -> dict:
        """Return the request in the exact shape this backend puts on the wire."""
        return to_chat(request)

    def send(self, request: dict) -> dict:
        """Send one request to the local server and return the assistant turn."""
        return from_chat(self.post(f"{self.base_url}/chat/completions", to_chat(request)))
