"""Derive the tools a runner is offered from the vendor's grammar and the server's schema."""

from __future__ import annotations

from pathlib import Path

ACT = "act"
STOP = "stop"

QUOTE_DESCRIPTION = (
    "The sentence of the vendor's own documentation that authorises this call, copied verbatim."
)

ACT_TEMPLATE = (
    "Run the vendor package once. `argv` is the whole command line and `argv[0]` must be "
    "exactly `{launcher}`. The subcommands it accepts are: {subcommands}. The harness "
    "refuses any call the vendor's grammar does not allow."
)

STOP_DESCRIPTION = (
    "End the run by naming the one place that answers the question."
)


class ToolsError(Exception):
    """Raised when the vendor's grammar and the server's listing do not agree."""


def _quote_property() -> dict:
    return {"type": "string", "description": QUOTE_DESCRIPTION}


def _schema(properties: dict, required: list[str]) -> dict:
    # inv: the runner may send no key the grammar was not written for, so the schema closes
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def act_tool(invocation: dict) -> dict:
    """Return the definition of the tool that runs the vendor package.

    Parameters
    ----------
    invocation : dict
        The system's `invocation` block, whose launcher and subcommands the description
        states.

    Returns
    -------
    dict
        The tool definition.
    """
    # why: a runner that has to guess the command's name spends its whole budget on refusals,
    # and the measurement then reads as the vendor's failure rather than the harness's silence
    launcher = Path(str((invocation.get("package") or {}).get("launcher") or "")).name
    allowed = sorted(invocation.get("subcommands") or {})
    described = ", ".join(f"`{name}`" for name in allowed) or "none"
    properties = {"argv": {"type": "array", "items": {"type": "string"}}, "quote": _quote_property()}
    return {"name": ACT, "description": ACT_TEMPLATE.format(launcher=launcher, subcommands=described),
            "input_schema": _schema(properties, ["argv", "quote"])}


def stop_tool() -> dict:
    """Return the definition of the tool that ends a run with the place it answers with."""
    properties = {
        "path": {"type": "string", "description": "Path of the place, as the vendor printed it."},
        "symbol": {"type": "string", "description": "Name of the function or method."},
        "start": {"type": "integer", "description": "First line of the place."},
    }
    return {"name": STOP, "description": STOP_DESCRIPTION,
            "input_schema": _schema(properties, ["path", "symbol", "start"])}


def derived(name: str, row: dict, server: dict) -> dict:
    """Return one runner tool, narrowed from the server's schema by the vendor's grammar.

    Parameters
    ----------
    name : str
        The tool's name, as the server offers it.
    row : dict
        The `invocation.tools` entry: its `keys` and their value bounds.
    server : dict
        The server's own listing for this tool: `description` and `inputSchema`.

    Returns
    -------
    dict
        A tool definition whose properties are the keys both sides know, plus `quote`.
    """
    schema = server.get("inputSchema") or {}
    offered_properties = schema.get("properties") or {}
    keys = row.get("keys") or {}
    properties: dict = {}
    # inv: the intersection runs both ways -- a key the server does not offer cannot be sent,
    # and a property the grammar never vetted must not be reachable
    for key, spec in keys.items():
        if key not in offered_properties:
            continue
        prop = dict(offered_properties[key])
        if "literal" in (spec or {}):
            prop["enum"] = list(spec["literal"])
        properties[key] = prop
    properties["quote"] = _quote_property()
    required = [key for key in (schema.get("required") or []) if key in properties]
    return {"name": name, "description": server.get("description", ""),
            "input_schema": _schema(properties, [*required, "quote"])}


def offered(invocation: dict, server_schemas: dict[str, dict]) -> list[dict]:
    """Return every tool the runner may call, in a fixed order.

    Parameters
    ----------
    invocation : dict
        The system's `invocation` block: its `tools` and, when set, `harness_only`.
    server_schemas : dict of str to dict
        The server's own listing, keyed by tool name.

    Returns
    -------
    list of dict
        `act`, then each derived tool by name, then `stop`.

    Raises
    ------
    ToolsError
        When a tool the grammar offers the runner is absent from the server's listing.
    """
    withheld = set(invocation.get("harness_only") or [])
    rows = invocation.get("tools") or {}
    wanted = set(rows) - withheld
    # inv: a tool the grammar vetted and the server does not serve is a mismatch between the
    # pinned grammar and the installed package; offering the rest would weaken the run in silence
    missing = sorted(wanted - set(server_schemas))
    if missing:
        raise ToolsError(f"the server offers no such tool: {', '.join(missing)}")
    # inv: the order is fixed by name, so two runs of one question send the same tool list
    names = sorted(wanted)
    return [act_tool(invocation), *(derived(name, rows[name], server_schemas[name]) for name in names), stop_tool()]
