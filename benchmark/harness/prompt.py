"""Build the prompt a runner is given, and split it at the harness heading."""

from __future__ import annotations

import re

HEADING = "# Executed by the harness before you start"

# inv: the heading must appear once, so the blind check knows exactly which part the owner wrote
MIN_FENCE = 3  # why: CommonMark's shortest code fence is three backticks

_BACKTICKS = re.compile(r"`+")

WITHHELD_NOTE = (
    "The harness does not offer this step; a call to it is refused and spends the run's budget."
)

ANSWER_RULE = (
    "Finish by calling `stop` with `path`, `symbol` and `start` naming the one place that "
    "answers the question. An answer written as prose is not read."
)


class PromptError(Exception):
    """Raised when a prompt cannot be split at the harness heading."""


def fence(text: str) -> str:
    """Return a backtick fence longer than every backtick run in `text`.

    Parameters
    ----------
    text : str
        The text the fence must enclose.

    Returns
    -------
    str
        A run of backticks, never shorter than markdown's own minimum.
    """
    longest = max((len(m.group()) for m in _BACKTICKS.finditer(text)), default=0)
    return "`" * max(MIN_FENCE, longest + 1)


def _offered(command: str, invocation: dict | None) -> bool:
    if invocation is None:
        return True
    rejected = set(invocation.get("rejected_subcommands") or [])
    # inv: a flag the grammar refuses is as unreachable as a subcommand it refuses, so both are
    # withheld; otherwise a prescribed mode would invite a call that can only be refused
    for row in (invocation.get("subcommands") or {}).values():
        rejected.update((row or {}).get("rejected") or [])
    # inv: the check reads the command's own words, because a prescribed command is prose the
    # vendor wrote, not a call the harness built
    return not any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", command) for name in rejected)


def authored(manifest: dict, question: dict, tokens: list[str], invocation: dict | None = None) -> str:
    """Return the part of the prompt the owner wrote, above the harness heading.

    Parameters
    ----------
    manifest : dict
        The system's `manifest.yaml`, whose `prescribed_workflow` is quoted.
    question : dict
        The question file, whose `text` is asked.
    tokens : list of str
        The expansion tokens of this system, empty when its recipe has no expansion step.
    invocation : dict or None
        The vendor's grammar; a prescribed command it rejects is named as unavailable
        rather than offered. None offers every command the vendor states.

    Returns
    -------
    str
        Markdown carrying the vendor's prescribed workflow, the question and the expansion.
    """
    workflow = manifest.get("prescribed_workflow") or {}
    lines = [f"# The system under measurement: {manifest['id']}", ""]
    if workflow.get("source"):
        lines += [f"Its own prescribed workflow, quoted from {workflow['source']}:", ""]
    # why: a rule the manifest quotes binds every step, so a prompt without it shows the runner a
    # workflow the vendor never wrote
    rules = [rule["quote"] for rule in (workflow.get("rules") or []) if rule.get("quote")]
    if rules:
        lines += ["## Its rules"] + [f"> {quote}" for quote in rules] + [""]
    for step in workflow.get("steps") or []:
        # inv: the two vendors write their workflows in different shapes -- one names and quotes
        # its steps, the other states a call and a note -- so every part is optional but the
        # step itself is always shown, or the runner would be given a workflow with holes in it
        heading = f"## Step {step['id']}"
        if step.get("name"):
            heading += f" — {step['name']}"
        lines.append(heading)
        if step.get("quote"):
            lines.append(f"> {step['quote']}")
        commands = (step.get("commands") or []) + [
            step[key] for key in ("command", "call") if step.get(key)]
        # why: inviting a runner into a call the grammar refuses spends its budget on a refusal
        # and measures the harness's own silence, not the vendor
        offered = [c for c in commands if _offered(c, invocation)]
        lines.extend(f"Its command: `{command}`" for command in offered)
        if commands and not offered:
            lines.append(WITHHELD_NOTE)
        if step.get("procedure"):
            # why: the procedure is the vendor's own instruction for the step; leaving it out
            # measures a workflow the runner was never shown in full
            lines.append(f"Its procedure: {str(step['procedure']).strip()}")
        modes = {k: v for k, v in (step.get("modes") or {}).items() if _offered(f"{k} {v}", invocation)}
        if modes:
            lines.append("Its modes: " + "; ".join(f"{k} — {v}" for k, v in modes.items()))
        if step.get("note"):
            lines.append(f"The vendor's note: {step['note']}")
        lines.append("")
    lines += ["# Your question", "", question["text"], ""]
    # why: a system whose recipe has no expansion step would be shown the heading over an empty
    # line, which reads as a step that ran and produced nothing rather than one that does not exist
    if tokens:
        lines += ["# The expansion the harness prepared", "", " ".join(tokens), ""]
    lines += ["# How to answer", "", ANSWER_RULE, ""]
    return "\n".join(lines)


def executed(steps: list[dict]) -> str:
    """Return one fenced block per step already executed by the harness.

    Parameters
    ----------
    steps : list of dict
        `{"name": ..., "cmd": ..., "out": ...}` per fixed step, in the order executed.

    Returns
    -------
    str
        Markdown carrying each step's command and its output.
    """
    lines: list[str] = []
    for step in steps:
        body = f"{step['cmd']}\n\n{step['out']}"
        # inv: the fence is measured over the body, so a vendor printing a fence of its own
        # cannot close the block and have the rest of its output read as prompt
        rail = fence(body)
        lines += [f"## {step['name']}", rail, body, rail, ""]
    return "\n".join(lines)


def build(manifest: dict, question: dict, tokens: list[str], steps: list[dict],
          invocation: dict | None = None) -> str:
    """Return the whole prompt: the authored part, the heading, then the executed steps.

    Parameters
    ----------
    manifest : dict
        The system's `manifest.yaml`.
    question : dict
        The question file.
    tokens : list of str
        The expansion tokens of this system, empty when its recipe has no expansion step.
    steps : list of dict
        The fixed steps already executed, in order.

    Returns
    -------
    str
        The prompt text, with the heading appearing exactly once.
    """
    return f"{authored(manifest, question, tokens, invocation)}\n{HEADING}\n\n{executed(steps)}"


def split(text: str) -> tuple[str, str]:
    """Split a prompt into the authored part and the executed part.

    Parameters
    ----------
    text : str
        A prompt built by `build`.

    Returns
    -------
    tuple of (str, str)
        The text above the heading and the text below it.

    Raises
    ------
    PromptError
        When the heading is absent, or appears more than once.
    """
    found = text.count(HEADING)
    if found == 0:
        raise PromptError("the prompt carries no harness heading")
    if found > 1:
        raise PromptError(f"the prompt carries the harness heading twice or more: {found}")
    above, _, below = text.partition(HEADING)
    return above, below
