"""Draft and render one snapshot's report: a template whose only numbers are `{{placeholders}}`."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple

from benchmark.harness import config, ledger, summary

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[1]
REPORTS = Path("record") / "reports"
SNAPSHOTS = Path("record") / "snapshots"

_PLACEHOLDER = re.compile(r"\{\{([^{}]*)\}\}")
_BACKTICK = re.compile(r"`([^`]*)`")
# why: en dash joins a `q<a>–q<b>` range in prose the way an ASCII hyphen does; both are the
# only separators the spec's range literal recognizes
_RANGE_SEPARATORS = ("-", "–")
# why: an id is at least seven characters long, git's default abbreviation, and carries at least
# one a-f; a shorter run and an all-decimal run are both read as figures, so an abbreviation that
# happens to be all digits is refused with them rather than admitted
_HEX_RUN = re.compile(r"^(?=[0-9a-f]*[a-f])[0-9a-f]{7,}$")
# NOT DERIVED: enough of the sentence on each side of the offending character to find it
_DIGIT_WINDOW_RADIUS = 20

# why: the exact number-word forms are the spec's own closed list of what reads as a figure in
# prose; a word outside it (like a name that happens to end in "-teen") is not refused
_NUMBER_WORDS: frozenset[str] = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "dozen", "half", "third", "quarter", "twice", "thrice", "double",
    "triple", "percent",
})
_NUMBER_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)

_COLUMN_NAMES = frozenset(summary.COLUMNS)


class _Literals(NamedTuple):
    """The closed sets a backticked digit run is checked against."""

    q_stems: frozenset[str]
    systems: frozenset[str]
    configurations: frozenset[str]
    version_clis: frozenset[str]
    models: frozenset[str]
    model_served: frozenset[str]
    run_ids: frozenset[str]


class _Placeholder(NamedTuple):
    """One `{{...}}` span's parsed fields."""

    column: str
    system: str
    configuration: str
    round_: str
    qualifier: str | None
    partial: bool


def _q_stems(benchmark: Path) -> frozenset[str]:
    return frozenset(config.question_ids(benchmark))


def _system_dirs(benchmark: Path) -> frozenset[str]:
    root = benchmark / "systems"
    if not root.is_dir():
        return frozenset()
    return frozenset(p.name for p in root.iterdir() if p.is_dir())


def _harness_docs(benchmark: Path) -> list[config.Harness]:
    root = benchmark / "systems"
    if not root.is_dir():
        return []
    names = sorted(p.parent.name for p in root.glob("*/harness.yaml"))
    return [config.load_harness(benchmark, name) for name in names]


def _configuration_keys(harnesses: list[config.Harness]) -> frozenset[str]:
    return frozenset(key for h in harnesses for key in h.configurations)


def _version_clis(harnesses: list[config.Harness]) -> frozenset[str]:
    out: set[str] = set()
    for h in harnesses:
        cli = (h.raw.get("version") or {}).get("cli")
        if cli is not None:
            out.add(str(cli))
    return frozenset(out)


def _model_names(harnesses: list[config.Harness]) -> frozenset[str]:
    """Return every `harness.yaml` `models:` key, `models--` prefix stripped."""
    prefix = "models--"
    return frozenset(
        key[len(prefix):] if key.startswith(prefix) else key
        for h in harnesses for key in h.models)


def _ledger_model_served(benchmark: Path) -> frozenset[str]:
    return frozenset(r["model_served"] for r in ledger.rows(benchmark) if r.get("model_served"))


def _ledger_run_ids(benchmark: Path) -> frozenset[str]:
    return frozenset(r["run_id"] for r in ledger.rows(benchmark) if r.get("run_id"))


def _literals(benchmark: Path) -> _Literals:
    """Collect the closed sets a backticked digit run may name.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/snapshots/`, `systems/` and `record/attempts.jsonl`.

    Returns
    -------
    _Literals
        Every class of the spec's literal set, as it stands on `benchmark`.
    """
    harnesses = _harness_docs(benchmark)
    return _Literals(
        q_stems=_q_stems(benchmark),
        systems=_system_dirs(benchmark),
        configurations=_configuration_keys(harnesses),
        version_clis=_version_clis(harnesses),
        models=_model_names(harnesses),
        model_served=_ledger_model_served(benchmark),
        run_ids=_ledger_run_ids(benchmark),
    )


def _is_range(content: str, q_stems: frozenset[str]) -> bool:
    for sep in _RANGE_SEPARATORS:
        if sep in content:
            parts = content.split(sep)
            if len(parts) == 2 and all(p in q_stems for p in parts):
                return True
    return False


def _is_literal(content: str, lit: _Literals) -> bool:
    """Return whether a backticked digit run names one member of the closed literal set."""
    known = (content in lit.q_stems or content in lit.systems or content in lit.configurations
            or content in lit.version_clis or content in lit.models
            or content in lit.model_served or content in lit.run_ids)
    if known:
        return True
    if _HEX_RUN.fullmatch(content):
        return True
    return _is_range(content, lit.q_stems)


def _strip_placeholders(template: str) -> str:
    return _PLACEHOLDER.sub("", template)


def _segments(text: str) -> list[tuple[bool, str]]:
    """Split `text` into alternating (is_backtick, content) pieces, delimiters dropped."""
    out: list[tuple[bool, str]] = []
    pos = 0
    for m in _BACKTICK.finditer(text):
        if m.start() > pos:
            out.append((False, text[pos:m.start()]))
        out.append((True, m.group(1)))
        pos = m.end()
    if pos < len(text):
        out.append((False, text[pos:]))
    return out


def _first_digit_index(content: str) -> int | None:
    for i, ch in enumerate(content):
        if unicodedata.category(ch) in ("Nd", "No"):
            return i
    return None


def _digit_window(content: str, index: int) -> str:
    """Return `content` sliced to `_DIGIT_WINDOW_RADIUS` characters either side of `index`."""
    start = max(0, index - _DIGIT_WINDOW_RADIUS)
    end = min(len(content), index + _DIGIT_WINDOW_RADIUS + 1)
    return content[start:index] + "»" + content[index] + "«" + content[index + 1:end]


def _check_prose(template: str, lit: _Literals) -> list[str]:
    """Return every digit/number-word refusal in `template`, `{{...}}` spans exempt."""
    problems: list[str] = []
    for is_backtick, content in _segments(_strip_placeholders(template)):
        if is_backtick:
            has_digit_or_numeral = any(unicodedata.category(ch) in ("Nd", "No") for ch in content)
            if has_digit_or_numeral and not _is_literal(content, lit):
                problems.append(f"backtick digit not in the literal set: `{content}`")
            continue
        index = _first_digit_index(content)
        if index is not None:
            problems.append("a digit or numeral character appears outside backticks: "
                            f"{_digit_window(content, index)!r}")
        for m in _NUMBER_WORD_RE.finditer(content):
            problems.append(f"a number word appears outside backticks: {m.group(0)!r}")
    return problems


def _parse_placeholder(inner: str) -> _Placeholder | None:
    """Return `inner`'s parsed fields, or None when its grammar is malformed."""
    tokens = [t for t in inner.split(" ") if t]
    if len(tokens) < 4:
        return None
    column, system, configuration, round_, *rest = tokens
    qualifier: str | None = None
    partial = False
    for tok in rest:
        if tok.startswith("@") and qualifier is None:
            qualifier = tok[1:]
        elif tok == "partial" and not partial:
            partial = True
        else:
            return None
    return _Placeholder(column, system, configuration, round_, qualifier, partial)


def _match_groups(groups: list[dict], snapshot: str, placeholder: _Placeholder) -> list[dict]:
    """Return every group of `snapshot` `placeholder` names, narrowed by its `@`/`/` qualifier."""
    matches = [g for g in groups if g["snapshot"] == snapshot and g["system"] == placeholder.system
              and g["configuration"] == placeholder.configuration and g["round"] == placeholder.round_]
    if placeholder.qualifier is None:
        return matches
    seal_part, _, recipe_part = placeholder.qualifier.partition("/")
    if seal_part:
        matches = [g for g in matches if isinstance(g["instrument_sha256"], str)
                  and g["instrument_sha256"].startswith(seal_part)]
    if recipe_part:
        matches = [g for g in matches if isinstance(g["harness_sha256"], str)
                  and g["harness_sha256"].startswith(recipe_part)]
    return matches


def _ambiguous_message(span: str, matches: list[dict]) -> str:
    """Return the ambiguous-match refusal, naming a fix only when one could narrow `matches`."""
    seals = {m["instrument_sha256"] for m in matches}
    recipes = {m["harness_sha256"] for m in matches}
    if len(seals) > 1 or len(recipes) > 1:
        return f"ambiguous match, {len(matches)} groups: {span} (add @<seal8> or /<recipe8>)"
    # inv: the group key holds snapshot, system, configuration, recipe and seal, so two matches
    # agreeing on all five are one group and this sentence cannot be reached; it stands for the
    # day the key loses a field
    return f"ambiguous match, {len(matches)} groups: {span} (the groups are indistinguishable)"


def _check_placeholders(template: str, snapshot: str, groups: list[dict]) -> list[str]:
    """Return every unknown-column, unknown-group, ambiguous-match or missing-partial refusal."""
    problems: list[str] = []
    for m in _PLACEHOLDER.finditer(template):
        span = m.group(0)
        placeholder = _parse_placeholder(m.group(1))
        if placeholder is None:
            problems.append(f"malformed placeholder: {span}")
            continue
        if placeholder.column not in _COLUMN_NAMES:
            problems.append(f"unknown column {placeholder.column!r}: {span}")
            continue
        matches = _match_groups(groups, snapshot, placeholder)
        if not matches:
            problems.append(f"unknown group under {snapshot}: {span}")
            continue
        if len(matches) > 1:
            problems.append(_ambiguous_message(span, matches))
            continue
        group = matches[0]
        if group["questions"] < group["sealed_questions"] and not placeholder.partial:
            problems.append(f"{span} is partial ({group['questions']} of {group['sealed_questions']} "
                            "questions); add partial")
        if group[placeholder.column] is None:
            problems.append(f"column {placeholder.column!r} is null for {span}")
    return problems


def _check_with_groups(template: str, snapshot: str, lit: _Literals, groups: list[dict]) -> list[str]:
    """Return `check`'s refusals, given `lit` and `groups` computed once by the caller."""
    return _check_prose(template, lit) + _check_placeholders(template, snapshot, groups)


def check(benchmark: Path, snapshot: str, template: str) -> list[str]:
    """Return every reason `template` is refused; empty when it passes.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/snapshots/`, `systems/` and `record/attempts.jsonl`.
    snapshot : str
        The snapshot every placeholder's group is drawn from.
    template : str
        The report template, `{{placeholders}}` and backtick spans included.

    Returns
    -------
    list of str
        One line per refusal, one entry per occurrence; empty when the template passes.
    """
    lit = _literals(benchmark)
    groups = summary.summarize(benchmark)["groups"]
    return _check_with_groups(template, snapshot, lit, groups)


def _render_value(group: dict, placeholder: _Placeholder) -> str:
    value = group[placeholder.column]
    # inv: a null column is refused by _check_placeholders before anything is substituted, so
    # the empty string cannot be reached; it stands for the day that refusal is narrowed
    text = "" if value is None else str(value)
    if placeholder.partial:
        return f"{text} of {group['questions']} run of {group['sealed_questions']}"
    return text


def render(benchmark: Path, snapshot: str, template: str) -> str:
    """Check `template`, then substitute every placeholder with its committed figure.

    Parameters
    ----------
    benchmark : Path
        Root `check` reads figures and literals from.
    snapshot : str
        The snapshot every placeholder's group is drawn from.
    template : str
        The report template.

    Returns
    -------
    str
        `template` with every `{{placeholder}}` replaced by the value `summary.summarize`
        computes for it.

    Raises
    ------
    SystemExit
        Joining every refusal `check` finds, one per line; nothing is substituted.
    """
    lit = _literals(benchmark)
    groups = summary.summarize(benchmark)["groups"]
    problems = _check_with_groups(template, snapshot, lit, groups)
    if problems:
        raise SystemExit("\n".join(problems))

    def substitute(m: re.Match[str]) -> str:
        placeholder = _parse_placeholder(m.group(1))
        if placeholder is None:
            raise RuntimeError(f"malformed placeholder survived check: {m.group(0)}")
        group = _match_groups(groups, snapshot, placeholder)[0]
        return _render_value(group, placeholder)

    return _PLACEHOLDER.sub(substitute, template)


def _require_snapshot(benchmark: Path, snapshot: str) -> None:
    # inv: the snapshot is one bare directory name under record/snapshots/, so a separator or a
    # leading dot is refused before the join that would otherwise reach outside record/snapshots/
    # and write the draft under a name of the caller's choosing
    if "/" in snapshot or snapshot.startswith("."):
        raise SystemExit(f"{snapshot} is not a snapshot name; it names one directory "
                         f"under benchmark/{SNAPSHOTS.as_posix()}/")
    if not (benchmark / SNAPSHOTS / snapshot).is_dir():
        raise SystemExit(f"no snapshot {snapshot}; report needs "
                         f"benchmark/{SNAPSHOTS.as_posix()}/{snapshot}/")


def _draft(benchmark: Path, snapshot: str) -> Path:
    _require_snapshot(benchmark, snapshot)
    template = sys.stdin.read()
    problems = check(benchmark, snapshot, template)
    if problems:
        raise SystemExit("\n".join(problems))
    out_dir = benchmark / REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{snapshot}.md.in"
    out.write_text(template, encoding="utf-8")
    return out


def _render_snapshot(benchmark: Path, snapshot: str) -> Path:
    _require_snapshot(benchmark, snapshot)
    in_path = benchmark / REPORTS / f"{snapshot}.md.in"
    if not in_path.is_file():
        raise SystemExit(f"no draft for {snapshot}; run report draft first")
    template = in_path.read_text(encoding="utf-8")
    rendered = render(benchmark, snapshot, template)
    out_dir = benchmark / REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{snapshot}.md"
    out.write_text(rendered, encoding="utf-8")
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark.harness report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("draft", "render"):
        sub.add_parser(name).add_argument("snapshot")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Dispatch `draft` or `render` and return the process exit code.

    Parameters
    ----------
    argv : list of str or None
        `draft <snapshot>` or `render <snapshot>`; None reads `sys.argv`.

    Returns
    -------
    int
        Zero.

    Raises
    ------
    SystemExit
        Joining every refusal `check` finds, one per line; nothing is written.
    """
    args = _parser().parse_args(argv)
    if args.cmd == "draft":
        _draft(DEFAULT_BENCHMARK, args.snapshot)
    else:
        _render_snapshot(DEFAULT_BENCHMARK, args.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
