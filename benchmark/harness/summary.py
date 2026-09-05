"""Read the ledger and the record's questions, and print the table the figures come from."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark.harness import config, ledger, questions, seal

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[1]

# why: 5 is the shortest list any shipped system prints (code-review-graph is called with the
# vendor's `limit: 5`), so a rank within it exists on every system's scale and hit_at_5 is the
# one hit figure that compares across every system; hit_at_1 marks an answer that lands first
HIT_AT_RANKS: tuple[int, ...] = (1, 5)

GROUP_KEY = ("snapshot", "system", "configuration", "harness_sha256", "instrument_sha256", "round")

_COLUMNS = (
    *GROUP_KEY,
    "sealed_questions", "questions", "attempts", "repeats", "aborted", "voided",
    "hit", "hit_at_1", "hit_at_5", "named", "hit_by_runner", "refused",
    "tokens_median", "system_calls_median", "ceiling_calls_median",
    "model_input_tokens", "model_output_tokens", "ceiling_reached", "agreement",
)
# why: report's placeholder grammar names a column by the same string this table prints it
# under, so it reuses this list rather than keeping a second one that could drift from it
COLUMNS = _COLUMNS

# NOT DERIVED: 8 hex characters keep the table's hash columns readable; render refuses a table
# whose groups collide at that length, so the abbreviation never silently hides a real difference
_SHA_DISPLAY_LEN = 8


def round_of(row: dict) -> str:
    """Return the round one ledger row belongs to.

    Parameters
    ----------
    row : dict
        One ledger row.

    Returns
    -------
    str
        `baseline` for a row with `runner: false`; `driven/local/<name>/<effort>/<max_actions>/
        <max_tokens>` for a driven row that names a `model` -- `<name>` is `model_served` once
        collect has written it, else the requested `model` itself; `driven/legacy/<driver>` for
        a driven row that carries `driver` and no `model`.

    Raises
    ------
    ValueError
        When the row carries no `runner` key.
    SystemExit
        When the row is driven but names neither a `driver` nor a `model`.
    """
    if "runner" not in row:
        raise ValueError(f"row {row.get('run_id')!r} carries no runner key")
    if not row["runner"]:
        return "baseline"
    # inv: prepare writes `model`, `effort`, `max_actions` and `max_tokens` into a driven row
    # when it opens it, before drive or collect ever runs; `model_served` is collect's own
    # addition, so a row a kill left aborted still names the model it asked for, never served
    if "model" in row:
        served = row.get("model_served") or row["model"]
        return f"driven/local/{served}/{row['effort']}/{row['max_actions']}/{row['max_tokens']}"
    if "driver" in row:
        return f"driven/legacy/{row['driver']}"
    raise SystemExit(f"row {row.get('run_id')!r} is driven but names no driver and no model")


def _review_passes(benchmark: Path, qid: str) -> bool:
    """Return whether `qid` may run under today's review, mirroring `prepare.prepare`'s gate."""
    return questions.admitted(benchmark, qid)


def _sealed_questions(benchmark: Path, snapshot: str, cache: dict[str, int]) -> int:
    """Return how many recorded questions name `snapshot` and pass review, cached by snapshot."""
    if snapshot not in cache:
        count = 0
        for qid in config.question_ids(benchmark):
            question = config.load_question(benchmark, qid)
            if question.get("snapshot") == snapshot and _review_passes(benchmark, qid):
                count += 1
        cache[snapshot] = count
    return cache[snapshot]


def _scored_rows(question_rows: list[dict], driven: bool) -> list[dict]:
    """Return the rows of one question, within one group, that `summary` scores.

    Parameters
    ----------
    question_rows : list of dict
        The question's completed, non-repeat rows of the group.
    driven : bool
        Whether the group's round is a driven one.

    Returns
    -------
    list of dict
        Every row for a baseline question; the single row with the lowest `attempt` for a
        driven one.
    """
    if not driven:
        return question_rows
    return [min(question_rows, key=lambda r: r["attempt"])]


def _hit_at(ordered: list[dict], hit: bool, k: int) -> bool | None:
    """Return whether every one of a question's scored rows ranked within `k`.

    Parameters
    ----------
    ordered : list of dict
        A question's scored rows, in any order.
    hit : bool
        Whether every scored row already hit.
    k : int
        The rank cut, 1 or 5.

    Returns
    -------
    bool or None
        False when the question missed, or a scored row's rank exceeds `k`; None when it hit
        but some scored row carries no rank; True when every scored row's rank is within `k`.
    """
    if not hit:
        return False
    ranks = [r.get("hit_rank") for r in ordered]
    known = [rank for rank in ranks if rank is not None]
    if len(known) != len(ranks):
        return None
    return all(rank <= k for rank in known)


def _question_summary(scored: list[dict], completed: list[dict], driven: bool) -> dict:
    """Score one question's rows the way `hit`, `hit_at_1`, `hit_at_5` and `agreement` publish them.

    Parameters
    ----------
    scored : list of dict
        The question's scored rows within one group; never empty.
    completed : list of dict
        The question's completed rows within the group, repeats included; never empty.
    driven : bool
        Whether the group's round is a driven one.

    Returns
    -------
    dict
        `attempts`, `hit`, `hit_at`, `hit_rank`, `stop_hit`, `hit_by`, `tokens`, `system_calls`,
        `ceiling_calls`, `ceiling_reached`, `agree` and the raw scored `rows`.
    """
    # inv: ordered rows sort lowest to highest attempt, so `last` is the highest-attempt scored
    # row -- a baseline question's latest retry, a driven question's only scored row -- and its
    # tokens, hit_rank and calls are what the question reports
    ordered = sorted(scored, key=lambda r: r["attempt"])
    last = ordered[-1]
    hit = all(bool(r.get("hit")) for r in ordered)
    # inv: a rank belongs to a question that hit, so a miss never publishes the rank of the one
    # attempt that happened to find it
    rank = last.get("hit_rank") if hit else None
    hit_at = {k: _hit_at(ordered, hit, k) for k in HIT_AT_RANKS}
    # why: agreement is a fact about every attempt a question spent, repeats included -- a repeat
    # that came back different is still a disagreement the question's cell has to own
    # inv: a completed row carrying no canonical list counts as disagreement, never as a silent
    # match with the rows that do carry one
    has_canonical = all(r.get("canonical") is not None for r in completed)
    canonical_lists = {tuple(r["canonical"]) for r in completed} if has_canonical else set()
    return {
        "attempts": [r["run_id"] for r in ordered],
        "hit": hit,
        "hit_at": hit_at,
        "hit_rank": rank,
        "stop_hit": last.get("stop_hit") if driven else None,
        "hit_by": last.get("hit_by") if driven else None,
        "tokens": last.get("tokens"),
        "system_calls": last.get("system_calls"),
        "ceiling_calls": last.get("ceiling_calls"),
        "ceiling_reached": any(r.get("stop") == "ceiling_reached" for r in ordered),
        "agree": has_canonical and len(canonical_lists) == 1,
        "rows": ordered,
    }


def _median_low(rows_: list[dict], field: str) -> int | None:
    values = [r[field] for r in rows_ if r.get(field) is not None]
    return statistics.median_low(values) if values else None


def _summarize_group(benchmark: Path, key: tuple, rows_: list[dict], sealed_cache: dict[str, int]) -> dict:
    snapshot, system, configuration, harness_sha256, instrument_sha256, round_ = key
    driven = round_ != "baseline"
    completed = [r for r in rows_ if r.get("outcome") == "completed"]
    repeats = [r for r in completed if r.get("repeat")]
    aborted = [r for r in rows_ if r.get("outcome") == "aborted"]
    voided = [r for r in rows_ if r.get("outcome") == "void"]
    completed_by_question: dict[str, list[dict]] = {}
    for row in completed:
        completed_by_question.setdefault(row["question"], []).append(row)
    by_question: dict[str, list[dict]] = {}
    for row in completed:
        if not row.get("repeat"):
            by_question.setdefault(row["question"], []).append(row)
    per_question = {qid: _question_summary(_scored_rows(qrows, driven), completed_by_question[qid], driven)
                    for qid, qrows in by_question.items()}
    questions_ = sorted(per_question)
    all_scored = [r for qid in questions_ for r in per_question[qid]["rows"]]
    named = hit_by_runner = refused = model_input = model_output = agreement = None
    if driven:
        named = sum(1 for qid in questions_ if per_question[qid]["stop_hit"] is True)
        hit_by_runner = sum(1 for qid in questions_ if per_question[qid]["hit_by"] == "runner")
        refused = sum(int(r.get("refused") or 0) for r in all_scored)
        usage = [r.get("model_usage") or {} for r in all_scored]
        if any(usage):
            model_input = sum(int(u.get("input_tokens") or 0) for u in usage)
            model_output = sum(int(u.get("output_tokens") or 0) for u in usage)
    else:
        agreement = sum(1 for qid in questions_ if per_question[qid]["agree"])
    return {
        "snapshot": snapshot, "system": system, "configuration": configuration,
        "harness_sha256": harness_sha256, "instrument_sha256": instrument_sha256, "round": round_,
        "sealed_questions": _sealed_questions(benchmark, snapshot, sealed_cache),
        "questions": len(questions_),
        "attempts": len(completed), "repeats": len(repeats), "aborted": len(aborted), "voided": len(voided),
        "hit": sum(1 for qid in questions_ if per_question[qid]["hit"]),
        "hit_at_1": sum(1 for qid in questions_ if per_question[qid]["hit_at"][1] is True),
        "hit_at_5": sum(1 for qid in questions_ if per_question[qid]["hit_at"][5] is True),
        "named": named, "hit_by_runner": hit_by_runner, "refused": refused,
        "tokens_median": _median_low(all_scored, "tokens"),
        "system_calls_median": _median_low(all_scored, "system_calls"),
        "ceiling_calls_median": _median_low(all_scored, "ceiling_calls"),
        "model_input_tokens": model_input, "model_output_tokens": model_output,
        "ceiling_reached": sum(1 for qid in questions_ if per_question[qid]["ceiling_reached"]),
        "agreement": agreement,
        "per_question": {qid: {
            "attempts": per_question[qid]["attempts"], "hit": per_question[qid]["hit"],
            "hit_rank": per_question[qid]["hit_rank"], "stop_hit": per_question[qid]["stop_hit"],
            "tokens": per_question[qid]["tokens"], "system_calls": per_question[qid]["system_calls"],
            "ceiling_calls": per_question[qid]["ceiling_calls"], "added_hit": None,
        } for qid in questions_},
    }


def _fill_added_hit(groups: list[dict]) -> None:
    """Set each driven question's `added_hit` against its matching baseline group, in place."""
    baseline_by_cell = {
        tuple(g[k] for k in GROUP_KEY[:5]): g for g in groups if g["round"] == "baseline"
    }
    for group in groups:
        if group["round"] == "baseline":
            continue
        base = baseline_by_cell.get(tuple(group[k] for k in GROUP_KEY[:5]))
        for qid, entry in group["per_question"].items():
            base_question = base["per_question"].get(qid) if base else None
            # why: nothing to have added to reads as an absent figure, never as a zero that
            # would claim the drive added nothing when no baseline verdict exists to compare
            entry["added_hit"] = (None if base_question is None
                                  else bool(entry["hit"] and not base_question["hit"]))


def _sort_key(group: dict) -> tuple[str, ...]:
    return tuple(str(group[k] or "") for k in GROUP_KEY)


def summarize(benchmark: Path) -> dict:
    """Compute the summary the ledger currently supports.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/` and `INSTRUMENT.yaml`.

    Returns
    -------
    dict
        `seal_reason` (the seal's reason, or None while unsealed) and `groups`, sorted by
        `(snapshot, system, configuration, harness_sha256, instrument_sha256, round)`.
    """
    snapshot_cache: dict[str, str] = {}

    def snapshot_of(qid: str) -> str:
        if qid not in snapshot_cache:
            question = config.load_question(benchmark, qid)
            snapshot_cache[qid] = config.question_snapshot(question, qid)
        return snapshot_cache[qid]

    buckets: dict[tuple, list[dict]] = {}
    for row in ledger.rows(benchmark):
        key = (snapshot_of(row["question"]), row["system"], row["configuration"],
              row.get("harness_sha256"), row.get("instrument_sha256"), round_of(row))
        buckets.setdefault(key, []).append(row)

    sealed_cache: dict[str, int] = {}
    groups = [_summarize_group(benchmark, key, rows_, sealed_cache) for key, rows_ in buckets.items()]
    _fill_added_hit(groups)
    groups.sort(key=_sort_key)
    return {"seal_reason": (seal.load(benchmark) or {}).get("reason"), "groups": groups}


def _cell(group: dict, column: str) -> str:
    value = group[column]
    if value is None:
        return ""
    if column in ("harness_sha256", "instrument_sha256") and isinstance(value, str):
        return value[:_SHA_DISPLAY_LEN]
    return str(value)


def _check_sha_prefix_collisions(groups: list[dict]) -> None:
    """Refuse to render when two groups' full hashes share an abbreviated column value.

    Parameters
    ----------
    groups : list of dict
        The groups `render` is about to print.

    Raises
    ------
    SystemExit
        Naming the column and the two full hashes that collide at `_SHA_DISPLAY_LEN`.
    """
    for column in ("harness_sha256", "instrument_sha256"):
        seen: dict[str, str] = {}
        for group in groups:
            value = group[column]
            if not isinstance(value, str):
                continue
            prefix = value[:_SHA_DISPLAY_LEN]
            clash = seen.setdefault(prefix, value)
            if clash != value:
                raise SystemExit(f"{column} values {clash} and {value} share the table prefix {prefix}")


def render(summary: dict) -> str:
    """Render `summary` as the one table `record/SUMMARY.md` publishes.

    Parameters
    ----------
    summary : dict
        What `summarize` returns.

    Returns
    -------
    str
        A title, the seal's reason, and one Markdown table row per group.

    Raises
    ------
    SystemExit
        When two groups' hashes collide at the table's abbreviation length.
    """
    _check_sha_prefix_collisions(summary["groups"])
    reason = summary["seal_reason"]
    lines = [
        "# summary", "", f"seal reason: {reason if reason is not None else ''}", "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "|".join("---" for _ in _COLUMNS) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(group, column) for column in _COLUMNS) + " |"
                for group in summary["groups"])
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="benchmark.harness summary",
                                   description="Print the one table the ledger currently supports.")


def main(argv: list[str] | None = None) -> int:
    """Compute the summary and write `record/summary.json` and `record/SUMMARY.md`.

    Parameters
    ----------
    argv : list of str or None
        Always empty; `summary` takes no arguments.

    Returns
    -------
    int
        Zero.

    Raises
    ------
    SystemExit
        When `argv` names any argument.
    """
    _parser().parse_args(argv)
    doc = summarize(DEFAULT_BENCHMARK)
    # inv: the table is rendered before either file is written, so a refusal inside render
    # leaves the two files as they were rather than a fresh json beside a stale table
    table = render(doc)
    (DEFAULT_BENCHMARK / "record" / "summary.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEFAULT_BENCHMARK / "record" / "SUMMARY.md").write_text(table, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
