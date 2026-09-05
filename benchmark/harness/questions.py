"""Author questions under a blindness refusal, and gate them on a written review."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import TYPE_CHECKING

import yaml

from benchmark.harness import blind, config, expand, rules

if TYPE_CHECKING:
    from pathlib import Path

# why: graphify's vocabulary step selects at most 12 tokens (its query.md), and a question's
# expansion block must never claim more than the system that reads it can use
MAX_TOKENS = 12

# why: an id is `q` followed by digits, and the trailing run of digits is parsed rather than
# trusted to sort correctly, because `q10` sorts before `q9` as a string
_QID_NUMBER = re.compile(r"(\d+)$")

# why: a reviewer is handed the request file -- the question, the reference place with the
# author's own why, and this paragraph -- plus read access to the snapshot's source/, and nothing
# else, so the whole review protocol lives in one fixed sentence set rather than in an evolving
# set of ad hoc instructions
REVIEW_INSTRUCTION = (
    "You are reviewing one question before any retrieval system is allowed to see it. Read "
    "nothing but this file and the snapshot's source/ tree -- not questions/, references/, "
    "record/ or systems/, and do not run or consult any retrieval system. Decide two things: "
    "reference_is_right, whether the place named under `reference` answers `question` exactly as "
    "it is written; and question_is_ambiguous, whether some other place in source/ answers the "
    "same question equally well. Then write questions/review/<qid>.yaml as a YAML mapping with "
    "exactly these six keys: reference_is_right (bool), question_is_ambiguous (bool), note (a "
    "string explaining a false or true verdict, or null when both verdicts need no explanation), "
    "reviewer_model (the name of the model that judged this), question_sha256 and reference_sha256 "
    "(copied verbatim from this request). Write no other file."
)


def _require_file(path: Path) -> Path:
    """Return `path`, or raise `SystemExit` naming it when it is not a file."""
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    return path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidates(candidates_dir: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    by_n: dict[int, dict] = {}
    for path in sorted(candidates_dir.glob("authored-*.jsonl")):
        for row in _jsonl(path):
            by_n[row["n"]] = row
    cands = {row["n"]: row for row in _jsonl(_require_file(candidates_dir / "candidates.jsonl"))}
    return by_n, cands


def _highest_question_number(benchmark: Path) -> int:
    """Return the highest numeric id among every question under every snapshot, or zero."""
    numbers = [int(m.group(1)) for qid in config.question_ids(benchmark)
               if (m := _QID_NUMBER.search(qid))]
    return max(numbers, default=0)


def author(benchmark: Path, snapshot: str) -> int:
    """Write `questions/<qid>.yaml` and `references/<qid>.yaml` from a snapshot's candidates.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    snapshot : str
        The snapshot whose `questions/candidates/` holds the authored candidates.

    Returns
    -------
    int
        Zero.

    Raises
    ------
    config.ConfigError
        When `snapshot` is not a bare identifier, or when one question id is held by more than
        one snapshot.
    SystemExit
        Naming a missing candidates or graph file; at the first candidate whose authored text
        or expansion leaks its own reference; or naming every existing file whose bytes would
        change. Nothing is written in any of the three cases.
    """
    snap_dir = config.snapshot_dir(benchmark, snapshot)
    candidates_dir = snap_dir / "questions" / "candidates"
    by_n, cands = _candidates(candidates_dir)
    graph_json = _require_file(snap_dir / "indexes" / "graphify" / "graph.json")
    vocab = expand.vocabulary(graph_json)

    kept_ns = [n for n in sorted(by_n) if by_n[n]["verdict"] == "keep"]
    skipped = [(n, by_n[n]["reason"]) for n in sorted(by_n) if by_n[n]["verdict"] != "keep"]
    # why: a number already committed for this snapshot is reused positionally -- by n order,
    # the same order a fresh pass re-derives -- so re-authoring an unchanged candidate set never
    # renumbers a file every recorded row pins by hash
    existing = sorted(int(m.group(1)) for p in (snap_dir / "questions").glob("q*.yaml")
                      if (m := _QID_NUMBER.search(p.stem)))
    # why: a candidate beyond this snapshot's own committed count draws a number past every
    # snapshot's highest, so two snapshots never share an id and run_id, which does not carry
    # the snapshot, never collides in the ledger
    next_new = _highest_question_number(benchmark) + 1

    kept: list[tuple[str, dict, dict]] = []
    for position, n in enumerate(kept_ns):
        row, cand = by_n[n], cands[n]
        if position < len(existing):
            number = existing[position]
        else:
            number = next_new
            next_new += 1
        qid = f"q{number:03d}"
        text = row["question"]
        tokens = expand.expand(text, vocab, MAX_TOKENS)
        block = expand.expansion_block(text, vocab, MAX_TOKENS)
        question = {"id": qid, "snapshot": snapshot, "text": text,
                    "rule": expand.MECHANICAL, "expansion": block}
        reference = {"id": qid, "places": [{
            "path": cand["path"], "symbol": cand["bare"], "qualified_name": cand["fqname"],
            "start": cand["start"], "end": cand["end"], "why": row["why"]}]}
        authored = text + "\n" + " ".join(tokens["graphify"]) + "\n" + " ".join(tokens["code-review-graph"])
        found = blind.violations(authored, [], reference)
        if found:
            raise SystemExit(f"{qid} (candidate n={n}) leaks its own reference: {found}")
        kept.append((qid, question, reference))

    pending = [(snap_dir / directory / f"{qid}.yaml",
                yaml.safe_dump(body, sort_keys=False, allow_unicode=True).encode("utf-8"))
               for qid, question, reference in kept
               for directory, body in (("questions", question), ("references", reference))]
    # inv: every recorded row pins its question by question_sha256, so a file whose bytes would
    # change is refused here and nothing at all is written; an identical file is left alone
    differing = [path for path, body in pending if path.is_file() and path.read_bytes() != body]
    if differing:
        raise SystemExit("authoring would rewrite a committed file, which every recorded "
                         f"question_sha256 pins: {', '.join(str(p) for p in differing)}")
    (snap_dir / "questions").mkdir(exist_ok=True)
    (snap_dir / "references").mkdir(exist_ok=True)
    fresh = [(path, body) for path, body in pending if not path.is_file()]
    for path, body in fresh:
        path.write_bytes(body)

    print(f"files written: {len(fresh)}; already present: {len(pending) - len(fresh)}; "
          f"skipped by the author: {len(skipped)}")
    return 0


def review_request(benchmark: Path, qid: str) -> int:
    """Write `questions/review/<qid>.request.yaml`, the one document handed to a reviewer.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    qid : str
        The question to request a review for.

    Returns
    -------
    int
        Zero.

    Raises
    ------
    SystemExit
        Naming the id when no snapshot holds the question, and naming the reference file
        when it does not exist.
    """
    try:
        qpath = config.question_path(benchmark, qid)
    except config.ConfigError as exc:
        # why: the verb is a command line, so an unresolvable id leaves as one named line
        # rather than a traceback the operator has to read the resolver out of
        raise SystemExit(str(exc)) from exc
    rpath = _require_file(config.reference_path(benchmark, qid))
    question = yaml.safe_load(qpath.read_text(encoding="utf-8"))
    reference = yaml.safe_load(rpath.read_text(encoding="utf-8"))
    request = {
        "qid": qid,
        "snapshot": question["snapshot"],
        "question": question["text"],
        "question_sha256": rules.sha256_file(qpath),
        "reference": reference,
        "reference_sha256": rules.sha256_file(rpath),
        "instruction": REVIEW_INSTRUCTION,
    }
    out_dir = config.review_path(benchmark, qid).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{qid}.request.yaml").write_text(
        yaml.safe_dump(request, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return 0


def check_review(benchmark: Path, qid: str) -> str | None:
    """Return why `qid` cannot run yet, or None when its review passes.

    Parameters
    ----------
    benchmark : Path
        The `benchmark/` directory.
    qid : str
        The question whose review is checked.

    Returns
    -------
    str or None
        A one-line refusal message naming `qid` -- including the resolver's own when no
        snapshot holds it -- or None when the review file exists,
        carries a non-empty `reviewer_model`, and its two hashes match both the request
        file (when one exists) and the committed question and reference files as they
        stand now.
    """
    try:
        review_path = config.review_path(benchmark, qid)
    except config.ConfigError as exc:
        # why: this gate answers one question -- may this id run -- so an id no snapshot holds
        # is refused here with its own reason instead of raising past every caller of the gate
        return str(exc)
    if not review_path.is_file():
        return f"question {qid} has no review; a question runs only after review"
    review = yaml.safe_load(review_path.read_text(encoding="utf-8")) or {}
    live_question = rules.sha256_file(config.question_path(benchmark, qid))
    live_reference = rules.sha256_file(config.reference_path(benchmark, qid))
    request_path = review_path.parent / f"{qid}.request.yaml"
    if request_path.is_file():
        request = yaml.safe_load(request_path.read_text(encoding="utf-8")) or {}
        expected_question, expected_reference = request.get("question_sha256"), request.get("reference_sha256")
        # inv: a request frozen before either committed file changed underneath it must still
        # fail here, or an edit made after a passing review would slip past this gate
        if expected_question != live_question or expected_reference != live_reference:
            return f"question {qid} review hashes do not match the question or reference"
    else:
        expected_question, expected_reference = live_question, live_reference
    if review.get("question_sha256") != expected_question or review.get("reference_sha256") != expected_reference:
        return f"question {qid} review hashes do not match the question or reference"
    reviewer_model = review.get("reviewer_model")
    if not reviewer_model or not isinstance(reviewer_model, str):
        return f"question {qid} review lacks reviewer_model"
    return None


def admitted(benchmark: Path, qid: str) -> bool:
    """Return whether `qid` passes review and was not withdrawn by it.

    Parameters
    ----------
    benchmark : Path
        Root holding `record/snapshots/`.
    qid : str
        The question to test.

    Returns
    -------
    bool
        False when `check_review` refuses the question, or when its review sets
        `reference_is_right` to false or `question_is_ambiguous` to true.
    """
    if check_review(benchmark, qid) is not None:
        return False
    review = yaml.safe_load(config.review_path(benchmark, qid).read_text(encoding="utf-8")) or {}
    return not (review.get("reference_is_right") is False or review.get("question_is_ambiguous") is True)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmark.harness questions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("author")
    a.add_argument("snapshot")
    for name in ("review-request", "review-check"):
        sub.add_parser(name).add_argument("qid")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Dispatch the `questions` subcommand and return the process exit code.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments, or None to use `sys.argv`.

    Returns
    -------
    int
        The subcommand's exit status.
    """
    args = _parser().parse_args(argv)
    # inv: questions is an instrument verb -- it writes questions/, references/ and
    # questions/review/ of the tree this package lives in, and of no other
    benchmark = rules._ROOT
    try:
        if args.cmd == "author":
            code = author(benchmark, args.snapshot)
        elif args.cmd == "review-request":
            code = review_request(benchmark, args.qid)
        elif args.cmd == "review-check":
            refusal = check_review(benchmark, args.qid)
            code = 0
            if refusal is not None:
                print(refusal, file=sys.stderr)
                code = 1
        else:
            _parser().error(f"unknown questions command: {args.cmd}")
    except config.ConfigError as exc:
        # why: every verb here is a command line, so an id the resolver cannot place reaches the
        # operator as one named line rather than as a traceback
        print(str(exc), file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
