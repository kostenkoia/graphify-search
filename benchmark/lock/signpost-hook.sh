#!/bin/bash
# PreToolUse signpost: tells an agent where the key is instead of "Operation not permitted". Carries no weight.
input="$(cat)"
path="$(printf '%s' "$input" | /usr/bin/jq -r '.tool_input.file_path // empty')"
[ -n "$path" ] || exit 0
case "$path" in
  */benchmark/harness/*|*/benchmark/systems/*|*/benchmark/record/snapshots/*/questions/*|*/benchmark/record/snapshots/*/references/*|*/benchmark/record/snapshots/known_transitions.yaml|*/benchmark/record/snapshots/*/meta.yaml|*/benchmark/record/snapshots/*/fileset.sha256|*/benchmark/record/snapshots/*/symbols.sha256|*/benchmark/record/snapshots/*/indexes/*/build.yaml|*/benchmark/INSTRUMENT.yaml|*/benchmark/PROTOCOL.md|*/tests/benchmark/*)
    printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"benchmark instrument is LOCKED. The owner opens it in a terminal: sudo benchmark/lock/unlock \"why\" — and closes it after the change is committed: sudo benchmark/lock/lock. Runs are refused while it is open."}}' ;;
  */benchmark/record/*)
    printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"benchmark record is never written by hand. Only prepare, drive, score, collect and summary write it."}}' ;;
esac
exit 0
