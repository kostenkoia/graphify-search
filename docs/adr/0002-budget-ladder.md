---
status: "accepted"
date: 2026-08-28
decision-makers: Ilia Kostenko
---

# ADR-0002: A coarser complete answer, never a truncated one

## Context and Problem Statement

An answer has to fit inside a caller's context budget. The vendor's own convention
(`benchmark/systems/graphify/docs/query.md`) ranks first and cuts second. It orders the NODE lines
by how many of the query's terms appear in a node's label — each term counted once however often it
occurs, equally scoring nodes left in whatever order the traversal set yields — appends the EDGE
lines in traversal order after them, and then slices the joined text at `char_budget =
token_budget * 4`: 8000 characters standing in for a 2000-token budget at an assumed four
characters per token. That factor is exactly the estimate the second decision driver below
refuses: the budget is checked in characters while the caller's limit is in tokens, so what the
reader receives can sit either side of the number the cut was made against. A list cut off partway
through also loses the answer entirely, since a place near the end of the list is as likely to be
the one the reader needed as one near the start. How should an answer that is too large be brought
inside budget?

## Decision Drivers

* The reader must receive a complete list of places, even a coarser one, never a list cut off
  partway through.
* The number the budget is checked against must be the number the reader actually receives, not
  an estimate that can drift from it.

## Considered Options

* Follow the vendor's own convention: order by the count of query terms in each node's label,
  then cut the rendered text at four characters per budgeted token.
* Shrink the answer in ordered stages instead of cutting text: drop every snippet, then every set
  of edges, then the results beyond a floor, and only then admit the budget was exceeded.

## Decision Outcome

Chosen: shrink in stages — drop every `snippet`, then every `edges`, then shrink to five results,
then print anyway with `exceeded: true` — because a cut list loses the answer, while a coarser but
complete one does not. `used_chars` measures the compact `results` array alone, the same bytes the
reader receives, so the figure a rung is checked against never depends on an estimate of itself.
`dropped` names each rung that fired.

### Consequences

* Good, because the reader always gets a whole list and knows exactly what is missing.
* Bad, because a tightly budgeted answer can lose every snippet and every edge and still fall back
  to five results, well short of what the caller asked for.

### Confirmation

Argued, not measured: `used_chars` is computed by rendering the same compact `results` array the
reader receives, so the number that gates each rung is the number the reader would actually get,
by construction rather than by a separate check.

## More Information

Revisit this decision if a caller needs partial detail preserved for its top-ranked results
specifically, rather than an all-or-nothing drop applied to every result at once.
