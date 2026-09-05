---
status: "accepted"
date: 2026-08-28
decision-makers: Ilia Kostenko
---

# ADR-0001: Only code and document nodes are places

> **"The reference corpus"** below is the private codebase these decisions were taken
> against. It is not part of this repository, and no figure drawn from it can be recounted
> from a clone — see [benchmark/PROTOCOL.md](../../benchmark/PROTOCOL.md).

## Context and Problem Statement

A query answer must point at somewhere an agent can jump to: a path and a line. Graphify emits
several node types in one graph, and not every one of them can serve that purpose. On the
reference corpus, every docstring node starts inside a code node's span rather than at a location
of its own, and a docstring node's label is prose, so it can never satisfy a symbol reference.
Concept nodes name an idea rather than a place in a file, even where they carry a location of
their own. Which node types may become a result?

## Decision Drivers

* A result must name a location an agent can act on directly: a path and a parsable start line.
* A docstring's content should still reach the ranking, even though the docstring node itself
  cannot be a place.

## Considered Options

* Treat every graph node as a candidate place, docstring and concept nodes included.
* Restrict places to node types that carry a real, parsable location: `file_type` `code` or
  `document`.

## Decision Outcome

Chosen: results are nodes of `file_type` `code` or `document` with a parsable `L<start>`, because
a docstring node's label is prose that cannot satisfy a symbol reference and a concept node names
an idea rather than a place an agent can jump to. A docstring's text is folded into the code node
its `rationale_for` edge names, so it still informs the ranking. `kind` is `document`, then `file`
(start `1` and a label that either ends with the basename or spells no identifier), else
`symbol`. Edges printed alongside a result
are one hop of `calls`, `extends`, `imports`, `imports_from`, `inherits`, `uses`, `references`, at
most five.

### Consequences

* Good, because a docstring's content still reaches the ranking through the code node its
  `rationale_for` edge names, so nothing described only in prose is lost.
* Bad, because docstring and concept nodes can never be returned as a place in their own right,
  even when a docstring's own wording is the closest match to the question.

### Confirmation

Counted directly against the reference corpus's node set: roughly two thirds of the graph's nodes
satisfy this rule (`file_type` `code` or `document` with a parsable start line); the remainder are
docstring and concept nodes, together with the code and document nodes graphify emits with an empty
`source_location`, all ineligible by design. **That corpus is private and is not part of this
repository, so the proportion is recorded as how the decision was reached and cannot be recounted
here.** The rule itself is `Graph.eligible()` and is checked by `tests/test_graph.py`;
`graphify-search index` reports how many nodes it kept for a graph of your own.

## More Information

Related: the `symbol` kind's body slice is a fixed window of `BODY_LINES` lines from the node's
start (`text.body_from`). Revisit this decision if graphify begins emitting a node type
that carries both a real location and content a caller could act on directly, distinct from the
code node it currently folds into.
