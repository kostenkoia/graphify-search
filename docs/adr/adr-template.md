---
# Optional metadata. Keep these three; MADR's `consulted`/`informed` are
# deliberately dropped for this project (single decision-maker, nobody to
# inform one-way).
status: "{proposed | accepted | rejected | deprecated | superseded by ADR-0123}"
date: { YYYY-MM-DD when the decision was last updated }
decision-makers: { who actually decided }
---

# ADR-NNNN: {short title stating the decision, not the topic}

<!--
FORMAT: MADR 4.0.0 (https://adr.github.io/madr/), trimmed to seven sections.
Kept: Context and Problem Statement, Decision Drivers, Considered Options,
Decision Outcome, Consequences, Confirmation, More Information.
Dropped: `consulted`/`informed` frontmatter and "Pros and Cons of the
Options" — at this project's scale the latter only restates Considered
Options plus Consequences.

FILE NAMING (MADR convention): `NNNN-title-with-dashes.md`, four digits,
consecutive, never reused. The number is an identity, not an ordering of
importance.

WHAT AN ADR IS HERE: a description of the architecture AS IT STANDS, with
the reasoning that produced it and what was done to get there. Not a diary.
If a fact does not help someone understand or safely change the current
system, it does not belong in an ADR.

DECISIONS, NOT INSTANCE DATA (binding): an ADR records the decision and its
invariants. Concrete values, paths, sizes, corpus names and identities live
in code, config and artefacts — REFERENCE them, never embed them. "The
derived index lives outside the vendor's output directory" is a decision;
the path is not. "The bar must exceed what a single row of the reference
set can move" is a decision; `+0.10` is a value derived from the set's
current size and belongs in the measurement spec.

Two reasons this is strict here. A number copied into an ADR silently
outlives the assumption that produced it — a threshold derived for a
10-row set keeps looking authoritative against a 20-row one. And these
files are tracked in git, while some corpora we measure are private: an
ADR that names no corpus has nothing to leak.

PUBLICATION RULE (binding — this directory ships in a public repository):

* Never name a measured corpus, a customer repository, or any repository
  other than this one. "A second, private corpus" carries the whole
  meaning; the name carries only risk.
* Never publish a characterisation of a third party's project — size,
  defect rate, release cadence, quality. Where such a figure is the
  evidence for a decision, state the decision and its shape here and keep
  the figure in the internal backlog. Our own capacity is ours to describe;
  someone else's engineering is not.
* Public numbers about THIS project are fine and encouraged; private-corpus
  numbers appear only qualitatively ("the second corpus moved the same
  way"), never as values.

NO FILE REFERENCES (binding, kia 2026-08-12): an ADR may point only at
CLAUDE.md / AGENTS.md, README, and other ADRs. No paths to source files,
working documents, specs, backlogs or measurement artefacts — not even our
own. Describe the mechanism in prose and name a symbol when a symbol is the
clearest anchor; a path is not. Two reasons: working documents move and get
deleted (a pointer into them rots the day it is written, and this directory
is meant to outlive them), and a public ADR should be readable by someone
who has only this file.

CLAUDE.md is git-ignored (kia 2026-08-12), so a reader outside this machine
cannot open it. Citing it is still allowed for a rule we work by, but never
as evidence a claim can be checked — an ADR's evidence has to be reachable
by whoever is reading the ADR.

REVIEW RULE (binding, kia 2026-08-12), three tiers:

1. Every ADR is reviewed TWICE after it is written, in two distinct passes
   — first for factual accuracy against the code (including: were the
   "considered options" real, or plausible inventions?), then for
   publication safety, format compliance and standalone readability.
2. The whole set gets ONE more review once it is complete, for what
   per-document passes structurally cannot see: contradictions between
   ADRs, duplication, gaps, and terminology that drifted between authors.
3. An ADR that has not been through both of its own passes is a draft, and
   a set that has not had the final pass is not finished.
4. **A reviewed ADR is the reference.** Its claims were verified against the
   source; the README and other prose were not, and can lag. So: never cite
   the README as evidence a behaviour exists — verify in the code and cite
   that. A README that contradicts a reviewed ADR is a bug in the README,
   and reconciling it is part of finishing the set, not a separate wish.

The publication test is NOT "did I write the name". A characterisation with
no name attached is still a characterisation when the README resolves it in
one hop. The test is: would a maintainer of the project being described
reasonably object to reading this in someone else's public repository?

SOURCED-OR-OMITTED (binding, learned the hard way): every entry in
Considered Options, and every claim that something happened, must be
QUOTED FROM A SOURCE before it is written — a spec, the backlog, a commit,
a code comment. Do not write an option because it is the plausible third
alternative; do not write "this has happened before" because it would
explain the decision well. The first two ADRs each shipped a fabricated
considered-option, and one shipped an invented incident; all three read
perfectly naturally and all three were false. If no source can be found,
the honest form is two options, or none — an ADR with a short options list
is fine, an ADR with a confabulated one is worthless.

NAME COLLISION, stated once so nobody re-derives it: this project once had a
PRODUCT feature also called "ADR" (`.decisions/**/*.yaml` injected into
search results), removed 2026-08-03. These documents are unrelated to it.
-->

## Context and Problem Statement

{Two or three sentences: what forced a decision. State the fork we stood at,
not the history of how we got to the fork. If a question phrases it better
than a statement, use the question.}

## Decision Drivers

<!--
The criteria the choice was judged against — especially any rule that was
fixed BEFORE the evidence was gathered. Separating these from Context is
load-bearing here: "what happened" and "by what bar we judged it" are
different claims, and conflating them is how a bar gets moved after the fact.
-->

* {driver, e.g. a pre-registered threshold, a constraint, a cost ceiling}
* {driver}

## Considered Options

<!--
One line each, including options that were rejected. This is where a dead
hypothesis belongs — a rejected option recorded here needs no file of its
own, and deleting its working notes then costs nothing.
-->

* {option 1}
* {option 2}

## Decision Outcome

Chosen: {option}, because {justification tied to the drivers above}.

<!--
"Change nothing" is a legitimate outcome and is written the same way:
Chosen: keep X, because ...

The scaffold above is the house form: the option is unquoted, and the
justification may follow as its own paragraph instead of trailing the clause.
MADR writes it `Chosen option: "…"`; every ADR in this set uses the house form,
so a new one should start from it.
-->

### Consequences

* Good, because {what this buys}
* Bad, because {what this costs — state it even when the decision is right}

### Confirmation

<!--
How we know. For a measured decision: the run-ids, the numbers, and the rule
that was fixed before the measurement. For a decision taken without
measurement, say exactly that — an honest "argued, not measured" is worth
more than a confident sentence with nothing behind it.
-->

{evidence}

## More Information

<!--
Artefact paths, links to superseded or related ADRs, and — when it applies —
the condition under which this decision should be revisited. If a decision
is cheap to reverse, say so; it changes how much later evidence is needed to
justify reopening it.
-->

{links, artefacts, revisit condition}
