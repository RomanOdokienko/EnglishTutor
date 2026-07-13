# ADR 0005: Focus selection and closure methodology

Status: Accepted
Date: 2026-07-13

## Context

The learning loop needs an explicit focus mechanic: pick 2-3 recurring error
categories, work on them between calls, and get a clear answer on the next
session whether they closed. Earlier UI versions guessed a weekly focus
automatically from a hidden score with invented weights; that produced
unexplainable priorities and no accountability loop. Evidence cards (Session
page) now show only measured facts per category.

## Decision

A focus is a human decision recorded in data/focus.json:
{id, participant, category_code, note, examples[], status: active|closed,
set_date, closed_date}. The backend exposes GET/POST /api/focus with actions
set, close and remove; at most 3 active focuses per participant.

The system proposes but never decides:

- Proposal rule: evidence cards are ordered recurring-first (category density
  at or above 0.3 per 100 words in at least 2 of the last 3 comparable
  sessions; a comparable session has at least 120 English words), then by
  count and density. The top cards are the natural focus candidates.
- Closure verdict (rule v1): compare the focus category's density in the
  viewed session against the session where the focus was set. If the density
  fell by 40% or more and the viewed session has at least 120 English words,
  the UI shows "ready to close". Closing remains a button press by a human.
- Densities always come from derived.grammar.by_category_density in
  history.json (ADR-0002); the verdict is recomputed at render time, never
  stored.

Closed focuses are the victory log, shown on the Progress page.

All thresholds (0.3/100w presence floor, 2-of-3 persistence, 40% drop,
120-word sample) are v1 working hypotheses to be recalibrated against real
data now that annotation extraction is stable.

## Consequences

- focus.json lives in DATA_DIR: it lands on the Railway volume and is covered
  by the git backup path (plan task 1.4).
- Re-annotating the baseline session changes the stored densities and thereby
  the verdict; the verdict is honest to current data rather than to a
  snapshot.
- If the baseline session is deleted, the focus shows "no verdict" instead of
  guessing.
- Threshold changes are methodology changes: update this ADR (or supersede
  it) and the UI constants together.
