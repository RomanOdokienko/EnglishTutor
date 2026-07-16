# ADR 0007: Error severity is a per-finding attribute judged by the model

Status: Accepted
Date: 2026-07-16

## Context

Session and This-week ranking is frequency-based (density × persistence), so a
rare but gross error stays visible in the evidence cards yet falls out of the
top-3 priorities and the weekly briefing. Severity cannot be a category weight:
that would revive the invented impact constants removed in task 2.2, and the
same category can be trivial in one utterance and gross in another. Judging a
*found* phrase is the measured-stable half of the annotation task (see
eval/README.md), so the model judges each instance.

## Decision

Every annotation finding carries `severity`, assigned by the model at
annotation time, by impact on the listener and explicitly independent of
`confidence`:

- `blocking` — the meaning is distorted or the listener must strain to
  reconstruct it;
- `noticeable` — a clear error the listener registers, but the meaning
  survives;
- `minor` — a small slip most listeners would not notice.

Enabled by default since 2026-07-16; `OPENAI_ANNOTATION_SEVERITY=0` is the kill
switch that removes the field from the schema and prompt. The finding dict is
assembled in exactly one place (`build_finding` in cli.py), so the field cannot
silently vanish on any path.

**No METRICS/TAXONOMY version bump.** By the change procedure a bump marks
changed meaning of existing values; severity changes neither the counting gate
(`is_countable_annotation` ignores it) nor the category set nor any formula.
Error counts and densities stay comparable across the rollout. Items annotated
before the rollout have an empty severity — "unrated", not "minor", the same
convention as legacy items without `confidence`.

Pilot measurement (2026-07-16, eval set, config unchanged otherwise): 82%
precision / 87% recall vs the 82%/90% baseline — asking for severity does not
degrade extraction. Distribution on the eval session: blocking 0,
noticeable 73, minor 57; severity is orthogonal to confidence (39 of 57 minor
items are high-confidence). Zero `blocking` is honest for these speakers, so
consumers MUST treat levels ordinally — "the grossest errors of a session" are
the items of the highest level present, not "the blocking ones".

## Consequences

- The local archive is re-annotated so history-wide severity views are
  possible; per-session finding counts shift within the known extraction
  variance (union of 2 passes, ±~2 findings). Production data needs one
  `POST /api/rebuild-annotations` (fire once, poll GET — see CLAUDE.md).
- Planned consumers (plan.md 2.7/2.8): a severe-errors density band next to
  the overall density chart, a guaranteed "grossest this session" slot in
  priorities/briefing, and card selection on Andrey's simple review page.
  None exist yet; nothing in the UI reads severity today.
- The eval set does not label severity; the pilot validated it by hand-review
  only. Before severity drives any *counted* metric (not just ordering and
  display), label severity for the REAL items of the eval set and measure
  agreement.
