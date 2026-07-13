# ADR 0002: Derived metrics are canonical

Status: Accepted
Date: 2026-07-13

## Context

Legacy progress values are partly based on LLM outputs. They can change when the
model, prompt, response quality or fallback path changes. The refactor adds a
deterministic layer based on the retained transcript, speaker mapping and stored
annotations.

## Decision

Use participant.derived as the canonical progress and comparison data.
analysis_version.metrics and analysis_version.taxonomy define the meaning of
those values. Legacy LLM fields remain compatibility data during the migration
and are not used as the authoritative chart series.

## Consequences

- Formula and taxonomy changes require a version increment and historical
  recomputation.
- UI must make unavailable derived data explicit instead of silently mixing
  series.
- LLM annotations remain inputs to grammar density and a source of coaching
  content, not a directly comparable score.
- The metrics specification and fixtures become release-critical.
