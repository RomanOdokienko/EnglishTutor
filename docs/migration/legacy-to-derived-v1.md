# Migration: legacy indicators to derived metrics v1

## Purpose

Move the product from a UI and history model centred on LLM-produced summaries
to a reproducible, versioned metrics layer. The resulting progress series is
participant.derived. LLM outputs remain valuable annotations and coaching
material, but are not the canonical cross-session measure.

## Decision boundary

| Area | Legacy state | Target state |
| --- | --- | --- |
| Canonical progress data | fluency_score, LLM grammar rates and chunked values | participant.derived.metrics and participant.derived.grammar |
| Reproducibility | May vary by model, prompt and retry outcome | Recomputed from retained transcript, speaker map and stored annotation items |
| Comparability | Implicit and potentially mixed across sessions | Explicit metrics and taxonomy versions on every analysis |
| Frontend source | Generated web artifacts were edited directly | web/ is source; out/web is generated |
| Static deployment | Generated files only | Vercel static frontend plus separately deployed API backend |

## Current state

The first target implementation is already present in the working tree:

- analysis_version contains metrics and taxonomy versions.
- The pipeline calculates deterministic per-speaker metrics and grammar
  densities.
- The history builder carries derived data to the progress UI.
- The progress UI reads derived data when present.
- Legacy fields are still written to preserve existing session screens and to
  avoid a breaking one-step conversion.

The transition is not complete until the checked-in schemas, tests and
documentation describe the current output shape.

## Migration steps

### 1. Freeze the v1 contract

- Treat the field names and formulas in metrics-and-taxonomy.md as v1.
- Record a version bump and an ADR before changing a formula, category or
  denominator.
- Define JSON Schema 2020-12 contracts for session input, analysis and history.
  The existing Draft-07 schema files are legacy until aligned and validated in
  the build.
- Add representative legacy and v1 analysis fixtures.

### 2. Make derived data the frontend default

- Every progress chart and comparison uses derived values.
- Display the metrics/taxonomy version alongside progress data.
- When derived data is absent, render an explicit unavailable state rather than
  silently falling back to legacy LLM values.
- Keep LLM fields only on detailed session views where their provenance is
  clear.

### 3. Recompute historical data

- Run the CLI with --recompute-derived, or invoke POST /api/reanalyze.
- Review every generated analysis_version and history entry.
- Regenerate out/web from web/ and commit the resulting versioned artifacts.
- Do not compare sessions that have different metrics or taxonomy versions in
  one undifferentiated chart.

### 4. Remove compatibility after adoption

Only after all supported UI paths use derived data and the historical archive
has been recomputed:

- Mark legacy history fields deprecated in the contract.
- Remove frontend reads of legacy values.
- Remove legacy output fields in a planned breaking contract version, not in a
  silent cleanup.

## Rollback

The session archive, transcript and stored annotation items are retained in
Git. If v1 is found defective, restore the prior implementation from Git,
regenerate out/ from the retained inputs, and label the affected output version
as superseded. Do not overwrite the metric meaning without incrementing the
version.

## Definition of done

- Every committed analysis and history entry validates against the active
  contract.
- Every historical entry has matching metrics and taxonomy versions.
- Progress and session pages use derived data as their canonical input.
- Source is edited only in web/; out/web is generated.
- README, architecture, runbook and the ADR log describe the same deployment
  topology.
- Fixture-based tests cover parsing, formulas, history generation and the
  legacy-to-v1 conversion.
