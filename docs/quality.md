# Quality and evaluation strategy

## Goal

Protect the reproducibility of derived metrics, compatibility of the migration
and usefulness of LLM-assisted annotations. The repository currently has no
test suite or CI configuration; this document defines the minimum quality gate
to add during the refactor.

## Required test layers

| Layer | Minimum coverage |
| --- | --- |
| Transcript parsing | Speaker labels, blank lines, malformed turns and aliases |
| Deterministic metrics | Fixed transcript fixtures with expected counts, ratios and MATTR values |
| Taxonomy | Explicit categories, inferred categories and uncategorised items |
| Output contracts | Validate session, analysis and history fixtures against active schemas |
| Migration | Legacy analysis fixture reprocessed into v1 with correct analysis_version and derived history fields |
| HTTP API | Upload, date validation, delete protection, rebuild modes and expected error statuses |
| Static artifact | Build copies every web source, injects API base and publishes history/session files |

## LLM evaluation

LLM output is not deterministic, so annotation changes are measured, not
eyeballed. The harness lives in `eval/` (see `eval/README.md`):

- `eval_set_2026-07-14.json` — one session's findings hand-labeled REAL / FP /
  ART / ASR, grown as new configs surfaced findings earlier runs had missed.
- `run_eval.py` — scores a candidate run against the set by span overlap
  (rewording a finding does not break the match) and refuses to score findings
  nobody has labeled.
- `run_annotations.py` — annotates a transcript through the production path into
  a candidate file, writing only where told (never `out/sessions`).

Read the numbers with their bias in mind: **precision is trustworthy**
(every item was judged on its merits); **recall is optimistic**, because the set
is built from model output, so whatever every config missed is absent by
construction. A true recall figure needs an exhaustive hand-pass on a short
segment (a recall probe), not yet done. Current prod config (gpt-5-mini, medium,
2-pass union) scores ~84% precision / ~88% recall on this set.

Still worth adding to the set over time: expected speaker attribution, an
expected empty/skipped-annotation case, and provider failure/fallback scenarios.

Evaluate model or prompt changes against `eval/` before making them the default
annotation configuration. Record model name and annotation status in
analysis_version so a result can be interpreted after a model change.

## Release gate

Before publishing a metrics/taxonomy change:

1. Run deterministic and contract tests.
2. Reanalyse representative historical fixtures.
3. Review the generated progress page for mixed versions and low-sample labels.
4. Verify the static Vercel artifact communicates only with the configured
   backend.
5. Commit source inputs and regenerated outputs together.
