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

LLM output is not deterministic. Maintain a small redacted evaluation set with:

- expected speaker attribution;
- acceptable annotation examples and categories;
- known false positives to avoid;
- an expected empty or skipped-annotation case;
- provider failure and fallback scenarios.

Evaluate model or prompt changes before making them the default annotation
configuration. Record model name and annotation status in analysis_version so a
result can be interpreted after a model change.

## Release gate

Before publishing a metrics/taxonomy change:

1. Run deterministic and contract tests.
2. Reanalyse representative historical fixtures.
3. Review the generated progress page for mixed versions and low-sample labels.
4. Verify the static Vercel artifact communicates only with the configured
   backend.
5. Commit source inputs and regenerated outputs together.
