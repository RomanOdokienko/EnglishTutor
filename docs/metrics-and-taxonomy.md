# Metrics and grammar taxonomy

## Scope and status

Derived metrics v1 are the canonical progress series. They are calculated
deterministically from retained transcript turns, speaker mapping and stored
annotation items. The process makes no new LLM request, but grammar density
still depends on the quality and availability of previously stored annotations.

The version is written to analysis_version:

- metrics: 1
- taxonomy: 1

Any change in a formula, tokenizer, denominator, category mapping or category
meaning increments the corresponding version and requires a historical
recomputation.

## Derived metrics v1

All values live at participants[].derived.metrics.

| Metric | Formula or definition | Interpretation |
| --- | --- | --- |
| word_count | English word count plus Russian word count | Total attributed spoken words |
| english_word_count | Latin-letter tokens with apostrophes allowed | Denominator for English-normalised rates |
| russian_word_count | Cyrillic-word tokens | Source for L1 fallback |
| turn_count | Number of attributed speaker turns | Input to turn-based rates |
| avg_words_per_turn | word_count / turn_count | Turn length, not fluency quality |
| speaking_share_pct | speaker word_count / all participant word_count × 100 | Share within this session |
| l1_fallback_pct | russian_word_count / word_count × 100 | Russian use within attributed speech |
| filler_per_100w | recognised filler count / english_word_count × 100 | Rate of listed fillers |
| question_pct | turns containing a question mark / turn_count × 100 | Question-form usage signal |
| lexical_diversity_mattr | Mean type-token ratio over a 50-token moving window; short text uses its token ratio | Lexical variety adjusted for transcript length |

Recognised fillers in v1 are um, uh, er, erm, hmm, like, you know, i mean,
kind of and sort of. Metrics are descriptive signals and must not be presented
as a CEFR or examination-grade assessment.

## Grammar taxonomy v1

Grammar values live at participants[].derived.grammar.

| Code | Label |
| --- | --- |
| TENSE | Verb Tense |
| VERB | Verb Form |
| ARTICLE | Articles |
| PREP | Prepositions |
| ORDER | Word Order |
| WORD | Wrong Word |
| COLLOC | Collocation |

For each participant:

- error_count is the number of attributed annotation items in a recognised
  category.
- error_density_per_100w is error_count / english_word_count × 100.
- by_category_count contains every v1 category with its count.
- by_category_density contains every v1 category count /
  english_word_count × 100.

An annotation item first uses its assigned category. If absent, the pipeline
may infer a category from its explanation, correction and text. Uncategorised
items are not added to a named category density. A session without annotations
is not evidence of zero grammatical errors; it has no reliable annotation-based
grammar measurement.

Classification source note (July 2026): the annotation call enforces the v1
category enum through the response JSON schema and the prompt spells out the
seven definitions with examples, so new items carry a model-assigned category;
keyword inference remains only as a fallback for legacy items. The category
set and meanings are unchanged, so the taxonomy version stays 1, but sessions
annotated before this change classified categories by keyword inference and
their per-category splits may show a small step relative to newer sessions.

## Comparability rules

- Compare values only when metrics and taxonomy versions match.
- Display the version near multi-session progress data.
- Exclude small samples from progress lines and chart scales; the current UI
  requires at least 120 English words per participant. Keep excluded sessions
  visible in the audit table and explain why they do not affect the trend.
- Recalculate the whole archive after a version bump before drawing a new
  trendline.
- Preserve original transcript and annotations so a result can be reproduced or
  audited.

## Change procedure

1. Propose the formula or taxonomy change with examples and fixtures.
2. Add an ADR if the change affects a durable architectural decision.
3. Increment the relevant version constant.
4. Update this document and the machine-readable contract.
5. Recompute all stored analyses and rebuild out/web.
6. Verify that the UI neither mixes versions nor silently falls back.
