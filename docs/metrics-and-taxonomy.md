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

## Timing metrics (additive, optional — ADR-0006)

Present only for sessions transcribed from audio, where
`sessions/<date>/timings.json` holds word-level timings. They live in the same
`participants[].derived.metrics` object; **an absent key means "not measured",
never zero** — text uploads legitimately never get them, and comparisons may
only use sessions where the key exists on both sides. They are deterministic
(no LLM) and are refreshed from the timing source on every
`--recompute-derived`. Adding them did NOT bump the metrics version: the v1
text metrics are computed exactly as before (additive-metrics policy,
ADR-0006).

| Metric | Formula or definition | Interpretation |
| --- | --- | --- |
| timed_word_count | Words with timestamps attributed to the speaker | Denominator basis; counts all languages |
| speaking_time_sec | Σ (utterance end − start) for the speaker | Own speech time incl. internal pauses |
| speech_rate_wpm | timed words / speaking minutes | Overall delivery speed |
| articulation_rate_wpm | timed words / (speaking − pause) minutes | Speed while actually talking |
| pauses_per_min | hesitation pauses / speaking minutes | Hesitation frequency |
| mean_pause_sec | pause time / pause count | Hesitation depth; 0.0 when no pauses |
| mean_length_of_run_words | timed words / run count | Fluent stretch between hesitations |

Rules v1 (constants `TIMING_VERSION = 1`, `PAUSE_THRESHOLD_MS = 500` in
cli.py): a hesitation pause is a gap ≥ 500 ms between consecutive words inside
one utterance; silence between utterances is not counted (the partner usually
speaks there). A run is a maximal word sequence without such a pause; an
utterance boundary always ends a run. The per-session `analysis["timing"]`
block records the version, threshold and per-label raw sums so any number can
be audited.

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

- error_count is the number of **countable** attributed annotation items (see
  the counting gate below), whether or not they carry a recognised category.
- error_density_per_100w is error_count / english_word_count × 100.
- by_category_count contains every v1 category with its count.
- by_category_density contains every v1 category count /
  english_word_count × 100.

An annotation item first uses its assigned category. If absent, the pipeline
may infer a category from its explanation, correction and text. Uncategorised
items still count toward error_count but are added to no named category
density — "we could not label it" is not "it was not an error". A session
without annotations is not evidence of zero grammatical errors; it has no
reliable annotation-based grammar measurement.

### Severity (per-finding attribute, ADR-0007)

Since 2026-07-16 every new finding carries `severity`
(blocking/noticeable/minor), judged by the model per instance by impact on the
listener, independent of confidence. It does **not** affect the counting gate
below and required no version bump: counts and densities mean exactly what
they meant. Empty severity = annotated before the rollout ("not rated", never
"minor"). Consumers treat levels ordinally: "the grossest errors of a session"
= items of the highest level present.

### Counting gate (`is_countable_annotation`)

Both the Session-page counter and the derived-metrics counter pass every item
through one function, so the two numbers cannot diverge (they once did, by up
to 69% on stored data). An item counts unless it is any of: empty text or
correction; span under 2 or over 24 words; a disfluency/filler; a correction
equal to the text; `is_stylistic: true`; or `confidence: "low"`. Items with
`confidence` of high or medium count. Legacy items predating the confidence
field have no such key and still count — excluding them would rewrite history
without re-running it.

### Annotation extraction (affects reproducibility, not the taxonomy)

Findings are produced by `annotate_chunk`: two independent model passes per
chunk, unioned by span overlap, over char-budgeted chunks
(`ANNOTATION_CHUNK_MAX_CHARS`). Extraction — *finding* the errors — is the
unstable half of the task; a single pass finds a different subset each run, so
the union stabilises the density series. Judgment of a found phrase is
comparatively stable. The taxonomy is unchanged by this, so the version stays 1,
but a re-extraction changes which errors are found and must recompute the whole
archive for comparability. Measure any prompt/effort/pass change against
`eval/` before making it the default (see docs/quality.md).

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

## Personal comparison v1

Session, This week and the Progress summary use one stored, chronological
comparison rule. For a participant in a selected session:

- use at most the three comparable sessions strictly before the selected date;
- comparable means completed annotations, at least 120 English words, and the
  same metrics and taxonomy versions;
- calculate the reference average independently for overall error density and
  every grammar category;
- because only earlier dates are eligible, adding a future session never
  changes the comparison shown for an old session;
- re-analysing, deleting or backfilling an earlier source session can change a
  later comparison, because the historical evidence itself changed.

Lower is better. A change is meaningful only when its absolute size reaches
`max(0.15 errors/100w, 10% of the reference average)`. Status is `improving`,
`needs_attention` or `steady`; sessions without a valid current sample use
`short_sample`, `annotations_unavailable`, `metrics_unavailable` or
`version_unavailable`, and a first eligible session uses `no_baseline`.

Focus progress is deliberately different: it compares with the exact session
on which the user set the focus, not with comparison v1's rolling reference.

## Change procedure

1. Propose the formula or taxonomy change with examples and fixtures.
2. Add an ADR if the change affects a durable architectural decision.
3. Increment the relevant version constant.
4. Update this document and the machine-readable contract.
5. Recompute all stored analyses and rebuild out/web.
6. Verify that the UI neither mixes versions nor silently falls back.
