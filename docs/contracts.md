# Contract baseline

## Status

This document records the contract implemented by the current refactor. It is
the baseline for replacing the existing legacy Draft-07 schema files with
validated versioned contracts.

The checked-in files schemas/analysis.schema.json and
schemas/history.schema.json are not authoritative today: their strict
additional-property rules do not include analysis_version, derived or all
history fields already written by the pipeline. Do not use them to reject a v1
analysis until they are aligned and validation is added to the build.

## File contracts

### Session input

Path: sessions/<YYYY-MM-DD>/

| File | Required fields or content | Notes |
| --- | --- | --- |
| meta.json | date, topic, participants, duration_minutes | speaker_map and speaker_labels are present when mapping is supplied |
| transcript.txt | Speaker-labelled conversational turns | Original retained source text |
| audio.<ext> | Optional raw recorded audio | Temporary during transcription; retained on provider failure and removed after a successful transcript |
| timings.json | Optional word-level timings: version, source, transcript_id, audio_duration_sec, utterances[] with speaker_label, channel, start_ms/end_ms and words[] {text, start_ms, end_ms} | Written on transcription from audio (ADR-0006); source data for timing metrics. Text uploads never have it. `cli.py --backfill-timings` creates it from a stored raw provider response |

Date must be a real YYYY-MM-DD date. A participant has name and role. Speaker
mapping maps a transcript label to a participant name.

### Per-session analysis

Path: out/sessions/<YYYY-MM-DD>/analysis.json

Top-level fields are date, session, participants, transcript, speaker_map,
chunks, llm and analysis_version. Some LLM fields are optional when model
analysis or annotations have not run. Sessions with word timings additionally
carry `timing` (version, pause_threshold_ms, source, audio_duration_sec and
per-label raw sums), refreshed from `sessions/<date>/timings.json` on every
derived recompute; the final timing metrics land in
`participants[].derived.metrics` (see docs/metrics-and-taxonomy.md).

| Field | Meaning |
| --- | --- |
| date | Session identifier |
| session | Topic and duration |
| participants[] | Per-participant source metrics, optional LLM results and canonical derived data |
| transcript | Source transcript copied into the analysis artifact |
| speaker_map | Label-to-person mapping used for attribution |
| chunks | Transcript chunks used by annotation and chunk metrics |
| llm | Status, model metadata, annotations and optional LLM summaries |
| analysis_version | metrics, taxonomy, annotation model and annotation status |

Each item in `llm.annotation_items` has: `start`, `end` (transcript offsets),
`text`, `correction`, `explanation`, `category` (a v1 code or empty), and —
since July 2026 — `confidence` (high/medium/low) and `is_stylistic` (bool). The
last two drive the counting gate (see docs/metrics-and-taxonomy.md); items from
before they existed omit them and are still counted. `llm.annotations_meta`
carries `per_chunk` (findings and pass count per chunk) for drift visibility.
Items also carry `severity` (blocking/noticeable/minor by communication
impact, ADR-0007), on by default since 2026-07-16
(`OPENAI_ANNOTATION_SEVERITY=0` is the kill switch). An empty severity means
"annotated before the rollout / not rated", not "minor". Severity never
affects the counting gate; consumers treat levels ordinally. The finding dict
is assembled only by `build_finding` in cli.py.

The canonical participant fields are:

| Path | Status | Meaning |
| --- | --- | --- |
| participants[].derived.metrics | Canonical | Deterministic per-speaker measures |
| participants[].derived.grammar | Canonical | Annotation-derived error counts and densities |
| participants[].metrics | Compatibility/detail | Previous deterministic summary |
| participants[].fluency | Compatibility/detail | Previous local fluency estimate |
| participants[].llm | Optional enrichment | Model-generated assessment and coaching material |

### History

Path: out/history.json

history.sessions[] contains date, topic, analysis_version and participants[].
Each history participant has name, role, derived and temporary legacy values:
fluency_score, llm_fluency_score, grammar_error_count, llm_error_rate and
chunked_error_rate. Progress features must read derived rather than those legacy
values. Every participant also has `comparison` version 1: eligibility,
reference dates/count, the overall comparison, and a comparison for every
grammar category. A metric comparison contains current, reference_average,
delta, threshold and status. The reference is calculated chronologically from
up to three comparable sessions strictly before that history entry.

### This-week briefing

Path: `out/briefing.json` (also copied to `out/web/briefing.json`). It is a
deterministic read model rebuilt after session analysis, derived reanalysis and
focus updates. Version 2 contains one primary active or suggested focus, at
most two patterns from the latest three comparable calls, at most two matching
annotation examples and one recent grammar direction record per participant.
Russian fallback and lexical diversity are intentionally absent from this
briefing. The browser treats it as a convenience read model; Session analysis
and history comparison v1 remain the canonical source for detailed evidence.

### Import bundle (POST /api/import-session)

Delivers locally computed session artifacts to a server without re-running any
model there (the model cost is paid once, locally, measured against eval/).
Requires ENGLISH_TUTOR_TOKEN on the server and the same value in the
`X-ET-Token` header (or `?token=`); disabled (503) when unconfigured. Body:
`{"analysis": <full analysis.json>, "session_files": {"meta.json"?, "transcript.txt"?, "timings.json"?}}`.
Only those three source files are writable. The server re-runs
`finalize_derived_metrics`, updates history and rebuilds web assets itself.
`push_to_prod.py` is the client.

## HTTP contract

All endpoints are served by the Python backend. GET history and per-session
analysis are read paths. POST paths change files or invoke provider-backed work.
Date bodies use JSON unless noted otherwise.

| Method and path | Request | Successful result | Notes |
| --- | --- | --- | --- |
| GET /history.json | None | History JSON | Static read contract |
| GET /sessions/<date>/analysis.json | None | Analysis JSON | Static read contract |
| POST /api/upload | transcript required; optional topic, date, duration and speaker mapping fields | date and topic | Creates session, analyses and rebuilds artifacts |
| POST /api/upload-audio | Audio bytes; query includes date, topic, recorder, other, duration, language and ext | date, topic, channels, analysis_status and optional warning | Requires ASSEMBLYAI_API_KEY; returns after transcription and deterministic metrics, while model analysis may continue in the background |
| POST /api/rebuild | Optional date body | Number of rebuilt sessions, optionally date | Rebuilds metrics and annotations according to configured keys |
| POST /api/rebuild-metrics | Optional date body | Number of rebuilt sessions, optionally date | Metrics only |
| POST /api/rebuild-annotations | Optional date body | Number of rebuilt sessions, optionally date | Annotations only; requires OpenAI configuration |
| POST /api/reanalyze | No body | reanalyzed count | Recomputes derived metrics for stored analyses without an LLM request |
| POST /api/delete | date required | deleted date | Removes session source, derived analysis and its history entry |
| POST /api/highlight-exercise | participant_name, category_code and category_title required; focus_text and examples optional | model and exercise | Requires OpenAI configuration |
| POST /api/test-gpt5 | No body | model and output | Diagnostic endpoint; not a user-facing production feature |

Validation errors use 400; missing sessions use 404; provider or rebuild
failures use 500 or 502 as applicable.

## Contract rules

- API clients must not write files directly; use the backend.
- A change to canonical derived field meaning requires a metrics or taxonomy
  version bump and a migration entry.
- New output fields are additive only within a version. Removal or semantic
  change requires an explicit contract migration.
- A client must treat an absent derived value as unavailable, not substitute a
  legacy score.
- The generated frontend and backend must use the same contract version at
  deployment time.

## Required follow-up implementation

1. Replace or update the current schema files to the active output shape.
2. Choose a JSON Schema draft and add schema validation to build and test flows.
3. Add API request/response fixtures, including malformed date and mapping
   cases.
4. Publish an OpenAPI 3.1 file once the backend contract is stable enough to
   generate or validate from it.
