# ADR 0006: Word-timing capture and additive optional metrics

Status: Accepted
Date: 2026-07-16

## Context

The learning loop measures grammar well but fluency not at all: the stored
`participants[].fluency` score is a word-count formula from the MVP and the
LLM 0-10 score is a single unanchored judgement — neither is shown in the UI
anymore. Real fluency is temporal (speech rate, hesitation pauses, run length)
and needs word timings. AssemblyAI returns word-level timestamps on every
transcription, but `utterances_to_transcript` used to discard them; only a
debug `assemblyai_response.json` occasionally survived.

Separately, the project needs a rule for growing its measurement ability
without breaking session comparability: bumping `analysis_version.metrics` for
every new metric would mark all older sessions incomparable even though the
existing formulas did not change.

## Decision

1. **Capture timings at transcription time.** `transcribe_audio_file` returns a
   slim timing payload (`build_utterance_timings`) and the server persists it as
   `sessions/<date>/timings.json` next to `transcript.txt`. It is source data
   (versioned in git, lands on the Railway volume), not a build artifact. The
   raw provider response stays a debug artifact; `python cli.py
   --backfill-timings` converts a surviving raw response into the contract
   format for sessions recorded before capture existed.

2. **Compute timing metrics deterministically in Layer A.** During
   `finalize_derived_metrics`, `refresh_timing_block` reads `timings.json` when
   present, stores per-label aggregates in `analysis["timing"]`, and the final
   metrics land additively in `participants[].derived.metrics`: speaking_time_sec,
   speech_rate_wpm, articulation_rate_wpm, pauses_per_min, mean_pause_sec,
   mean_length_of_run_words, timed_word_count. No LLM is involved; a re-run is
   reproducible. Definitions live in docs/metrics-and-taxonomy.md.

3. **Additive optional metrics do not bump METRICS_VERSION.** A version bump is
   for changed meaning of existing values. New metrics enter as optional keys:
   absent means "not measured" (text uploads have no timings and legitimately
   never get these keys), and absence must never be rendered or averaged as
   zero. Any comparison of an optional metric requires the key present on both
   sides. The timing computation carries its own `timing.version` +
   `pause_threshold_ms` inside the analysis, so a threshold change is visible
   per session and is propagated by `--recompute-derived`.

4. **Pause rule v1** (working hypothesis, recalibrate on real calls): a pause
   is a gap ≥ 500 ms between consecutive words inside one utterance; silence
   between utterances is not counted (the partner is usually speaking — that
   would measure turn-taking, not fluency). A run is a maximal word sequence
   without such a pause; utterance boundaries end runs. Rates count all timed
   words regardless of language; `l1_fallback_pct` already reports the mix.

## Consequences

- Fluency finally has honest, deterministic measurements, but only for
  recorded sessions. The existing text-only archive shows no timing metrics —
  that is correct, not a bug. Legacy `participants[].fluency` and the LLM
  fluency score remain stored for compatibility but stay deprecated.
- `finalize_derived_metrics` now reads one file outside the analysis artifact
  (`sessions/<date>/timings.json`) when it exists. If session sources are
  absent (an artifact copied alone), the previously stored timing block is
  kept as-is rather than deleted.
- UI work (a fluency strip / charts) can gate purely on key presence and will
  light up with the first real recorded session.
- Changing `PAUSE_THRESHOLD_MS` or the aggregate formulas is a methodology
  change: bump `TIMING_VERSION`, update this ADR or supersede it, and re-run
  `--recompute-derived` over the archive.
