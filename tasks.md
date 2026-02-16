# Tasks: Local English Session Evaluator (MVP)

## Milestone 1: Project scaffolding
- [x] Create base folders: `sessions/`, `out/`, `out/sessions/`, `out/web/`.
- [x] Add sample session folder with `meta.json` and `transcript.txt` for development.
- [x] Define JSON schema (or example structure) for `analysis.json` and `history.json`.

## Milestone 2: Ingest + Analyzer
- [x] Implement ingest module to read `meta.json` + `transcript.txt`.
- [x] Implement deterministic metrics (word counts, speaker turns, token ratios).
- [x] Add LLM-backed fields: fluency score/level, grammar errors, top-3 recommendations.
- [x] Emit `out/sessions/<YYYY-MM-DD>/analysis.json`.

## Milestone 3: History builder
- [x] Implement history builder to add/replace session in `out/history.json`.
- [x] Ensure history is sorted by date for charting.

## Milestone 4: Static web viewer
- [x] Build `out/web/index.html` with dropdown, stats, and recommendations.
- [x] Build `out/web/app.js` to load `history.json` and render views.
- [x] Build `out/web/styles.css` for minimal styling.
- [x] Add progress chart (fluency score over dates; optional error count).

## Milestone 4.1: Local web upload flow (iterative testing)
- [x] Add a simple local HTML page with a file input for `.txt` transcripts.
- [x] Add client-side parsing for a single transcript file and send it to the analyzer.
- [x] Add a minimal server endpoint to accept uploads and trigger analysis.
- [x] Save uploaded files into a dated `sessions/<YYYY-MM-DD>/` folder.
- [x] Rebuild `out/history.json` and `out/web/` after each upload.
- [x] Provide a manual "re-run analysis" button for iterative testing.

## Milestone 5: CLI command
- [x] Provide a single local command to run ingest -> analyze -> history -> web build.
- [x] Document usage in a README.

## Milestone 6: QA + iteration
- [x] Validate with at least two sessions.
- [x] Adjust metrics and recommendation prompts as needed.

## Milestone 7: LLM integration + metrics
- [x] Add OpenAI optional analysis (parallel to local metrics).
- [x] Persist LLM status (ok/error/skipped) and model in analysis output.
- [x] Add LLM grammar error count, top recurring errors, and examples with corrections.
- [x] Add grammar error rate per 100 words.
- [x] Add lexical diversity metric.
- [x] Replace single progress chart with two charts and per-speaker lines.

## Milestone 8: Data hygiene + mapping
- [x] Add speaker mapping (Speaker A/B -> Roman/Andrey) on upload.
- [x] Persist mapping in `meta.json` and aliases in `data/people.json`.
- [x] Add delete-session flow and update history/output.
- [x] Preserve LLM results on local rebuilds without OpenAI.

## Milestone 9: Annotated transcript (planned)
- [x] Add two-column transcript UI (Original + Annotated placeholder).
- [x] Define chunked LLM annotation format (errors with positions + corrections).
- [x] Implement chunked LLM analysis across full transcript.
- [x] Render annotated transcript with highlights + explanations.
- [x] Add per-line Issues column aligned with the transcript.
- [x] Add incremental annotation saves with progress and resume.

## Notes: GPT-5 mini probe (working config)
- Endpoint: `POST /v1/responses`
- Model used: `gpt-5-mini` (also observed `gpt-5-mini-2025-08-07`)
- Payload that returns output:
  - `input`: "Reply with the single word OK."
  - `reasoning`: `{"effort":"low"}`
  - `text`: `{"format":{"type":"text"}}`
  - `max_output_tokens`: 200
- Do NOT include `temperature` (unsupported by gpt-5-mini)
