# Tasks: Local English Session Evaluator (MVP)

## Milestone 1: Project scaffolding
- [ ] Create base folders: `sessions/`, `out/`, `out/sessions/`, `out/web/`.
- [ ] Add sample session folder with `meta.json` and `transcript.txt` for development.
- [ ] Define JSON schema (or example structure) for `analysis.json` and `history.json`.

## Milestone 2: Ingest + Analyzer
- [ ] Implement ingest module to read `meta.json` + `transcript.txt`.
- [ ] Implement deterministic metrics (word counts, speaker turns, token ratios).
- [ ] Add LLM-backed fields: fluency score/level, grammar errors, top-3 recommendations.
- [ ] Emit `out/sessions/<YYYY-MM-DD>/analysis.json`.

## Milestone 3: History builder
- [ ] Implement history builder to add/replace session in `out/history.json`.
- [ ] Ensure history is sorted by date for charting.

## Milestone 4: Static web viewer
- [ ] Build `out/web/index.html` with dropdown, stats, and recommendations.
- [ ] Build `out/web/app.js` to load `history.json` and render views.
- [ ] Build `out/web/styles.css` for minimal styling.
- [ ] Add progress chart (fluency score over dates; optional error count).

## Milestone 4.1: Local web upload flow (iterative testing)
- [ ] Add a simple local HTML page with a file input for `.txt` transcripts.
- [ ] Add client-side parsing for a single transcript file and send it to the analyzer.
- [ ] Add a minimal server endpoint to accept uploads and trigger analysis.
- [ ] Save uploaded files into a dated `sessions/<YYYY-MM-DD>/` folder.
- [ ] Rebuild `out/history.json` and `out/web/` after each upload.
- [ ] Provide a manual "re-run analysis" button for iterative testing.

## Milestone 5: CLI command
- [ ] Provide a single local command to run ingest → analyze → history → web build.
- [ ] Document usage in a README.

## Milestone 6: QA + iteration
- [ ] Validate with at least two sessions.
- [ ] Adjust metrics and recommendation prompts as needed.
