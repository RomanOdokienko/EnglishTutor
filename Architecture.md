# Architecture: Local English Session Evaluator (MVP)

## 1) What this product is
A local-first MVP that helps two people improve spoken English by analyzing weekly conversation transcripts.

Input: a transcript file for each session (already transcribed; no audio processing in MVP).

Output: a local static web page where you can:
1) pick a session date and view stats for that session
2) see two progress charts (LLM Proficiency and Grammar error rate per 100 words) per speaker
3) see top recurring grammar errors + recommendations per speaker

Non-goals for MVP:
- audio transcription/diarization
- pronunciation/accent scoring
- exam-grade CEFR calibration
- user accounts/auth/cloud hosting

---

## 2) User flow
1) After a weekly call, produce a transcript text file.
2) Upload via `upload.html` and map Speaker A/B to people (Roman/Andrey).
3) The system stores session files and builds outputs.
4) Open the dashboard to inspect metrics and progress.

---

## 3) Where the state lives (source of truth)

### Session inputs
`sessions/<YYYY-MM-DD>/`
- `meta.json`
- `transcript.txt`

### Derived outputs
`out/`
- `out/sessions/<YYYY-MM-DD>/analysis.json`
- `out/history.json`
- `out/web/` (static site)

### Mapping data
`data/people.json` (aliases for speaker labels)

---

## 4) Components

### A) Ingest
Reads `meta.json` + `transcript.txt`, treats each speaker block as one turn.

### B) Analyzer
Creates:
- deterministic metrics (word count, turns, lexical diversity)
- optional LLM analysis (proficiency, grammar errors, recommendations)
- grammar error rate per 100 words (from LLM)

### C) History builder
Maintains `out/history.json` sorted by date.

### D) Local web server
`server.py` provides:
- upload endpoint
- re-run metrics endpoint
- re-run annotations endpoint (chunked, incremental save)
- test endpoint for gpt-5-mini probe
- delete session endpoint

### E) Static web viewer
Loads `history.json` and renders:
- per-session cards
- progress charts (per speaker)
- recommendations and recurring errors
- transcript in three columns (Original / Annotated / Issues)

---

## 5) Folder structure (current)
```
.
|-- sessions/
|   `-- <YYYY-MM-DD>/
|       |-- meta.json
|       `-- transcript.txt
|-- out/
|   |-- sessions/
|   |   `-- <YYYY-MM-DD>/
|   |       `-- analysis.json
|   |-- history.json
|   `-- web/
|       |-- index.html
|       |-- upload.html
|       |-- app.js
|       `-- styles.css
|-- data/
|   `-- people.json
|-- cli.py
`-- server.py
```

---

## 6) Chunked LLM annotation (implemented)
Goal: annotate the full transcript with error highlights and explanations.
Approach: chunk the transcript, analyze chunks via LLM, merge annotations, and render:
- highlighted annotated text
- per-line issue explanations in a third column

Incremental behavior:
- Each chunk is saved after processing to avoid losing progress.
- Resume is supported via `OPENAI_ANNOTATION_RESUME=1`.
- Fallback from gpt-5 to gpt-4o can be disabled via `OPENAI_ANNOTATION_NO_FALLBACK=1`.

## 7) Chunked LLM metrics (additional)
Goal: compute grammar error counts per chunk and aggregate into a more complete error rate.
Approach:
- Split transcript into chunks (same as annotations).
- For each chunk, LLM returns `error_count` + `error_types`.
- Words per chunk are counted locally.
- Aggregate totals and top error types are stored under `llm.chunked_metrics`.
Incremental behavior:
- Each chunk save persists `per_chunk` and `summary`.
- Resume is supported via `OPENAI_CHUNK_METRICS_RESUME=1`.
