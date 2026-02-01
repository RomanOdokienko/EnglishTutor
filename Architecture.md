# Architecture: Local English Session Evaluator (MVP)

## 1) What this product is
A local-first MVP that helps two people improve spoken English by analyzing weekly conversation transcripts.

**Input:** a transcript file for each session (already transcribed; no audio processing in MVP).

**Output:** a simple local web page (static files on your hard drive) where you can:
1) pick a session date and view stats for that session
2) see a progress chart across sessions
3) see top-3 recommendations for the selected session

This MVP is designed to be:
- extremely simple to run
- transparent (all data is readable on disk)
- incremental (easy to extend later)

**Non-goals for MVP:**
- audio transcription/diarization
- pronunciation/accent scoring
- real “exam-grade” CEFR calibration
- user accounts/auth/cloud hosting

---

## 2) User flow (how the owner uses it)
1) After a weekly call, you produce a transcript text file.
2) You place it into a dated folder inside `sessions/`.
3) You run one local “process” command (CLI).
4) The system generates:
   - a session analysis JSON
   - an updated history dataset
   - a local static web page you can open in a browser
5) You open the web page from disk and:
   - choose the session date
   - review the stats + top-3 recommendations
   - check the progress chart

---

## 3) Where the state lives (source of truth)
All state lives locally in the repository folders (hard drive).

### Session inputs (source of truth for raw data)
`sessions/<YYYY-MM-DD>/`
- `meta.json`
- `transcript.txt`

### Derived outputs (source of truth for analysis and UI)
`out/`
- `out/sessions/<YYYY-MM-DD>/analysis.json` (analysis for one session)
- `out/history.json` (aggregated dataset across sessions)
- `out/web/` (static website files)

**Rationale:**
- `analysis.json` is the stable, structured result for one date
- `history.json` is the single dataset the web page uses for charts and browsing
- the website is just static HTML/JS that reads `history.json`

No database in MVP. The history file is enough and keeps everything simple.

---

## 4) Components (conceptual)
### A) Ingest (reads files)
Reads `meta.json` + `transcript.txt` from a session folder.

### B) Analyzer (produces structured evaluation)
Creates:
- per-speaker metrics (simple deterministic metrics from transcript)
- per-speaker fluency score/level (can be returned by the LLM in MVP)
- grammar errors list (LLM)
- top recurring patterns (MVP can approximate per session; cross-session patterns are a later enhancement)
- top-3 recommendations per speaker (LLM)

### C) History builder
Updates `out/history.json` by adding or replacing the record for that session date.

### D) Static web viewer
A local web page that:
- loads `out/history.json`
- renders:
  1) date dropdown (choose a session)
  2) session stats (metrics + errors summary)
  3) top-3 recommendations for the chosen session
  4) progress chart across sessions (fluency score over dates; optionally error count too)

---

## 5) Folder structure (MVP)
```
.
├── sessions/
│   └── <YYYY-MM-DD>/
│       ├── meta.json
│       └── transcript.txt
├── out/
│   ├── sessions/
│   │   └── <YYYY-MM-DD>/
│   │       └── analysis.json
│   ├── history.json
│   └── web/
│       ├── index.html
│       ├── app.js
│       └── styles.css
└── (cli + scripts)
```
