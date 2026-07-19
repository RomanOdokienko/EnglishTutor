# Development and operations guide

## Local development

Prerequisite: Python 3.13. The current Python code uses the standard library
for HTTP and provider requests; the repository does not yet define a dependency
manifest or lock file.

| Goal | Action |
| --- | --- |
| Build all saved sessions | python cli.py |
| Build with OpenAI analysis | Set OPENAI_API_KEY, then run python cli.py --use-openai |
| Recompute canonical derived data | python cli.py --recompute-derived |
| Start local app | python server.py |
| Open local session UI | http://127.0.0.1:8000/web/highlights.html |
| Upload transcript | http://127.0.0.1:8000/web/upload.html |

The server starts by rebuilding the static artifact and can write session and
output files. Do not run destructive API operations against a working tree with
uncommitted data that has not been reviewed.

## Environment variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| OPENAI_API_KEY | Backend and CLI | Enables model analysis, annotations and exercises |
| OPENAI_MODEL | Backend and CLI | Overrides general analysis model |
| OPENAI_ANNOTATION_MODEL | Backend and CLI | Overrides annotation model (default gpt-5-mini) |
| OPENAI_ANNOTATION_EFFORT | Backend and CLI | gpt-5 reasoning effort for annotation: low/medium/high (default medium). Effort buys recall; low is where the 0–26 findings-per-chunk spread was measured |
| OPENAI_ANNOTATION_PASSES | Backend and CLI | Independent annotation passes unioned per chunk (default 2). Stabilises recall against run-to-run variance |
| OPENAI_EXERCISE_MODEL | Backend | Overrides exercise model |
| OPENAI_TEST_MODEL | Backend | Model for diagnostic probe |
| OPENAI_ANNOTATION_RESUME | Backend and CLI | Resumes saved annotation chunks |
| OPENAI_ANNOTATION_NO_FALLBACK | Backend and CLI | Disables annotation fallback |
| OPENAI_CHUNK_METRICS | Backend and CLI | Enables chunk grammar metrics |
| OPENAI_CHUNK_METRICS_RESUME | Backend and CLI | Resumes chunk metric work |
| ASSEMBLYAI_API_KEY | Backend | Enables audio upload and transcription |
| ASSEMBLYAI_LANGUAGE | Backend | Optional transcription language |
| ENGLISH_TUTOR_API_BASE_URL | Static build and browser | Public backend base injected into pages |
| ENGLISH_TUTOR_CORS_ORIGIN | Backend | Allowed static frontend origin |
| ENGLISH_TUTOR_DATA_ROOT | Backend and CLI | Moves mutable data (sessions/, data/, out/) to a mounted volume; unset means repository paths |
| ENGLISH_TUTOR_TOKEN | Backend and push_to_prod.py | Shared secret for POST /api/import-session; the endpoint answers 503 until this is set. Same value on the server and in the local .env |
| OPENAI_ANNOTATION_SEVERITY | Backend and CLI | Kill switch for the severity field (ADR-0007): set 0 to drop it from the annotation schema; default on |
| ENGLISH_TUTOR_KEEP_AUDIO | Backend | Set 1 to keep `sessions/<date>/audio.<ext>` after a successful transcription. Unset on prod so the Railway volume stays small; useful for a local run, where disk is free |
| HOST and PORT | Backend | Bind settings for local or hosted server |

Never commit the values of credential variables. `cli.py` reads a `.env` file in
the repository root at import time (dependency-free; only fills variables not
already set, so Railway's dashboard values still win). `.env` is git-ignored —
put local keys there for local runs and eval.

## Production topology (Railway backend + Vercel frontend)

The live deployment (July 2026):

| Piece | Where | Value |
| --- | --- | --- |
| Backend (server.py) | Railway, auto-deploy from git | https://englishtutor-production-ab18.up.railway.app |
| Frontend (out/web) | Vercel, auto-deploy from git | https://english-tutor-delta.vercel.app |
| Mutable data | Railway Volume mounted at /data | ENGLISH_TUTOR_DATA_ROOT=/data |

### Railway data persistence

The Railway filesystem is ephemeral: anything the server writes is lost on
redeploy unless it lives on a Volume. The backend therefore reads
ENGLISH_TUTOR_DATA_ROOT and, when it is set, keeps sessions/, data/ and out/
under that directory. On startup the server seeds the data root from the
repository copies, file by file and without ever overwriting existing files,
so the first boot migrates the versioned history and every later boot is a
no-op for files that already exist. out/web is regenerated from web/ on every
startup, so stale static assets on the volume are not a concern.

One-time Railway setup:

1. In the Railway service, add a Volume and mount it at /data.
2. Set the service variable ENGLISH_TUTOR_DATA_ROOT=/data.
3. Keep ENGLISH_TUTOR_CORS_ORIGIN set to the exact Vercel origin
   (https://english-tutor-delta.vercel.app, no trailing slash).
4. Redeploy and verify /history.json, then upload a test session and redeploy
   again to confirm it survives.

### Frontend to backend wiring

Pages resolve the API base in this order: ?api_base= query parameter (also
persisted to localStorage), the inline value injected at build time from
ENGLISH_TUTOR_API_BASE_URL, then localStorage. The committed out/web is built
with the Railway URL injected, so any browser that opens the Vercel pages
reads and writes against Railway with no manual step. To rebuild after
frontend changes:

    ENGLISH_TUTOR_API_BASE_URL=https://englishtutor-production-ab18.up.railway.app python cli.py

(or regenerate assets only via write_web_assets), then commit out/web; Vercel
picks the commit up automatically.

Remaining production caution: do not leave write APIs publicly exposed for
long — the access token task (plan 1.3) covers this.

## Recovery actions

| Situation | Action |
| --- | --- |
| Derived formula or taxonomy changed | Increment version, run reanalysis, rebuild out/web and commit results |
| Annotation process stops | Re-run annotation; saved chunks resume by default |
| Re-annotating a session on prod | `rebuild-annotations` is a long synchronous LLM job (~6–9 min/session). Fire the POST once, then poll read-only GET on `sessions/<date>/analysis.json`; never put the POST in a poll loop. Done when `annotations_meta.per_chunk` is present and `chunks_processed == total_chunks`. Do sessions one at a time — `update_history` writes one shared file |
| Metrics are stale | Run the derived reanalysis endpoint or CLI command |
| Static site has old data | Rebuild out/web, commit/publish the artifact, then invalidate any hosting cache |
| Accidental session deletion | Restore sessions/<date> and its analysis from Git, rebuild history and out/web |
| Provider credential missing | Set the relevant environment variable; do not add it to source files |
| Audio upload or transcription fails | Use Retry upload from the still-open Record page or upload the automatically downloaded local recording; inspect the retained sessions/<date>/audio.<ext> on the Railway volume if provider diagnosis is needed |

## Keeping the audio

Audio is the only input that cannot be re-derived: `transcript.txt` and
`timings.json` come from it, never the reverse. Losing it closes off
re-transcription with better settings and every sound-based analysis
(pronunciation, intonation) permanently.

The record page already downloads a copy to the operator's machine when the
recording stops (`keepLocalCopy` in web/record.js), named
`english-tutor-<date>-<you>-<partner>.<ext>`. Prod then deletes its own copy so
the Railway volume stays small. Storage is therefore local disk, not the volume
and not git.

File a downloaded recording into the repository layout with:

    python cli.py --import-audio ~/Downloads/english-tutor-2026-07-18-roman-andrey.webm

It reads the date from the filename (`--date` overrides), creates
`sessions/<date>/` when the call was recorded on prod and the folder is missing,
and copies the file to `sessions/<date>/audio.<ext>`. Audio is git-ignored, so
the repository layout is used as an index, not as storage.
