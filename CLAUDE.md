# CLAUDE.md — orientation for a fresh session

Read this first. It is the map and the list of things that bite. It does not
re-describe what the reference docs already cover well — it points at them and
records the non-obvious implementation logic that lives nowhere else.

## What this is

English Tutor analyses recorded/uploaded English conversations between **exactly
two people** (Roman + Andrey) on their weekly remote call, and shows whether
grammar is improving. Permanently two users: no accounts, no database — JSON
files on disk + a stdlib Python server is the right architecture, not a
limitation. Invest in the learning loop, not infrastructure.

- Product direction & roadmap: [plan.md](plan.md) (active), memory north-star.
- How it's built: [Architecture.md](Architecture.md).
- Doc index: [docs/README.md](docs/README.md).

## Repo map

| Path | What it is |
| --- | --- |
| `cli.py` | The whole pipeline (~3000 lines): transcript parsing, metrics, LLM annotation, history, briefing, web build. Big on purpose — one place. |
| `server.py` | Stdlib `ThreadingHTTPServer`. Upload, rebuild, delete, reanalyze endpoints. Imports data paths from `cli.py`. |
| `transcribe.py` | AssemblyAI multichannel client (mic=ch1, system=ch2 → deterministic Speaker A/B). |
| `web/` | **Frontend source — edit here.** |
| `out/web/` | **Generated. NEVER edit by hand** — `write_web_assets` overwrites it from `web/` on every build/startup. |
| `out/sessions/<date>/analysis.json`, `out/history.json`, `out/briefing.json` | Generated artifacts, committed. |
| `sessions/<date>/` | Source of truth: `meta.json`, `transcript.txt`, optional `audio.<ext>`. |
| `eval/` | Annotation-quality harness (see Quality below). |
| `docs/` | Reference docs + ADRs + migration. |

## The pipeline in one breath

`transcript.txt` → **Layer A** deterministic metrics (`compute_deterministic_metrics`, no LLM, reproducible) + **Layer B** LLM annotations (`annotate_chunk` per chunk) → `finalize_derived_metrics` attaches `participant.derived` (the canonical progress series) → `update_history` → `write_web_assets`. Every output carries `analysis_version` so old and new sessions are comparable only when versions match.

## Load-bearing gotchas (this is why the file exists)

- **Edit `web/`, never `out/web/`.** The latter is regenerated and your change vanishes.
- **Two error counters must agree.** `build_annotation_metrics` (Session page) and `finalize_derived_metrics` (Progress charts) both count through the single gate `is_countable_annotation`. They once diverged by up to 69% on stored data. If you change what counts, change only that function.
- **An annotation finding dict is rebuilt field-by-field in THREE places**: `normalize_annotations`, `build_chunk_annotations`, and the incremental rebuild path in `annotate_session`. A new field (like `confidence`) must be threaded through all three or it silently vanishes before it reaches the counter — and the diff still looks correct.
- **Confidence gate:** the model returns `confidence` (high/medium/low) and `is_stylistic`. Counted = high or medium, not stylistic. `low` and `is_stylistic` are dropped. Legacy items (pre-confidence, no field) still count — don't gate them out or you delete history. The gate is deliberately silent about category: ~39% of gpt-4o-era items have none, and "unlabeled" ≠ "not an error".
- **Extraction is the unstable half, not judgment.** Annotation quality is lost in *finding* errors, not in judging a found phrase. Hence `annotate_chunk` runs 2 concurrent passes unioned by span overlap, over ~1200-char chunks (`ANNOTATION_CHUNK_MAX_CHARS`). Prompt buys precision, effort buys recall, second pass buys recall + stability — measured, not guessed.
- **`reanalyze` ≠ `rebuild-annotations`.** `/api/reanalyze` recomputes derived metrics from stored findings with **no LLM** (cheap, idempotent). `/api/rebuild-annotations` **calls the model** and is a long synchronous job (~6–9 min/session).
- **Re-annotating on prod:** fire the POST **once**, then poll read-only GET on `sessions/<date>/analysis.json`. Never put a mutating call in a poll loop (a poll loop once ran 7h doing exactly that). Completion marker: `annotations_meta.per_chunk` present and `chunks_processed == total_chunks`.
- **Railway FS is ephemeral.** Mutable data survives only under the volume via `ENGLISH_TUTOR_DATA_ROOT=/data`. Data paths are defined once in `cli.py` and imported by `server.py`.
- **Recording is one synchronous HTTP request** through upload+transcribe+analysis. The "15-minute limit" (`max_wait=900` in `transcribe.py`) is a transcription-**wait** timeout, not a recording-length cap. A full ~30-min live recording is **not yet proven end-to-end** (only a ~30s clip and text uploads are). If it times out on Railway, the fix is to make transcription async like annotations already are.

## Measuring annotation quality

`eval/` is how you know if an annotation change helped, instead of guessing.
`eval/eval_set_2026-07-14.json` is hand-labeled (REAL/FP/ART). `run_eval.py`
scores a run by span overlap; `run_annotations.py` produces a candidate offline
without touching prod. See [eval/README.md](eval/README.md). Current prod config
(gpt-5-mini, medium, 2-pass): ~84% precision / ~88% recall. Precision numbers are
trustworthy; recall is measured against a model-built set, so it is optimistic.

## What does NOT exist yet (so you don't assume it does)

No test suite, no CI, no dependency manifest (stdlib only, Python 3.13). JSON
Schema files in `schemas/` are stale and **not** enforced. `docs/quality.md`
describes tests to *add*, not tests that exist. Backend has no auth — write
endpoints must not be publicly exposed for real use (deferred, see
[docs/data-scope.md](docs/data-scope.md)).

## Deploy

Backend on Railway (auto-deploy from `main`), frontend `out/web` on Vercel
(auto-deploy). Full topology + env vars in [docs/operations.md](docs/operations.md).
Keys are backend-only; a local `.env` in repo root is read by `cli.py` and is
git-ignored.
