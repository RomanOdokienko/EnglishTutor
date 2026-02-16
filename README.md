# Local English Session Evaluator

A local-first MVP that analyzes English conversation transcripts and produces a static HTML report.

## Quick start (CLI)

1) Drop a session into `sessions/<YYYY-MM-DD>/` with:
   - `meta.json`
   - `transcript.txt`
2) Run the CLI:

```bash
python cli.py
```

## Local web app (upload + dashboard)

Start a local server:

```bash
python server.py
```

Open in a browser:
- `http://127.0.0.1:8000/web/index.html` (dashboard)
- `http://127.0.0.1:8000/web/upload.html` (upload)

Upload flow includes manual speaker mapping (Speaker A/B -> Roman/Andrey) and creates:
- `sessions/<YYYY-MM-DD>/meta.json`
- `sessions/<YYYY-MM-DD>/transcript.txt`

You can delete a session from the dashboard (removes session files and history entry).

## OpenAI API (optional)

Set your token to enable model-based analysis in parallel with local metrics:

```powershell
$env:OPENAI_API_KEY="your-token"
```

Optional model override (metrics):

```powershell
$env:OPENAI_MODEL="gpt-4o-mini"
```

Optional annotation model override:

```powershell
$env:OPENAI_ANNOTATION_MODEL="gpt-5-mini"
```

Annotation controls:

```powershell
# Continue from last processed chunk (default = 1)
$env:OPENAI_ANNOTATION_RESUME="1"

# Disable fallback to gpt-4o if gpt-5 returns empty responses
$env:OPENAI_ANNOTATION_NO_FALLBACK="1"
```

Chunked grammar metrics (additional):

```powershell
# Enable/disable (default = 1)
$env:OPENAI_CHUNK_METRICS="1"

# Resume from last processed chunk (default = 1)
$env:OPENAI_CHUNK_METRICS_RESUME="1"
```

CLI usage:

```bash
python cli.py --use-openai
```

For the local server (`python server.py`), if `OPENAI_API_KEY` is set, uploads and re-runs will include LLM output automatically.

## Output

Running the CLI writes to `out/`:

- `out/sessions/<YYYY-MM-DD>/analysis.json` - per-session analysis
- `out/history.json` - aggregated history for the web UI
- `out/web/` - static HTML/CSS/JS viewer

Open `out/web/index.html` in your browser to explore the results.

## Notes
- Speaker mapping aliases are stored in `data/people.json`.
- Progress charts show LLM Proficiency and Grammar error rate per 100 words (per speaker).
- Transcript view is three-column (Original / Annotated / Issues) with per-line issues from LLM annotations.
- Re-run buttons are split: **Re-run metrics** and **Re-run annotations**.
- Annotations run in chunks with incremental save; status shows progress (e.g., `running 3/17`).
- Chunked grammar metrics are computed per chunk and aggregated (shown as “Chunked grammar”).

## Troubleshooting
- **Annotations fail with WinError 10054**: re-run annotations; the server retries and saves progress per chunk. If it still fails, resume is on by default.  
- **Want to restart annotations from scratch**: set `OPENAI_ANNOTATION_RESUME=0` and re-run annotations.  
- **Need to force gpt-5 only (no fallback)**: set `OPENAI_ANNOTATION_NO_FALLBACK=1`.  
- **Model not shown in UI**: re-run annotations; the UI reads `annotations_attempted_model` or `annotations_model`.  
