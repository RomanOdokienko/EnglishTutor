# Local English Session Evaluator

This repository contains a local-first MVP that analyzes English conversation transcripts and
produces a static HTML report.

## Quick start

1. Drop a session into `sessions/<YYYY-MM-DD>/` with:
   - `meta.json`
   - `transcript.txt`
2. Run the CLI:

```bash
python3 cli.py
```

## Output

Running the CLI writes to `out/`:

- `out/sessions/<YYYY-MM-DD>/analysis.json` — per-session analysis
- `out/history.json` — aggregated history for the web UI
- `out/web/` — static HTML/CSS/JS viewer

Open `out/web/index.html` in your browser to explore the results.
