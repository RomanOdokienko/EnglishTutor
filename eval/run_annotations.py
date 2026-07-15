#!/usr/bin/env python3
"""Annotate a transcript offline and write a candidate file for run_eval.py.

Runs the production annotation path (call_openai_chunk_annotations ->
normalize_annotations -> offset to transcript coordinates) chunk by chunk, so
what gets measured is what ships. It writes only to the path given by --out and
never touches out/sessions, so a measurement run cannot overwrite real data.

    python eval/run_annotations.py --out cand_medium.json --effort medium
    python eval/run_eval.py cand_medium.json

Cost is a rounding error: one session is ~9 chunks, a few cents at most.
Time is not — the whole script is bounded by --deadline (default 7 min) and it
stops cleanly between chunks rather than running away. Expect 9 calls at
roughly 15-40s each on medium effort. Partial results are still written and
still scoreable; the meta block records that they are partial.

Needs OPENAI_API_KEY in the environment or in a local .env (cli.py loads it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cli  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, default=EVAL_DIR / "transcript_2026-07-14.txt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=cli.clean_env("OPENAI_ANNOTATION_MODEL", "gpt-5-mini"))
    parser.add_argument("--effort", default=None, help="gpt-5 reasoning effort: low|medium|high")
    parser.add_argument("--passes", type=int, default=2, help="independent passes unioned per chunk")
    parser.add_argument("--deadline", type=float, default=480.0, help="seconds for the whole run")
    args = parser.parse_args()

    api_key = cli.clean_env("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set (env or .env). Nothing to run.", flush=True)
        return 2
    if args.effort:
        # call_openai_chunk_annotations reads this per request.
        import os

        os.environ["OPENAI_ANNOTATION_EFFORT"] = args.effort

    transcript = args.transcript.read_text(encoding="utf-8")
    blocks = cli.parse_transcript_blocks(transcript)
    chunks = cli.build_chunks(blocks, transcript)
    effort = args.effort or cli.clean_env("OPENAI_ANNOTATION_EFFORT", "medium")
    sizes = [c["range"]["end"] - c["range"]["start"] for c in chunks]
    print(
        f"model={args.model} effort={effort} passes={args.passes}"
        f" chunks={len(chunks)} ({min(sizes)}-{max(sizes)} chars) deadline={args.deadline:.0f}s",
        flush=True,
    )

    started = time.monotonic()
    items: list[dict] = []
    processed = 0
    stopped_early = None

    for chunk in sorted(chunks, key=lambda c: c["range"]["start"]):
        elapsed = time.monotonic() - started
        if elapsed > args.deadline:
            stopped_early = f"deadline hit after {processed}/{len(chunks)} chunks"
            print(f"  ! {stopped_early}", flush=True)
            break

        chunk_text = chunk["text"]
        offset = chunk["range"]["start"]
        call_started = time.monotonic()
        # The production path: N concurrent passes unioned, then normalized.
        normalized, error, chunk_meta = cli.annotate_chunk(chunk_text, api_key, args.model, args.passes)
        took = time.monotonic() - call_started

        if error:
            # One bad chunk should not throw away the rest.
            print(f"  chunk {chunk['index']}: ERROR after {took:.0f}s: {error[:120]}", flush=True)
            processed += 1
            continue

        for item in normalized:
            items.append({**item, "start": item["start"] + offset, "end": item["end"] + offset})
        processed += 1
        kept = sum(1 for i in normalized if cli.is_countable_annotation(i))
        raw = "+".join(str(n) for n in chunk_meta.get("raw_counts", []))
        print(
            f"  chunk {chunk['index']}: passes {raw} raw -> {len(normalized)} union"
            f" -> {kept} countable  ({took:.0f}s, {len(chunk_text)} chars)",
            flush=True,
        )

    payload = {
        "annotation_items": items,
        "meta": {
            "model": args.model,
            "reasoning_effort": effort,
            "passes": args.passes,
            "chunks_processed": processed,
            "total_chunks": len(chunks),
            "elapsed_sec": round(time.monotonic() - started, 1),
            "partial": bool(stopped_early),
            "stopped_early": stopped_early,
            "transcript": str(args.transcript),
        },
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    countable = sum(1 for i in items if cli.is_countable_annotation(i))
    print(flush=True)
    print(f"wrote {args.out}: {len(items)} findings, {countable} countable", flush=True)
    print(f"elapsed {payload['meta']['elapsed_sec']}s", flush=True)
    if stopped_early:
        print(f"PARTIAL RUN — {stopped_early}; scores from it are not comparable", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
