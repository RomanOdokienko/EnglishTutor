#!/usr/bin/env python3
"""Diagnose an empty transcription against an already-recorded audio file.

Uploads the file once, then submits several transcript configs so we can see
which one actually returns text, and prints the raw AssemblyAI response.

    $env:ASSEMBLYAI_API_KEY="..."
    python diag_transcribe.py [path\\to\\audio.webm]
"""
import json
import os
import sys

from transcribe import upload_audio, poll_transcript, _request, TRANSCRIPT_ENDPOINT

DEFAULT_AUDIO = "sessions/2026-07-13/audio.webm"

CONFIGS = [
    ("A: multichannel + language_code=en", {"multichannel": True, "language_code": "en"}),
    ("B: multichannel + language_detection", {"multichannel": True, "language_detection": True}),
    ("C: speaker_labels + language_detection", {"speaker_labels": True, "language_detection": True}),
]


def summarize(label: str, tr: dict) -> None:
    utt = tr.get("utterances") or []
    words = tr.get("words") or []
    text = tr.get("text") or ""
    print(f"\n=== {label} ===")
    print("  status         :", tr.get("status"))
    print("  error          :", tr.get("error"))
    print("  audio_channels :", tr.get("audio_channels"))
    print("  language_code  :", tr.get("language_code"))
    print("  audio_duration :", tr.get("audio_duration"))
    print("  text length    :", len(text))
    print("  #utterances    :", len(utt))
    print("  #words         :", len(words))
    if text:
        print("  text[:200]     :", repr(text[:200]))
    for u in utt[:3]:
        print("    utt: ch=", u.get("channel"), "spk=", u.get("speaker"),
              "text=", repr((u.get("text") or "")[:80]))


def main() -> None:
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        print("Set ASSEMBLYAI_API_KEY first.")
        sys.exit(1)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not os.path.exists(audio_path):
        print("Audio not found:", audio_path)
        sys.exit(1)

    print("Uploading", audio_path, "…")
    upload_url, error = upload_audio(api_key, audio_path)
    if error:
        print("Upload failed:", error)
        sys.exit(1)
    print("upload_url:", upload_url)

    for label, extra in CONFIGS:
        body = {"audio_url": upload_url, **extra}
        data = json.dumps(body).encode("utf-8")
        created, error = _request(TRANSCRIPT_ENDPOINT, api_key, data=data,
                                  method="POST", content_type="application/json", timeout=60)
        if error:
            print(f"\n=== {label} ===\n  request failed:", error)
            continue
        tid = created.get("id")
        result, error = poll_transcript(api_key, tid, interval=3.0, max_wait=600.0)
        if error:
            print(f"\n=== {label} ===\n  poll error:", error)
            continue
        summarize(label, result)
        # persist the first (current-product) config's full response for the record
        if label.startswith("A"):
            out = os.path.join(os.path.dirname(audio_path), "assemblyai_response.json")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            print("  full response saved to:", out)


if __name__ == "__main__":
    main()
