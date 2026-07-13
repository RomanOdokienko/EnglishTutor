#!/usr/bin/env python3
"""AssemblyAI multichannel transcription.

Records from a remote call arrive as a 2-channel file: channel 1 is the
microphone (the person at this computer), channel 2 is the system audio
(the remote partner). AssemblyAI's multichannel mode transcribes each
channel separately, so speaker separation is deterministic - no diarization
guessing and no manual Speaker A/B mapping.

The public entry point is `transcribe_audio_file`, which returns a transcript
in the existing "Speaker A:/Speaker B:" format so the rest of the pipeline is
untouched.
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.assemblyai.com/v2"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
TRANSCRIPT_ENDPOINT = f"{BASE_URL}/transcript"

CHANNEL_TO_LABEL = {"1": "Speaker A", "2": "Speaker B"}


def _request(url: str, api_key: str, data: bytes | None = None,
             method: str | None = None, content_type: str | None = None,
             timeout: int = 120) -> tuple[dict | None, str | None]:
    headers = {"authorization": api_key}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload), None
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read().decode("utf-8")
        except Exception:
            pass
        return None, f"HTTP {error.code}: {detail or error.reason}"
    except urllib.error.URLError as error:
        return None, f"Connection error: {error.reason}"
    except (TimeoutError, json.JSONDecodeError) as error:
        return None, str(error)


def upload_audio(api_key: str, file_path: str) -> tuple[str | None, str | None]:
    """Upload a local audio file, return its temporary upload_url."""
    with open(file_path, "rb") as handle:
        audio_bytes = handle.read()
    result, error = _request(
        UPLOAD_ENDPOINT,
        api_key,
        data=audio_bytes,
        method="POST",
        content_type="application/octet-stream",
        timeout=300,
    )
    if error:
        return None, f"Upload failed: {error}"
    upload_url = (result or {}).get("upload_url")
    if not upload_url:
        return None, "Upload response missing upload_url."
    return upload_url, None


def create_transcript(api_key: str, audio_url: str, language_code: str = "",
                      multichannel: bool = True) -> tuple[str | None, str | None]:
    """Queue a transcript job, return its id.

    If `language_code` is empty, auto-detect the language. Forcing a wrong
    language (e.g. "en" on Russian speech) makes AssemblyAI return no text.
    """
    body: dict = {"audio_url": audio_url, "multichannel": multichannel}
    if language_code:
        body["language_code"] = language_code
    else:
        body["language_detection"] = True
    data = json.dumps(body).encode("utf-8")
    result, error = _request(
        TRANSCRIPT_ENDPOINT,
        api_key,
        data=data,
        method="POST",
        content_type="application/json",
        timeout=60,
    )
    if error:
        return None, f"Transcript request failed: {error}"
    transcript_id = (result or {}).get("id")
    if not transcript_id:
        return None, "Transcript response missing id."
    return transcript_id, None


def poll_transcript(api_key: str, transcript_id: str, interval: float = 3.0,
                    max_wait: float = 900.0) -> tuple[dict | None, str | None]:
    """Poll until the job completes, errors, or times out."""
    url = f"{TRANSCRIPT_ENDPOINT}/{transcript_id}"
    waited = 0.0
    while waited <= max_wait:
        result, error = _request(url, api_key, method="GET", timeout=60)
        if error:
            return None, error
        status = (result or {}).get("status")
        if status == "completed":
            return result, None
        if status == "error":
            return None, (result or {}).get("error") or "AssemblyAI reported an error."
        time.sleep(interval)
        waited += interval
    return None, f"Timed out after {int(max_wait)}s waiting for transcription."


def _channel_of(utterance: dict) -> str:
    """Multichannel utterances carry the channel in `channel` or `speaker`."""
    for key in ("channel", "speaker"):
        value = utterance.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def utterances_to_transcript(transcript: dict) -> tuple[str, list[str]]:
    """Convert AssemblyAI utterances to "Speaker A:/B:" lines.

    Consecutive utterances from the same channel are merged into one turn.
    Returns (transcript_text, channels_seen).
    """
    utterances = transcript.get("utterances") or []
    lines: list[str] = []
    channels_seen: list[str] = []
    current_label: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_label and current_parts:
            lines.append(f"{current_label}: {' '.join(current_parts).strip()}")

    for utterance in utterances:
        text = (utterance.get("text") or "").strip()
        if not text:
            continue
        channel = _channel_of(utterance)
        if channel and channel not in channels_seen:
            channels_seen.append(channel)
        label = CHANNEL_TO_LABEL.get(channel, f"Speaker {channel or '?'}")
        if label != current_label:
            flush()
            current_label = label
            current_parts = [text]
        else:
            current_parts.append(text)
    flush()

    if not lines:
        # Fall back to the flat transcript text if utterances are absent.
        flat = (transcript.get("text") or "").strip()
        if flat:
            lines.append(f"Speaker A: {flat}")
    return "\n".join(lines), channels_seen


def transcribe_audio_file(api_key: str, file_path: str, language_code: str = "",
                          poll_interval: float = 3.0,
                          max_wait: float = 900.0) -> tuple[dict | None, str | None]:
    """Full flow: upload -> queue -> poll -> convert.

    Returns ({transcript, channels, audio_channels, id}, None) or (None, error).
    """
    upload_url, error = upload_audio(api_key, file_path)
    if error:
        return None, error
    transcript_id, error = create_transcript(api_key, upload_url, language_code=language_code)
    if error:
        return None, error
    result, error = poll_transcript(api_key, transcript_id, interval=poll_interval, max_wait=max_wait)
    if error:
        return None, error
    result = result or {}

    # Persist the raw response next to the audio for inspection/debugging.
    try:
        debug_path = os.path.join(os.path.dirname(file_path) or ".", "assemblyai_response.json")
        with open(debug_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    except OSError:
        pass

    transcript_text, channels = utterances_to_transcript(result)
    if not transcript_text.strip():
        detail = (
            f"status={result.get('status')}, "
            f"audio_channels={result.get('audio_channels')}, "
            f"language={result.get('language_code')}, "
            f"utterances={len(result.get('utterances') or [])}, "
            f"words={len(result.get('words') or [])}, "
            f"text_len={len(result.get('text') or '')}"
        )
        api_error = result.get("error")
        if api_error:
            detail += f", api_error={api_error}"
        return None, f"Transcription returned no text ({detail})."
    return {
        "transcript": transcript_text,
        "channels": channels,
        "audio_channels": result.get("audio_channels"),
        "id": transcript_id,
    }, None
