#!/usr/bin/env python3
import hmac
import json
import os
import re
import threading
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


import shutil

from transcribe import transcribe_audio_file


def delete_session(date: str) -> bool:
    if not date:
        return False
    if not is_valid_session_date(date):
        return False
    session_dir = SESSIONS_DIR / date
    out_session_dir = OUT_DIR / "sessions" / date
    removed = False
    if session_dir.exists():
        shutil.rmtree(session_dir)
        removed = True
    if out_session_dir.exists():
        shutil.rmtree(out_session_dir)
        removed = True

    history_path = OUT_DIR / "history.json"
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        history["sessions"] = [
            session for session in history.get("sessions", []) if session.get("date") != date
        ]
        apply_history_comparisons(history["sessions"])
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return removed

from cli import (
    DATA_DIR,
    OUT_DIR,
    SESSIONS_DIR,
    analyze_session,
    annotate_session,
    apply_history_comparisons,
    build_all,
    build_briefing,
    call_openai_highlight_exercise,
    call_openai_probe,
    clean_env,
    ensure_data_root_seeded,
    load_json,
    reanalyze_derived_all,
    update_history,
    write_analysis,
    write_web_assets,
)


REGISTRY_PATH = DATA_DIR / "people.json"
FOCUS_PATH = DATA_DIR / "focus.json"
DISMISSED_PATH = DATA_DIR / "dismissed_examples.json"
FOCUS_CATEGORY_CODES = {"TENSE", "VERB", "ARTICLE", "PREP", "ORDER", "WORD", "COLLOC"}
FOCUS_ACTIVE_LIMIT = 3
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LINE_RE = re.compile(r"^\s*([^:]+):\s*(.+)$")
ANALYSIS_LOCK = threading.Lock()


def is_valid_session_date(raw_value: str | None) -> bool:
    if not raw_value:
        return False
    value = raw_value.strip()
    if not DATE_RE.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def normalize_date(raw_value: str | None) -> str:
    if not raw_value:
        return datetime.now().strftime("%Y-%m-%d")
    if is_valid_session_date(raw_value):
        return raw_value.strip()
    try:
        parsed = datetime.fromisoformat(raw_value)
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        return datetime.now().strftime("%Y-%m-%d")


def detect_participants(transcript_text: str) -> list[dict]:
    speakers: list[str] = []
    for raw_line in transcript_text.splitlines():
        match = LINE_RE.match(raw_line.strip())
        if not match:
            continue
        speaker = match.group(1).strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)

    if not speakers:
        speakers = ["Student"]

    participants = []
    for index, name in enumerate(speakers):
        lowered = name.lower()
        if any(keyword in lowered for keyword in ("tutor", "teacher", "coach")):
            role = "tutor"
        elif index == 0:
            role = "student"
        else:
            role = "partner"
        participants.append({"name": name, "role": role})
    return participants


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"people": {}}
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_registry(mapping: dict) -> None:
    if not mapping:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    people = registry.setdefault("people", {})
    for label, person in mapping.items():
        entry = people.setdefault(person, {"aliases": []})
        aliases = entry.setdefault("aliases", [])
        if label not in aliases:
            aliases.append(label)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def create_session_files(
    transcript_text: str,
    topic: str,
    date: str,
    duration: int,
    participants: list[dict] | None = None,
    speaker_map: dict | None = None,
    timings: dict | None = None,
) -> Path:
    session_dir = SESSIONS_DIR / date
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")
    if timings:
        # Word-level timings from the transcriber (see docs/contracts.md).
        # Source data for fluency metrics; only recorded sessions have it —
        # text uploads legitimately never get this file.
        (session_dir / "timings.json").write_text(
            json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    meta = {
        "date": date,
        "topic": topic,
        "participants": participants or detect_participants(transcript_text),
        "duration_minutes": duration,
    }
    if speaker_map:
        meta["speaker_map"] = speaker_map
        meta["speaker_labels"] = list(speaker_map.keys())
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return session_dir


def openai_config() -> tuple[bool, str | None]:
    api_key = clean_env("OPENAI_API_KEY")
    model = clean_env("OPENAI_MODEL") or None
    return bool(api_key), model


def finish_session_analysis(session_dir: Path, model: str | None) -> None:
    """Finish slow model analysis after the transcript is already durable."""
    try:
        with ANALYSIS_LOCK:
            analysis = analyze_session(
                session_dir,
                use_openai=True,
                openai_model=model,
                out_dir=OUT_DIR,
            )
            write_analysis(OUT_DIR, analysis)
            update_history(OUT_DIR, analysis)
            write_web_assets(OUT_DIR)
        print(f"Background analysis complete: {session_dir.name}", flush=True)
    except Exception as error:
        print(f"Background analysis failed for {session_dir.name}: {error}", flush=True)


def load_focus_data() -> dict:
    data = load_json(FOCUS_PATH, {"focuses": []})
    if not isinstance(data.get("focuses"), list):
        data = {"focuses": []}
    return data


def save_focus_data(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FOCUS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_dismissed_examples() -> dict:
    data = load_json(DISMISSED_PATH, {"dismissed": []})
    if not isinstance(data.get("dismissed"), list):
        data = {"dismissed": []}
    return data


def save_dismissed_examples(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DISMISSED_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class UploadHandler(SimpleHTTPRequestHandler):
    def send_plain_response(self, status_code: int, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        payload = (body or "").encode("utf-8", errors="replace")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def cors_origin(self) -> str:
        return (os.getenv("ENGLISH_TUTOR_CORS_ORIGIN") or "*").strip() or "*"

    def should_send_cors(self) -> bool:
        path = (self.path or "").split("?", 1)[0]
        return (
            path.startswith("/api/")
            or path == "/history.json"
            or path == "/briefing.json"
            or path.startswith("/sessions/")
        )

    def end_headers(self) -> None:
        if self.should_send_cors():
            self.send_header("Access-Control-Allow-Origin", self.cors_origin())
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if self.should_send_cors():
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(204)
        self.end_headers()

    def send_json_response(self, payload: dict, status_code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_plain_response(200, "OK")
            return
        if urlparse(self.path).path == "/api/focus":
            self.send_json_response(load_focus_data())
            return
        if urlparse(self.path).path == "/api/dismiss-example":
            self.send_json_response(load_dismissed_examples())
            return
        if self.path in ("/web", "/web/"):
            location = "/web/highlights.html"
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()
            return
        super().do_GET()

    def handle_upload_audio(self) -> None:
        api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
        if not api_key:
            self.send_plain_response(400, "Missing ASSEMBLYAI_API_KEY.")
            return

        content_length = self.headers.get("Content-Length")
        if not content_length:
            self.send_plain_response(400, "Missing audio body.")
            return
        try:
            body_length = int(content_length)
        except ValueError:
            self.send_plain_response(400, "Invalid Content-Length.")
            return
        if body_length <= 0:
            self.send_plain_response(400, "Empty audio body.")
            return

        audio_bytes = self.rfile.read(body_length)

        params = parse_qs(urlparse(self.path).query)

        def q(name: str, default: str = "") -> str:
            values = params.get(name)
            return values[0].strip() if values else default

        date = normalize_date(q("date"))
        topic = q("topic") or "Recorded session"
        recorder = q("recorder") or "Speaker A"
        other = q("other") or "Speaker B"
        try:
            duration = max(1, int(q("duration", "30")))
        except ValueError:
            duration = 30
        # Empty = auto-detect language (safer than forcing a language and getting no text).
        language = q("language") or os.getenv("ASSEMBLYAI_LANGUAGE", "")
        ext = re.sub(r"[^a-z0-9]", "", q("ext", "webm").lower()) or "webm"

        # Persist the raw recording next to the session so it can be re-processed.
        session_dir = SESSIONS_DIR / date
        session_dir.mkdir(parents=True, exist_ok=True)
        audio_path = session_dir / f"audio.{ext}"
        audio_path.write_bytes(audio_bytes)

        result, error = transcribe_audio_file(api_key, str(audio_path), language_code=language)
        if error:
            self.log_error("Transcription failed: %s", error)
            self.send_plain_response(502, f"Transcription failed: {error}")
            return

        transcript_text = result["transcript"]

        # Channel 1 (mic) = the person recording; channel 2 = the remote partner.
        speaker_map = {"Speaker A": recorder, "Speaker B": other}
        participants = [
            {"name": recorder, "role": "student"},
            {"name": other, "role": "partner"},
        ]
        if recorder and other and recorder != other:
            update_registry(speaker_map)

        create_session_files(
            transcript_text,
            topic,
            date,
            duration,
            participants=participants,
            speaker_map=speaker_map,
            timings=result.get("timings"),
        )

        # Publish the transcript and deterministic metrics immediately. Model
        # annotations run after the response so this request does not remain
        # open for the slowest part of the pipeline.
        use_openai, model = openai_config()
        analysis = analyze_session(
            session_dir,
            use_openai=False,
            openai_model=model,
            out_dir=OUT_DIR,
        )
        write_analysis(OUT_DIR, analysis)
        update_history(OUT_DIR, analysis)
        write_web_assets(OUT_DIR)

        # Audio is the one input that cannot be re-derived: transcript.txt and
        # timings.json come from it, never the other way round. Prod still
        # deletes it so the Railway volume stays small — the record page has
        # already saved a copy on the operator's machine (keepLocalCopy in
        # web/record.js). Set ENGLISH_TUTOR_KEEP_AUDIO=1 where disk is free
        # (a local run) to retain it next to the session instead.
        if clean_env("ENGLISH_TUTOR_KEEP_AUDIO").lower() not in ("1", "true", "yes", "on"):
            try:
                audio_path.unlink(missing_ok=True)
            except OSError as error:
                self.log_error("Could not remove temporary audio %s: %s", audio_path, error)

        analysis_status = "ready"
        if use_openai:
            analysis_status = "processing"
            threading.Thread(
                target=finish_session_analysis,
                args=(session_dir, model),
                daemon=True,
                name=f"analysis-{date}",
            ).start()

        channels = result.get("channels") or []
        warning = None
        if len(channels) < 2:
            warning = (
                "Only one audio channel was detected, so both speakers are merged. "
                "When recording, make sure to share the whole screen with system audio."
            )

        payload = json.dumps({
            "date": date,
            "topic": topic,
            "channels": channels,
            "audio_channels": result.get("audio_channels"),
            "warning": warning,
            "analysis_status": analysis_status,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_focus(self) -> None:
        content_length = self.headers.get("Content-Length")
        try:
            body = self.rfile.read(int(content_length or "0"))
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.send_plain_response(400, "Expected JSON payload.")
            return

        action = str(payload.get("action") or "").strip().lower()
        data = load_focus_data()
        focuses = data["focuses"]

        if action == "set":
            participant = str(payload.get("participant") or "").strip()
            category = str(payload.get("category_code") or "").strip().upper()
            session_date = str(payload.get("session_date") or "").strip()
            if not participant or category not in FOCUS_CATEGORY_CODES or not is_valid_session_date(session_date):
                self.send_plain_response(400, "Need participant, a valid category_code and session_date.")
                return
            active = [
                item for item in focuses
                if item.get("participant") == participant and item.get("status") == "active"
            ]
            if any(item.get("category_code") == category for item in active):
                self.send_plain_response(409, "This category is already an active focus.")
                return
            if len(active) >= FOCUS_ACTIVE_LIMIT:
                self.send_plain_response(409, f"Limit is {FOCUS_ACTIVE_LIMIT} active focuses per participant - close one first.")
                return
            raw_examples = payload.get("examples") if isinstance(payload.get("examples"), list) else []
            examples = []
            for item in raw_examples[:3]:
                if not isinstance(item, dict):
                    continue
                error = str(item.get("error") or "").strip()
                correction = str(item.get("correction") or "").strip()
                if error and correction:
                    examples.append({"error": error, "correction": correction})
            base_id = f"{participant.lower()}-{category.lower()}-{session_date}"
            focus_id = base_id
            suffix = 2
            while any(item.get("id") == focus_id for item in focuses):
                focus_id = f"{base_id}-{suffix}"
                suffix += 1
            focuses.append(
                {
                    "id": focus_id,
                    "participant": participant,
                    "category_code": category,
                    "note": str(payload.get("note") or "").strip(),
                    "examples": examples,
                    "status": "active",
                    "set_date": session_date,
                    "closed_date": None,
                }
            )
        elif action in ("close", "remove"):
            focus_id = str(payload.get("id") or "").strip()
            entry = next((item for item in focuses if item.get("id") == focus_id), None)
            if entry is None:
                self.send_plain_response(404, "Focus not found.")
                return
            if action == "remove":
                focuses.remove(entry)
            else:
                entry["status"] = "closed"
                entry["closed_date"] = datetime.now().strftime("%Y-%m-%d")
        else:
            self.send_plain_response(400, "Unknown action. Use set, close or remove.")
            return

        save_focus_data(data)
        build_briefing(OUT_DIR)
        self.send_json_response(data)

    def handle_dismiss_example(self) -> None:
        """Wave a Rehearse example off This week.

        Mirrors handle_focus: appends to a small data/*.json and rebuilds the
        briefing so the skipped card is replaced by the next-best candidate.
        The (participant, error, correction) triple is enough to hold the pair
        out on every future build; build_briefing keys on it lower-cased.
        """
        content_length = self.headers.get("Content-Length")
        try:
            body = self.rfile.read(int(content_length or "0"))
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.send_plain_response(400, "Expected JSON payload.")
            return

        action = str(payload.get("action") or "dismiss").strip().lower()
        participant = str(payload.get("participant") or "").strip()
        error = str(payload.get("error") or "").strip()
        correction = str(payload.get("correction") or "").strip()
        if not participant or not error or not correction:
            self.send_plain_response(400, "Need participant, error and correction.")
            return

        data = load_dismissed_examples()
        rows = data["dismissed"]
        key = (participant, error.lower(), correction.lower())

        def matches(row: dict) -> bool:
            return (
                str(row.get("participant") or "") == participant
                and str(row.get("error") or "").strip().lower() == error.lower()
                and str(row.get("correction") or "").strip().lower() == correction.lower()
            )

        if action == "restore":
            data["dismissed"] = [row for row in rows if not matches(row)]
        else:
            if not any(matches(row) for row in rows):
                rows.append({
                    "participant": participant,
                    "error": error,
                    "correction": correction,
                    "at": datetime.now().strftime("%Y-%m-%d"),
                })

        save_dismissed_examples(data)
        build_briefing(OUT_DIR)
        self.send_json_response(data)

    def handle_import_session(self) -> None:
        """Accept a locally computed session bundle and store it as-is.

        The cheap counterpart of /api/rebuild-annotations: annotations are
        produced and measured locally (eval harness), then delivered as files,
        so production pays neither OpenAI nor an hour of synchronous rebuild.
        The endpoint writes files, so it never runs unauthenticated: it is
        disabled until ENGLISH_TUTOR_TOKEN is configured, and every request
        must present that token (X-ET-Token header or ?token=).
        """
        expected_token = clean_env("ENGLISH_TUTOR_TOKEN")
        if not expected_token:
            self.send_plain_response(503, "Import disabled: ENGLISH_TUTOR_TOKEN is not configured.")
            return
        params = parse_qs(urlparse(self.path).query)
        supplied = (self.headers.get("X-ET-Token") or (params.get("token") or [""])[0] or "").strip()
        if not hmac.compare_digest(supplied, expected_token):
            self.send_plain_response(401, "Invalid or missing token.")
            return

        try:
            body_length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            body_length = 0
        if body_length <= 0 or body_length > 50_000_000:
            self.send_plain_response(400, "JSON body required (max 50 MB).")
            return
        try:
            bundle = json.loads(self.rfile.read(body_length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self.send_plain_response(400, f"Invalid JSON: {error}")
            return

        analysis = bundle.get("analysis")
        if not isinstance(analysis, dict):
            self.send_plain_response(400, "Bundle must contain an 'analysis' object.")
            return
        date = str(analysis.get("date") or "").strip()
        if not is_valid_session_date(date):
            self.send_plain_response(400, f"Invalid analysis date: {date!r}")
            return
        session_files = bundle.get("session_files") or {}
        # Only the documented session source files are writable — nothing else,
        # so a valid token still cannot write outside sessions/<date>/.
        writable = {"meta.json", "transcript.txt", "timings.json"}
        unknown = sorted(set(session_files) - writable)
        if unknown:
            self.send_plain_response(400, f"Unknown session files: {', '.join(unknown)}")
            return

        try:
            with ANALYSIS_LOCK:
                session_dir = SESSIONS_DIR / date
                session_dir.mkdir(parents=True, exist_ok=True)
                for name, content in session_files.items():
                    target = session_dir / name
                    if name.endswith(".json"):
                        target.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
                    else:
                        target.write_text(str(content), encoding="utf-8")
                # write_analysis runs finalize_derived_metrics, which re-reads a
                # timings.json written above; history and web assets follow.
                write_analysis(OUT_DIR, analysis)
                update_history(OUT_DIR, analysis)
                write_web_assets(OUT_DIR)
        except Exception as error:
            self.send_plain_response(500, f"Import failed: {error}")
            return

        items = (analysis.get("llm") or {}).get("annotation_items") or []
        payload = json.dumps({
            "imported": date,
            "session_files": sorted(session_files),
            "findings": len(items),
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path

        if request_path == "/api/import-session":
            self.handle_import_session()
            return

        if request_path == "/api/focus":
            self.handle_focus()
            return

        if request_path == "/api/dismiss-example":
            self.handle_dismiss_example()
            return

        if request_path == "/api/upload-audio":
            self.handle_upload_audio()
            return

        if request_path == "/api/reanalyze":
            # Recompute derived metrics for all sessions (no LLM) so the whole
            # series is comparable under the current metrics/taxonomy version.
            try:
                count = reanalyze_derived_all(OUT_DIR)
            except Exception as error:
                self.send_plain_response(500, f"Re-analyze failed: {error}")
                return
            payload = json.dumps({"reanalyzed": count}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/api/test-gpt5":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                self.send_error(400, "Missing OPENAI_API_KEY.")
                return
            model = os.getenv("OPENAI_TEST_MODEL", "gpt-5-mini")
            output, error = call_openai_probe(api_key, model)
            if error:
                self.send_error(500, f"Test failed: {error}")
                return
            payload = json.dumps({"model": model, "output": output}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/api/highlight-exercise":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                self.send_plain_response(400, "Missing OPENAI_API_KEY.")
                return
            content_length = self.headers.get("Content-Length")
            if not content_length:
                self.send_plain_response(400, "Missing request body.")
                return
            try:
                body = self.rfile.read(int(content_length))
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_plain_response(400, "Invalid JSON payload.")
                return

            participant_name = str(payload.get("participant_name") or "").strip()
            category_code = str(payload.get("category_code") or "").strip().upper()
            category_title = str(payload.get("category_title") or "").strip()
            focus_text = str(payload.get("focus_text") or "").strip()
            examples = payload.get("examples") if isinstance(payload.get("examples"), list) else []

            if not participant_name or not category_code or not category_title:
                self.send_plain_response(400, "Missing exercise context.")
                return

            model = clean_env("OPENAI_EXERCISE_MODEL") or clean_env("OPENAI_MODEL") or "gpt-4o-mini"
            exercise, error = call_openai_highlight_exercise(
                api_key=api_key,
                model=model,
                participant_name=participant_name,
                category_code=category_code,
                category_title=category_title,
                focus_text=focus_text,
                examples=examples,
            )
            if error:
                self.log_error("Exercise generation failed: %s", error)
                self.send_plain_response(500, f"Exercise generation failed: {error}")
                return

            response_payload = json.dumps({"model": model, "exercise": exercise}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_payload)))
            self.end_headers()
            self.wfile.write(response_payload)
            return

        if self.path == "/api/delete":
            content_length = self.headers.get("Content-Length")
            if not content_length:
                self.send_error(400, "Missing request body.")
                return
            try:
                body = self.rfile.read(int(content_length))
                payload = json.loads(body.decode("utf-8"))
                date = payload.get("date")
            except Exception:
                self.send_error(400, "Invalid JSON payload.")
                return

            if not date:
                self.send_error(400, "Missing date.")
                return
            if not is_valid_session_date(date):
                self.send_error(400, "Invalid date. Expected YYYY-MM-DD.")
                return

            if not delete_session(date):
                self.send_error(404, "Session not found")
                return

            write_web_assets(OUT_DIR)
            payload = json.dumps({"deleted": date}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path in ("/api/rebuild", "/api/rebuild-metrics", "/api/rebuild-annotations"):
            use_openai, model = openai_config()
            content_length = self.headers.get("Content-Length")
            date = None
            if content_length:
                try:
                    body = self.rfile.read(int(content_length))
                    payload = json.loads(body.decode("utf-8"))
                    raw_date = payload.get("date")
                    if raw_date not in (None, ""):
                        if not is_valid_session_date(raw_date):
                            self.send_error(400, "Invalid date. Expected YYYY-MM-DD.")
                            return
                        date = raw_date.strip()
                except Exception:
                    date = None

            if date:
                session_dir = SESSIONS_DIR / date
                if not session_dir.exists():
                    self.send_error(404, "Session not found")
                    return
                try:
                    if self.path == "/api/rebuild-annotations":
                        analysis = annotate_session(session_dir, OUT_DIR, openai_model=model, force_reannotate=True)
                    else:
                        analysis = analyze_session(
                            session_dir,
                            use_openai=use_openai,
                            openai_model=model,
                            run_annotations=self.path != "/api/rebuild-metrics",
                            out_dir=OUT_DIR,
                        )
                    now = datetime.now().strftime("%Y-%m-%d %H:%M")
                    analysis.setdefault("llm", {})
                    if self.path == "/api/rebuild-annotations":
                        analysis["llm"]["annotations_updated_at"] = now
                    else:
                        analysis["llm"]["metrics_updated_at"] = now
                    write_analysis(OUT_DIR, analysis)
                    update_history(OUT_DIR, analysis)
                    write_web_assets(OUT_DIR)
                    result = {"sessions": 1, "date": date}
                    if self.path == "/api/rebuild-annotations":
                        llm = analysis.get("llm", {})
                        result["annotations_status"] = llm.get("annotations_status")
                        result["annotations_error"] = llm.get("annotations_error")
                        result["annotation_items"] = len(llm.get("annotation_items", []))
                    payload = json.dumps(result).encode("utf-8")
                except Exception as error:
                    self.send_error(500, f"Rebuild failed: {error}")
                    return
            else:
                try:
                    if self.path == "/api/rebuild-annotations":
                        count = 0
                        for session_dir in sorted(SESSIONS_DIR.iterdir()):
                            if not session_dir.is_dir():
                                continue
                            if not (session_dir / "meta.json").exists():
                                continue
                            analysis = annotate_session(session_dir, OUT_DIR, openai_model=model, force_reannotate=True)
                            now = datetime.now().strftime("%Y-%m-%d %H:%M")
                            analysis.setdefault("llm", {})
                            analysis["llm"]["annotations_updated_at"] = now
                            write_analysis(OUT_DIR, analysis)
                            update_history(OUT_DIR, analysis)
                            count += 1
                        write_web_assets(OUT_DIR)
                    else:
                        count = build_all(
                            SESSIONS_DIR,
                            OUT_DIR,
                            use_openai=use_openai,
                            openai_model=model,
                        )
                        if self.path == "/api/rebuild-metrics":
                            now = datetime.now().strftime("%Y-%m-%d %H:%M")
                            for session_dir in sorted(SESSIONS_DIR.iterdir()):
                                out_analysis_path = OUT_DIR / "sessions" / session_dir.name / "analysis.json"
                                if not out_analysis_path.exists():
                                    continue
                                analysis = load_json(out_analysis_path, {})
                                analysis.setdefault("llm", {})
                                analysis["llm"]["metrics_updated_at"] = now
                                write_analysis(OUT_DIR, analysis)
                    payload = json.dumps({"sessions": count}).encode("utf-8")
                except Exception as error:
                    self.send_error(500, f"Rebuild failed: {error}")
                    return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path != "/api/upload":
            self.send_error(404, "Not found")
            return

        content_length = self.headers.get("Content-Length")
        if not content_length:
            self.send_error(400, "Missing request body.")
            return
        try:
            body_length = int(content_length)
        except ValueError:
            self.send_error(400, "Invalid Content-Length.")
            return

        body = self.rfile.read(body_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400, "Expected JSON payload.")
            return

        transcript_text = payload.get("transcript", "").strip()
        if not transcript_text:
            self.send_error(400, "Missing transcript text.")
            return

        speaker_a_label = (payload.get("speaker_a_label") or "").strip()
        speaker_b_label = (payload.get("speaker_b_label") or "").strip()
        speaker_a_person = (payload.get("speaker_a_person") or "").strip()
        speaker_b_person = (payload.get("speaker_b_person") or "").strip()

        topic = payload.get("topic") or "Uploaded session"
        date = normalize_date(payload.get("date"))
        duration_raw = payload.get("duration") or "30"
        try:
            duration = max(1, int(duration_raw))
        except ValueError:
            duration = 30

        speaker_map = {}
        participants = None
        if speaker_a_label and speaker_b_label and speaker_a_person and speaker_b_person:
            if speaker_a_label != speaker_b_label and speaker_a_person != speaker_b_person:
                speaker_map = {
                    speaker_a_label: speaker_a_person,
                    speaker_b_label: speaker_b_person,
                }
                participants = [
                    {"name": speaker_a_person, "role": "student"},
                    {"name": speaker_b_person, "role": "partner"},
                ]
                update_registry(speaker_map)

        session_dir = create_session_files(
            transcript_text,
            topic,
            date,
            duration,
            participants=participants,
            speaker_map=speaker_map or None,
        )

        use_openai, model = openai_config()
        analysis = analyze_session(
            session_dir,
            use_openai=use_openai,
            openai_model=model,
            out_dir=OUT_DIR,
        )
        write_analysis(OUT_DIR, analysis)
        update_history(OUT_DIR, analysis)
        write_web_assets(OUT_DIR)

        payload = json.dumps({"date": date, "topic": topic}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def resolve_bind_settings(default_port: int = 8000) -> tuple[str, int]:
    raw_port = (os.getenv("PORT") or "").strip()
    try:
        port = int(raw_port) if raw_port else default_port
    except ValueError:
        port = default_port

    raw_host = (os.getenv("HOST") or "").strip()
    if raw_host:
        host = raw_host
    elif raw_port:
        host = "0.0.0.0"
    else:
        host = "127.0.0.1"
    return host, port


def run_server(port: int = 8000) -> None:
    ensure_data_root_seeded()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_web_assets(OUT_DIR)
    build_all(SESSIONS_DIR, OUT_DIR, use_openai=False, openai_model=None)
    handler = partial(UploadHandler, directory=str(OUT_DIR))
    host, resolved_port = resolve_bind_settings(port)
    server = ThreadingHTTPServer((host, resolved_port), handler)
    print(f"Serving on http://{host}:{resolved_port}/web/highlights.html")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
