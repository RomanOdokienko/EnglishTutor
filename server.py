#!/usr/bin/env python3
import json
import os
import re
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


import shutil


def delete_session(date: str) -> bool:
    if not date:
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
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return removed

from cli import (
    analyze_session,
    annotate_session,
    build_all,
    call_openai_probe,
    update_history,
    write_analysis,
    write_web_assets,
)


ROOT_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT_DIR / "sessions"
OUT_DIR = ROOT_DIR / "out"
DATA_DIR = ROOT_DIR / "data"
REGISTRY_PATH = DATA_DIR / "people.json"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LINE_RE = re.compile(r"^\s*([^:]+):\s*(.+)$")


def normalize_date(raw_value: str | None) -> str:
    if not raw_value:
        return datetime.now().strftime("%Y-%m-%d")
    if DATE_RE.match(raw_value):
        return raw_value
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
) -> Path:
    session_dir = SESSIONS_DIR / date
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")
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
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL")
    return bool(api_key), model


class UploadHandler(SimpleHTTPRequestHandler):
    def do_POST(self) -> None:
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
                    date = payload.get("date")
                except Exception:
                    date = None

            if date:
                session_dir = SESSIONS_DIR / date
                if not session_dir.exists():
                    self.send_error(404, "Session not found")
                    return
                try:
                    if self.path == "/api/rebuild-annotations":
                        analysis = annotate_session(session_dir, OUT_DIR, openai_model=model)
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
                    payload = json.dumps({"sessions": 1, "date": date}).encode("utf-8")
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
                            analysis = annotate_session(session_dir, OUT_DIR, openai_model=model)
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


def run_server(port: int = 8000) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_web_assets(OUT_DIR)
    build_all(SESSIONS_DIR, OUT_DIR, use_openai=False, openai_model=None)
    handler = partial(UploadHandler, directory=str(OUT_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving on http://127.0.0.1:{port}/web/index.html")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
