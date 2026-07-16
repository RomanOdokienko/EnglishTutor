#!/usr/bin/env python3
"""Deliver locally computed session artifacts to the production server.

Production data (Railway volume) is canonical; when annotations are rebuilt
LOCALLY — measured against eval/ first, model cost paid once — this script
ships the resulting files via POST /api/import-session, so production pays
neither OpenAI nor an hour of synchronous rebuild.

    python push_to_prod.py 2026-01-19 2026-02-09
    python push_to_prod.py --all

Reads local files only (never writes); the server rebuilds history and web
assets itself after each import. Needs in env or .env:
  ENGLISH_TUTOR_TOKEN         — must match the value configured on the server
  ENGLISH_TUTOR_API_BASE_URL  — the production base URL
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli  # noqa: E402  — reuses .env loading and the data paths

SESSION_SOURCE_FILES = ("meta.json", "transcript.txt", "timings.json")


def build_bundle(date: str) -> dict | None:
    analysis_path = cli.OUT_DIR / "sessions" / date / "analysis.json"
    if not analysis_path.exists():
        print(f"{date}: no local analysis.json, skipping")
        return None
    bundle: dict = {
        "analysis": json.loads(analysis_path.read_text(encoding="utf-8")),
        "session_files": {},
    }
    session_dir = cli.SESSIONS_DIR / date
    for name in SESSION_SOURCE_FILES:
        path = session_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        bundle["session_files"][name] = json.loads(text) if name.endswith(".json") else text
    return bundle


def push(base: str, token: str, date: str) -> bool:
    bundle = build_bundle(date)
    if bundle is None:
        return False
    request = urllib.request.Request(
        f"{base.rstrip('/')}/api/import-session",
        data=json.dumps(bundle, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-ET-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            print(f"{date}: {response.status} {response.read().decode('utf-8')[:200]}")
        return True
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read().decode("utf-8")[:300]
        except Exception:
            pass
        print(f"{date}: HTTP {error.code} {detail}")
        return False
    except urllib.error.URLError as error:
        print(f"{date}: connection error: {error.reason}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dates", nargs="*", help="session dates (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="push every session under out/sessions")
    parser.add_argument("--api-base", default=cli.clean_env("ENGLISH_TUTOR_API_BASE_URL"))
    args = parser.parse_args()

    token = cli.clean_env("ENGLISH_TUTOR_TOKEN")
    if not token:
        print("ENGLISH_TUTOR_TOKEN is not set (env or .env). Nothing to push.")
        return 2
    if not args.api_base:
        print("ENGLISH_TUTOR_API_BASE_URL is not set and --api-base not given.")
        return 2

    dates = args.dates
    if args.all:
        dates = sorted(
            path.name for path in (cli.OUT_DIR / "sessions").iterdir()
            if path.is_dir() and (path / "analysis.json").exists()
        )
    if not dates:
        print("No dates given. Use dates or --all.")
        return 2

    failures = sum(0 if push(args.api_base, token, date) else 1 for date in dates)
    print(f"done: {len(dates) - failures}/{len(dates)} pushed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
