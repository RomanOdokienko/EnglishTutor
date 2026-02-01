#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class SpeakerMetrics:
    word_count: int
    turn_count: int

    @property
    def avg_words_per_turn(self) -> float:
        if self.turn_count == 0:
            return 0.0
        return round(self.word_count / self.turn_count, 2)


WORD_RE = re.compile(r"[A-Za-z']+")
LINE_RE = re.compile(r"^\s*([^:]+):\s*(.+)$")

ERROR_PATTERNS = [
    {
        "pattern": re.compile(r"\bI was tell\b", re.IGNORECASE),
        "correction": "I was telling",
        "explanation": "Use the -ing form with 'was' to show past continuous.",
    },
    {
        "pattern": re.compile(r"\bI meet\b", re.IGNORECASE),
        "correction": "I met",
        "explanation": "Use the past tense of 'meet' for finished actions.",
    },
    {
        "pattern": re.compile(r"\bmistakes with past tense\b", re.IGNORECASE),
        "correction": "mistakes with the past tense",
        "explanation": "Add the article to refer to a specific tense.",
    },
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_transcript(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        speaker, text = match.groups()
        turns.append((speaker.strip(), text.strip()))
    return turns


def compute_metrics(turns: Iterable[tuple[str, str]]) -> dict[str, SpeakerMetrics]:
    metrics: dict[str, SpeakerMetrics] = {}
    for speaker, text in turns:
        words = WORD_RE.findall(text)
        if speaker not in metrics:
            metrics[speaker] = SpeakerMetrics(word_count=0, turn_count=0)
        metrics[speaker].word_count += len(words)
        metrics[speaker].turn_count += 1
    return metrics


def compute_fluency(metrics: SpeakerMetrics) -> tuple[float, str]:
    score = round((metrics.avg_words_per_turn * 0.8) + (metrics.word_count / 25), 1)
    score = max(1.0, min(10.0, score))
    if score < 4:
        level = "A2"
    elif score < 6:
        level = "B1"
    elif score < 8:
        level = "B2"
    else:
        level = "C1"
    return score, level


def collect_errors(texts: Iterable[str]) -> list[dict]:
    errors = []
    seen = set()
    for text in texts:
        for pattern in ERROR_PATTERNS:
            if pattern["pattern"].search(text):
                key = pattern["correction"]
                if key in seen:
                    continue
                seen.add(key)
                errors.append(
                    {
                        "text": text.strip(),
                        "correction": pattern["correction"],
                        "explanation": pattern["explanation"],
                    }
                )
    return errors


def build_recommendations(errors: list[dict]) -> list[str]:
    recommendations = []
    for error in errors:
        recommendations.append(f"Review: {error['correction']}")
    if not recommendations:
        recommendations = [
            "Practice expanding answers with an extra detail.",
            "Use past-tense verbs consistently when telling stories.",
            "Ask follow-up questions to keep the conversation going.",
        ]
    return recommendations[:3]


def analyze_session(session_dir: Path) -> dict:
    meta = load_json(session_dir / "meta.json")
    turns = load_transcript(session_dir / "transcript.txt")
    metrics_by_speaker = compute_metrics(turns)
    turns_by_speaker: dict[str, list[str]] = {}
    for speaker, text in turns:
        turns_by_speaker.setdefault(speaker, []).append(text)

    participants = []
    for participant in meta["participants"]:
        name = participant["name"]
        metrics = metrics_by_speaker.get(name, SpeakerMetrics(word_count=0, turn_count=0))
        fluency_score, fluency_level = compute_fluency(metrics)
        errors = collect_errors(turns_by_speaker.get(name, []))
        recommendations = build_recommendations(errors)
        participants.append(
            {
                "name": name,
                "role": participant["role"],
                "metrics": {
                    "word_count": metrics.word_count,
                    "turn_count": metrics.turn_count,
                    "avg_words_per_turn": metrics.avg_words_per_turn,
                },
                "fluency": {
                    "score": fluency_score,
                    "level": fluency_level,
                },
                "grammar": {
                    "errors": errors,
                    "top_recommendations": recommendations,
                },
            }
        )

    return {
        "date": meta["date"],
        "session": {
            "topic": meta["topic"],
            "duration_minutes": meta["duration_minutes"],
        },
        "participants": participants,
    }


def write_analysis(out_dir: Path, analysis: dict) -> Path:
    date = analysis["date"]
    session_out = out_dir / "sessions" / date
    session_out.mkdir(parents=True, exist_ok=True)
    analysis_path = session_out / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis_path


def update_history(out_dir: Path, analysis: dict) -> Path:
    history_path = out_dir / "history.json"
    if history_path.exists():
        history = load_json(history_path)
    else:
        history = {"sessions": []}

    history_sessions = [
        session for session in history["sessions"] if session["date"] != analysis["date"]
    ]
    history_sessions.append(
        {
            "date": analysis["date"],
            "topic": analysis["session"]["topic"],
            "participants": [
                {
                    "name": participant["name"],
                    "role": participant["role"],
                    "fluency_score": participant["fluency"]["score"],
                    "grammar_error_count": len(participant["grammar"]["errors"]),
                }
                for participant in analysis["participants"]
            ],
        }
    )
    history_sessions.sort(key=lambda item: item["date"])
    history["sessions"] = history_sessions
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history_path


def write_web_assets(out_dir: Path) -> None:
    web_dir = out_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (web_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (web_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")


def iter_session_dirs(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    return sorted([path for path in sessions_dir.iterdir() if path.is_dir()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysis and history for sessions.")
    parser.add_argument(
        "--sessions-dir",
        default="sessions",
        help="Directory containing session folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="out",
        help="Output directory for analysis, history, and web assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions_dir = Path(args.sessions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = iter_session_dirs(sessions_dir)
    if not session_dirs:
        print("No sessions found.")
        return

    for session_dir in session_dirs:
        analysis = analyze_session(session_dir)
        write_analysis(out_dir, analysis)
        update_history(out_dir, analysis)

    write_web_assets(out_dir)
    print(f"Processed {len(session_dirs)} session(s) at {datetime.now():%Y-%m-%d %H:%M}.")


INDEX_HTML = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>English Session Evaluator</title>
    <link rel=\"stylesheet\" href=\"styles.css\" />
  </head>
  <body>
    <main>
      <header>
        <h1>English Session Evaluator</h1>
        <p class=\"subtitle\">Review fluency progress and session details.</p>
      </header>
      <section class=\"controls\">
        <label for=\"session-select\">Session date</label>
        <select id=\"session-select\"></select>
      </section>
      <section class=\"summary\">
        <h2 id=\"session-title\">Session</h2>
        <div id=\"session-details\" class=\"cards\"></div>
      </section>
      <section class=\"progress\">
        <h2>Progress</h2>
        <canvas id=\"progress-chart\" width=\"640\" height=\"240\"></canvas>
      </section>
      <section class=\"recommendations\">
        <h2>Recommendations</h2>
        <div id=\"recommendations\"></div>
      </section>
    </main>
    <script src=\"app.js\"></script>
  </body>
</html>
"""


APP_JS = """const sessionSelect = document.getElementById('session-select');
const sessionTitle = document.getElementById('session-title');
const sessionDetails = document.getElementById('session-details');
const recommendationsEl = document.getElementById('recommendations');
const chart = document.getElementById('progress-chart');
const ctx = chart.getContext('2d');

const state = {
  history: null,
  analysisCache: new Map(),
};

async function loadHistory() {
  const response = await fetch('../history.json');
  if (!response.ok) {
    throw new Error('Unable to load history.json');
  }
  return response.json();
}

async function loadAnalysis(date) {
  if (state.analysisCache.has(date)) {
    return state.analysisCache.get(date);
  }
  const response = await fetch(`../sessions/${date}/analysis.json`);
  if (!response.ok) {
    throw new Error(`Unable to load analysis for ${date}`);
  }
  const data = await response.json();
  state.analysisCache.set(date, data);
  return data;
}

function renderDropdown() {
  sessionSelect.innerHTML = '';
  state.history.sessions.forEach((session) => {
    const option = document.createElement('option');
    option.value = session.date;
    option.textContent = `${session.date} · ${session.topic}`;
    sessionSelect.appendChild(option);
  });
}

function renderParticipants(analysis) {
  sessionTitle.textContent = `${analysis.date} · ${analysis.session.topic}`;
  sessionDetails.innerHTML = '';
  analysis.participants.forEach((participant) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h3>${participant.name} <span>${participant.role}</span></h3>
      <p><strong>Words:</strong> ${participant.metrics.word_count}</p>
      <p><strong>Turns:</strong> ${participant.metrics.turn_count}</p>
      <p><strong>Avg words/turn:</strong> ${participant.metrics.avg_words_per_turn}</p>
      <p><strong>Fluency:</strong> ${participant.fluency.score} (${participant.fluency.level})</p>
    `;
    sessionDetails.appendChild(card);
  });
}

function renderRecommendations(analysis) {
  recommendationsEl.innerHTML = '';
  analysis.participants.forEach((participant) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'card';
    const items = participant.grammar.top_recommendations
      .map((item) => `<li>${item}</li>`)
      .join('');
    wrapper.innerHTML = `
      <h3>${participant.name}</h3>
      <ul>${items}</ul>
    `;
    recommendationsEl.appendChild(wrapper);
  });
}

function renderChart() {
  const width = chart.width;
  const height = chart.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, width, height);

  const sessions = state.history.sessions;
  if (sessions.length === 0) {
    return;
  }
  const scores = sessions.map((session) =>
    session.participants.map((p) => p.fluency_score)
  );
  const avgScores = scores.map((values) =>
    values.reduce((sum, value) => sum + value, 0) / values.length
  );

  const maxScore = Math.max(...avgScores, 10);
  const minScore = 0;
  const padding = 30;
  const stepX = (width - padding * 2) / Math.max(avgScores.length - 1, 1);

  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  avgScores.forEach((score, index) => {
    const x = padding + index * stepX;
    const normalized = (score - minScore) / (maxScore - minScore);
    const y = height - padding - normalized * (height - padding * 2);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    ctx.fillStyle = '#0ea5e9';
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.stroke();

  ctx.fillStyle = '#334155';
  ctx.font = '12px sans-serif';
  avgScores.forEach((score, index) => {
    const x = padding + index * stepX;
    ctx.fillText(score.toFixed(1), x - 8, height - 8);
  });
}

async function handleSelection() {
  const date = sessionSelect.value;
  const analysis = await loadAnalysis(date);
  renderParticipants(analysis);
  renderRecommendations(analysis);
}

async function init() {
  try {
    state.history = await loadHistory();
    renderDropdown();
    renderChart();
    sessionSelect.addEventListener('change', handleSelection);
    await handleSelection();
  } catch (error) {
    sessionTitle.textContent = 'Failed to load data.';
    sessionDetails.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
""";


STYLES_CSS = """* {
  box-sizing: border-box;
}

body {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  margin: 0;
  background: #f1f5f9;
  color: #0f172a;
}

main {
  max-width: 960px;
  margin: 0 auto;
  padding: 32px 20px 48px;
}

header {
  margin-bottom: 24px;
}

h1 {
  margin: 0 0 8px;
  font-size: 32px;
}

.subtitle {
  margin: 0;
  color: #475569;
}

.controls {
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

select {
  max-width: 320px;
  padding: 8px 12px;
  font-size: 15px;
}

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.card {
  background: #ffffff;
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
}

.card h3 {
  margin: 0 0 8px;
  font-size: 18px;
  display: flex;
  justify-content: space-between;
}

.card h3 span {
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
}

.progress {
  margin-top: 24px;
}

canvas {
  width: 100%;
  background: #ffffff;
  border-radius: 12px;
}

.recommendations {
  margin-top: 24px;
}

.recommendations ul {
  padding-left: 20px;
  margin: 0;
}
"""


if __name__ == "__main__":
    main()
