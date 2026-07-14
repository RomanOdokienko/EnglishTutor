
#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent
# Mutable data (sessions/, data/, out/) can be relocated to a mounted volume
# (e.g. Railway) via ENGLISH_TUTOR_DATA_ROOT; without it everything stays in
# the repository as before. server.py imports these, so this is the single
# source of truth for data paths.
_DATA_ROOT_ENV = (os.getenv("ENGLISH_TUTOR_DATA_ROOT") or "").strip()
DATA_ROOT = Path(_DATA_ROOT_ENV).resolve() if _DATA_ROOT_ENV else ROOT_DIR
SESSIONS_DIR = DATA_ROOT / "sessions"
OUT_DIR = DATA_ROOT / "out"
DATA_DIR = DATA_ROOT / "data"
WEB_SRC_DIR = ROOT_DIR / "web"

WORD_RE = re.compile(r"[A-Za-z']+")
LINE_RE = re.compile(r"^\s*([^:]+):\s*(.*)$")
BLOCK_RE = re.compile(r"^\s*([^:]+):\s*(.*)$", re.MULTILINE)
FILLER_RE = re.compile(r"[A-Za-zА-Яа-яЁё']+")

def _copy_missing(src: Path, dst: Path) -> None:
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _copy_missing(child, dst / child.name)
    elif not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ensure_data_root_seeded() -> None:
    """Seed ENGLISH_TUTOR_DATA_ROOT with the repo-versioned data.

    Idempotent per file: existing files under DATA_ROOT are never overwritten,
    so restarts and redeploys only fill in whatever is missing. No-op when
    ENGLISH_TUTOR_DATA_ROOT is not set.
    """
    if DATA_ROOT == ROOT_DIR:
        return
    for name in ("sessions", "data", "out"):
        src = ROOT_DIR / name
        if src.exists():
            _copy_missing(src, DATA_ROOT / name)


def clean_env(name: str, default: str = "") -> str:
    """Read an env var trimmed of stray whitespace.

    A dashboard-set value like "gpt-5-mini " (trailing space) would otherwise
    reach the API verbatim and 404 as an unknown model.
    """
    value = (os.getenv(name) or "").strip()
    return value or default


OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANNOTATION_MODEL = "gpt-4o"
ANNOTATION_MAX_CHUNKS = 0
ANNOTATION_BLOCKS_PER_CHUNK = 6


@dataclass
class SpeakerMetrics:
    word_count: int
    turn_count: int
    unique_word_count: int

    @property
    def avg_words_per_turn(self) -> float:
        if self.turn_count == 0:
            return 0.0
        return round(self.word_count / self.turn_count, 2)

    @property
    def lexical_diversity(self) -> float:
        if self.word_count == 0:
            return 0.0
        return round(self.unique_word_count / self.word_count, 2)


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_transcript(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    current_speaker: str | None = None
    buffer: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if match:
            if current_speaker is not None and buffer:
                turns.append((current_speaker, " ".join(buffer).strip()))
            current_speaker, first_text = match.groups()
            current_speaker = current_speaker.strip()
            buffer = [first_text.strip()]
        else:
            if current_speaker is not None:
                buffer.append(line)
    if current_speaker is not None and buffer:
        turns.append((current_speaker, " ".join(buffer).strip()))
    return turns


def parse_transcript_blocks(transcript_text: str) -> list[dict]:
    blocks: list[dict] = []
    matches = list(BLOCK_RE.finditer(transcript_text))
    if not matches:
        return blocks

    for index, match in enumerate(matches):
        speaker = match.group(1).strip()
        block_start = match.start()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(transcript_text)
        text = transcript_text[block_start:content_end].strip()
        blocks.append(
            {
                "index": index,
                "speaker": speaker,
                "text": text,
                "range": {"start": block_start, "end": content_end},
            }
        )
    return blocks


def build_chunks(
    blocks: list[dict],
    transcript_text: str,
    max_blocks: int = ANNOTATION_BLOCKS_PER_CHUNK,
) -> list[dict]:
    if not blocks:
        return []
    chunks: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        start_pos = current[0]["range"]["start"]
        end_pos = current[-1]["range"]["end"]
        chunk_text = transcript_text[start_pos:end_pos]
        chunks.append(
            {
                "index": len(chunks),
                "block_indices": [item["index"] for item in current],
                "text": chunk_text,
                "range": {"start": start_pos, "end": end_pos},
            }
        )
        current = []

    for block in blocks:
        if current and len(current) >= max_blocks:
            flush()
        current.append(block)

    flush()
    return chunks


def compute_metrics(turns: Iterable[tuple[str, str]]) -> dict[str, SpeakerMetrics]:
    metrics: dict[str, SpeakerMetrics] = {}
    unique_words: dict[str, set[str]] = {}
    for speaker, text in turns:
        words = WORD_RE.findall(text)
        if speaker not in metrics:
            metrics[speaker] = SpeakerMetrics(word_count=0, turn_count=0, unique_word_count=0)
            unique_words[speaker] = set()
        metrics[speaker].word_count += len(words)
        unique_words[speaker].update(word.lower() for word in words)
        metrics[speaker].turn_count += 1
        metrics[speaker].unique_word_count = len(unique_words[speaker])
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


def normalize_llm_fluency_score(raw_score: object) -> float | None:
    if raw_score is None:
        return None
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None

    # Keep one canonical scale in storage/UI: 0..10.
    if 0.0 <= score <= 1.0:
        score *= 10.0
    elif 10.0 < score <= 100.0:
        score /= 10.0

    score = max(0.0, min(10.0, score))
    return round(score, 1)


def normalize_analysis_fluency_scores(analysis: dict) -> None:
    for participant in analysis.get("participants", []):
        llm = participant.get("llm")
        if not isinstance(llm, dict):
            continue
        fluency = llm.get("fluency")
        if not isinstance(fluency, dict):
            continue
        normalized = normalize_llm_fluency_score(fluency.get("score"))
        if normalized is None:
            continue
        fluency["score"] = normalized


def extract_output_text(response: dict) -> str:
    direct_text = response.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    text_block = response.get("text")
    if isinstance(text_block, dict):
        value = text_block.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()

    outputs = response.get("output", [])
    texts: list[str] = []
    for item in outputs:
        for content in item.get("content", []):
            if content.get("type") == "output_text" and "text" in content:
                text_value = content["text"]
                if isinstance(text_value, dict):
                    text_value = text_value.get("value")
                if isinstance(text_value, str):
                    texts.append(text_value)
            elif content.get("type") == "text":
                text_value = content.get("text")
                if isinstance(text_value, dict):
                    text_value = text_value.get("value")
                if isinstance(text_value, str):
                    texts.append(text_value)
            elif "text" in content and isinstance(content["text"], str):
                texts.append(content["text"])
    return "\n".join(texts).strip()


def escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def escape_attr(value: str) -> str:
    return escape_html(value).replace('"', "&quot;")


def call_openai_probe(api_key: str, model: str) -> tuple[str | None, str | None]:
    if not api_key:
        return None, "Missing API key"

    payload = {
        "model": model,
        "input": "Reply with the single word OK.",
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "text"}},
        "max_output_tokens": 200,
    }
    request = urllib.request.Request(
        OPENAI_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8")
        except Exception:
            body = str(error)
        return None, body
    except Exception as error:
        return None, str(error)

    output_text = extract_output_text(response_payload)
    if not output_text:
        preview = json.dumps(response_payload)[:600]
        return None, f"Empty response. Raw: {preview}"
    return output_text.strip(), None


def normalize_annotations(text: str, errors: list[dict]) -> list[dict]:
    valid_categories = {"TENSE", "VERB", "ARTICLE", "PREP", "ORDER", "WORD", "COLLOC"}
    excluded_categories = {"ASR_LOW_CONF", "NOISE", "FILLER_NATIVE", "FRAGMENT"}
    normalized: list[dict] = []
    for item in errors:
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        snippet = (item.get("text") or "").strip()
        if snippet and snippet in text:
            if snippet not in text[start:end]:
                found_at = text.find(snippet)
                if found_at != -1:
                    start = found_at
                    end = found_at + len(snippet)
        if not snippet:
            snippet = text[start:end].strip()
        if len(snippet) < 4:
            continue

        # Discard single-word spans to reduce noise.
        if len(snippet.split()) < 2:
            continue
        if len(snippet.split()) > 24:
            continue

        # Expand to word boundaries to avoid single-character highlights.
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        while end < len(text) and text[end].isalnum():
            end += 1
        snippet = text[start:end].strip()
        if len(snippet) < 4:
            continue
        if len(snippet.split()) < 2:
            continue
        if len(snippet.split()) > 24:
            continue
        category = (item.get("category") or "").strip().upper()
        if category and category not in valid_categories:
            category = ""
        if not category:
            category = infer_category_code(item)
        if category in excluded_categories:
            continue
        if is_punctuation_only_change(snippet, item.get("correction", "")):
            continue
        if is_filler_text(snippet):
            continue
        if has_disfluency_pattern(snippet):
            continue
        if is_non_evaluated_explanation(item.get("explanation", "")):
            continue
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": snippet,
                "correction": item.get("correction", ""),
                "explanation": item.get("explanation", ""),
                "category": category,
            }
        )

    normalized.sort(key=lambda item: (item["start"], item["end"]))
    cleaned: list[dict] = []
    last_end = -1
    for item in normalized:
        if item["start"] < last_end:
            continue
        cleaned.append(item)
        last_end = item["end"]
    return cleaned


def is_punctuation_only_change(text: str, correction: str) -> bool:
    if not text or not correction:
        return False
    def strip_punct(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", value).lower()
    return strip_punct(text) == strip_punct(correction)


def is_filler_text(text: str) -> bool:
    tokens = [token.lower() for token in FILLER_RE.findall(text)]
    if not tokens:
        return False
    fillers = {
        "uh",
        "um",
        "er",
        "hmm",
        "mmm",
        "yeah",
        "yeahh",
        "mm",
        "mmm",
        "ээ",
        "эээ",
        "ээм",
        "эм",
        "мм",
        "ну",
        "нуу",
    }
    return all(token in fillers for token in tokens)


def is_non_evaluated_explanation(explanation: str) -> bool:
    text = (explanation or "").lower()
    if not text:
        return False
    keywords = (
        "asr",
        "low confidence",
        "noise",
        "non-speech",
        "nonspeech",
        "filler",
        "hesitation",
        "unclear audio",
        "audio unclear",
    )
    return any(keyword in text for keyword in keywords)


def build_annotated_html(text: str, errors: list[dict]) -> str:
    if not errors:
        return escape_html(text)
    parts: list[str] = []
    cursor = 0
    for item in errors:
        start = item["start"]
        end = item["end"]
        if start > cursor:
            parts.append(escape_html(text[cursor:start]))
        segment = escape_html(text[start:end])
        correction = item.get("correction", "")
        explanation = item.get("explanation", "")
        category = item.get("category", "")
        title = correction or explanation
        if correction and explanation:
            title = f"Correct: {correction} | {explanation}"
        elif correction:
            title = f"Correct: {correction}"
        elif explanation:
            title = explanation
        else:
            title = "Grammar issue"
        if category:
            title = f"{category}: {title}"
        parts.append(
            f'<mark class="grammar-error" title="{escape_attr(title)}">{segment}</mark>'
        )
        cursor = end
    if cursor < len(text):
        parts.append(escape_html(text[cursor:]))
    return "".join(parts)

ERROR_TYPE_RULES = [
    {
        "key": "subject_verb_agreement",
        "match": ["subject-verb agreement", "subject verb agreement", "subject verb"],
        "title": "Subject-verb agreement",
        "guidance": "Ensure the verb matches the subject in number and tense.",
        "keywords": ["subject-verb", "subject verb", "agreement", "agree in number", "agree in tense"],
    },
    {
        "key": "missing_article",
        "match": ["missing article", "missing articles", "article usage", "article"],
        "title": "Missing articles",
        "guidance": "Add a/an/the before singular count nouns when needed.",
        "keywords": ["article", "a/an/the", "missing article"],
    },
    {
        "key": "verb_form",
        "match": ["incorrect verb form", "verb form", "verb tense", "incorrect tense"],
        "title": "Incorrect verb form/tense",
        "guidance": "Use the correct verb form for tense and aspect (e.g., past vs. present).",
        "keywords": ["verb form", "verb tense", "tense", "conjugation"],
    },
    {
        "key": "sentence_fragment",
        "match": ["sentence fragment", "fragment"],
        "title": "Sentence fragments",
        "guidance": "Make sure each sentence has a clear subject and verb.",
        "keywords": ["fragment", "incomplete sentence"],
    },
    {
        "key": "run_on",
        "match": ["run-on sentence", "run on sentence", "run-on"],
        "title": "Run-on sentences",
        "guidance": "Split long sentences or use proper conjunctions/punctuation.",
        "keywords": ["run-on", "run on", "comma splice"],
    },
    {
        "key": "repetition",
        "match": ["repetition", "repeated word", "word repetition"],
        "title": "Repetition",
        "guidance": "Avoid repeating the same word or phrase in close proximity.",
        "keywords": ["repetition", "repeated"],
    },
    {
        "key": "word_choice",
        "match": ["incorrect word choice", "word choice", "wrong word"],
        "title": "Word choice",
        "guidance": "Choose words that fit the intended meaning and context.",
        "keywords": ["word choice", "wrong word", "incorrect word"],
    },
    {
        "key": "preposition",
        "match": ["preposition", "incorrect preposition"],
        "title": "Prepositions",
        "guidance": "Use the correct preposition for common phrases and verbs.",
        "keywords": ["preposition"],
    },
    {
        "key": "pronoun",
        "match": ["pronoun"],
        "title": "Pronouns",
        "guidance": "Ensure pronouns agree with their antecedents in number and case.",
        "keywords": ["pronoun"],
    },
    {
        "key": "word_order",
        "match": ["word order", "incorrect order"],
        "title": "Word order",
        "guidance": "Keep standard English word order (subject–verb–object).",
        "keywords": ["word order", "incorrect order"],
    },
]

CATEGORY_LABELS = {
    "TENSE": "Verb Tense",
    "VERB": "Verb Form",
    "ARTICLE": "Articles",
    "PREP": "Prepositions",
    "ORDER": "Word Order",
    "WORD": "Wrong Word",
    "COLLOC": "Collocation",
}

# Taxonomy v1 spelled out for the annotation prompt. Definitions and examples
# pin the category boundaries so repeated runs classify the same error the
# same way (docs/metrics-and-taxonomy.md is the canonical reference).
ANNOTATION_TAXONOMY_PROMPT = (
    "Classify every error with exactly one category code from this fixed taxonomy:\n"
    "- TENSE: wrong verb tense or aspect for the timeline. Example: 'Yesterday I go there' -> 'Yesterday I went there'.\n"
    "- VERB: wrong verb form: subject-verb agreement, infinitive vs gerund, modal, participle. Example: 'He don't know' -> 'He doesn't know'.\n"
    "- ARTICLE: missing, extra, or wrong a/an/the. Example: 'I have car' -> 'I have a car'.\n"
    "- PREP: wrong or missing preposition. Example: 'It depends of him' -> 'It depends on him'.\n"
    "- ORDER: wrong word order, broken or incomplete sentence structure. Example: 'I know what should I do' -> 'I know what I should do'.\n"
    "- WORD: wrong, non-existent, or redundant word that makes the phrase imprecise. Example: 'something like else' -> 'something else'.\n"
    "- COLLOC: unnatural word combination, literal translation instead of the standard pairing. Example: 'make a photo' -> 'take a photo'.\n"
    "The category MUST be exactly one of: TENSE, VERB, ARTICLE, PREP, ORDER, WORD, COLLOC. Never use any other value.\n"
    "If several codes could apply, prefer the specific grammar code (TENSE, VERB, ARTICLE, PREP, ORDER) over WORD, "
    "and use COLLOC only when the word combination is unnatural rather than a single word being wrong."
)

CATEGORY_KEYWORDS = {
    "TENSE": ["tense", "past", "present", "future", "aspect"],
    "VERB": ["verb form", "conjugation", "infinitive", "agreement", "modal", "verb"],
    "ARTICLE": ["article", "a/an/the"],
    "PREP": ["preposition"],
    "ORDER": [
        "word order",
        "sentence structure",
        "fragment",
        "run-on",
        "incomplete",
        "missing subject",
        "missing verb",
        "pronoun reference",
        "conjunction",
        "negation",
        "misplaced modifier",
        "awkward structure",
        "incorrect structure",
    ],
    "WORD": [
        "word choice",
        "wrong word",
        "incorrect word",
        "awkward",
        "clarity",
        "redundant",
        "repetition",
        "phrase",
        "wordiness",
    ],
    "COLLOC": ["collocation"],
}

PRACTICAL_GUIDANCE = {
    "TENSE": "Match verb tense to time markers and keep tense consistent within a sentence.",
    "VERB": "Use the correct verb form and ensure subject-verb agreement and modal usage.",
    "ARTICLE": "Use a/an/the (or zero article) correctly before nouns.",
    "PREP": "Choose the correct preposition for common phrases and verbs.",
    "ORDER": "Keep standard word order and avoid fragments or run-ons.",
    "WORD": "Pick the most precise word and avoid awkward or redundant phrasing.",
    "COLLOC": "Use natural word combinations instead of literal translations.",
}

INSIGHT_IMPACT = {
    "TENSE": "This creates timeline confusion in your message.",
    "VERB": "This makes grammar sound inconsistent and less natural.",
    "ARTICLE": "This reduces grammatical accuracy in noun phrases.",
    "PREP": "This changes meaning in key phrases and verb patterns.",
    "ORDER": "This makes ideas harder to follow on first read.",
    "WORD": "This lowers precision and can sound unnatural.",
    "COLLOC": "This sounds translated and less native-like.",
}


def normalize_error_type(title: str) -> dict:
    normalized = (title or "").strip()
    lowered = normalized.lower()
    for rule in ERROR_TYPE_RULES:
        if any(token in lowered for token in rule["match"]):
            return {
                "key": rule["key"],
                "title": rule["title"],
                "guidance": rule["guidance"],
                "keywords": rule["keywords"],
            }
    return {
        "key": lowered or "unknown",
        "title": normalized or "Other",
        "guidance": "Review this pattern and compare against recent examples.",
        "keywords": [lowered] if lowered else [],
    }


IGNORED_CHUNK_ERROR_KEYWORDS = (
    "punctuation",
    "capitalization",
    "comma",
    "apostrophe",
    "quote",
    "quotes",
    "bracket",
    "parenthesis",
    "dash",
    "hyphen",
)


def normalize_chunk_error_title(title: str) -> str | None:
    raw = (title or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if any(keyword in lowered for keyword in IGNORED_CHUNK_ERROR_KEYWORDS):
        return None
    if "article" in lowered:
        return "Articles"
    if "preposition" in lowered:
        return "Prepositions"
    if "collocation" in lowered:
        return "Collocation"
    if "word order" in lowered or "incorrect order" in lowered:
        return "Word Order"
    if "verb tense" in lowered or "tense" in lowered:
        return "Verb Tense"
    if (
        "verb form" in lowered
        or "conjugation" in lowered
        or "word form" in lowered
        or "infinitive" in lowered
        or "subject-verb" in lowered
        or "subject verb" in lowered
        or "modal" in lowered
        or "agreement" in lowered
        or "not agree" in lowered
    ):
        return "Verb Form"
    if (
        "word choice" in lowered
        or "incorrect word" in lowered
        or "wrong word" in lowered
        or "wordiness" in lowered
        or "redundant" in lowered
        or "redundancy" in lowered
        or "repetition" in lowered
        or "repeated word" in lowered
        or "repeated words" in lowered
        or "noun form" in lowered
        or "clarity of expression" in lowered
        or "misused word" in lowered
        or "phrasing error" in lowered
        or "confusing phrasing" in lowered
        or "clarity issues" in lowered
        or "clarity" in lowered
        or "incorrect phrase" in lowered
        or "repetitive phrase" in lowered
        or "repetitive language" in lowered
        or "improper phrase" in lowered
        or "awkward phrase" in lowered
        or "awkward phrasing" in lowered
        or "incorrect phrasing" in lowered
        or "confusing phrase" in lowered
        or "sentence clarity" in lowered
    ):
        return "Wrong Word"
    if (
        "sentence fragment" in lowered
        or "fragment" in lowered
        or "run-on" in lowered
        or "run on" in lowered
        or "sentence structure" in lowered
        or "incorrect structure" in lowered
        or "unclear structure" in lowered
        or "unclear" in lowered
        or "phrase structure" in lowered
        or "incomplete sentence" in lowered
        or "incomplete statement" in lowered
        or "missing subject" in lowered
        or "missing verb" in lowered
        or "pronoun reference" in lowered
        or "incorrect pronoun" in lowered
        or "ambiguous reference" in lowered
        or "pronoun confusion" in lowered
        or "pronoun referencing" in lowered
        or "conjunction misuse" in lowered
        or "overusing conjunctions" in lowered
        or "negation" in lowered
        or "awkward structure" in lowered
        or "misplaced modifier" in lowered
        or "incomplete clause" in lowered
    ):
        return "Word Order"
    if "incorrect use of verb" in lowered:
        return "Verb Form"
    if "comparative form" in lowered:
        return "Verb Form"
    return raw


def map_annotation_items_to_speakers(
    annotation_items: list[dict],
    transcript_text: str,
    speaker_map: dict,
) -> dict[str, list[dict]]:
    blocks = parse_transcript_blocks(transcript_text)
    if not blocks:
        return {}
    sorted_items = sorted(annotation_items, key=lambda item: int(item.get("start", 0)))
    by_name: dict[str, list[dict]] = {}
    block_index = 0
    for item in sorted_items:
        try:
            start = int(item.get("start", 0))
        except (TypeError, ValueError):
            continue
        while block_index < len(blocks) and start >= blocks[block_index]["range"]["end"]:
            block_index += 1
        if block_index >= len(blocks):
            break
        block = blocks[block_index]
        if not (block["range"]["start"] <= start < block["range"]["end"]):
            continue
        label = block.get("speaker", "")
        name = speaker_map.get(label, label) if label else ""
        if not name:
            continue
        by_name.setdefault(name, []).append(item)
    return by_name


def infer_category_code(item: dict) -> str:
    haystack = f"{item.get('explanation', '')} {item.get('correction', '')} {item.get('text', '')}".lower()
    for code, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return code
    return ""


def normalize_compare_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def has_meaningful_correction(text: str, correction: str) -> bool:
    if not text or not correction:
        return False
    return normalize_compare_text(text) != normalize_compare_text(correction)


def build_insight_reason(code: str, count: int, total_errors: int) -> str:
    if total_errors <= 0:
        return "This issue appears repeatedly in the session."
    share = round((count / total_errors) * 100)
    impact = INSIGHT_IMPACT.get(code, "This pattern reduces clarity and accuracy.")
    return f"{count} cases ({share}% of all grammar issues). {impact}"


def tokenize_words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (value or "").lower())


def token_diff_size(text: str, correction: str) -> int:
    src = Counter(tokenize_words(text))
    dst = Counter(tokenize_words(correction))
    removed = src - dst
    added = dst - src
    return sum(removed.values()) + sum(added.values())


def has_disfluency_pattern(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if lowered.startswith("i mean") or lowered.startswith("you know"):
        return True
    # Example: "I. I", "the the", "we, we"
    if re.search(r"\b([a-z]{1,5})\b[\s,.;:!?-]+\1\b", lowered):
        return True
    return False


def is_clean_example_pair(text: str, correction: str) -> bool:
    if not has_meaningful_correction(text, correction):
        return False
    if "\n" in text or "\n" in correction:
        return False
    words = tokenize_words(text)
    correction_words = tokenize_words(correction)
    if len(words) < 2 or len(words) > 18:
        return False
    if len(correction_words) < 2 or len(correction_words) > 24:
        return False
    if len(re.findall(r"[.!?]", text)) > 1:
        return False
    if has_disfluency_pattern(text) or is_filler_text(text):
        return False
    diff_size = token_diff_size(text, correction)
    if diff_size == 0:
        return False
    # If the correction rewrites too much, it's usually noisy for examples.
    if diff_size > max(8, int(len(words) * 0.8)):
        return False
    return True


def example_quality_score(text: str, correction: str) -> int:
    words_len = len(tokenize_words(text))
    diff_size = token_diff_size(text, correction)
    score = 0
    if 3 <= words_len <= 10:
        score += 3
    elif words_len <= 14:
        score += 2
    elif words_len <= 18:
        score += 1
    if 1 <= diff_size <= 4:
        score += 3
    elif diff_size <= 7:
        score += 1
    else:
        score -= 2
    if re.search(r"[.!?]", text):
        score -= 1
    if has_disfluency_pattern(text):
        score -= 3
    return score


def build_annotation_metrics(analysis: dict, transcript_text: str) -> None:
    annotation_items = analysis.get("llm", {}).get("annotation_items") or []
    if not annotation_items:
        return
    speaker_map = analysis.get("speaker_map", {})
    by_name = map_annotation_items_to_speakers(annotation_items, transcript_text, speaker_map)
    for participant in analysis.get("participants", []):
        name = participant.get("name")
        if not name:
            continue
        items = by_name.get(name, [])
        counts: dict[str, int] = {}
        for item in items:
            text = (item.get("text") or "").strip()
            correction = (item.get("correction") or "").strip()
            if not text or not correction:
                continue
            word_count = len(tokenize_words(text))
            if word_count < 2 or word_count > 24:
                continue
            if has_disfluency_pattern(text) or is_filler_text(text):
                continue
            if not has_meaningful_correction(text, correction):
                continue
            code = (item.get("category") or "").strip().upper()
            if not code:
                code = infer_category_code(item)
            if not code:
                continue
            counts[code] = counts.get(code, 0) + 1
            item["category"] = code
        error_types = [
            {"code": code, "title": CATEGORY_LABELS.get(code, code), "count": count}
            for code, count in sorted(counts.items(), key=lambda it: it[1], reverse=True)
        ]
        total_errors = sum(item["count"] for item in error_types)
        word_count = participant.get("metrics", {}).get("word_count", 0)
        rate = round((total_errors / word_count) * 100, 2) if word_count else 0.0
        participant.setdefault("llm", {})
        participant["llm"]["annotation_grammar"] = {
            "total_errors": total_errors,
            "error_rate_per_100_words": rate,
            "error_types": error_types,
        }


def match_annotation_to_type(item: dict, keywords: list[str]) -> bool:
    if not keywords:
        return False
    haystack = f"{item.get('explanation', '')} {item.get('correction', '')}".lower()
    return any(keyword in haystack for keyword in keywords)


def build_practical_recommendations(analysis: dict, transcript_text: str) -> None:
    annotation_items = analysis.get("llm", {}).get("annotation_items") or []
    if not annotation_items:
        return
    speaker_map = analysis.get("speaker_map", {})
    by_name = map_annotation_items_to_speakers(annotation_items, transcript_text, speaker_map)

    for participant in analysis.get("participants", []):
        name = participant.get("name")
        annotation_grammar = participant.get("llm", {}).get("annotation_grammar") if participant.get("llm") else None
        if not name or not annotation_grammar:
            continue
        top_errors = annotation_grammar.get("error_types") or []
        ordered = top_errors[:3]
        total_errors = int(annotation_grammar.get("total_errors", 0) or 0)

        items_for_speaker = by_name.get(name, [])
        used_texts: set[str] = set()
        practical: list[dict] = []
        insights: list[dict] = []
        for bucket in ordered:
            code = (bucket.get("code") or "").strip().upper()
            label = bucket.get("title") or CATEGORY_LABELS.get(code, code)
            guidance = PRACTICAL_GUIDANCE.get(code, "")
            count = int(bucket.get("count", 0) or 0)
            examples: list[dict] = []
            candidates: list[tuple[int, int, str, str]] = []
            for item in items_for_speaker:
                text = (item.get("text") or "").strip()
                correction = (item.get("correction") or "").strip()
                if not text or not correction:
                    continue
                if text in used_texts:
                    continue
                item_code = (item.get("category") or "").strip().upper()
                if code and item_code != code:
                    continue
                if not is_clean_example_pair(text, correction):
                    continue
                quality = example_quality_score(text, correction)
                start = int(item.get("start", 0) or 0)
                candidates.append((quality, start, text, correction))

            candidates.sort(key=lambda row: (-row[0], row[1]))
            for _quality, _start, text, correction in candidates:
                examples.append({"error": text, "correction": correction})
                used_texts.add(text)
                if len(examples) >= 2:
                    break
            insight = {
                "code": code,
                "title": label,
                "count": count,
                "why": build_insight_reason(code, count, total_errors),
                "focus": guidance,
                "examples": examples,
            }
            insights.append(insight)
            practical.append(
                {
                    "title": label,
                    "guidance": guidance,
                    "why": insight["why"],
                    "examples": examples,
                }
            )
        if practical:
            participant.setdefault("llm", {})
            participant["llm"]["practical_recommendations"] = practical
            participant["llm"]["top3_insights"] = insights

def call_openai_analysis(
    transcript_text: str,
    participants: list[dict],
    api_key: str,
    model: str,
) -> tuple[dict | None, str | None]:
    if not api_key:
        return None, "Missing API key"

    system_prompt = (
        "You are an English tutor analyzing a conversation transcript. "
        "Return strictly valid JSON matching the schema."
    )
    participant_lines = "\n".join(
        f"- {participant['name']} ({participant['role']})" for participant in participants
    )
    user_prompt = (
        "Participants:\n"
        f"{participant_lines}\n\n"
        "Transcript:\n"
        f"{transcript_text}\n\n"
        "Fluency score must be on a 0-10 scale (decimals allowed). "
        "For each participant, analyze ONLY clear grammatical errors (ignore fillers, hesitations, false starts). "
        "Return error_count (count of grammatical error instances), top_errors (top 3 most frequent error types with counts), "
        "and 3 practical recommendations based on the typical errors. Keep responses concise."
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "participants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "fluency": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "score": {"type": "number", "minimum": 0, "maximum": 10},
                                "level": {"type": "string"},
                            },
                            "required": ["score", "level"],
                        },
                        "grammar": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "error_count": {"type": "integer", "minimum": 0},
                                "top_errors": {
                                    "type": "array",
                                    "minItems": 0,
                                    "maxItems": 3,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "title": {"type": "string"},
                                            "count": {"type": "integer", "minimum": 0},
                                        },
                                        "required": ["title", "count"],
                                    },
                                },
                                "top_recommendations": {
                                    "type": "array",
                                    "minItems": 0,
                                    "maxItems": 3,
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["error_count", "top_errors", "top_recommendations"],
                        },
                    },
                    "required": ["name", "fluency", "grammar"],
                },
            }
        },
        "required": ["participants"],
    }

    payload = {
        "model": model,
        "temperature": 0.2,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "session_analysis",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 800,
    }

    request = urllib.request.Request(
        OPENAI_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8")
        except Exception:
            body = str(error)
        return None, body
    except Exception as error:
        return None, str(error)

    output_text = extract_output_text(response_payload)
    if not output_text:
        preview = json.dumps(response_payload)[:600]
        return None, f"Empty response. Raw: {preview}"

    try:
        return json.loads(output_text), None
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"


def call_openai_highlight_exercise(
    *,
    api_key: str,
    model: str,
    participant_name: str,
    category_code: str,
    category_title: str,
    focus_text: str,
    examples: list[dict] | None = None,
) -> tuple[dict | None, str | None]:
    if not api_key:
        return None, "Missing API key"

    example_lines = []
    for item in examples or []:
        error = str(item.get("error") or "").strip()
        correction = str(item.get("correction") or "").strip()
        if error and correction:
            example_lines.append(f"- {error} -> {correction}")
    examples_text = "\n".join(example_lines) if example_lines else "- No trusted examples available"

    system_prompt = (
        "You are an English tutor creating one short practice exercise for a learner preparing "
        "for a 30-minute speaking call. Return strictly valid JSON only."
    )
    user_prompt = (
        f"Learner: {participant_name}\n"
        f"Focus category: {category_title} ({category_code})\n"
        f"Focus guidance: {focus_text}\n"
        "Reference examples:\n"
        f"{examples_text}\n\n"
        "Create exactly one practical multiple-choice exercise with 2 or 3 options. "
        "Keep it short, concrete, and directly related to the examples. "
        "The learner should be able to solve it in under 20 seconds. "
        "Prefer sentence-level speaking examples, not abstract theory. "
        "Return one correct answer, short explanation, and a very short title."
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": ["multiple_choice"]},
            "title": {"type": "string"},
            "prompt": {"type": "string"},
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "answer": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["type", "title", "prompt", "question", "options", "answer", "explanation"],
    }

    payload = {
        "model": model,
        "temperature": 0.3,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "highlight_exercise",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 400,
    }

    request = urllib.request.Request(
        OPENAI_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8")
        except Exception:
            body = str(error)
        return None, body
    except Exception as error:
        return None, str(error)

    output_text = extract_output_text(response_payload)
    if not output_text:
        preview = json.dumps(response_payload)[:600]
        return None, f"Empty response. Raw: {preview}"

    try:
        exercise = json.loads(output_text)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"

    options = [str(option).strip() for option in exercise.get("options", []) if str(option).strip()]
    answer = str(exercise.get("answer") or "").strip()
    if len(options) < 2 or answer not in options:
        return None, "Model returned an invalid exercise payload."
    return exercise, None


def locate_annotation_spans(chunk_text: str, errors: list[dict]) -> list[dict]:
    """Derive start/end offsets for phrases the model quoted verbatim.

    A per-snippet cursor advances past each match so repeated occurrences of
    the same mistake map to distinct spans. Quotes not found verbatim in the
    chunk are dropped.
    """
    located: list[dict] = []
    cursors: dict[str, int] = {}
    for item in errors:
        snippet = (item.get("text") or "").strip()
        if not snippet:
            continue
        pos = chunk_text.find(snippet, cursors.get(snippet, 0))
        if pos == -1:
            pos = chunk_text.find(snippet)
        if pos == -1:
            continue
        cursors[snippet] = pos + 1
        located.append({**item, "text": snippet, "start": pos, "end": pos + len(snippet)})
    return located


def call_openai_chunk_annotations(
    chunk_text: str,
    api_key: str,
    model: str,
) -> tuple[list[dict] | None, str | None]:
    if not api_key:
        return None, "Missing API key"

    system_prompt = (
        "You are an English tutor annotating a transcript of spoken English. "
        "Identify clear grammatical errors. "
        "Be exhaustive: include every clear grammar error you see. "
        "Ignore fillers, hesitations, false starts, punctuation, capitalization, "
        "and purely stylistic improvements."
    )
    # No character indices in the contract: computing them is the hardest part
    # of the task for the model and made gpt-5-mini at low reasoning effort
    # randomly return empty lists for whole chunks (measured July 2026:
    # 0-26 findings on the same chunk). The model quotes the phrase verbatim
    # and locate_annotation_spans() derives start/end deterministically.
    user_prompt = (
        "Return JSON with a list of grammar errors found in the transcript chunk. "
        "For each error the text field must quote the erroneous phrase EXACTLY as it appears in the chunk: "
        "copy it verbatim, 2-12 words, never multi-sentence, at least 4 characters; include surrounding words if needed. "
        "Include correction, a short explanation, and a category.\n"
        f"{ANNOTATION_TAXONOMY_PROMPT}\n"
        "Exclude transcription artifacts and disfluencies: repeated starts (e.g., 'I. I'), fillers, and discourse markers like 'I mean'/'you know', "
        "unless there is a clear grammar-rule violation. "
        "Do not limit the number of errors.\n\n"
        f"Chunk text:\n{chunk_text}"
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "correction": {"type": "string"},
                        "explanation": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["TENSE", "VERB", "ARTICLE", "PREP", "ORDER", "WORD", "COLLOC"],
                        },
                    },
                    "required": ["text", "correction", "explanation", "category"],
                },
            }
        },
        "required": ["errors"],
    }

    # json_schema with the category enum is enforced by the API for gpt-5
    # models too (verified against /v1/responses with gpt-5-mini), so every
    # item arrives with a valid taxonomy code.
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "chunk_annotations",
                "schema": schema,
                "strict": True,
            }
        },
        # Dense chunks can carry 20+ errors with explanations, and for gpt-5
        # models reasoning tokens draw from the same budget; a tight cap
        # truncates the JSON mid-string and loses the whole chunk.
        "max_output_tokens": 8000,
    }
    if model.startswith("gpt-5"):
        # gpt-5-mini rejects temperature; keep reasoning effort low so the
        # budget stays with the answer.
        payload["reasoning"] = {"effort": "low"}

    request = urllib.request.Request(
        OPENAI_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8")
            except Exception:
                body = str(error)
            return None, body
        except Exception as error:
            if attempts >= 3:
                return None, str(error)
            time.sleep(1.5 * attempts)

    output_text = extract_output_text(response_payload)
    if not output_text:
        preview = json.dumps(response_payload)[:600]
        return None, f"Empty response. Raw: {preview}"

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"
    return locate_annotation_spans(chunk_text, parsed.get("errors", [])), None


def count_words_in_chunk(chunk_text: str) -> int:
    words: list[str] = []
    for raw_line in chunk_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if match:
            line = match.group(2)
        words.extend(WORD_RE.findall(line))
    return len(words)


def count_words_by_speaker_in_chunk(chunk_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    current_speaker: str | None = None
    for raw_line in chunk_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if match:
            current_speaker = match.group(1).strip()
            line = match.group(2)
        if not current_speaker:
            continue
        words = WORD_RE.findall(line)
        if not words:
            continue
        counts[current_speaker] = counts.get(current_speaker, 0) + len(words)
    return counts


def call_openai_chunk_metrics(
    chunk_text: str,
    api_key: str,
    model: str,
) -> tuple[dict | None, str | None]:
    if not api_key:
        return None, "Missing API key"

    system_prompt = (
        "You are an English tutor. Count clear grammatical errors only. "
        "Ignore fillers, hesitations, false starts, and stylistic improvements."
    )
    speakers = []
    for raw_line in chunk_text.splitlines():
        match = LINE_RE.match(raw_line.strip())
        if match:
            label = match.group(1).strip()
            if label and label not in speakers:
                speakers.append(label)
    speaker_list = ", ".join(speakers) if speakers else "Speaker"
    user_prompt = (
        "Return JSON with per-speaker error counts and top error types (up to 5). "
        "Use only these speaker labels: "
        f"{speaker_list}. "
        "Count each grammatical error instance. Provide concise type names.\n\n"
        f"Chunk text:\n{chunk_text}"
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "error_count": {"type": "integer", "minimum": 0},
                        "error_types": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "title": {"type": "string"},
                                    "count": {"type": "integer", "minimum": 0},
                                },
                                "required": ["title", "count"],
                            },
                        },
                    },
                    "required": ["name", "error_count", "error_types"],
                },
            }
        },
        "required": ["speakers"],
    }

    if model.startswith("gpt-5"):
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nReturn ONLY valid JSON."},
            ],
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "text"}},
            "max_output_tokens": 1200,
        }
    else:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "chunk_metrics",
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": 1200,
        }

    request = urllib.request.Request(
        OPENAI_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8")
            except Exception:
                body = str(error)
            return None, body
        except Exception as error:
            if attempts >= 3:
                return None, str(error)
            time.sleep(1.5 * attempts)

    output_text = extract_output_text(response_payload)
    if not output_text:
        preview = json.dumps(response_payload)[:600]
        return None, f"Empty response. Raw: {preview}"

    try:
        return json.loads(output_text), None
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"


def build_chunked_speaker_summary(per_speaker_totals: dict[str, dict]) -> list[dict]:
    summary: list[dict] = []
    for label, totals in per_speaker_totals.items():
        error_types_all = [
            {"title": title, "count": count}
            for title, count in sorted(totals["error_types"].items(), key=lambda item: item[1], reverse=True)
        ]
        total_errors = sum(item["count"] for item in error_types_all)
        total_words = totals["total_words"]
        rate = round((total_errors / total_words) * 100, 2) if total_words else 0.0
        summary.append(
            {
                "name": label,
                "total_words": total_words,
                "total_errors": total_errors,
                "error_rate_per_100_words": rate,
                "error_types": error_types_all,
                "top_error_types": error_types_all[:5],
            }
        )
    return summary


def run_annotations_with_fallback(
    chunks: list[dict],
    transcript_text: str,
    api_key: str,
    annotation_model: str,
) -> tuple[str | None, list[dict], str | None, dict, str]:
    no_fallback = os.getenv("OPENAI_ANNOTATION_NO_FALLBACK") in ("1", "true", "yes")
    annotated_html, annotation_items, annotation_error, annotation_meta = build_chunk_annotations(
        chunks, transcript_text, api_key, annotation_model
    )
    if (
        not no_fallback
        and annotation_error
        and annotation_error.startswith("Empty response")
        and annotation_model.startswith("gpt-5")
    ):
        fallback_model = "gpt-4o"
        annotated_html, annotation_items, annotation_error, annotation_meta = build_chunk_annotations(
            chunks, transcript_text, api_key, fallback_model
        )
        annotation_meta["fallback_from"] = annotation_model
        annotation_meta["fallback_to"] = fallback_model
        annotation_meta["fallback_reason"] = "Empty response from gpt-5"
        return annotated_html, annotation_items, annotation_error, annotation_meta, fallback_model
    return annotated_html, annotation_items, annotation_error, annotation_meta, annotation_model


def call_annotation_chunk_with_fallback(
    chunk_text: str,
    api_key: str,
    annotation_model: str,
) -> tuple[list[dict] | None, str | None, str, dict]:
    errors, error = call_openai_chunk_annotations(chunk_text, api_key, annotation_model)
    no_fallback = os.getenv("OPENAI_ANNOTATION_NO_FALLBACK") in ("1", "true", "yes")
    if (
        not no_fallback
        and error
        and (error.startswith("Empty response") or error.startswith("Invalid JSON"))
        and annotation_model.startswith("gpt-5")
    ):
        fallback_model = "gpt-4o"
        errors, error = call_openai_chunk_annotations(chunk_text, api_key, fallback_model)
        return errors, error, fallback_model, {
            "fallback_from": annotation_model,
            "fallback_to": fallback_model,
            "fallback_reason": "Unusable response from gpt-5",
        }
    return errors, error, annotation_model, {}


def merge_existing_llm(analysis: dict, existing: dict) -> None:
    if not existing:
        return
    existing_llm = existing.get("llm")
    if existing_llm:
        analysis["llm"] = existing_llm
    existing_chunked = existing.get("llm", {}).get("chunked_metrics")
    if existing_chunked:
        analysis.setdefault("llm", {})
        analysis["llm"]["chunked_metrics"] = existing_chunked

    existing_by_name = {
        participant.get("name"): participant
        for participant in existing.get("participants", [])
        if participant.get("name")
    }
    for participant in analysis.get("participants", []):
        existing_participant = existing_by_name.get(participant.get("name"))
        if existing_participant and existing_participant.get("llm"):
            participant["llm"] = existing_participant["llm"]


def merge_missing_llm(analysis: dict, existing: dict) -> None:
    if not existing:
        return
    existing_by_name = {
        participant.get("name"): participant
        for participant in existing.get("participants", [])
        if participant.get("name")
    }
    for participant in analysis.get("participants", []):
        if participant.get("llm"):
            continue
        existing_participant = existing_by_name.get(participant.get("name"))
        if existing_participant and existing_participant.get("llm"):
            participant["llm"] = existing_participant["llm"]

def apply_llm_results(analysis: dict, llm_data: dict) -> None:
    if not llm_data:
        return
    llm_participants = llm_data.get("participants", [])
    llm_by_name = {}
    for item in llm_participants:
        name = item.get("name")
        if not name:
            continue
        llm_by_name[name] = item
        llm_by_name[name.lower()] = item

    used_indices = set()
    for index, participant in enumerate(analysis.get("participants", [])):
        name = participant.get("name", "")
        entry = llm_by_name.get(name) or llm_by_name.get(name.lower())
        if entry is None and index < len(llm_participants) and index not in used_indices:
            entry = llm_participants[index]
            used_indices.add(index)
        if not entry:
            continue
        llm_info = {
            "fluency": entry.get("fluency") or {},
            "grammar": entry.get("grammar") or {},
        }
        normalized_score = normalize_llm_fluency_score(llm_info.get("fluency", {}).get("score"))
        if normalized_score is not None:
            llm_info.setdefault("fluency", {})
            llm_info["fluency"]["score"] = normalized_score
        participant["llm"] = llm_info

        word_count = participant.get("metrics", {}).get("word_count", 0)
        error_count = llm_info.get("grammar", {}).get("error_count")
        if error_count is not None:
            rate = 0.0
            if word_count:
                rate = round((error_count / word_count) * 100, 2)
            llm_info["grammar"]["error_rate_per_100_words"] = rate


def build_chunk_annotations(
    chunks: list[dict],
    transcript_text: str,
    api_key: str,
    model: str,
) -> tuple[str | None, list[dict], str | None, dict]:
    if not chunks:
        return None, [], None, {"chunks_processed": 0, "total_chunks": 0, "processed_chars": 0}
    ordered_chunks = sorted(chunks, key=lambda item: item["range"]["start"])
    if ANNOTATION_MAX_CHUNKS > 0:
        ordered_chunks = ordered_chunks[:ANNOTATION_MAX_CHUNKS]
    annotated_parts: list[str] = []
    all_errors: list[dict] = []
    last_end = 0
    processed_chars = 0
    total_chunks = len(chunks)

    last_error = None
    for chunk in ordered_chunks:
        chunk_text = chunk.get("text", "")
        errors, error = call_openai_chunk_annotations(chunk_text, api_key, model)
        if error:
            last_error = error
            meta = {
                "chunks_processed": len(annotated_parts),
                "total_chunks": total_chunks,
                "processed_chars": processed_chars,
            }
            return None, all_errors, error, meta
        normalized = normalize_annotations(chunk_text, errors or [])
        chunk_html = build_annotated_html(chunk_text, normalized)

        start = chunk["range"]["start"]
        end = chunk["range"]["end"]
        if start > last_end:
            annotated_parts.append(escape_html(transcript_text[last_end:start]))
        annotated_parts.append(chunk_html)
        last_end = end
        processed_chars += len(chunk_text)

        for item in normalized:
            all_errors.append(
                {
                    "start": item["start"] + start,
                    "end": item["end"] + start,
                    "text": item["text"],
                    "correction": item.get("correction", ""),
                    "explanation": item.get("explanation", ""),
                    "category": item.get("category", ""),
                }
            )

    if last_end < len(transcript_text):
        annotated_parts.append(escape_html(transcript_text[last_end:]))

    meta = {
        "chunks_processed": len(ordered_chunks),
        "total_chunks": total_chunks,
        "processed_chars": processed_chars,
    }
    meta["attempted_model"] = model
    if last_error:
        meta["last_error"] = last_error
    return "".join(annotated_parts), all_errors, None, meta


def annotate_session(
    session_dir: Path,
    out_dir: Path,
    openai_model: str | None = None,
    force_reannotate: bool = False,
) -> dict:
    transcript_path = session_dir / "transcript.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript.txt in {session_dir}")

    transcript_text = transcript_path.read_text(encoding="utf-8")
    out_analysis_path = out_dir / "sessions" / session_dir.name / "analysis.json"
    existing_analysis = load_json(out_analysis_path, {}) if out_analysis_path.exists() else {}

    analysis = existing_analysis if existing_analysis else analyze_session(
        session_dir,
        use_openai=False,
        openai_model=openai_model,
        out_dir=out_dir,
    )
    normalize_analysis_fluency_scores(analysis)

    blocks = parse_transcript_blocks(transcript_text)
    analysis["chunks"] = build_chunks(blocks, transcript_text)

    api_key = clean_env("OPENAI_API_KEY")
    model = (openai_model or "").strip() or clean_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    annotation_model = clean_env("OPENAI_ANNOTATION_MODEL", DEFAULT_ANNOTATION_MODEL)
    analysis.setdefault("llm", {"status": "skipped", "model": model})
    analysis["llm"]["annotations_model"] = annotation_model

    if not api_key:
        analysis["llm"]["annotations_status"] = "skipped"
        analysis["llm"]["annotations_error"] = "Missing API key"
        return analysis

    ordered_chunks = sorted(analysis["chunks"], key=lambda item: item["range"]["start"])
    total_chunks = len(ordered_chunks)
    # Resume only recovers an interrupted run. An explicit user "Re-run
    # annotations" (force_reannotate) must start fresh, otherwise a fully
    # annotated session is skipped chunk-by-chunk and the stale items are kept.
    resume = (not force_reannotate) and os.getenv("OPENAI_ANNOTATION_RESUME", "1") in ("1", "true", "yes")
    if force_reannotate:
        analysis["llm"].pop("annotation_items", None)
        analysis["llm"].pop("annotations_meta", None)
    existing_meta = analysis["llm"].get("annotations_meta", {}) if resume else {}
    start_index = int(existing_meta.get("chunks_processed", 0)) if resume else 0
    existing_items = analysis["llm"].get("annotation_items", []) if resume else []
    all_errors = list(existing_items)
    processed_chars = 0
    if start_index > 0:
        for chunk in ordered_chunks[:start_index]:
            processed_chars += len(chunk.get("text", ""))

    fallback_info: dict = {}
    model_used = annotation_model

    for index, chunk in enumerate(ordered_chunks):
        if index < start_index:
            continue
        chunk_text = chunk.get("text", "")
        errors, error, model_used, fallback_meta = call_annotation_chunk_with_fallback(
            chunk_text, api_key, annotation_model
        )
        if fallback_meta:
            fallback_info.update(fallback_meta)
        if error:
            analysis["llm"]["annotations_status"] = "error"
            analysis["llm"]["annotations_error"] = error
            analysis["llm"]["annotations_attempted_model"] = annotation_model
            analysis["llm"]["annotations_model"] = model_used
            analysis["llm"]["annotation_items"] = all_errors
            analysis["llm"]["annotated_transcript_html"] = build_annotated_html(
                transcript_text, all_errors
            )
            analysis["llm"]["annotations_meta"] = {
                "chunks_processed": index,
                "total_chunks": total_chunks,
                "processed_chars": processed_chars,
                "attempted_model": annotation_model,
                **fallback_info,
            }
            write_analysis(out_dir, analysis)
            return analysis

        normalized = normalize_annotations(chunk_text, errors or [])
        for item in normalized:
            all_errors.append(
                {
                    "start": item["start"] + chunk["range"]["start"],
                    "end": item["end"] + chunk["range"]["start"],
                    "text": item["text"],
                    "correction": item.get("correction", ""),
                    "explanation": item.get("explanation", ""),
                    "category": item.get("category", ""),
                }
            )
        processed_chars += len(chunk_text)

        analysis["llm"]["annotations_status"] = "in_progress"
        analysis["llm"]["annotations_attempted_model"] = annotation_model
        analysis["llm"]["annotations_model"] = model_used
        analysis["llm"]["annotation_items"] = all_errors
        analysis["llm"]["annotated_transcript_html"] = build_annotated_html(
            transcript_text, all_errors
        )
        analysis["llm"]["annotations_meta"] = {
            "chunks_processed": index + 1,
            "total_chunks": total_chunks,
            "processed_chars": processed_chars,
            "attempted_model": annotation_model,
            **fallback_info,
        }
        write_analysis(out_dir, analysis)

    analysis["llm"]["annotations_status"] = "ok"
    analysis["llm"]["annotations_attempted_model"] = annotation_model
    analysis["llm"]["annotations_model"] = model_used
    analysis["llm"]["annotation_items"] = all_errors
    analysis["llm"]["annotated_transcript_html"] = build_annotated_html(transcript_text, all_errors)
    analysis["llm"]["annotations_meta"] = {
        "chunks_processed": total_chunks,
        "total_chunks": total_chunks,
        "processed_chars": processed_chars,
        "attempted_model": annotation_model,
        **fallback_info,
    }
    analysis["llm"].pop("annotations_error", None)
    build_annotation_metrics(analysis, transcript_text)
    build_practical_recommendations(analysis, transcript_text)
    write_analysis(out_dir, analysis)
    return analysis

def analyze_session(
    session_dir: Path,
    use_openai: bool = False,
    openai_model: str | None = None,
    existing_analysis: dict | None = None,
    run_annotations: bool = True,
    out_dir: Path | None = None,
) -> dict:
    out_dir = out_dir or OUT_DIR
    meta_path = session_dir / "meta.json"
    transcript_path = session_dir / "transcript.txt"
    if not meta_path.exists() or not transcript_path.exists():
        raise FileNotFoundError(f"Missing meta.json or transcript.txt in {session_dir}")

    meta = load_json(meta_path, {})
    transcript_text = transcript_path.read_text(encoding="utf-8")
    turns = load_transcript(transcript_path)

    speaker_map = meta.get("speaker_map", {})
    mapped_turns = [(speaker_map.get(label, label), text) for label, text in turns]

    metrics_by_speaker = compute_metrics(mapped_turns)

    participants_meta = meta.get("participants", [])
    label_order = [item.get("name") for item in participants_meta if item.get("name")]
    if not label_order:
        label_order = []
        for label, _text in turns:
            if label not in label_order:
                label_order.append(label)

    name_to_role: dict[str, str] = {}
    ordered_names: list[str] = []
    for label in label_order:
        name = speaker_map.get(label, label)
        if name not in ordered_names:
            ordered_names.append(name)
        role = "student"
        for item in participants_meta:
            if item.get("name") == label:
                role = item.get("role") or role
                break
        name_to_role.setdefault(name, role)

    participants: list[dict] = []
    for name in ordered_names:
        metrics = metrics_by_speaker.get(name) or SpeakerMetrics(0, 0, 0)
        fluency_score, fluency_level = compute_fluency(metrics)
        participants.append(
            {
                "name": name,
                "role": name_to_role.get(name, "student"),
                "metrics": {
                    "word_count": metrics.word_count,
                    "turn_count": metrics.turn_count,
                    "avg_words_per_turn": metrics.avg_words_per_turn,
                    "unique_word_count": metrics.unique_word_count,
                    "lexical_diversity": metrics.lexical_diversity,
                },
                "fluency": {"score": fluency_score, "level": fluency_level},
            }
        )

    analysis = {
        "date": meta.get("date") or session_dir.name,
        "session": {
            "topic": meta.get("topic") or "Session",
            "duration_minutes": meta.get("duration_minutes") or 30,
        },
        "participants": participants,
        "transcript": transcript_text,
        "speaker_map": speaker_map,
    }

    blocks = parse_transcript_blocks(transcript_text)
    analysis["chunks"] = build_chunks(blocks, transcript_text)

    if not use_openai:
        if existing_analysis:
            merge_existing_llm(analysis, existing_analysis)
        analysis.setdefault("llm", {"status": "skipped", "model": openai_model or DEFAULT_OPENAI_MODEL})
        normalize_analysis_fluency_scores(analysis)
        build_annotation_metrics(analysis, transcript_text)
        build_practical_recommendations(analysis, transcript_text)
        return analysis

    api_key = clean_env("OPENAI_API_KEY")
    model = (openai_model or "").strip() or clean_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    annotation_model = clean_env("OPENAI_ANNOTATION_MODEL", DEFAULT_ANNOTATION_MODEL)
    if not api_key:
        if existing_analysis:
            merge_existing_llm(analysis, existing_analysis)
        analysis["llm"] = {"status": "skipped", "model": model}
        normalize_analysis_fluency_scores(analysis)
        build_annotation_metrics(analysis, transcript_text)
        build_practical_recommendations(analysis, transcript_text)
        return analysis

    llm_data, error = call_openai_analysis(transcript_text, participants, api_key, model)
    if llm_data:
        apply_llm_results(analysis, llm_data)
        analysis["llm"] = {"status": "ok", "model": model}
    else:
        analysis["llm"] = {"status": "error", "model": model, "error": error}
        return analysis

    if existing_analysis:
        merge_missing_llm(analysis, existing_analysis)

    if run_annotations:
        analysis.setdefault("llm", {"status": "ok", "model": model})
        analysis["llm"]["annotations_model"] = annotation_model
        annotated_html, annotation_items, annotation_error, annotation_meta, model_used = run_annotations_with_fallback(
            analysis["chunks"], transcript_text, api_key, annotation_model
        )
        if annotation_error:
            analysis["llm"]["annotations_status"] = "error"
            analysis["llm"]["annotations_error"] = annotation_error
            analysis["llm"].pop("annotations_meta", None)
            analysis["llm"].pop("annotation_items", None)
            analysis["llm"].pop("annotated_transcript_html", None)
        else:
            analysis["llm"]["annotations_status"] = "ok"
            analysis["llm"]["annotation_items"] = annotation_items
            analysis["llm"]["annotations_meta"] = annotation_meta
            analysis["llm"]["annotations_model"] = model_used
            analysis["llm"].pop("annotations_error", None)
            if annotated_html:
                analysis["llm"]["annotated_transcript_html"] = annotated_html
    # Chunked LLM metrics (additional, per speaker)
    chunk_metrics_enabled = os.getenv("OPENAI_CHUNK_METRICS", "1") not in ("0", "false", "no")
    if chunk_metrics_enabled:
        resume = os.getenv("OPENAI_CHUNK_METRICS_RESUME", "1") in ("1", "true", "yes")
        existing_chunked = analysis.get("llm", {}).get("chunked_metrics", {}) if resume else {}
        per_chunk = existing_chunked.get("per_chunk", []) if resume else []
        start_index = len(per_chunk)

        per_speaker_totals: dict[str, dict] = {}
        if resume and per_chunk:
            for entry in per_chunk:
                for speaker_entry in entry.get("speakers", []):
                    label = speaker_entry.get("name")
                    if not label:
                        continue
                    totals = per_speaker_totals.setdefault(
                        label,
                        {"total_words": 0, "total_errors": 0, "error_types": {}},
                    )
                    totals["total_words"] += int(speaker_entry.get("word_count", 0))
                    totals["total_errors"] += int(speaker_entry.get("error_count", 0))
                    for item in speaker_entry.get("error_types", []):
                        title = item.get("title")
                        count = int(item.get("count", 0))
                        if title:
                            totals["error_types"][title] = totals["error_types"].get(title, 0) + count

        for index, chunk in enumerate(analysis["chunks"]):
            if index < start_index:
                continue
            chunk_text = chunk.get("text", "")
            word_counts = count_words_by_speaker_in_chunk(chunk_text)
            llm_chunk, chunk_error = call_openai_chunk_metrics(chunk_text, api_key, model)
            if chunk_error or not llm_chunk:
                per_speaker_summary = build_chunked_speaker_summary(per_speaker_totals)
                analysis.setdefault("llm", {"status": "ok", "model": model})
                analysis["llm"]["chunked_metrics"] = {
                    "status": "error",
                    "error": chunk_error,
                    "per_chunk": per_chunk,
                    "per_speaker": per_speaker_summary,
                    "meta": {
                        "chunks_processed": len(per_chunk),
                        "total_chunks": len(analysis["chunks"]),
                    },
                }
                write_analysis(out_dir, analysis)
                break

            speaker_items = llm_chunk.get("speakers", []) or []
            per_chunk_entry = {"index": index, "speakers": []}
            for speaker_item in speaker_items:
                label = speaker_item.get("name")
                if not label:
                    continue
                word_count = int(word_counts.get(label, 0))
                raw_error_types = speaker_item.get("error_types", []) or []
                normalized_types: dict[str, int] = {}
                for item in raw_error_types:
                    title = normalize_chunk_error_title(item.get("title", ""))
                    if not title:
                        continue
                    count = int(item.get("count", 0))
                    normalized_types[title] = normalized_types.get(title, 0) + count
                error_types = [
                    {"title": title, "count": count}
                    for title, count in sorted(normalized_types.items(), key=lambda it: it[1], reverse=True)
                ]
                type_sum = sum(item["count"] for item in error_types)
                error_count = type_sum if type_sum > 0 else int(speaker_item.get("error_count", 0))
                per_chunk_entry["speakers"].append(
                    {
                        "name": label,
                        "word_count": word_count,
                        "error_count": error_count,
                        "error_types": error_types,
                    }
                )
                totals = per_speaker_totals.setdefault(
                    label,
                    {"total_words": 0, "total_errors": 0, "error_types": {}},
                )
                totals["total_words"] += word_count
                totals["total_errors"] += error_count
                for item in error_types:
                    title = item.get("title")
                    count = int(item.get("count", 0))
                    if title:
                        totals["error_types"][title] = totals["error_types"].get(title, 0) + count

            per_chunk.append(per_chunk_entry)

            per_speaker_summary = build_chunked_speaker_summary(per_speaker_totals)

            analysis.setdefault("llm", {"status": "ok", "model": model})
            analysis["llm"]["chunked_metrics"] = {
                "status": "in_progress" if index < len(analysis["chunks"]) - 1 else "ok",
                "per_chunk": per_chunk,
                "per_speaker": per_speaker_summary,
                "meta": {
                    "chunks_processed": len(per_chunk),
                    "total_chunks": len(analysis["chunks"]),
                },
            }
            write_analysis(out_dir, analysis)

        per_speaker_summary = build_chunked_speaker_summary(per_speaker_totals)

        analysis["llm"]["chunked_metrics"] = {
            "status": "ok",
            "per_chunk": per_chunk,
            "per_speaker": per_speaker_summary,
            "meta": {
                "chunks_processed": len(per_chunk),
                "total_chunks": len(analysis["chunks"]),
            },
        }

        # Attach per-speaker chunked stats to participants (mapped names).
        label_to_name = analysis.get("speaker_map", {})
        per_speaker_by_name = {}
        for entry in per_speaker_summary:
            label = entry.get("name")
            name = label_to_name.get(label, label)
            per_speaker_by_name[name] = entry
        for participant in analysis.get("participants", []):
            name = participant.get("name")
            if name and name in per_speaker_by_name:
                participant.setdefault("llm", {})
                participant["llm"]["chunked_grammar"] = per_speaker_by_name[name]
    normalize_analysis_fluency_scores(analysis)
    build_annotation_metrics(analysis, transcript_text)
    build_practical_recommendations(analysis, transcript_text)
    return analysis


# --- Derived metrics (Layer A: deterministic + per-category grammar densities) ---
# Bump these when the computation changes so old/new sessions are comparable only
# within the same version. Derived metrics are recomputed from stored transcript +
# annotation_items, so re-analysis never needs the LLM.
METRICS_VERSION = 1
TAXONOMY_VERSION = 1

LATIN_WORD_RE = re.compile(r"[A-Za-z']+")
RU_WORD_RE = re.compile(r"[А-Яа-яЁё]+")
FILLER_TERMS = ["um", "uh", "er", "erm", "hmm", "like", "you know", "i mean",
                "kind of", "sort of"]
FILLER_REGEX = re.compile(r"\b(?:" + "|".join(FILLER_TERMS) + r")\b")


def _mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio: lexical diversity that does not sag with
    length, so sessions of different lengths stay comparable (unlike raw TTR)."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return round(len(set(tokens)) / n, 3)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(n - window + 1)]
    return round(sum(ratios) / len(ratios), 3)


def compute_deterministic_metrics(mapped_turns: list[tuple[str, str]]) -> dict[str, dict]:
    """Reproducible, LLM-free per-speaker metrics computed purely from text."""
    agg: dict[str, dict] = {}
    for name, text in mapped_turns:
        if not name:
            continue
        bucket = agg.setdefault(name, {
            "turns": 0, "eng_tokens": [], "eng_words": 0, "rus_words": 0,
            "fillers": 0, "questions": 0,
        })
        low = text.lower()
        eng = LATIN_WORD_RE.findall(low)
        bucket["turns"] += 1
        bucket["eng_tokens"].extend(eng)
        bucket["eng_words"] += len(eng)
        bucket["rus_words"] += len(RU_WORD_RE.findall(low))
        bucket["fillers"] += len(FILLER_REGEX.findall(low))
        if "?" in text:
            bucket["questions"] += 1

    total_all = sum(b["eng_words"] + b["rus_words"] for b in agg.values()) or 1
    out: dict[str, dict] = {}
    for name, b in agg.items():
        total_words = b["eng_words"] + b["rus_words"]
        eng = b["eng_words"]
        turns = b["turns"]
        out[name] = {
            "word_count": total_words,
            "english_word_count": eng,
            "russian_word_count": b["rus_words"],
            "turn_count": turns,
            "avg_words_per_turn": round(total_words / turns, 2) if turns else 0.0,
            "speaking_share_pct": round(total_words / total_all * 100, 1),
            "l1_fallback_pct": round(b["rus_words"] / total_words * 100, 1) if total_words else 0.0,
            "filler_per_100w": round(b["fillers"] / eng * 100, 2) if eng else 0.0,
            "question_pct": round(b["questions"] / turns * 100, 1) if turns else 0.0,
            "lexical_diversity_mattr": _mattr(b["eng_tokens"], 50),
        }
    return out


def finalize_derived_metrics(analysis: dict) -> None:
    """Attach Layer-A metrics + per-category grammar densities + analysis_version.

    Everything here is derived deterministically from already-stored fields
    (transcript, speaker_map, annotation_items), so this is safe and cheap to
    re-run over historical sessions to keep the whole series comparable.
    """
    transcript_text = analysis.get("transcript", "") or ""
    speaker_map = analysis.get("speaker_map", {}) or {}

    mapped: list[tuple[str, str]] = []
    for line in transcript_text.splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        label = match.group(1).strip()
        mapped.append((speaker_map.get(label, label), match.group(2)))
    det = compute_deterministic_metrics(mapped)

    items = (analysis.get("llm") or {}).get("annotation_items") or []
    by_name = map_annotation_items_to_speakers(items, transcript_text, speaker_map) if items else {}

    for participant in analysis.get("participants", []):
        name = participant.get("name")
        metrics = det.get(name, {})
        words = metrics.get("english_word_count", 0) or 0

        cat_counts = {code: 0 for code in CATEGORY_LABELS}
        total_errors = 0
        for item in by_name.get(name, []):
            code = (item.get("category") or infer_category_code(item) or "").upper()
            total_errors += 1
            if code in cat_counts:
                cat_counts[code] += 1

        def density(count: int) -> float:
            return round(count / words * 100, 2) if words else 0.0

        participant["derived"] = {
            "metrics": metrics,
            "grammar": {
                "error_count": total_errors,
                "error_density_per_100w": density(total_errors),
                "by_category_count": cat_counts,
                "by_category_density": {code: density(c) for code, c in cat_counts.items()},
            },
        }

    llm = analysis.get("llm") or {}
    analysis["analysis_version"] = {
        "metrics": METRICS_VERSION,
        "taxonomy": TAXONOMY_VERSION,
        "annotation_model": llm.get("annotations_model") or llm.get("model"),
        "annotations_status": llm.get("annotations_status") or llm.get("status"),
    }


def reanalyze_derived_all(out_dir: Path) -> int:
    """Recompute derived metrics for every stored session (no LLM) and rebuild
    history + web. Use after changing the metrics/taxonomy version."""
    sessions_dir = out_dir / "sessions"
    count = 0
    if sessions_dir.exists():
        for session_dir in sorted(sessions_dir.iterdir()):
            analysis_path = session_dir / "analysis.json"
            if not analysis_path.exists():
                continue
            analysis = load_json(analysis_path, {})
            if not analysis:
                continue
            write_analysis(out_dir, analysis)   # finalize_derived_metrics runs here
            update_history(out_dir, analysis)
            count += 1
    write_web_assets(out_dir)
    return count


def write_analysis(out_dir: Path, analysis: dict) -> None:
    finalize_derived_metrics(analysis)
    session_dir = out_dir / "sessions" / analysis["date"]
    write_json(session_dir / "analysis.json", analysis)


def update_history(out_dir: Path, analysis: dict) -> None:
    history_path = out_dir / "history.json"
    history = load_json(history_path, {"sessions": []})
    sessions = history.get("sessions", [])

    entry = {
        "date": analysis.get("date"),
        "topic": analysis.get("session", {}).get("topic", "Session"),
        "analysis_version": analysis.get("analysis_version"),
        "participants": [],
    }

    for participant in analysis.get("participants", []):
        llm = participant.get("llm") or {}
        grammar = llm.get("grammar") or {}
        annotation_grammar = llm.get("annotation_grammar") or {}
        llm_fluency_score = normalize_llm_fluency_score(llm.get("fluency", {}).get("score"))
        entry["participants"].append(
            {
                "name": participant.get("name"),
                "role": participant.get("role"),
                # Layer A + per-category grammar densities (the trustworthy series).
                "derived": participant.get("derived"),
                # Legacy LLM fields (kept for the existing charts; being superseded).
                "fluency_score": participant.get("fluency", {}).get("score"),
                "llm_fluency_score": llm_fluency_score,
                "grammar_error_count": grammar.get("error_count"),
                "llm_error_rate": annotation_grammar.get("error_rate_per_100_words") or grammar.get("error_rate_per_100_words"),
                "chunked_error_rate": llm.get("chunked_grammar", {}).get("error_rate_per_100_words"),
            }
        )

    sessions = [session for session in sessions if session.get("date") != entry["date"]]
    sessions.append(entry)
    sessions.sort(key=lambda item: item.get("date", ""))
    history["sessions"] = sessions
    write_json(history_path, history)


def _briefing_participant(session: dict, participant_name: str) -> dict | None:
    for participant in session.get("participants", []):
        if participant.get("name") == participant_name:
            return participant
    return None


def _briefing_annotation_available(session: dict) -> bool:
    return (session.get("analysis_version") or {}).get("annotations_status") == "ok"


def _briefing_metric_trend(
    sessions: list[dict], participant_name: str, path: tuple[str, ...], higher_is_better: bool,
) -> dict | None:
    values: list[dict] = []
    for session in sessions:
        participant = _briefing_participant(session, participant_name)
        derived = (participant or {}).get("derived") or {}
        current: object = derived
        for key in path:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, (int, float)):
            values.append({"date": session.get("date"), "value": current})
    if not values:
        return None
    recent = values[-3:]
    first, last = recent[0]["value"], recent[-1]["value"]
    delta = round(last - first, 2)
    if len(recent) == 1 or delta == 0:
        direction = "steady"
    elif (delta > 0) == higher_is_better:
        direction = "improving"
    else:
        direction = "worsening"
    return {"values": recent, "current": last, "delta": delta, "direction": direction}


def build_briefing(out_dir: Path) -> dict:
    """Build the deterministic pre-call briefing consumed by the Home page.

    The briefing is deliberately assembled from stored history, focus decisions
    and annotation evidence. It makes no LLM call and remains reproducible.
    """
    history = load_json(out_dir / "history.json", {"sessions": []})
    sessions = sorted(history.get("sessions", []), key=lambda item: item.get("date", ""))
    focus_data = load_json(DATA_DIR / "focus.json", {"focuses": []})
    participant_names: list[str] = []
    for session in sessions:
        for participant in session.get("participants", []):
            name = participant.get("name")
            if name and name not in participant_names:
                participant_names.append(name)

    analyses: dict[str, dict] = {}
    for session in sessions:
        date = session.get("date")
        if date:
            analyses[date] = load_json(out_dir / "sessions" / date / "analysis.json", {})

    people: list[dict] = []
    active_focuses = [item for item in focus_data.get("focuses", []) if item.get("status") == "active"]
    for name in participant_names:
        comparable = [
            session for session in sessions
            if _briefing_annotation_available(session)
            and ((_briefing_participant(session, name) or {}).get("derived", {}).get("metrics", {}).get("english_word_count", 0) >= 120)
        ]
        recent_comparable = comparable[-3:]
        category_rows: list[dict] = []
        for code, title in CATEGORY_LABELS.items():
            densities = []
            for session in recent_comparable:
                grammar = ((_briefing_participant(session, name) or {}).get("derived", {}).get("grammar", {}))
                density_value = (grammar.get("by_category_density") or {}).get(code)
                if isinstance(density_value, (int, float)):
                    densities.append(density_value)
            if densities:
                category_rows.append({
                    "code": code,
                    "title": title,
                    "average_density": round(sum(densities) / len(densities), 2),
                    "seen_sessions": sum(1 for value in densities if value >= 0.3),
                    "sessions_considered": len(densities),
                })
        category_rows.sort(key=lambda item: (-item["average_density"], -item["seen_sessions"], item["title"]))

        focus_rows: list[dict] = []
        for focus in active_focuses:
            if focus.get("participant") != name:
                continue
            code = focus.get("category_code") or ""
            baseline_session = next((item for item in sessions if item.get("date") == focus.get("set_date")), None)
            baseline_participant = _briefing_participant(baseline_session or {}, name) or {}
            baseline = ((baseline_participant.get("derived") or {}).get("grammar") or {}).get("by_category_density", {}).get(code)
            newer = [item for item in comparable if item.get("date", "") > str(focus.get("set_date") or "")]
            latest = newer[-1] if newer else None
            latest_participant = _briefing_participant(latest or {}, name) or {}
            current = ((latest_participant.get("derived") or {}).get("grammar") or {}).get("by_category_density", {}).get(code)
            change_pct = None
            if isinstance(baseline, (int, float)) and baseline > 0 and isinstance(current, (int, float)):
                change_pct = round((current - baseline) / baseline * 100)
            focus_rows.append({
                "id": focus.get("id"),
                "code": code,
                "title": CATEGORY_LABELS.get(code, code),
                "set_date": focus.get("set_date"),
                "baseline_density": baseline,
                "latest_density": current,
                "latest_date": (latest or {}).get("date"),
                "change_pct": change_pct,
                "ready_to_close": bool(change_pct is not None and change_pct <= -40),
            })

        examples: list[dict] = []
        seen_examples: set[tuple[str, str]] = set()
        for session in reversed(sessions):
            if not _briefing_annotation_available(session):
                continue
            analysis = analyses.get(session.get("date", ""), {})
            items = (analysis.get("llm") or {}).get("annotation_items") or []
            by_name = map_annotation_items_to_speakers(items, analysis.get("transcript", ""), analysis.get("speaker_map", {}) or {})
            for item in by_name.get(name, []):
                error = str(item.get("text") or "").strip()
                correction = str(item.get("correction") or "").strip()
                key = (error.lower(), correction.lower())
                if not error or not correction or key in seen_examples:
                    continue
                seen_examples.add(key)
                code = (item.get("category") or infer_category_code(item) or "").upper()
                examples.append({
                    "date": session.get("date"),
                    "code": code,
                    "category_title": CATEGORY_LABELS.get(code, "Grammar issue"),
                    "error": error,
                    "correction": correction,
                })
                if len(examples) >= 4:
                    break
            if len(examples) >= 4:
                break

        metric_sessions = [
            item for item in sessions
            if ((_briefing_participant(item, name) or {}).get("derived", {}).get("metrics", {}).get("english_word_count", 0) >= 120)
        ]
        grammar_sessions = [item for item in metric_sessions if _briefing_annotation_available(item)]
        trends = [
            {"key": "error_density", "label": "Error density", "unit": "/100w", "trend": _briefing_metric_trend(grammar_sessions, name, ("grammar", "error_density_per_100w"), False)},
            {"key": "russian_fallback", "label": "Russian fallback", "unit": "%", "trend": _briefing_metric_trend(metric_sessions, name, ("metrics", "l1_fallback_pct"), False)},
            {"key": "lexical_diversity", "label": "Lexical diversity", "unit": "", "trend": _briefing_metric_trend(metric_sessions, name, ("metrics", "lexical_diversity_mattr"), True)},
        ]
        people.append({
            "name": name,
            "active_focuses": focus_rows,
            "top_categories": category_rows[:3],
            "examples": examples,
            "trends": [item for item in trends if item["trend"]],
        })

    briefing = {"version": 1, "participants": people}
    write_json(out_dir / "briefing.json", briefing)
    return briefing


def build_all(
    sessions_dir: Path,
    out_dir: Path,
    use_openai: bool = False,
    openai_model: str | None = None,
) -> int:
    count = 0
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        if not (session_dir / "meta.json").exists():
            continue
        existing_analysis = None
        out_analysis_path = out_dir / "sessions" / session_dir.name / "analysis.json"
        if out_analysis_path.exists():
            existing_analysis = load_json(out_analysis_path, {})
        analysis = analyze_session(
            session_dir,
            use_openai=use_openai,
            openai_model=openai_model,
            existing_analysis=existing_analysis,
            out_dir=out_dir,
        )
        write_analysis(out_dir, analysis)
        update_history(out_dir, analysis)
        count += 1
    write_web_assets(out_dir)
    return count


def write_web_assets(out_dir: Path) -> None:
    web_dir = out_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    api_base = json.dumps((os.getenv("ENGLISH_TUTOR_API_BASE_URL") or "").strip().rstrip("/"))
    build_briefing(out_dir)
    nav_html = "".join(
        f'<a class="app-nav-link{(" is-accent" if accent else "")}" data-nav-key="{key}" href="{href}">{label}</a>'
        for key, label, href, accent in (
            ("home", "This week", "home.html", False),
            ("session", "Session", "highlights.html", False),
            ("progress", "Progress", "progress.html", False),
            ("method", "How it works", "method.html", False),
            ("record", "Record", "record.html", True),
        )
    )

    # Frontend sources live as real files under web/; copy them to out/web,
    # injecting the API base URL into HTML templates on the way.
    for src in sorted(WEB_SRC_DIR.iterdir()):
        if not src.is_file():
            continue
        content = src.read_text(encoding="utf-8")
        if src.suffix == ".html":
            content = content.replace("__ENGLISH_TUTOR_API_BASE_URL__", api_base)
            content = content.replace("__ENGLISH_TUTOR_NAV__", nav_html)
        (web_dir / src.name).write_text(content, encoding="utf-8")

    history_path = out_dir / "history.json"
    if history_path.exists():
        (web_dir / "history.json").write_text(history_path.read_text(encoding="utf-8"), encoding="utf-8")
    briefing_path = out_dir / "briefing.json"
    if briefing_path.exists():
        (web_dir / "briefing.json").write_text(briefing_path.read_text(encoding="utf-8"), encoding="utf-8")
    sessions_src = out_dir / "sessions"
    sessions_dst = web_dir / "sessions"
    if sessions_src.exists():
        if sessions_dst.exists():
            shutil.rmtree(sessions_dst)
        shutil.copytree(sessions_src, sessions_dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local analysis outputs.")
    parser.add_argument("--sessions", type=Path, default=SESSIONS_DIR, help="Sessions directory")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI analysis")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model override")
    parser.add_argument(
        "--recompute-derived",
        action="store_true",
        help="Recompute derived metrics for all stored sessions (no LLM) and rebuild history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recompute_derived:
        count = reanalyze_derived_all(args.out)
        print(f"Recomputed derived metrics for {count} session(s).")
        return
    count = build_all(args.sessions, args.out, use_openai=args.use_openai, openai_model=args.model)
    print(f"Built {count} session(s).")


if __name__ == "__main__":
    main()
