
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
SESSIONS_DIR = ROOT_DIR / "sessions"
OUT_DIR = ROOT_DIR / "out"
DATA_DIR = ROOT_DIR / "data"

WORD_RE = re.compile(r"[A-Za-z']+")
LINE_RE = re.compile(r"^\s*([^:]+):\s*(.*)$")
BLOCK_RE = re.compile(r"^\s*([^:]+):\s*(.*)$", re.MULTILINE)
FILLER_RE = re.compile(r"[A-Za-zА-Яа-яЁё']+")

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
                                "score": {"type": "number"},
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


def call_openai_chunk_annotations(
    chunk_text: str,
    api_key: str,
    model: str,
) -> tuple[list[dict] | None, str | None]:
    if not api_key:
        return None, "Missing API key"

    system_prompt = (
        "You are an English tutor. Identify clear grammatical errors. "
        "Be exhaustive: include every clear grammar error you see. "
        "Ignore fillers, hesitations, false starts, and stylistic improvements."
    )
    user_prompt = (
        "Return JSON with a list of grammar errors found in the transcript chunk. "
        "Provide 0-based character indices into the chunk text (start inclusive, end exclusive). "
        "Spans must be short and local (prefer 2-12 words), never multi-sentence. "
        "Spans must cover the full erroneous phrase (at least 4 characters, include surrounding words if needed). "
        "Include correction, a short explanation, and a category code from this list: "
        "TENSE, VERB, ARTICLE, PREP, ORDER, WORD, COLLOC. "
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
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 0},
                        "text": {"type": "string"},
                        "correction": {"type": "string"},
                        "explanation": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["TENSE", "VERB", "ARTICLE", "PREP", "ORDER", "WORD", "COLLOC"],
                        },
                    },
                    "required": ["start", "end", "text", "correction", "explanation", "category"],
                },
            }
        },
        "required": ["errors"],
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
            "max_output_tokens": 2000,
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
                    "name": "chunk_annotations",
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": 2000,
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
        parsed = json.loads(output_text)
        return parsed.get("errors", []), None
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"


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
        and error.startswith("Empty response")
        and annotation_model.startswith("gpt-5")
    ):
        fallback_model = "gpt-4o"
        errors, error = call_openai_chunk_annotations(chunk_text, api_key, fallback_model)
        return errors, error, fallback_model, {
            "fallback_from": annotation_model,
            "fallback_to": fallback_model,
            "fallback_reason": "Empty response from gpt-5",
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

    blocks = parse_transcript_blocks(transcript_text)
    analysis["chunks"] = build_chunks(blocks, transcript_text)

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    annotation_model = os.getenv("OPENAI_ANNOTATION_MODEL") or DEFAULT_ANNOTATION_MODEL
    annotation_model = os.getenv("OPENAI_ANNOTATION_MODEL") or DEFAULT_ANNOTATION_MODEL
    annotation_model = os.getenv("OPENAI_ANNOTATION_MODEL") or DEFAULT_ANNOTATION_MODEL
    analysis.setdefault("llm", {"status": "skipped", "model": model})
    analysis["llm"]["annotations_model"] = annotation_model

    if not api_key:
        analysis["llm"]["annotations_status"] = "skipped"
        analysis["llm"]["annotations_error"] = "Missing API key"
        return analysis

    ordered_chunks = sorted(analysis["chunks"], key=lambda item: item["range"]["start"])
    total_chunks = len(ordered_chunks)
    resume = os.getenv("OPENAI_ANNOTATION_RESUME", "1") in ("1", "true", "yes")
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
        build_annotation_metrics(analysis, transcript_text)
        build_practical_recommendations(analysis, transcript_text)
        return analysis

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    if not api_key:
        if existing_analysis:
            merge_existing_llm(analysis, existing_analysis)
        analysis["llm"] = {"status": "skipped", "model": model}
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
    build_annotation_metrics(analysis, transcript_text)
    build_practical_recommendations(analysis, transcript_text)
    return analysis


def write_analysis(out_dir: Path, analysis: dict) -> None:
    session_dir = out_dir / "sessions" / analysis["date"]
    write_json(session_dir / "analysis.json", analysis)


def update_history(out_dir: Path, analysis: dict) -> None:
    history_path = out_dir / "history.json"
    history = load_json(history_path, {"sessions": []})
    sessions = history.get("sessions", [])

    entry = {
        "date": analysis.get("date"),
        "topic": analysis.get("session", {}).get("topic", "Session"),
        "participants": [],
    }

    for participant in analysis.get("participants", []):
        llm = participant.get("llm") or {}
        grammar = llm.get("grammar") or {}
        annotation_grammar = llm.get("annotation_grammar") or {}
        entry["participants"].append(
            {
                "name": participant.get("name"),
                "role": participant.get("role"),
                "fluency_score": participant.get("fluency", {}).get("score"),
                "llm_fluency_score": llm.get("fluency", {}).get("score"),
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

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>English Session Evaluator</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main>
      <header>
        <h1>English Session Evaluator</h1>
        <p class="subtitle">Review fluency progress and session details.</p>
        <div class="header-actions">
          <div class="header-links">
            <a class="inline-link" href="upload.html">Upload a transcript</a>
            <a class="inline-link" href="highlights.html">Highlights</a>
          </div>
          <div class="header-status-grid">
            <div class="status-block">
              <button class="ghost" id="rebuild-metrics-button">Re-run metrics</button>
              <p class="helper" id="llm-status"></p>
            </div>
            <div class="status-block">
              <button class="ghost" id="rebuild-annotations-button">Re-run annotations</button>
              <p class="helper" id="rebuild-meta"></p>
            </div>
          </div>
          <div class="header-tools">
            <button class="ghost" id="test-model-button">Test gpt-5-mini</button>
          </div>
        </div>
        <p class="helper" id="rebuild-status"></p>
      </header>
      <section class="controls">
        <label for="session-select">Session date</label>
        <div class="session-row">
          <select id="session-select"></select>
          <button class="ghost danger" id="delete-button">Delete session</button>
        </div>
      </section>
      <section class="summary">
        <h2 id="session-title">Session</h2>
        <div id="session-details" class="cards"></div>
      </section>
      <section class="progress">
        <h2>Progress</h2>
        <div class="chart-grid">
          <div class="chart-card">
            <h3>LLM Proficiency</h3>
            <canvas id="progress-proficiency" width="640" height="220"></canvas>
          </div>
          <div class="chart-card">
            <h3>Grammar errors / 100 words</h3>
            <canvas id="progress-errors" width="640" height="220"></canvas>
          </div>
        </div>
      </section>
      <section class="recommendations">
        <h2>Recommendations</h2>
        <div id="recommendations"></div>
      </section>
      <section class="transcript">
        <h2>Transcript</h2>
        <div class="transcript-rows">
          <div class="transcript-row transcript-header">
            <div class="transcript-cell">
              <h3>Original</h3>
            </div>
            <div class="transcript-cell">
              <h3>Annotated</h3>
            </div>
            <div class="transcript-cell">
              <h3>Issues</h3>
            </div>
          </div>
          <div id="transcript-rows"></div>
        </div>
        <p class="helper" id="transcript-note">Annotations will appear after the chunked LLM analysis is implemented.</p>
      </section>
    </main>
    <script src="app.js"></script>
  </body>
</html>
"""

UPLOAD_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Upload Transcript</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main>
      <header>
        <h1>Upload a transcript</h1>
        <p class="subtitle">
          Select a .txt transcript file to analyze locally.
        </p>
        <a class="inline-link" href="index.html">Back to dashboard</a>
      </header>
      <section class="card upload-card">
        <form id="upload-form">
          <label class="file-label" for="transcript-file">
            Transcript file (.txt)
          </label>
          <input id="transcript-file" name="transcript" type="file" accept=".txt" />
          <label class="file-label" for="topic">Topic</label>
          <input id="topic" name="topic" type="text" placeholder="Travel stories" />
          <div class="row">
            <div>
              <label class="file-label" for="date">Date</label>
              <input id="date" name="date" type="date" />
            </div>
            <div>
              <label class="file-label" for="duration">Duration (minutes)</label>
              <input id="duration" name="duration" type="number" min="1" max="240" value="30" />
            </div>
          </div>
          <div class="mapping">
            <h2 class="section-title">Speaker mapping</h2>
            <p class="helper">Assign people to detected speakers.</p>
            <p class="helper" id="detected-speakers"></p>
            <div class="row">
              <div>
                <label class="file-label" for="speaker-a-person">Speaker A</label>
                <select id="speaker-a-person">
                  <option value="Roman">Roman</option>
                  <option value="Andrey">Andrey</option>
                </select>
              </div>
              <div>
                <label class="file-label" for="speaker-b-person">Speaker B</label>
                <select id="speaker-b-person">
                  <option value="Andrey">Andrey</option>
                  <option value="Roman">Roman</option>
                </select>
              </div>
            </div>
          </div>
          <button class="primary" type="submit">Upload + Analyze</button>
          <p class="helper" id="upload-status">
            This runs a local upload to the analysis server.
          </p>
        </form>
      </section>
    </main>
    <script>
      const form = document.getElementById('upload-form');
      const statusEl = document.getElementById('upload-status');
      const detectedEl = document.getElementById('detected-speakers');
      const speakerAPerson = document.getElementById('speaker-a-person');
      const speakerBPerson = document.getElementById('speaker-b-person');
      let detectedLabels = [];

      async function readFileAsText(file) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = () => reject(reader.error);
          reader.readAsText(file);
        });
      }

      function isValidTranscript(text) {
        const linePattern = /^\s*[^:]+:\s*.+/m;
        return linePattern.test(text);
      }

      function detectSpeakers(text) {
        const speakers = [];
        text.split(/\r?\n/).forEach((line) => {
          const match = line.match(/^\s*([^:]+):\s*.+/);
          if (match) {
            const name = match[1].trim();
            if (name && !speakers.includes(name)) {
              speakers.push(name);
            }
          }
        });
        return speakers;
      }

      async function handleFileChange(file) {
        if (!file) {
          return;
        }
        const text = await readFileAsText(file);
        const speakers = detectSpeakers(text);
        detectedLabels = speakers;
        if (speakers.length === 2) {
          detectedEl.textContent = `Detected speakers: ${speakers.join(', ')}. Speaker A = ${speakers[0]}, Speaker B = ${speakers[1]}.`;
        } else if (speakers.length === 0) {
          detectedEl.textContent = 'No speakers detected. Use format: Name: text';
        } else {
          detectedEl.textContent = `Detected ${speakers.length} speakers. Expected exactly 2.`;
        }
      }

      const fileInput = document.getElementById('transcript-file');
      fileInput.addEventListener('change', () => {
        handleFileChange(fileInput.files[0]);
      });

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!fileInput.files.length) {
          statusEl.textContent = 'Please pick a transcript file.';
          return;
        }
        statusEl.textContent = 'Uploading...';
        try {
          const transcript = await readFileAsText(fileInput.files[0]);
          if (!isValidTranscript(transcript)) {
            statusEl.textContent = 'Invalid format. Use "Name: text" per line.';
            return;
          }

          if (!detectedLabels.length) {
            detectedLabels = detectSpeakers(transcript);
          }
          if (detectedLabels.length !== 2) {
            statusEl.textContent = 'Expected exactly 2 speakers in the transcript.';
            return;
          }

          const speakerAPersonValue = speakerAPerson.value;
          const speakerBPersonValue = speakerBPerson.value;
          if (speakerAPersonValue === speakerBPersonValue) {
            statusEl.textContent = 'Speaker A and B persons must be different.';
            return;
          }
          const payload = {
            transcript,
            topic: document.getElementById('topic').value,
            date: document.getElementById('date').value,
            duration: document.getElementById('duration').value,
            speaker_a_label: detectedLabels[0],
            speaker_b_label: detectedLabels[1],
            speaker_a_person: speakerAPersonValue,
            speaker_b_person: speakerBPersonValue,
          };
          const response = await fetch('/api/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || 'Upload failed.');
          }
          const result = await response.json();
          statusEl.textContent = `Uploaded session ${result.date}. Open the dashboard to review.`;
        } catch (error) {
          statusEl.textContent = error.message || 'Upload failed.';
        }
      });
    </script>
  </body>
</html>
"""

HIGHLIGHTS_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Session Highlights</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main>
      <header>
        <h1>Session Highlights</h1>
        <p class="subtitle">Three focused, practical takeaways for the next session.</p>
        <div class="header-actions">
          <div class="header-links">
            <a class="inline-link" href="index.html">Detailed statistics</a>
          </div>
        </div>
      </header>
      <section class="controls controls-highlights">
        <label for="highlight-session-select">Session date</label>
        <select id="highlight-session-select"></select>
      </section>
      <section id="highlights-root" class="highlight-grid"></section>
    </main>
    <script src="highlights.js"></script>
  </body>
</html>
"""

HIGHLIGHTS_JS = r"""const sessionSelect = document.getElementById('highlight-session-select');
const highlightsRoot = document.getElementById('highlights-root');

const state = {
  history: null,
  analysisCache: new Map(),
};

async function loadHistory() {
  const response = await fetch('history.json');
  if (!response.ok) {
    throw new Error('Unable to load history.json');
  }
  return response.json();
}

async function loadAnalysis(date) {
  if (state.analysisCache.has(date)) {
    return state.analysisCache.get(date);
  }
  const response = await fetch(`sessions/${date}/analysis.json`);
  if (!response.ok) {
    throw new Error(`Unable to load analysis for ${date}`);
  }
  const data = await response.json();
  state.analysisCache.set(date, data);
  return data;
}

function formatSessionDate(value) {
  const raw = String(value || '').trim();
  if (!raw) {
    return '';
  }
  const parsed = new Date(`${raw}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) {
    return raw;
  }
  return parsed.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

function renderDropdown() {
  sessionSelect.innerHTML = '';
  state.history.sessions.forEach((session) => {
    const option = document.createElement('option');
    option.value = session.date;
    option.textContent = formatSessionDate(session.date);
    sessionSelect.appendChild(option);
  });
}

function sortParticipants(participants) {
  const list = Array.isArray(participants) ? [...participants] : [];
  return list.sort((a, b) => {
    const aName = (a?.name || '').toLowerCase();
    const bName = (b?.name || '').toLowerCase();
    if (aName === 'roman' && bName !== 'roman') return -1;
    if (bName === 'roman' && aName !== 'roman') return 1;
    return aName.localeCompare(bName);
  });
}

function escapeHtml(value) {
  if (!value) {
    return '';
  }
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function getTopInsights(participant) {
  const insights = participant.llm?.top3_insights || [];
  if (insights.length) {
    return insights.slice(0, 3);
  }
  const practical = participant.llm?.practical_recommendations || [];
  return practical.slice(0, 3).map((item) => ({
    title: item.title,
    why: item.why || '',
    focus: item.guidance || '',
    examples: item.examples || [],
  }));
}

function buildLead(participant, top3) {
  const totalErrors = participant.llm?.annotation_grammar?.total_errors;
  const labels = top3.map((item) => item.title).filter(Boolean);
  if (typeof totalErrors === 'number' && labels.length) {
    return `${participant.name}: ${totalErrors} total grammar issues. Top recurring patterns are ${labels.join(', ')}.`;
  }
  if (typeof totalErrors === 'number') {
    return `${participant.name}: ${totalErrors} total grammar issues in this session.`;
  }
  return 'Focused patterns to work on before the next session.';
}

function renderHighlights(analysis) {
  highlightsRoot.innerHTML = '';
  const participants = sortParticipants(analysis.participants || []);
  participants.forEach((participant) => {
    const card = document.createElement('div');
    card.className = 'highlight-card';

    const top3 = getTopInsights(participant);
    const items = top3.map((rec, index) => {
      const examples = (rec.examples || [])
        .map(
          (example) =>
            `<li><span class="example-error">${escapeHtml(example.error)}</span><span class="example-arrow">&rarr;</span><span class="example-fix">${escapeHtml(example.correction)}</span></li>`
        )
        .join('');
      const examplesBlock = examples
        ? `<ul class="errors compact">${examples}</ul>`
        : '<p class="metric-note">No examples captured yet.</p>';
      const why = rec.why || 'Recurring pattern in this session.';
      const focus = rec.focus ? `<p class="highlight-focus">${escapeHtml(rec.focus)}</p>` : '';
      return `
        <div class="highlight-item">
          <div class="highlight-rank">${index + 1}</div>
          <div class="highlight-body">
            <h3>${escapeHtml(rec.title)}</h3>
            <p class="highlight-why">${escapeHtml(why)}</p>
            ${focus}
            ${examplesBlock}
          </div>
        </div>
      `;
    }).join('');
    const lead = buildLead(participant, top3);

    const summary = participant.llm?.annotation_grammar?.total_errors
      ? `${participant.llm.annotation_grammar.total_errors} total errors.`
      : 'Key focuses for next session.';

    card.innerHTML = `
      <div class="highlight-header">
        <h2>${escapeHtml(participant.name)}</h2>
        <span class="highlight-meta">${escapeHtml(summary)}</span>
      </div>
      <p class="highlight-lead">${escapeHtml(lead)}</p>
      ${items || '<p class="metric-note">No highlights yet. Run annotations to generate focus areas.</p>'}
    `;
    highlightsRoot.appendChild(card);
  });
}

async function handleSelection() {
  const date = sessionSelect.value;
  const analysis = await loadAnalysis(date);
  renderHighlights(analysis);
}

async function init() {
  try {
    state.history = await loadHistory();
    renderDropdown();
    if (state.history.sessions.length) {
      sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
    }
    sessionSelect.addEventListener('change', handleSelection);
    await handleSelection();
  } catch (error) {
    highlightsRoot.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
"""

APP_JS = r"""const sessionSelect = document.getElementById('session-select');
const sessionTitle = document.getElementById('session-title');
const sessionDetails = document.getElementById('session-details');
const recommendationsEl = document.getElementById('recommendations');
const transcriptRowsEl = document.getElementById('transcript-rows');
const transcriptNoteEl = document.getElementById('transcript-note');
const proficiencyChart = document.getElementById('progress-proficiency');
const errorsChart = document.getElementById('progress-errors');
const rebuildMetricsButton = document.getElementById('rebuild-metrics-button');
const rebuildAnnotationsButton = document.getElementById('rebuild-annotations-button');
const testModelButton = document.getElementById('test-model-button');
const deleteButton = document.getElementById('delete-button');
const rebuildStatus = document.getElementById('rebuild-status');
const rebuildMeta = document.getElementById('rebuild-meta');
const llmStatus = document.getElementById('llm-status');

const state = {
  history: null,
  analysisCache: new Map(),
};

async function loadHistory() {
  const response = await fetch('history.json');
  if (!response.ok) {
    throw new Error('Unable to load history.json');
  }
  return response.json();
}

async function loadAnalysis(date) {
  if (state.analysisCache.has(date)) {
    return state.analysisCache.get(date);
  }
  const response = await fetch(`sessions/${date}/analysis.json`);
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
    option.textContent = `${session.date} - ${session.topic}`;
    sessionSelect.appendChild(option);
  });
}

function renderParticipants(analysis) {
  if (!sessionTitle || !sessionDetails) {
    return;
  }
  sessionTitle.textContent = `${analysis.date} - ${analysis.session.topic}`;
  sessionDetails.innerHTML = '';
  sortParticipants(analysis.participants).forEach((participant) => {
    const card = document.createElement('div');
    card.className = 'card';
    const grammar = participant.llm?.annotation_grammar || participant.llm?.chunked_grammar;
    card.innerHTML = `
      <h3>${participant.name} <span>${participant.role}</span></h3>
      <p class="metric"><strong>Words:</strong> ${participant.metrics.word_count}<span class="metric-note">Count of A-Z words in this speaker's turns.</span></p>
      <p class="metric"><strong>Turns:</strong> ${participant.metrics.turn_count}<span class="metric-note">Number of blocks starting with "${participant.name}:".</span></p>
      <p class="metric"><strong>Avg words/turn:</strong> ${participant.metrics.avg_words_per_turn}<span class="metric-note">Words divided by turns.</span></p>
      <p class="metric"><strong>Lexical diversity:</strong> ${participant.metrics.lexical_diversity}<span class="metric-note">Unique words / total words.</span></p>
      ${participant.llm ? `<p class="metric"><strong>LLM Proficiency:</strong> ${participant.llm.fluency.score} (${participant.llm.fluency.level})<span class="metric-note">Model-based assessment.</span></p>` : ''}
      ${grammar?.error_rate_per_100_words !== undefined ? `<p class="metric"><strong>Grammar error rate:</strong> ${grammar.error_rate_per_100_words} / 100 words<span class="metric-note">Derived from annotations when available.</span></p>` : ''}
      ${grammar?.error_types?.length ? `<ul class="errors">${grammar.error_types.map((item) => `<li><strong>${item.title}</strong> (${item.count})</li>`).join('')}</ul>` : ''}
    `;
    sessionDetails.appendChild(card);
  });
}

function renderRecommendations(analysis) {
  recommendationsEl.innerHTML = '';
  const annotationItems = analysis.llm?.annotation_items || [];
  const annotationByName = buildAnnotationMap(analysis.transcript, annotationItems, analysis.speaker_map || {});
  sortParticipants(analysis.participants).forEach((participant) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'card';
    const grammar = participant.llm?.annotation_grammar || participant.llm?.chunked_grammar;
    const chunkedErrors = grammar?.error_types
      ?.map((error) => {
        const categoryCode = getCategoryCode(error);
        const examples = findAnnotationExamples(annotationByName, participant.name, categoryCode);
        const exampleItems = examples.length
          ? examples.map((item) => `<li><strong>${escapeHtml(item.text)}</strong> &rarr; ${escapeHtml(item.correction)}</li>`).join('')
          : '<li class="metric-note">No examples yet.</li>';
        return `
          <details class="error-group">
            <summary>${escapeHtml(error.title)} <span class="count">(${error.count})</span></summary>
            <ul class="errors compact">${exampleItems}</ul>
          </details>
        `;
      })
      .join('');
    const practicalList = participant.llm?.practical_recommendations || [];
    const practicalItems = practicalList
      .map((rec) => {
        const examples = (rec.examples || [])
          .map(
            (example) =>
              `<li><span class="example-error">${escapeHtml(example.error)}</span><span class="example-arrow">&rarr;</span><span class="example-fix">${escapeHtml(example.correction)}</span></li>`
          )
          .join('');
        const guidance = rec.guidance ? `<p class="metric-note">${escapeHtml(rec.guidance)}</p>` : '';
        const examplesBlock = examples ? `<ul class="errors compact">${examples}</ul>` : '';
        return `
          <li class="recommendation-item">
            <p class="recommendation-title">${escapeHtml(rec.title)}</p>
            ${guidance}
            ${examplesBlock}
          </li>
        `;
      })
      .join('');
    const chunkedBlock = grammar
      ? `
        <h4 class="subheading">Grammar breakdown</h4>
        <p class="metric-note">Total grammar errors: ${grammar.total_errors}</p>
        ${chunkedErrors ? `<div class="error-groups">${chunkedErrors}</div>` : ''}
      `
      : '';
    const summaryText = buildRecommendationSummary(participant, practicalList);
    const practicalBlock = practicalItems
      ? `
        <h4 class="subheading">Practical recommendations</h4>
        <p class="recommendations-summary">${escapeHtml(summaryText)}</p>
        <p class="recommendations-note">Top 3 most frequent patterns this session.</p>
        <ol class="recommendations-list">${practicalItems}</ol>
      `
      : '';
    wrapper.innerHTML = `
      <h3>${participant.name}</h3>
      ${practicalBlock}
      ${chunkedBlock}
    `;
    recommendationsEl.appendChild(wrapper);
  });

  const chunkedContainer = analysis.llm?.chunked_metrics;
  if (chunkedContainer?.status) {
    const wrapper = document.createElement('div');
    wrapper.className = 'card';
    wrapper.innerHTML = `
      <h3>Chunked grammar (${chunkedContainer.status})</h3>
      <p class="metric-note">Chunked metrics run per speaker and appear in the speaker cards.</p>
    `;
    recommendationsEl.appendChild(wrapper);
  }
}

function sortParticipants(participants) {
  const list = Array.isArray(participants) ? [...participants] : [];
  return list.sort((a, b) => {
    const aName = (a?.name || '').toLowerCase();
    const bName = (b?.name || '').toLowerCase();
    if (aName === 'roman' && bName !== 'roman') return -1;
    if (bName === 'roman' && aName !== 'roman') return 1;
    return aName.localeCompare(bName);
  });
}

function escapeHtml(value) {
  if (!value) {
    return '';
  }
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function buildRecommendationSummary(participant, practicalList) {
  const name = participant?.name || 'This speaker';
  const totalErrors = participant?.llm?.annotation_grammar?.total_errors ?? participant?.llm?.chunked_grammar?.total_errors;
  const countText = typeof totalErrors === 'number' ? `${totalErrors} total grammar errors.` : 'Grammar errors detected.';
  const topTitles = practicalList.slice(0, 3).map((rec) => rec.title).filter(Boolean);
  if (!topTitles.length) {
    return `${name}: ${countText} Focus on the examples below to correct recurring patterns.`;
  }
  return `${name}: ${countText} The main recurring issues are ${topTitles.join(', ')}.`;
}

const CATEGORY_LABEL_TO_CODE = {
  'Verb Tense': 'TENSE',
  'Verb Form': 'VERB',
  Articles: 'ARTICLE',
  Prepositions: 'PREP',
  'Word Order': 'ORDER',
  'Wrong Word': 'WORD',
  Collocation: 'COLLOC',
};

function getCategoryCode(error) {
  if (!error) return '';
  if (error.code) return String(error.code);
  return CATEGORY_LABEL_TO_CODE[error.title] || '';
}

function parseTranscriptBlocks(text) {
  const pattern = /^\s*([^:]+):/gm;
  const matches = [...text.matchAll(pattern)];
  if (!matches.length) {
    return [];
  }
  return matches.map((match, index) => {
    const start = match.index ?? 0;
    const end = index + 1 < matches.length ? (matches[index + 1].index ?? text.length) : text.length;
    return { speaker: match[1].trim(), start, end };
  });
}

function buildAnnotationMap(transcriptText, items, speakerMap) {
  if (!items.length) {
    return {};
  }
  const blocks = parseTranscriptBlocks(transcriptText);
  const sortedItems = [...items].sort((a, b) => (a.start || 0) - (b.start || 0));
  const byName = {};
  let blockIndex = 0;
  sortedItems.forEach((item) => {
    const start = Number(item.start || 0);
    while (blockIndex < blocks.length && start >= blocks[blockIndex].end) {
      blockIndex += 1;
    }
    if (blockIndex >= blocks.length) {
      return;
    }
    const block = blocks[blockIndex];
    if (start < block.start || start >= block.end) {
      return;
    }
    const label = block.speaker || '';
    const name = speakerMap[label] || label;
    if (!name) {
      return;
    }
    byName[name] = byName[name] || [];
    byName[name].push(item);
  });
  return byName;
}

function findAnnotationExamples(annotationByName, name, categoryCode) {
  const items = annotationByName[name] || [];
  if (!items.length) {
    return [];
  }
  if (!categoryCode) {
    return [];
  }
  return items.filter((item) => String(item.category || '') === String(categoryCode));
}

function buildAnnotatedLine(lineText, items, lineStart) {
  if (!items.length) {
    return escapeHtml(lineText);
  }
  const parts = [];
  let cursor = 0;
  const sorted = [...items].sort((a, b) => a.start - b.start);
  sorted.forEach((item) => {
    const start = Math.max(item.start - lineStart, 0);
    const end = Math.min(item.end - lineStart, lineText.length);
    if (end <= start) {
      return;
    }
    if (start > cursor) {
      parts.push(escapeHtml(lineText.slice(cursor, start)));
    }
    const segment = escapeHtml(lineText.slice(start, end));
    const correction = item.correction ? `Correct: ${item.correction}` : '';
    const explanation = item.explanation || '';
    let title = correction || explanation || 'Grammar issue';
    if (correction && explanation) {
      title = `${correction} | ${explanation}`;
    }
    parts.push(`<mark class="grammar-error" title="${escapeHtml(title)}">${segment}</mark>`);
    cursor = end;
  });
  if (cursor < lineText.length) {
    parts.push(escapeHtml(lineText.slice(cursor)));
  }
  return parts.join('');
}

function renderTranscript(analysis) {
  if (!transcriptRowsEl) {
    return;
  }
  const transcript = analysis.transcript || '';
  const items = analysis.llm?.annotation_items || analysis.annotation_items || [];

  if (transcriptNoteEl) {
    const meta = analysis.llm?.annotations_meta;
    if (meta && meta.total_chunks !== undefined) {
      transcriptNoteEl.textContent = `Annotated ${meta.chunks_processed}/${meta.total_chunks} chunks (${meta.processed_chars} chars).`;
    } else {
      transcriptNoteEl.textContent = items.length
        ? 'Annotated transcript generated.'
        : 'Annotations will appear after the chunked LLM analysis is implemented.';
    }
  }

  const lines = transcript.split('\n');
  let offset = 0;
  transcriptRowsEl.innerHTML = '';

  lines.forEach((line, index) => {
    const lineStart = offset;
    const lineEnd = offset + line.length;
    offset += line.length + 1;

    const lineItems = items.filter((item) => item.start < lineEnd && item.end > lineStart);

    const row = document.createElement('div');
    row.className = 'transcript-row';
    if (!lineItems.length) {
      row.classList.add('row-empty');
    }

    const originalCell = document.createElement('div');
    originalCell.className = 'transcript-cell transcript-text';
    originalCell.innerHTML = line ? escapeHtml(line) : '&nbsp;';

    const annotatedCell = document.createElement('div');
    annotatedCell.className = 'transcript-cell transcript-text';
    annotatedCell.innerHTML = line ? buildAnnotatedLine(line, lineItems, lineStart) : '&nbsp;';

    const issuesCell = document.createElement('div');
    issuesCell.className = 'transcript-cell transcript-issues';
    if (!lineItems.length) {
      issuesCell.innerHTML = line ? '&nbsp;' : '&nbsp;';
    } else {
      const list = document.createElement('ul');
      list.className = 'issue-list compact';
      lineItems.forEach((item) => {
        const li = document.createElement('li');
        const explanation = item.explanation ? item.explanation : '—';
        const correction = item.correction ? ` (${item.correction})` : '';
        li.innerHTML = `<div class="issue-why">${explanation}${correction}</div>`;
        list.appendChild(li);
      });
      issuesCell.appendChild(list);
    }

    row.appendChild(originalCell);
    row.appendChild(annotatedCell);
    row.appendChild(issuesCell);
    transcriptRowsEl.appendChild(row);
  });
}

function renderLlmStatus(analysis) {
  if (!llmStatus) {
    return;
  }
  const info = analysis.llm || { status: 'skipped' };
  const metricsAt = info.metrics_updated_at || 'unknown';
  const annotationsAt = info.annotations_updated_at || 'unknown';

  let metricsStatus = 'skipped';
  if (info.status === 'ok') {
    metricsStatus = `success (${info.model || 'default model'})`;
  } else if (info.status === 'error') {
    metricsStatus = `error (${info.model || 'default model'})`;
  }

  let annotationsStatus = 'skipped';
  if (info.annotations_status === 'ok') {
    const attempted = info.annotations_attempted_model
      || info.annotations_model
      || info.annotations_meta?.attempted_model;
    const model = attempted ? ` (${attempted})` : '';
    const fallback = info.annotations_meta?.fallback_from
      ? ` via ${info.annotations_meta.fallback_to}`
      : '';
    annotationsStatus = `ok${model}${fallback}`;
  } else if (info.annotations_status === 'in_progress') {
    const attempted = info.annotations_attempted_model
      || info.annotations_model
      || info.annotations_meta?.attempted_model;
    const model = attempted ? ` (${attempted})` : '';
    const meta = info.annotations_meta;
    const progress = meta && meta.total_chunks
      ? ` ${meta.chunks_processed}/${meta.total_chunks}`
      : '';
    annotationsStatus = `running${model}${progress}`;
  } else if (info.annotations_status === 'error') {
    annotationsStatus = 'error';
  }

  llmStatus.textContent = `Metrics: ${metricsStatus} • updated ${metricsAt}`;
  if (rebuildMeta) {
    rebuildMeta.textContent = `Annotations: ${annotationsStatus} • updated ${annotationsAt}`;
  }
}

function renderRebuildMeta(analysis) {
  if (!rebuildMeta) {
    return;
  }
  // handled by renderLlmStatus
}
"""

APP_JS += r"""
function drawLineChart(canvas, series, labels, options) {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, width, height);

  const padding = { top: 28, right: 20, bottom: 28, left: 36 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const allValues = series.flatMap((item) => item.values.filter((v) => v !== null));
  const minValue = options.min !== undefined ? options.min : 0;
  const maxValue = allValues.length
    ? Math.max(...allValues, options.maxFallback || 1)
    : options.maxFallback || 1;

  const count = labels.length;
  const stepX = count > 1 ? plotWidth / (count - 1) : 0;

  function xAt(index) {
    return padding.left + (count > 1 ? index * stepX : plotWidth / 2);
  }

  function yAt(value) {
    const normalized = (value - minValue) / (maxValue - minValue || 1);
    return padding.top + (1 - normalized) * plotHeight;
  }

  // Grid
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  const ticks = options.ticks || [minValue, (minValue + maxValue) / 2, maxValue];
  ctx.fillStyle = '#64748b';
  ctx.font = '11px sans-serif';
  ticks.forEach((tick) => {
    const y = yAt(tick);
    ctx.fillText(tick.toFixed(options.tickDecimals || 0), 6, y + 4);
    ctx.strokeStyle = '#f1f5f9';
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  });

  // Series
  series.forEach((item) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.values.forEach((value, index) => {
      if (value === null || value === undefined) {
        return;
      }
      const x = xAt(index);
      const y = yAt(value);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    item.values.forEach((value, index) => {
      if (value === null || value === undefined) {
        return;
      }
      const x = xAt(index);
      const y = yAt(value);
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  // Legend
  let legendX = padding.left;
  series.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, padding.top - 18, 10, 10);
    ctx.fillStyle = '#334155';
    ctx.font = '12px sans-serif';
    ctx.fillText(item.label, legendX + 16, padding.top - 9);
    legendX += 120;
  });

  // X labels
  ctx.fillStyle = '#64748b';
  labels.forEach((label, index) => {
    const x = xAt(index);
    ctx.fillText(label, x - 18, height - 8);
  });
}

function renderChart() {
  const sessions = state.history.sessions;
  if (!sessions.length) {
    return;
  }

  const dates = sessions.map((session) => session.date);
  const speakers = new Set();
  sessions.forEach((session) => {
    session.participants.forEach((participant) => speakers.add(participant.name));
  });
  const speakerList = Array.from(speakers);
  const colors = ['#0ea5e9', '#f97316', '#10b981', '#a855f7'];

  const proficiencySeries = speakerList.map((name, index) => {
    const values = sessions.map((session) => {
      const participant = session.participants.find((p) => p.name === name);
      if (!participant) {
        return null;
      }
      return participant.llm_fluency_score !== null && participant.llm_fluency_score !== undefined
        ? participant.llm_fluency_score
        : participant.fluency_score;
    });
    return {
      label: name,
      color: colors[index % colors.length],
      values,
    };
  });

  const errorSeries = speakerList.map((name, index) => {
    const values = sessions.map((session) => {
      const participant = session.participants.find((p) => p.name === name);
      if (!participant) {
        return null;
      }
      return participant.chunked_error_rate ?? participant.llm_error_rate ?? null;
    });
    return {
      label: name,
      color: colors[index % colors.length],
      values,
    };
  });

  drawLineChart(proficiencyChart, proficiencySeries, dates, {
    min: 0,
    maxFallback: 10,
    ticks: [0, 5, 10],
    tickDecimals: 0,
  });

  drawLineChart(errorsChart, errorSeries, dates, {
    min: 0,
    maxFallback: 5,
    tickDecimals: 1,
  });
}

async function handleSelection() {
  const date = sessionSelect.value;
  const analysis = await loadAnalysis(date);
  renderParticipants(analysis);
  renderRecommendations(analysis);
  renderTranscript(analysis);
  renderLlmStatus(analysis);
  renderRebuildMeta(analysis);
}

function attachRebuildMetrics() {
  if (!rebuildMetricsButton) {
    return;
  }
  rebuildMetricsButton.addEventListener('click', async () => {
    rebuildMetricsButton.disabled = true;
    rebuildStatus.textContent = 'Rebuilding metrics...';
    try {
      const date = sessionSelect.value;
      const response = await fetch('/api/rebuild-metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Rebuild failed.');
      }
      const result = await response.json();
      rebuildStatus.textContent = result.date
        ? `Rebuilt ${result.date}.`
        : `Rebuilt ${result.sessions} session(s).`;
      state.analysisCache.delete(date);
      const activeDate = date;
      state.history = await loadHistory();
      renderDropdown();
      if (sessionSelect && activeDate) {
        sessionSelect.value = activeDate;
      }
      renderChart();
      await handleSelection();
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Rebuild failed.';
    } finally {
      rebuildMetricsButton.disabled = false;
    }
  });
}

function attachRebuildAnnotations() {
  if (!rebuildAnnotationsButton) {
    return;
  }
  rebuildAnnotationsButton.addEventListener('click', async () => {
    rebuildAnnotationsButton.disabled = true;
    rebuildStatus.textContent = 'Rebuilding annotations...';
    try {
      const date = sessionSelect.value;
      const response = await fetch('/api/rebuild-annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Rebuild failed.');
      }
      const result = await response.json();
      rebuildStatus.textContent = result.date
        ? `Rebuilt ${result.date}.`
        : `Rebuilt ${result.sessions} session(s).`;
      state.analysisCache.delete(date);
      const activeDate = date;
      state.history = await loadHistory();
      renderDropdown();
      if (sessionSelect && activeDate) {
        sessionSelect.value = activeDate;
      }
      renderChart();
      await handleSelection();
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Rebuild failed.';
    } finally {
      rebuildAnnotationsButton.disabled = false;
    }
  });
}

function attachTestModel() {
  if (!testModelButton) {
    return;
  }
  testModelButton.addEventListener('click', async () => {
    testModelButton.disabled = true;
    rebuildStatus.textContent = 'Testing gpt-5-mini...';
    try {
      const response = await fetch('/api/test-gpt5', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Test failed.');
      }
      const result = await response.json();
      const output = result.output ? `Output: ${result.output}` : 'No output.';
      rebuildStatus.textContent = `Test ${result.model}: ${output}`;
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Test failed.';
    } finally {
      testModelButton.disabled = false;
    }
  });
}
"""

APP_JS += r"""
function attachDelete() {
  if (!deleteButton) {
    return;
  }
  deleteButton.addEventListener('click', async () => {
    const date = sessionSelect.value;
    if (!date) {
      return;
    }
    if (!confirm(`Delete session ${date}? This cannot be undone.`)) {
      return;
    }
    deleteButton.disabled = true;
    rebuildStatus.textContent = 'Deleting...';
    try {
      const response = await fetch('/api/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Delete failed.');
      }
      const result = await response.json();
      rebuildStatus.textContent = `Deleted ${result.deleted}.`;
      state.analysisCache.delete(date);
      state.history = await loadHistory();
      renderDropdown();
      renderChart();
      if (state.history.sessions.length) {
        sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
        await handleSelection();
      } else {
        sessionTitle.textContent = 'No sessions.';
        sessionDetails.innerHTML = '';
        recommendationsEl.innerHTML = '';
        if (transcriptRowsEl) {
          transcriptRowsEl.innerHTML = '';
        }
      }
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Delete failed.';
    } finally {
      deleteButton.disabled = false;
    }
  });
}

async function init() {
  try {
    state.history = await loadHistory();
    renderDropdown();
    if (state.history.sessions.length) {
      sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
    }
    renderChart();
    sessionSelect.addEventListener('change', handleSelection);
    attachRebuildMetrics();
    attachRebuildAnnotations();
    attachTestModel();
    attachDelete();
    await handleSelection();
  } catch (error) {
    sessionTitle.textContent = 'Failed to load data.';
    sessionDetails.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
"""

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
  max-width: 1240px;
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

.inline-link {
  display: inline-block;
  margin-top: 12px;
  color: #0ea5e9;
  text-decoration: none;
  font-weight: 600;
}

.inline-link:hover {
  text-decoration: underline;
}

.header-actions {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  flex-wrap: wrap;
  margin-top: 10px;
}

.header-actions {
  flex-direction: column;
}

.header-links,
.header-buttons {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.header-links {
  padding-right: 0;
  border-right: none;
}

.header-status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  width: 100%;
}

.status-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.status-block .ghost {
  align-self: flex-start;
}

.header-tools {
  margin-top: 6px;
}

.ghost {
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid #cbd5f5;
  background: #ffffff;
  color: #0f172a;
  font-weight: 600;
  cursor: pointer;
}

.ghost:hover {
  background: #f8fafc;
}

.ghost.danger {
  border-color: #fca5a5;
  color: #b91c1c;
}

.ghost.danger:hover {
  background: #fee2e2;
}

.header-buttons .ghost {
  border-color: #e2e8f0;
  background: #f8fafc;
  color: #334155;
}

.header-buttons .ghost:hover {
  background: #eef2f7;
}

.header-buttons .ghost.danger {
  border-color: #fecaca;
  background: #fff5f5;
  color: #b91c1c;
}

.header-buttons .ghost.danger:hover {
  background: #fee2e2;
}

.ghost:disabled {
  opacity: 0.6;
  cursor: default;
}

.header-links .inline-link {
  margin-top: 0;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid transparent;
  font-weight: 600;
  text-decoration: none;
}

.header-links .inline-link:first-child {
  background: #0ea5e9;
  color: #ffffff;
}

.header-links .inline-link:first-child:hover {
  background: #0284c7;
}

.header-links .inline-link:last-child {
  background: #ecfeff;
  color: #0f766e;
  border-color: #a5f3fc;
}

.header-links .inline-link:last-child:hover {
  background: #cffafe;
}

.controls {
  margin-bottom: 24px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.controls-highlights {
  max-width: 360px;
  background: #ffffff;
  border: 1px solid #dbeafe;
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
}

.controls-highlights label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #475569;
}

.session-row {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.session-row select {
  flex: 1 1 260px;
}

select {
  max-width: 320px;
  padding: 8px 12px;
  font-size: 15px;
}

.controls-highlights select {
  max-width: none;
  width: 100%;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  background: #f8fafc;
  color: #0f172a;
  font-weight: 600;
}

.controls-highlights select:focus {
  outline: none;
  border-color: #38bdf8;
  box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
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

.highlight-grid {
  display: grid;
  gap: 16px;
}

.highlight-card {
  background: #ffffff;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.highlight-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 10px;
  margin-bottom: 12px;
}

.highlight-header h2 {
  margin: 0;
  font-size: 20px;
}

.highlight-lead {
  margin: 0 0 12px;
  color: #334155;
  font-size: 15px;
}

.highlight-meta {
  font-size: 12px;
  color: #64748b;
}

.highlight-item {
  display: grid;
  grid-template-columns: 32px 1fr;
  gap: 12px;
  padding: 12px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  margin-bottom: 12px;
}

.highlight-item:last-child {
  margin-bottom: 0;
}

.highlight-rank {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #0f172a;
  color: #ffffff;
  font-weight: 700;
  display: grid;
  place-items: center;
  font-size: 12px;
}

.highlight-body h3 {
  margin: 0 0 6px;
  font-size: 16px;
}

.highlight-why {
  margin: 0 0 8px;
  color: #0f172a;
  background: #ecfeff;
  border: 1px solid #bae6fd;
  border-radius: 10px;
  padding: 8px 10px;
}

.highlight-focus {
  margin: 0 0 8px;
  color: #475569;
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

.metric {
  margin: 8px 0 0;
}

.metric-note {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: #64748b;
}

.upload-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.upload-card form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.mapping {
  padding-top: 8px;
  border-top: 1px solid #e2e8f0;
}

.section-title {
  margin: 0 0 6px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #64748b;
}

.row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}

input[type='text'],
input[type='date'],
input[type='number'] {
  padding: 8px 12px;
  font-size: 15px;
  border: 1px solid #cbd5f5;
  border-radius: 8px;
}

.file-label {
  font-weight: 600;
}

.primary {
  padding: 10px 16px;
  border: none;
  border-radius: 999px;
  background: #0f172a;
  color: #ffffff;
  font-weight: 600;
  cursor: pointer;
}

.primary:hover {
  background: #1e293b;
}

.helper {
  margin: 6px 0 0;
  color: #64748b;
  font-size: 13px;
}

#llm-status,
#rebuild-meta,
#rebuild-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid #e2e8f0;
  background: #ffffff;
  margin-right: 8px;
}

.progress {
  margin-top: 24px;
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}

.chart-card {
  background: #ffffff;
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
}

.chart-card h3 {
  margin: 0 0 12px;
  font-size: 16px;
}

canvas {
  width: 100%;
  background: #ffffff;
  border-radius: 12px;
}

.transcript {
  margin-top: 24px;
}

.transcript-rows {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.transcript-row {
  display: grid;
  grid-template-columns: 1fr 1fr 0.7fr;
  gap: 16px;
  align-items: start;
}

.transcript-row.row-empty .transcript-cell {
  background: transparent;
  border-color: transparent;
  padding: 0;
  min-height: 0;
}

.transcript-header h3 {
  margin: 0 0 6px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #64748b;
}

.transcript-cell {
  padding: 8px 12px;
  background: #ffffff;
  border-radius: 10px;
  border: 1px solid #e2e8f0;
  font-family: 'SFMono-Regular', 'Consolas', 'Liberation Mono', monospace;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  min-height: 28px;
}

.transcript-cell.transcript-issues {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  font-size: 13px;
  line-height: 1.4;
  white-space: normal;
}

.issue-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.issue-list.compact li {
  padding: 6px 8px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  margin-bottom: 6px;
}

.issue-why {
  color: #475569;
}

.subheading {
  margin: 12px 0 8px;
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
}

.errors {
  list-style: none;
  padding: 0;
  margin: 8px 0 0;
}

.errors li {
  margin-bottom: 12px;
}

.errors.compact li {
  margin-bottom: 6px;
  padding: 6px 8px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}

.recommendations-list {
  margin: 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.recommendations-summary {
  margin: 0 0 6px;
  color: #334155;
  line-height: 1.5;
}

.recommendations-note {
  margin: 4px 0 12px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #0f172a;
  background: #e2e8f0;
  border: 1px solid #cbd5f5;
}

.recommendation-item {
  padding: 10px 12px;
  border-radius: 10px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}

.recommendation-item p {
  margin: 0 0 6px;
}

.recommendation-item .metric-note {
  margin: 0 0 8px;
  padding: 8px 10px;
  border-radius: 10px;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  color: #1e293b;
  font-weight: 600;
}

.recommendation-title {
  font-weight: 600;
  margin: 0 0 6px;
  color: #0f172a;
}

.error-groups {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.error-group {
  border-radius: 10px;
  border: 1px solid #e2e8f0;
  background: #ffffff;
}

.error-group summary {
  cursor: pointer;
  list-style: none;
  padding: 10px 12px;
  font-weight: 600;
  background: #f8fafc;
  border-radius: 10px;
}

.error-group summary::-webkit-details-marker {
  display: none;
}

.error-group summary::after {
  content: '▾';
  float: right;
  color: #64748b;
}

.error-group[open] summary::after {
  content: '▴';
}

.error-group .errors {
  margin: 0;
  padding: 10px 12px 12px;
}

.error-group .count {
  color: #64748b;
  font-weight: 500;
}

.example-error {
  font-weight: 600;
  color: #0f172a;
}

.example-fix {
  color: #0f172a;
}

.example-arrow {
  margin: 0 6px;
  color: #94a3b8;
}

.example {
  margin: 8px 0 0;
  padding: 8px 12px;
  background: #f8fafc;
  border-radius: 8px;
}

.correction {
  margin-top: 6px;
  font-weight: 600;
}

.correction.missing {
  color: #b91c1c;
}

mark.grammar-error {
  background: #fde68a;
  padding: 0 2px;
  border-radius: 3px;
  cursor: help;
}
"""


def write_web_assets(out_dir: Path) -> None:
    web_dir = out_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (web_dir / "upload.html").write_text(UPLOAD_HTML, encoding="utf-8")
    (web_dir / "highlights.html").write_text(HIGHLIGHTS_HTML, encoding="utf-8")
    (web_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (web_dir / "highlights.js").write_text(HIGHLIGHTS_JS, encoding="utf-8")
    (web_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    history_path = out_dir / "history.json"
    if history_path.exists():
        (web_dir / "history.json").write_text(history_path.read_text(encoding="utf-8"), encoding="utf-8")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_all(args.sessions, args.out, use_openai=args.use_openai, openai_model=args.model)
    print(f"Built {count} session(s).")


if __name__ == "__main__":
    main()
