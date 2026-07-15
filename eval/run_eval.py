#!/usr/bin/env python3
"""Score an annotation run against the hand-labeled 2026-07-14 set.

Why this exists: annotation quality was only ever judged by eye, so prompt and
model changes could not be compared. Every item of the 2026-07-14 baseline was
adjudicated by hand (REAL / FP / ART) and stored in eval_set_2026-07-14.json.
A candidate run is matched to those labels by transcript span overlap, which
survives the model rewording or resizing a span.

    python eval/run_eval.py <candidate.json> [--json]

<candidate.json> is either a full analysis.json or a bare list of annotation
items; each item needs start/end offsets into the same transcript.

Findings the candidate reports that no labeled item covers are NOT scored: they
are new and nobody has judged them. They are printed for manual review, and
their count is the honest uncertainty on any precision figure below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cli  # noqa: E402  — reuse prod speaker mapping so eval measures prod logic

EVAL_DIR = Path(__file__).resolve().parent
# Two spans describe the same finding when they overlap by at least half of the
# shorter one. Looser than exact text match (the model rewords spans between
# runs), tighter than "touches at all" (adjacent errors would collapse).
MATCH_THRESHOLD = 0.5
CATEGORIES = ["ARTICLE", "PREP", "VERB", "ORDER", "COLLOC", "TENSE", "WORD"]


def load_eval_set(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transcript = (EVAL_DIR / payload["transcript_file"]).read_text(encoding="utf-8")
    digest = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    if digest != payload["transcript_sha256"]:
        raise SystemExit(
            f"transcript hash mismatch: labels were made against a different text\n"
            f"  expected {payload['transcript_sha256']}\n  got      {digest}"
        )
    payload["transcript"] = transcript
    return payload


def load_candidate(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = (payload.get("llm") or {}).get("annotation_items")
        if items is None:
            items = payload.get("annotation_items")
        if items is None:
            raise SystemExit(f"{path}: no llm.annotation_items or annotation_items")
    else:
        items = payload
    for item in items:
        if "start" not in item or "end" not in item:
            raise SystemExit(f"{path}: every item needs start/end transcript offsets")
    return items


def overlap_ratio(a: dict, b: dict) -> float:
    span = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if span <= 0:
        return 0.0
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return span / shorter if shorter else 0.0


def match(candidate: list[dict], labeled: list[dict]) -> tuple[dict, dict]:
    """Greedy best-overlap pairing; each labeled item is claimed at most once."""
    pairs = []
    for ci, cand in enumerate(candidate):
        for li, lab in enumerate(labeled):
            ratio = overlap_ratio(cand, lab)
            if ratio >= MATCH_THRESHOLD:
                pairs.append((ratio, ci, li))
    pairs.sort(key=lambda row: -row[0])
    cand_to_lab: dict[int, int] = {}
    lab_to_cand: dict[int, int] = {}
    for _ratio, ci, li in pairs:
        if ci in cand_to_lab or li in lab_to_cand:
            continue
        cand_to_lab[ci] = li
        lab_to_cand[li] = ci
    return cand_to_lab, lab_to_cand


def speaker_of(item: dict, transcript: str, speaker_map: dict) -> str:
    by_name = cli.map_annotation_items_to_speakers([item], transcript, speaker_map)
    return next(iter(by_name), "?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--eval-set", type=Path, default=EVAL_DIR / "eval_set_2026-07-14.json")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = parser.parse_args()

    ev = load_eval_set(args.eval_set)
    labeled = ev["items"]
    transcript = ev["transcript"]
    candidate = load_candidate(args.candidate)
    speaker_map = {"Speaker A": "Roman", "Speaker B": "Andrey"}

    cand_to_lab, lab_to_cand = match(candidate, labeled)

    kept_real, kept_junk, new_items = [], [], []
    for ci, cand in enumerate(candidate):
        li = cand_to_lab.get(ci)
        if li is None:
            new_items.append(cand)
        elif labeled[li]["label"] == "REAL":
            kept_real.append((cand, labeled[li]))
        else:
            kept_junk.append((cand, labeled[li]))

    total_real = sum(1 for x in labeled if x["label"] == "REAL")
    total_junk = len(labeled) - total_real
    missed_real = [x for i, x in enumerate(labeled) if x["label"] == "REAL" and i not in lab_to_cand]
    dropped_junk = [x for i, x in enumerate(labeled) if x["label"] != "REAL" and i not in lab_to_cand]

    known = len(kept_real) + len(kept_junk)
    precision = len(kept_real) / known if known else 0.0
    recall = len(kept_real) / total_real if total_real else 0.0

    summary = {
        "candidate": str(args.candidate),
        "reported_findings": len(candidate),
        "kept_real": len(kept_real),
        "kept_junk": len(kept_junk),
        "precision_on_known": round(precision, 4),
        "missed_real": len(missed_real),
        "recall_on_known_real": round(recall, 4),
        "dropped_junk": len(dropped_junk),
        "junk_dropped_pct": round(len(dropped_junk) / total_junk, 4) if total_junk else 0.0,
        "new_unlabeled": len(new_items),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    print(f"candidate: {args.candidate.name}   baseline set: {args.eval_set.name}")
    print(f"reported findings: {len(candidate)}   (labeled baseline had {len(labeled)})")
    print()
    print(f"  kept REAL        {len(kept_real):3d}   of {total_real} known real errors")
    print(f"  kept junk        {len(kept_junk):3d}   of {total_junk} known non-errors  <- want low")
    print(f"  dropped junk     {len(dropped_junk):3d}   ({summary['junk_dropped_pct']:.0%} of known junk removed)")
    print(f"  missed REAL      {len(missed_real):3d}   ({1 - recall:.0%} of real errors lost)")
    print(f"  new unlabeled    {len(new_items):3d}   not covered by any label -> review by hand")
    print()
    print(f"  precision on known items : {precision:.0%}")
    print(f"  recall on known real     : {recall:.0%}")

    per_cat = defaultdict(lambda: [0, 0])
    for cand, lab in kept_real:
        per_cat[cand.get("category", "?")][0] += 1
        per_cat[cand.get("category", "?")][1] += 1
    for cand, lab in kept_junk:
        per_cat[cand.get("category", "?")][1] += 1
    print()
    print("  by category (known items only)")
    for code in CATEGORIES:
        real, shown = per_cat.get(code, [0, 0])
        if shown:
            print(f"    {code:<8} {real:3d}/{shown:<3d}  {real / shown:.0%}")

    per_sp = defaultdict(lambda: [0, 0])
    for cand, lab in kept_real:
        per_sp[lab["speaker"]][0] += 1
        per_sp[lab["speaker"]][1] += 1
    for cand, lab in kept_junk:
        per_sp[lab["speaker"]][1] += 1
    print()
    print("  by speaker (known items only)")
    for name, (real, shown) in sorted(per_sp.items()):
        print(f"    {name:<8} {real:3d}/{shown:<3d}  {real / shown:.0%}")

    if new_items:
        print()
        print(f"  NEW findings needing adjudication ({len(new_items)}):")
        for item in new_items[:40]:
            who = speaker_of(item, transcript, speaker_map)
            print(f"    [{item.get('category', '?'):<7}] {who:<7} {item.get('text', '')[:70]!r}")
        if len(new_items) > 40:
            print(f"    ... and {len(new_items) - 40} more")

    if missed_real:
        print()
        print(f"  REAL errors the candidate lost ({len(missed_real)}):")
        for item in missed_real[:25]:
            print(f"    [{item['category']:<7}] {item['speaker']:<7} {item['text'][:70]!r}")
        if len(missed_real) > 25:
            print(f"    ... and {len(missed_real) - 25} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
