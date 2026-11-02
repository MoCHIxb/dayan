from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .pipeline import write_jsonl


SYSTEM_PROMPT = (
    "You are a personalized rewriting assistant. Preserve the meaning of the requested "
    "tweet, infer the user's writing style from the profile examples, and return only the "
    "rewritten tweet without explanation."
)

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
_TOKEN = re.compile(r"[A-Za-z0-9']+")


def sanitize_text(text: str) -> str:
    """Decode HTML entities and redact direct identifiers without inventing content."""
    value = unicodedata.normalize("NFKC", html.unescape(str(text)))
    value = _URL.sub("<URL>", value)
    value = _EMAIL.sub("<EMAIL>", value)
    value = _HANDLE.sub("<USER>", value)
    value = _PHONE.sub("<PHONE>", value)
    value = re.sub(r"[\t\r\n]+", " ", value)
    value = re.sub(r" {2,}", " ", value)
    return value.strip()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text) if len(token) > 1]


def _task_text(task_input: str) -> str:
    """Remove the benchmark's repeated instruction prefix before retrieval."""
    _, separator, value = task_input.partition(":")
    return value.strip() if separator else task_input


def select_profile(profile: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    """Select existing history with per-profile BM25; no text is generated or altered."""
    tokenized = [_tokens(sanitize_text(item.get("text", ""))) for item in profile]
    query_tokens = _tokens(_task_text(sanitize_text(query)))
    document_count = max(len(tokenized), 1)
    average_length = sum(len(tokens) for tokens in tokenized) / document_count or 1.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    k1, b = 1.5, 0.75
    ranked: list[tuple[float, int, dict[str, str]]] = []
    for index, (item, tokens) in enumerate(zip(profile, tokenized)):
        text = sanitize_text(item.get("text", ""))
        frequencies = Counter(tokens)
        score = 0.0
        for term in set(query_tokens):
            tf = frequencies[term]
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * len(tokens) / average_length)
            score += idf * tf * (k1 + 1.0) / denominator
        ranked.append(
            (
                score,
                -index,
                {"id": str(item.get("id", "")), "text": text, "retrieval_score": round(score, 6)},
            )
        )
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in ranked[:limit] if item["text"]]


def build_user_message(task_input: str, profile: list[dict[str, Any]]) -> str:
    examples = "\n".join(f"{index}. {item['text']}" for index, item in enumerate(profile, 1))
    return f"User style examples:\n{examples}\n\nTask:\n{sanitize_text(task_input)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(raw_dir: Path, split: str, profile_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_path = raw_dir / split / f"{split}_questions.json"
    output_path = raw_dir / split / f"{split}_outputs.json"
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    outputs = {str(row["id"]): row["output"] for row in output_payload["golds"]}
    if len(outputs) != len(output_payload["golds"]):
        raise ValueError(f"{split}: duplicate output ids")

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for question in questions:
        record_id = str(question["id"])
        if record_id not in outputs:
            missing.append(record_id)
            continue
        task_input = sanitize_text(question["input"])
        selected = select_profile(question.get("profile", []), task_input, profile_k)
        response = sanitize_text(outputs[record_id])
        if not task_input or not response or not selected:
            continue
        records.append(
            {
                "id": record_id,
                "dataset": "LaMP-7",
                "official_split": split,
                "task": "personalized_tweet_paraphrasing",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(task_input, selected)},
                    {"role": "assistant", "content": response},
                ],
                "profile": selected,
                "profile_size_original": len(question.get("profile", [])),
                "profile_size_selected": len(selected),
                "profile_retrieval": "bm25_per_user_k1_1.5_b_0.75",
                "license_note": "LaMP code/data construction CC BY-NC-SA 4.0; underlying source terms apply",
            }
        )

    stats = {
        "split": split,
        "question_count": len(questions),
        "output_count": len(outputs),
        "record_count": len(records),
        "missing_output_ids": missing,
        "profile_size_original": dict(Counter(row["profile_size_original"] for row in records)),
        "question_sha256": _sha256(question_path),
        "output_sha256": _sha256(output_path),
    }
    return records, stats


def prepare_lamp7(
    raw_dir: str | Path,
    output_dir: str | Path,
    profile_k: int = 8,
    seed: int = 20251206,
    subset_sizes: Iterable[int] = (50, 100, 200, 500, 2000),
) -> dict[str, Any]:
    raw_root = Path(raw_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, Any] = {"dataset": "LaMP-7", "profile_k": profile_k, "seed": seed}
    prepared: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        records, stats = load_split(raw_root, split, profile_k)
        prepared[split] = records
        all_stats[split] = stats
        write_jsonl(records, output_root / f"{split}.jsonl")

    indices = list(range(len(prepared["train"])))
    random.Random(seed).shuffle(indices)
    previous: set[str] = set()
    subset_stats: dict[str, Any] = {}
    for size in sorted(set(int(value) for value in subset_sizes)):
        if size > len(indices):
            raise ValueError(f"subset size {size} exceeds train size {len(indices)}")
        subset = [prepared["train"][index] for index in indices[:size]]
        ids = {row["id"] for row in subset}
        if not previous.issubset(ids):
            raise AssertionError("subsets are not nested")
        previous = ids
        write_jsonl(subset, output_root / f"train_{size}.jsonl")
        subset_stats[str(size)] = {"count": len(subset), "id_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}
    all_stats["subsets"] = subset_stats
    (output_root / "dataset_report.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_stats
