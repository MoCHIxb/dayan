from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv"}
ANNOTATION_FIELDS = [
    "id",
    "source",
    "source_group",
    "text",
    "keep",
    "task_type",
    "style_label",
    "style_traits",
    "topic_label",
    "quality",
    "instruction",
    "input",
    "license",
    "consent",
    "redistributable",
    "pii_flags",
    "near_duplicate_of",
    "notes",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 不是 JSON object")
            records.append(value)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "output", "response", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def iter_documents(input_path: str | Path) -> Iterator[tuple[str, str]]:
    root = Path(input_path)
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    )
    for path in files:
        source = path.name if root.is_file() else path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            yield source, path.read_text(encoding="utf-8-sig")
        elif suffix == ".jsonl":
            for index, record in enumerate(read_jsonl(path)):
                text = _extract_text(record)
                if text:
                    yield f"{source}#{index}", text
        elif suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = value if isinstance(value, list) else [value]
            for index, record in enumerate(rows):
                if isinstance(record, dict):
                    text = _extract_text(record)
                    if text:
                        yield f"{source}#{index}", text
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for index, record in enumerate(csv.DictReader(handle)):
                    text = _extract_text(record)
                    if text:
                        yield f"{source}#{index}", text


def clean_text(text: str, remove_patterns: Iterable[str] = ()) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    for pattern in remove_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    text = re.sub(r"[\t\u00a0\u3000]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _sentences(paragraph: str) -> list[str]:
    pieces = re.findall(r".+?(?:[。！？!?；;]+[”’\"']?|$)", paragraph.strip(), flags=re.DOTALL)
    return [piece.strip() for piece in pieces if piece.strip()]


def segment_text(text: str, min_chars: int = 30, max_chars: int = 800) -> list[str]:
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("长度参数必须满足 1 <= min_chars <= max_chars")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    segments: list[str] = []
    for paragraph in paragraphs:
        sentences = _sentences(paragraph)
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    segments.append(current)
                    current = ""
                for start in range(0, len(sentence), max_chars):
                    segments.append(sentence[start : start + max_chars])
            elif not current or len(current) + len(sentence) <= max_chars:
                current += sentence
            else:
                segments.append(current)
                current = sentence
        if current:
            segments.append(current)
    return [segment for segment in segments if len(segment) >= min_chars]


def stable_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"seg_{digest}"


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def near_duplicate_similarity(left: str, right: str, n: int = 5) -> float:
    left_set = _char_ngrams(left, n)
    right_set = _char_ngrams(right, n)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def pii_flags(text: str) -> list[str]:
    patterns = {
        "email": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "phone_cn": r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)",
        "id_cn": r"(?<!\d)\d{17}[\dXx](?!\d)",
        "url": r"(?i)\b(?:https?://|www\.)\S+",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def prepare_records(input_path: str | Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for source, raw_text in iter_documents(input_path):
        cleaned = clean_text(raw_text, config.get("remove_patterns", []))
        for text in segment_text(
            cleaned,
            min_chars=int(config.get("min_chars", 30)),
            max_chars=int(config.get("max_chars", 800)),
        ):
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            record_id = stable_id(source, text)
            ngram = int(config.get("near_duplicate_ngram", 5))
            threshold = float(config.get("near_duplicate_threshold", 0.85))
            near_duplicate_of = ""
            for previous in records:
                if near_duplicate_similarity(text, str(previous["text"]), ngram) >= threshold:
                    near_duplicate_of = str(previous["id"])
                    break
            records.append(
                {
                    "id": record_id,
                    "source": source,
                    "source_group": source.split("#", 1)[0],
                    "text": text,
                    "keep": True,
                    "task_type": "rewrite",
                    "style_label": "",
                    "style_traits": "",
                    "topic_label": "",
                    "quality": 0,
                    "instruction": config.get("default_instruction", ""),
                    "input": "",
                    "license": config.get("default_license", "TO_BE_REVIEWED"),
                    "consent": bool(config.get("default_consent", False)),
                    "redistributable": bool(config.get("default_redistributable", False)),
                    "pii_flags": ",".join(pii_flags(text)),
                    "near_duplicate_of": near_duplicate_of,
                    "notes": "",
                }
            )
    return records


def export_annotation(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in ANNOTATION_FIELDS}
            row["keep"] = "1" if bool(record.get("keep", True)) else "0"
            row["consent"] = "1" if bool(record.get("consent", False)) else "0"
            row["redistributable"] = "1" if bool(record.get("redistributable", False)) else "0"
            writer.writerow(row)


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "保留"}


def import_annotation(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    for row in rows:
        row["keep"] = _parse_bool(row.get("keep", ""))
        row["consent"] = _parse_bool(row.get("consent", ""))
        row["redistributable"] = _parse_bool(row.get("redistributable", ""))
        raw_quality = str(row.get("quality", "")).strip()
        row["quality"] = int(raw_quality) if raw_quality else 0
        records.append(dict(row))
    return records


def validation_errors(records: Iterable[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for index, record in enumerate(records, 1):
        prefix = f"记录 {index}"
        for field in ("id", "source", "text"):
            if not str(record.get(field, "")).strip():
                errors.append(f"{prefix}: 缺少 {field}")
        record_id = str(record.get("id", ""))
        if record_id in seen_ids:
            errors.append(f"{prefix}: id 重复 {record_id}")
        seen_ids.add(record_id)
        text = str(record.get("text", "")).strip()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text and text_hash in seen_text:
            errors.append(f"{prefix}: text 精确重复")
        seen_text.add(text_hash)
        if bool(record.get("keep", False)):
            quality = record.get("quality", 0)
            if not isinstance(quality, int) or not 1 <= quality <= 5:
                errors.append(f"{prefix}: 保留样本 quality 必须为 1-5")
            if not str(record.get("style_label", "")).strip():
                errors.append(f"{prefix}: 保留样本缺少 style_label")
            if not str(record.get("topic_label", "")).strip():
                errors.append(f"{prefix}: 保留样本缺少 topic_label")
            if not str(record.get("instruction", "")).strip():
                errors.append(f"{prefix}: 保留样本缺少 instruction")
    return errors


def split_records(
    records: Iterable[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 20251206,
) -> dict[str, list[dict[str, Any]]]:
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/validation/test 比例必须为非负且总和为 1")
    kept = [record for record in records if bool(record.get("keep", False))]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in kept:
        source_group = str(record.get("source_group", "")).strip()
        if not source_group:
            source_group = str(record["source"]).split("#", 1)[0]
        groups[source_group].append(record)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    names = ("train", "validation", "test")
    active_names = tuple(name for index, name in enumerate(names) if ratios[index] > 0)
    targets = {name: ratios[i] * len(kept) for i, name in enumerate(names)}
    result = {name: [] for name in names}
    for group_index, (_, group) in enumerate(items):
        empty_positive = [name for i, name in enumerate(names) if ratios[i] > 0 and not result[name]]
        if empty_positive and len(items) - group_index <= len(empty_positive):
            chosen = empty_positive[0]
        else:
            chosen = max(active_names, key=lambda name: targets[name] - len(result[name]))
        result[chosen].extend(group)
    for name in names:
        result[name].sort(key=lambda record: str(record.get("id", "")))
    return result


def convert_record(record: dict[str, Any], output_format: str) -> dict[str, Any]:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    answer = str(record.get("text", "")).strip()
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    metadata = {
        "id": record.get("id", ""),
        "style_label": record.get("style_label", ""),
        "topic_label": record.get("topic_label", ""),
    }
    if output_format == "messages":
        return {**metadata, "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]}
    if output_format == "alpaca":
        return {**metadata, "instruction": instruction, "input": input_text, "output": answer}
    if output_format == "prompt-completion":
        return {**metadata, "prompt": user_content, "completion": answer}
    raise ValueError(f"不支持的格式: {output_format}")


def dataset_stats(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    kept = [row for row in rows if bool(row.get("keep", False))]
    lengths = [len(str(row.get("text", ""))) for row in kept]
    return {
        "total": len(rows),
        "kept": len(kept),
        "discarded": len(rows) - len(kept),
        "sources": len(
            {
                str(row.get("source_group", "")).strip()
                or str(row.get("source", "")).split("#", 1)[0]
                for row in kept
            }
        ),
        "mean_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "style_labels": dict(Counter(str(row.get("style_label", "")) for row in kept)),
        "topic_labels": dict(Counter(str(row.get("topic_label", "")) for row in kept)),
        "quality": dict(Counter(str(row.get("quality", "")) for row in kept)),
    }

