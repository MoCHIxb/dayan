"""Explicit real-pair import for developer examples; no answer generation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import unicodedata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Personal example importer accepts at most 16 MiB")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            return list(csv.DictReader(stream))
        if path.suffix.lower() != ".jsonl":
            raise ValueError("Input must be CSV or JSONL")
        return [json.loads(line) for line in stream if line.strip()]


def validate_pairs(rows):
    groups, ids, fingerprints = {}, set(), {}
    splits = {name: [] for name in ("train", "dev", "test")}
    if not rows or len(rows) > 10000:
        raise ValueError("Need 1–10000 explicitly split records")
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {number}: expected an object")
        for key in ("id", "source_group", "split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Row {number}: nonempty string {key} required")
        key, group, split = row["id"].strip(), row["source_group"].strip(), row["split"].strip()
        if split not in splits or key in ids:
            raise ValueError(f"Row {number}: invalid split or duplicate ID")
        if group in groups and groups[group] != split:
            raise ValueError(f"Row {number}: source group leaks across splits")
        messages = row.get("messages")
        if messages is None:
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("input", "output")):
                raise ValueError(f"Row {number}: authentic input/output pair required")
            messages = [{"role": "user", "content": row["input"]},
                        {"role": "assistant", "content": row["output"]}]
        if not isinstance(messages, list) or len(messages) not in (2, 3):
            raise ValueError(f"Row {number}: use user/assistant, optionally preceded by system")
        roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
        if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
            raise ValueError(f"Row {number}: invalid role order")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip()
               or "\x00" in m["content"] for m in messages):
            raise ValueError(f"Row {number}: nonempty text required")
        # Compare task/response, excluding a potentially shared system instruction.
        text = json.dumps([unicodedata.normalize("NFKC", " ".join(m["content"].split())).casefold()
                           for m in messages[-2:]], ensure_ascii=False)
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        if fingerprint in fingerprints:
            raise ValueError(f"Row {number}: duplicate task-response pair; review before splitting")
        ids.add(key)
        groups[group] = split
        fingerprints[fingerprint] = split
        splits[split].append(dict(id=key, source_group=group, messages=[
            dict(role=m["role"], content=m["content"]) for m in messages]))
    if not 50 <= len(splits["train"]) <= 500 or not splits["dev"] or not splits["test"]:
        raise ValueError("Need 50–500 train pairs plus separate nonempty dev and test")
    return splits


def prepare(source, destination, *, authorized, near_duplicates_reviewed):
    if not authorized or not near_duplicates_reviewed:
        raise ValueError("Confirm real-data authorization and manual cross-source near-duplicate review")
    source, destination = Path(source), Path(destination)
    splits = validate_pairs(read_rows(source))
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, rows in splits.items():
        path = destination/f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        files[path.name] = sha(path)
    manifest = dict(schema_version=1, purpose="personal_adapter_example_v1", source_sha256=sha(source),
                    counts={name: len(rows) for name, rows in splits.items()}, files_sha256=files,
                    real_authorization_declared=True, manual_near_duplicate_review_declared=True,
                    checks="unique IDs/pairs; source groups disjoint; explicit labels; no synthesis",
                    limitations="Declarations do not verify authorship or exhaustive semantic deduplication")
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def verify_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("purpose") != "personal_adapter_example_v1":
        raise ValueError("Unknown personal dataset manifest")
    expected = {f"{split}.jsonl" for split in ("train", "dev", "test")}
    if set(manifest["files_sha256"]) != expected:
        raise ValueError("Unexpected manifest file inventory")
    all_rows = []
    for split in ("train", "dev", "test"):
        path = directory/f"{split}.jsonl"
        if sha(path) != manifest["files_sha256"][path.name]:
            raise ValueError(f"Changed prepared data: {split}")
        rows = read_rows(path)
        if len(rows) != manifest["counts"][split]:
            raise ValueError("Prepared record count differs from manifest")
        all_rows.extend(dict(row, split=split) for row in rows)
    if not manifest.get("real_authorization_declared"):
        raise ValueError("Missing provenance authorization declaration")
    splits = validate_pairs(all_rows)
    if not manifest.get("manual_near_duplicate_review_declared"):
        from personal_llm_data.text_safety import verify_automated_audit
        recorded = manifest.get("automated_near_duplicate_review", {})
        if verify_automated_audit(splits, recorded) != recorded:
            raise ValueError("Automated near-duplicate audit receipt differs from recomputation")
    return splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confirm-real-authorized", action="store_true")
    parser.add_argument("--confirm-near-duplicate-review", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.input, args.output_dir, authorized=args.confirm_real_authorized,
                     near_duplicates_reviewed=args.confirm_near_duplicate_review)
    print(json.dumps(result, indent=2))
    return 0
