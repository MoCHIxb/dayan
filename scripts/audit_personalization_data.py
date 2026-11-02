"""Audit existing LaMP files without generating examples or changing splits."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from personal_llm_data.lamp7 import sanitize_text
from personal_llm_data.pipeline import read_jsonl


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {"scope": "exact matches after existing sanitization; overlap is not automatically user identity or leakage",
              "splits": {}, "cross_split": {}, "subsets": {}}
    assets = {}
    for split in ("train", "dev", "test"):
        folder = ROOT / "data/raw/lamp7" / split
        qp, op = folder / f"{split}_questions.json", folder / f"{split}_outputs.json"
        questions = json.loads(qp.read_text(encoding="utf-8"))
        golds = json.loads(op.read_text(encoding="utf-8"))["golds"]
        outputs = {str(r["id"]): sanitize_text(r["output"]) for r in golds}
        raw_outputs = {str(r["id"]): r["output"] for r in golds}
        ids = [str(q["id"]) for q in questions]
        assert len(ids) == len(set(ids)) == len(outputs) == len(golds)
        assert set(ids) == set(outputs)
        processed_path = ROOT / f"data/processed/lamp7_bm25_k12/{split}.jsonl"
        processed = {str(r["id"]): r for r in read_jsonl(processed_path)}
        assert set(processed) == set(ids)
        histories, history_ids, signatures, inputs = set(), set(), [], []
        counts = Counter()
        sizes = []
        for q in questions:
            key = str(q["id"])
            row = processed[key]
            texts = [sanitize_text(p["text"]) for p in q["profile"]]
            pids = [str(p["id"]) for p in q["profile"]]
            selected = [p["text"] for p in row["profile"]]
            assert row["messages"][-1]["content"] == outputs[key]
            assert all(t in texts for t in selected)
            counts["target_in_own_full_history"] += outputs[key] in texts
            counts["raw_target_in_own_full_history"] += raw_outputs[key] in [p["text"] for p in q["profile"]]
            counts["target_in_own_selected_history"] += outputs[key] in selected
            counts["duplicate_history_text_records"] += len(texts) != len(set(texts))
            counts["duplicate_history_id_records"] += len(pids) != len(set(pids))
            histories.update(texts)
            history_ids.update(pids)
            signatures.append(tuple(sorted(pids)))
            inputs.append(sanitize_text(q["input"]))
            sizes.append(len(texts))
        assets[split] = {"ids": set(ids), "targets": set(outputs.values()), "histories": histories,
                         "history_ids": history_ids, "signatures": set(signatures), "inputs": set(inputs)}
        report["splits"][split] = {"records": len(ids), "question_fields": sorted(set().union(*(q.keys() for q in questions))),
            "unique_profile_id_set_signatures": len(set(signatures)),
            "explicit_user_id_present": any("user_id" in q for q in questions),
            "history_min": min(sizes), "history_max": max(sizes),
            "duplicate_target_excess": len(outputs) - len(set(outputs.values())),
            **dict(counts), "sha256": {str(p.relative_to(ROOT)): digest(p) for p in (qp, op, processed_path)}}
    for a, b in combinations(assets, 2):
        x, y = assets[a], assets[b]
        report["cross_split"][f"{a}__{b}"] = {
            "shared_question_ids": len(x["ids"] & y["ids"]),
            "shared_target_texts": len(x["targets"] & y["targets"]),
            "shared_input_texts": len(x["inputs"] & y["inputs"]),
            "shared_history_ids": len(x["history_ids"] & y["history_ids"]),
            "shared_history_texts": len(x["histories"] & y["histories"]),
            "shared_profile_id_sets": len(x["signatures"] & y["signatures"]),
            f"{a}_targets_in_{b}_histories": len(x["targets"] & y["histories"]),
            f"{b}_targets_in_{a}_histories": len(y["targets"] & x["histories"])}
    previous = set()
    for n in (50, 100, 200, 500):
        path = ROOT / f"data/processed/lamp7_bm25_k12/train_{n}.jsonl"
        rows = read_jsonl(path)
        ids = {str(r["id"]) for r in rows}
        assert len(rows) == len(ids) == n and previous <= ids <= assets["train"]["ids"]
        previous = ids
        report["subsets"][str(n)] = {"records": n, "sha256": digest(path),
            "targets_also_in_dev": sum(r["messages"][-1]["content"] in assets["dev"]["targets"] for r in rows),
            "targets_also_in_test": sum(r["messages"][-1]["content"] in assets["test"]["targets"] for r in rows)}
    out = ROOT / "experiments/refocused/data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
