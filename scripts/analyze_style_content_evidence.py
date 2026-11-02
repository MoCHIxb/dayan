"""Descriptive analysis of ALL existing official test predictions; no new scores invented."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from personal_llm_data.pipeline import read_jsonl, write_jsonl


def tokens(text):
    return re.findall(r"<\w+>|\w+(?:'\w+)?|[^\w\s]", text.lower())


def features(text):
    letters = [c for c in text if c.isalpha()]
    n = max(len(text), 1)
    return {"characters": len(text), "uppercase_rate": sum(c.isupper() for c in letters) / max(len(letters), 1),
            "exclamation_rate": text.count("!") / n, "question_rate": text.count("?") / n,
            "ellipsis_rate": len(re.findall(r"\.{2,}", text)) / n}


def ngrams(text, n=4):
    ts = tokens(text)
    return {tuple(ts[i:i+n]) for i in range(len(ts)-n+1)}


def summarize(rows):
    keys = [k for k in rows[0] if k != "id"]
    return {"samples": len(rows), **{k: mean(r[k] for r in rows) for k in keys}}


def main():
    data_path = ROOT / "data/processed/lamp7_bm25_k12/test.jsonl"
    data = {str(r["id"]): r for r in read_jsonl(data_path)}
    train = {r["messages"][-1]["content"] for r in read_jsonl(ROOT / "data/processed/lamp7_bm25_k12/train.jsonl")}
    assert len(data) == 1496
    conditions, skipped = [], []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    paths = sorted((ROOT / "experiments/runs").glob("*/**/*test_metrics.predictions.jsonl"))
    for path in paths:
        predictions = read_jsonl(path)
        ids = [str(r["id"]) for r in predictions]
        if len(ids) != 1496 or len(set(ids)) != 1496 or set(ids) != set(data):
            skipped.append(str(path.relative_to(ROOT)))
            continue  # pilots are outside this complete 4B test analysis
        rows, overlapping, other = [], [], []
        texts_by_group = {"all": [], "reference_in_selected_history": [], "reference_not_in_selected_history": []}
        for p in predictions:
            key = str(p["id"])
            record = data[key]
            ref, hyp = p["reference"], p["prediction"]
            assert ref == record["messages"][-1]["content"]
            history = [h["text"] for h in record["profile"]]
            inp = record["messages"][-2]["content"].split("\n\nTask:\n", 1)[1]
            # Existing LaMP instruction prefix, same first-colon rule as retrieval.
            inp = inp.partition(":")[2].strip() if ":" in inp else inp
            hf = {k: mean(features(t)[k] for t in history) for k in features(hyp)}
            pf, rf = features(hyp), features(ref)
            hg = set().union(*(ngrams(t) for t in history))
            pg = ngrams(hyp)
            nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", inp))
            outnums = set(re.findall(r"\b\d+(?:\.\d+)?\b", hyp))
            r = {"id": key, "exact_reference": hyp == ref, "exact_input": hyp == inp,
                 "exact_selected_history": hyp in history, "exact_any_train_target": hyp in train,
                 "empty": not hyp, "output_characters": len(hyp),
                 "history_4gram_coverage": len(pg & hg) / len(pg) if pg else 0.,
                 "has_output_4grams": bool(pg),
                 "input_has_numbers": bool(nums), "number_set_changed": nums != outnums}
            for k in pf:
                r[f"history_mae_{k}"] = abs(pf[k] - hf[k])
                r[f"reference_mae_{k}"] = abs(pf[k] - rf[k])
            rows.append(r)
            (overlapping if ref in history else other).append(r)
            texts_by_group["all"].append((hyp, ref))
            texts_by_group["reference_in_selected_history" if ref in history else "reference_not_in_selected_history"].append((hyp, ref))
        result = {"predictions": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "all": summarize(rows),
                  "reference_in_selected_history": summarize(overlapping) if overlapping else None,
                  "reference_not_in_selected_history": summarize(other) if other else None}
        for group, pairs in texts_by_group.items():
            if not pairs:
                continue
            hyps, refs = zip(*pairs)
            result[group]["bleu"] = BLEU(effective_order=True).corpus_score(hyps, [refs]).score
            result[group]["chrf"] = CHRF().corpus_score(hyps, [refs]).score
            result[group]["rougeL_f"] = mean(rouge.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in pairs)
        conditions.append(result)
        item_path = ROOT / "experiments/refocused/per_item" / path.parent.name / path.name
        item_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, item_path)
        result["per_item"] = str(item_path.relative_to(ROOT))
    report = {"scope": "descriptive surface and copy proxies; not causal style proof, semantic scores, or human judgments",
        "history": "original BM25-k12 selected history; actually retained prompt history may differ after truncation",
        "number_warning": "set change is a flag, not factual error; omissions/additions and benign reformulations included",
        "inference_warning": "group metrics are descriptive sensitivity analysis without confidence intervals; full-test primary results unchanged",
        "fourgram_warning": "tokenized lowercase 4gram set coverage; outputs under 4 tokens have zero coverage",
        "test_samples": 1496, "skipped_incomplete_or_different_ids": skipped,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(), "conditions": conditions}
    out = ROOT / "experiments/refocused/style_content_evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Analyzed {len(conditions)} full-test prediction files; wrote {out}")
    for c in conditions:
        print(c["predictions"], json.dumps({k: c["all"][k] for k in ("exact_selected_history", "history_4gram_coverage", "exact_reference")}))


if __name__ == "__main__":
    main()
