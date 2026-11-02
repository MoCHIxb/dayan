from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bert_score import BERTScorer

from personal_llm_data.pipeline import read_jsonl


def load(path: str) -> dict[str, dict[str, str]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "observed": float(values[0]),
        "ci95_low": float(np.percentile(values[1:], 2.5)),
        "ci95_high": float(np.percentile(values[1:], 97.5)),
        "bootstrap_probability_gt_zero": float(np.mean(values[1:] > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BERTScore semantic evaluation for the 2x2 experiment")
    parser.add_argument("--base-no-profile", required=True)
    parser.add_argument("--base-profile", required=True)
    parser.add_argument("--adapter-no-profile", required=True)
    parser.add_argument("--adapter-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20251206)
    args = parser.parse_args()

    rows = {
        "base_no_profile": load(args.base_no_profile),
        "base_profile": load(args.base_profile),
        "adapter_no_profile": load(args.adapter_no_profile),
        "adapter_profile": load(args.adapter_profile),
    }
    ids = sorted(rows["base_profile"])
    references = [rows["base_profile"][item]["reference"] for item in ids]
    for name, condition in rows.items():
        if sorted(condition) != ids:
            raise ValueError(f"{name}: ids do not match")
        if [condition[item]["reference"] for item in ids] != references:
            raise ValueError(f"{name}: references do not match")

    scorer = BERTScorer(
        model_type=args.model,
        lang="en",
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        idf=True,
        idf_sents=references,
        rescale_with_baseline=True,
        use_fast_tokenizer=True,
    )
    per_item: dict[str, dict[str, np.ndarray]] = {}
    for name, condition in rows.items():
        candidates = [condition[item]["prediction"] for item in ids]
        precision, recall, f1 = scorer.score(candidates, references, batch_size=args.batch_size, verbose=True)
        per_item[name] = {
            "precision": precision.cpu().numpy(),
            "recall": recall.cpu().numpy(),
            "f1": f1.cpu().numpy(),
        }

    means = {
        name: {metric: float(values.mean()) for metric, values in condition.items()}
        for name, condition in per_item.items()
    }
    f1 = {name: condition["f1"] for name, condition in per_item.items()}
    contrasts = {
        "lora_effect_with_profile": f1["adapter_profile"] - f1["base_profile"],
        "lora_effect_without_profile": f1["adapter_no_profile"] - f1["base_no_profile"],
        "profile_effect_on_adapter": f1["adapter_profile"] - f1["adapter_no_profile"],
        "profile_effect_on_base": f1["base_profile"] - f1["base_no_profile"],
        "lora_profile_interaction": (f1["adapter_profile"] - f1["adapter_no_profile"])
        - (f1["base_profile"] - f1["base_no_profile"]),
    }
    rng = np.random.default_rng(args.seed)
    result_contrasts: dict[str, dict[str, float]] = {}
    for name, values in contrasts.items():
        bootstrap = np.empty(args.bootstrap + 1, dtype=np.float64)
        bootstrap[0] = values.mean()
        for index in range(args.bootstrap):
            sample = rng.integers(0, len(values), len(values))
            bootstrap[index + 1] = values[sample].mean()
        result_contrasts[name] = interval(bootstrap)

    report = {
        "samples": len(ids),
        "model": args.model,
        "idf": True,
        "rescale_with_baseline": True,
        "condition_scores": means,
        "f1_contrasts": result_contrasts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
