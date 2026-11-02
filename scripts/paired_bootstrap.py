from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF

from personal_llm_data.pipeline import read_jsonl


def load_predictions(path: str) -> dict[str, dict[str, str]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def summarize(samples: list[float], observed: float) -> dict[str, float]:
    return {
        "observed": observed,
        "ci95_low": percentile(samples, 2.5),
        "ci95_high": percentile(samples, 97.5),
        "bootstrap_probability_gt_zero": sum(value > 0 for value in samples) / len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap for the 2x2 personalization experiment")
    parser.add_argument("--base-no-profile", required=True)
    parser.add_argument("--base-profile", required=True)
    parser.add_argument("--adapter-no-profile", required=True)
    parser.add_argument("--adapter-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20251206)
    args = parser.parse_args()

    conditions = {
        "base_no_profile": load_predictions(args.base_no_profile),
        "base_profile": load_predictions(args.base_profile),
        "adapter_no_profile": load_predictions(args.adapter_no_profile),
        "adapter_profile": load_predictions(args.adapter_profile),
    }
    ids = sorted(conditions["base_profile"])
    for name, rows in conditions.items():
        if sorted(rows) != ids:
            raise ValueError(f"{name}: prediction ids do not match")
    references = [conditions["base_profile"][item]["reference"] for item in ids]
    predictions = {
        name: [rows[item]["prediction"] for item in ids] for name, rows in conditions.items()
    }
    for name, rows in conditions.items():
        if [rows[item]["reference"] for item in ids] != references:
            raise ValueError(f"{name}: references do not match")

    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_values = {
        name: np.asarray(
            [rouge.score(reference, hypothesis)["rougeL"].fmeasure for reference, hypothesis in zip(references, hypotheses)]
        )
        for name, hypotheses in predictions.items()
    }

    def corpus_scores(indices: np.ndarray) -> dict[str, dict[str, float]]:
        sampled_references = [references[int(index)] for index in indices]
        result: dict[str, dict[str, float]] = {}
        for name, hypotheses in predictions.items():
            sampled_hypotheses = [hypotheses[int(index)] for index in indices]
            result[name] = {
                "bleu": bleu.corpus_score(sampled_hypotheses, [sampled_references]).score,
                "chrf": chrf.corpus_score(sampled_hypotheses, [sampled_references]).score,
                "rougeL": float(rouge_values[name][indices].mean()),
            }
        return result

    full_indices = np.arange(len(ids))
    observed_scores = corpus_scores(full_indices)
    contrast_names: dict[str, Callable[[dict[str, float]], float]] = {
        "lora_effect_with_profile": lambda scores: scores["adapter_profile"] - scores["base_profile"],
        "lora_effect_without_profile": lambda scores: scores["adapter_no_profile"] - scores["base_no_profile"],
        "profile_effect_on_adapter": lambda scores: scores["adapter_profile"] - scores["adapter_no_profile"],
        "profile_effect_on_base": lambda scores: scores["base_profile"] - scores["base_no_profile"],
        "lora_profile_interaction": lambda scores: (scores["adapter_profile"] - scores["adapter_no_profile"])
        - (scores["base_profile"] - scores["base_no_profile"]),
    }
    observed_contrasts = {
        metric: {
            name: function({condition: observed_scores[condition][metric] for condition in conditions})
            for name, function in contrast_names.items()
        }
        for metric in ("bleu", "chrf", "rougeL")
    }
    bootstrap_values = {
        metric: {name: [] for name in contrast_names} for metric in ("bleu", "chrf", "rougeL")
    }
    rng = np.random.default_rng(args.seed)
    for iteration in range(args.samples):
        indices = rng.integers(0, len(ids), size=len(ids))
        scores = corpus_scores(indices)
        for metric in bootstrap_values:
            metric_scores = {condition: scores[condition][metric] for condition in conditions}
            for name, function in contrast_names.items():
                bootstrap_values[metric][name].append(function(metric_scores))
        if (iteration + 1) % 100 == 0:
            print(f"bootstrap {iteration + 1}/{args.samples}")

    report = {
        "samples": len(ids),
        "bootstrap_resamples": args.samples,
        "seed": args.seed,
        "condition_scores": observed_scores,
        "contrasts": {
            metric: {
                name: summarize(bootstrap_values[metric][name], observed_contrasts[metric][name])
                for name in contrast_names
            }
            for metric in bootstrap_values
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
