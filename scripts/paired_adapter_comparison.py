from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF

from personal_llm_data.pipeline import read_jsonl


def load_predictions(path: str) -> dict[str, dict[str, str]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Original corpus-level paired bootstrap for two adapter systems"
    )
    parser.add_argument("--reference-system", required=True)
    parser.add_argument("--candidate-system", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20251206)
    args = parser.parse_args()

    systems = {
        args.reference_name: load_predictions(args.reference_system),
        args.candidate_name: load_predictions(args.candidate_system),
    }
    if args.reference_name == args.candidate_name:
        raise ValueError("System names must differ")
    ids = sorted(systems[args.reference_name])
    if sorted(systems[args.candidate_name]) != ids:
        raise ValueError("Prediction IDs do not match")

    references = [systems[args.reference_name][item]["reference"] for item in ids]
    predictions: dict[str, list[str]] = {}
    for name, rows in systems.items():
        if [rows[item]["reference"] for item in ids] != references:
            raise ValueError(f"{name}: references do not match")
        predictions[name] = [rows[item]["prediction"] for item in ids]

    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_values = {
        name: np.asarray(
            [
                rouge.score(reference, hypothesis)["rougeL"].fmeasure
                for reference, hypothesis in zip(references, hypotheses)
            ]
        )
        for name, hypotheses in predictions.items()
    }

    def scores(indices: np.ndarray) -> dict[str, dict[str, float]]:
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

    observed_scores = scores(np.arange(len(ids)))
    observed_delta = {
        metric: observed_scores[args.candidate_name][metric]
        - observed_scores[args.reference_name][metric]
        for metric in ("bleu", "chrf", "rougeL")
    }
    bootstrap_values = {metric: [] for metric in observed_delta}
    rng = np.random.default_rng(args.seed)
    for iteration in range(args.samples):
        sampled = scores(rng.integers(0, len(ids), size=len(ids)))
        for metric in bootstrap_values:
            bootstrap_values[metric].append(
                sampled[args.candidate_name][metric] - sampled[args.reference_name][metric]
            )
        if (iteration + 1) % 100 == 0:
            print(f"bootstrap {iteration + 1}/{args.samples}", flush=True)

    report = {
        "samples": len(ids),
        "bootstrap_resamples": args.samples,
        "seed": args.seed,
        "difference": f"{args.candidate_name} - {args.reference_name}",
        "condition_scores": observed_scores,
        "contrasts": {
            metric: {
                "observed": observed_delta[metric],
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_probability_gt_zero": float(np.mean(np.asarray(values) > 0)),
            }
            for metric, values in bootstrap_values.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
