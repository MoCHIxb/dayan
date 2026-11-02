from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

from personal_llm_data.pipeline import read_jsonl


CONDITION_ARGS = (
    "base_no_profile",
    "base_profile",
    "adapter_no_profile",
    "adapter_profile",
)


def load_index(path: str) -> dict[str, dict[str, str]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def summarize_difference(values: np.ndarray) -> dict[str, float | int]:
    tolerance = 1e-12
    return {
        "mean_delta_rougeL": float(values.mean()),
        "median_delta_rougeL": float(np.median(values)),
        "wins": int(np.sum(values > tolerance)),
        "ties": int(np.sum(np.abs(values) <= tolerance)),
        "losses": int(np.sum(values < -tolerance)),
        "win_rate_excluding_ties": float(np.sum(values > tolerance) / max(np.sum(np.abs(values) > tolerance), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified full-test error analysis for the 2x2 experiment")
    parser.add_argument("--data", required=True)
    for name in CONDITION_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    records = {str(row["id"]): row for row in read_jsonl(args.data)}
    conditions = {name: load_index(getattr(args, name)) for name in CONDITION_ARGS}
    ids = sorted(records)
    for name, rows in conditions.items():
        if sorted(rows) != ids:
            raise ValueError(f"{name}: prediction IDs do not match processed test data")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    per_item: dict[str, np.ndarray] = {}
    for name, rows in conditions.items():
        per_item[name] = np.asarray(
            [scorer.score(rows[item]["reference"], rows[item]["prediction"])["rougeL"].fmeasure for item in ids]
        )

    contrasts = {
        "lora_effect_with_profile": per_item["adapter_profile"] - per_item["base_profile"],
        "lora_effect_without_profile": per_item["adapter_no_profile"] - per_item["base_no_profile"],
        "profile_effect_on_adapter": per_item["adapter_profile"] - per_item["adapter_no_profile"],
        "profile_effect_on_base": per_item["base_profile"] - per_item["base_no_profile"],
    }

    reference_lengths = np.asarray([len(conditions["base_profile"][item]["reference"].split()) for item in ids])
    length_edges = np.quantile(reference_lengths, [0.25, 0.5, 0.75])
    length_groups = np.digitize(reference_lengths, length_edges, right=True)
    length_labels = ["Q1_shortest", "Q2", "Q3", "Q4_longest"]

    profile_sizes = np.asarray([int(records[item]["profile_size_original"]) for item in ids])
    profile_groups = np.select(
        [profile_sizes <= 10, profile_sizes <= 20, profile_sizes <= 50],
        [0, 1, 2],
        default=3,
    )
    profile_labels = ["1-10", "11-20", "21-50", "51+"]

    def stratify(groups: np.ndarray, labels: list[str]) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for group_id, label in enumerate(labels):
            mask = groups == group_id
            result[label] = {
                "samples": int(mask.sum()),
                "base_profile_rougeL": float(per_item["base_profile"][mask].mean()),
                "adapter_profile_rougeL": float(per_item["adapter_profile"][mask].mean()),
                "lora_delta_with_profile": float(contrasts["lora_effect_with_profile"][mask].mean()),
                "profile_delta_on_adapter": float(contrasts["profile_effect_on_adapter"][mask].mean()),
            }
        return result

    primary = contrasts["lora_effect_with_profile"]
    order = np.argsort(primary)

    def examples(indices: np.ndarray) -> list[dict[str, float | int | str]]:
        return [
            {
                "id": ids[int(index)],
                "delta_rougeL": float(primary[int(index)]),
                "reference_words": int(reference_lengths[int(index)]),
                "profile_size_original": int(profile_sizes[int(index)]),
            }
            for index in indices
        ]

    result = {
        "samples": len(ids),
        "metric": "per-example ROUGE-L F1",
        "pairwise": {name: summarize_difference(values) for name, values in contrasts.items()},
        "reference_length_quartile_edges_words": [float(value) for value in length_edges],
        "by_reference_length": stratify(length_groups, length_labels),
        "by_original_profile_size": stratify(profile_groups, profile_labels),
        "largest_lora_regressions_with_profile": examples(order[: args.examples]),
        "largest_lora_improvements_with_profile": examples(order[-args.examples :][::-1]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
