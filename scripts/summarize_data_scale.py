from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SIZES = (50, 100, 200, 500, 2000, 10437)
ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    # Iterate physical lines: str.splitlines() also splits Unicode separators
    # which can legally appear inside a JSON string in the real corpus.
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def predictions(path: Path):
    rows = read_jsonl(path)
    indexed = {str(row["id"]): row for row in rows}
    if len(rows) != 1496 or len(indexed) != 1496:
        raise ValueError(f"Expected 1496 unique test records: {path}")
    return indexed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize the complete controlled data-scale experiment")
    parser.add_argument("--output", default="experiments/data_scale_summary.json")
    args = parser.parse_args()
    runs = ROOT / "experiments/runs"
    full_run = runs / "qwen3-4b-lamp7-full-lora-r16"
    baseline = predictions(full_run / "baseline_test_metrics.predictions.jsonl")
    full_config = read_json(full_run / "training_summary.json")["config"]
    results = []
    previous_ids: set[str] = set()
    for size in SIZES:
        run = full_run if size == 10437 else runs / f"qwen3-4b-lamp7-n{size}-lora-r16-a32"
        summary = read_json(run / "training_summary.json")
        config = summary["config"]
        shared = {key: value for key, value in config.items() if key not in ("train_file", "output_dir")}
        reference_config = {key: value for key, value in full_config.items() if key not in ("train_file", "output_dir")}
        if shared != reference_config or summary["train_records"] != size or summary["eval_records"] != 500:
            raise ValueError(f"Controlled configuration or record count mismatch: {run}")
        train_rows = read_jsonl(ROOT / config["train_file"])
        train_ids = {str(row["id"]) for row in train_rows}
        if len(train_rows) != size or len(train_ids) != size or not previous_ids.issubset(train_ids):
            raise ValueError(f"Invalid nested training subset: {run}")
        previous_ids = train_ids
        evals = [row for row in summary["log_history"] if "eval_loss" in row]
        if len(evals) != 2 or [row["epoch"] for row in evals] != [1.0, 2.0]:
            raise ValueError(f"Expected two epoch evaluations: {run}")
        best = min(evals, key=lambda row: row["eval_loss"])
        final_hash = sha256(run / "adapter/adapter_model.safetensors")
        best_hash = sha256(run / f"checkpoints/checkpoint-{best['step']}/adapter_model.safetensors")
        if final_hash != best_hash:
            raise ValueError(f"Final adapter differs from best checkpoint: {run}")
        conditions = {}
        for condition in ("profile", "no_profile"):
            stem = run / f"adapter_{condition}_test_metrics"
            rows = predictions(stem.with_suffix(".predictions.jsonl"))
            if rows.keys() != baseline.keys() or any(rows[key]["reference"] != baseline[key]["reference"] for key in baseline):
                raise ValueError(f"Test alignment mismatch: {stem}")
            metrics = read_json(stem.with_suffix(".json"))
            if metrics["samples"] != 1496:
                raise ValueError(f"Metric sample count mismatch: {stem}")
            conditions[condition] = metrics
        semantic = read_json(run / "bertscore.json")
        if semantic["samples"] != 1496 or semantic["model"] != "roberta-large" or not semantic["idf"] or not semantic["rescale_with_baseline"]:
            raise ValueError(f"BERTScore protocol mismatch: {run}")
        paired = None
        if size != 10437:
            paired = read_json(run / f"paired_n{size}_vs_full_profile_bootstrap.json")
            if paired["samples"] != 1496 or paired["bootstrap_resamples"] != 2000:
                raise ValueError(f"Incomplete bootstrap: {run}")
        results.append({"train_records": size, "run": str(run.relative_to(ROOT)),
                        "optimizer_steps": max(row.get("step", 0) for row in summary["log_history"]),
                        "dev_losses": [row["eval_loss"] for row in evals], "best_epoch": best["epoch"],
                        "elapsed_seconds": summary["elapsed_seconds"], "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
                        "adapter_sha256": final_hash, "conditions": conditions,
                        "bertscore": semantic, "paired_vs_full_profile": paired})
    report = {"training_seed": 20251206, "test_samples": 1496, "eval_records": 500,
              "design": "fixed_two_epochs_variable_optimizer_steps", "results": results}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audited all {len(results)} data scales; wrote {output}")


if __name__ == "__main__":
    main()
