"""Opt-in full-test developer evaluation, reusing the unchanged evaluator."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from personal_llm_data.personal_data import read_rows, sha
from personal_llm_data.personal_runtime import gpu_guard, write_json
from train_personal_adapter import ROOT, inputs_snapshot, validate_config, verify_checkpoint


def verify_predictions(records, predictions):
    expected = [(r["id"], r["messages"][-1]["content"]) for r in records]
    actual = [(r.get("id"), r.get("reference")) for r in predictions]
    if actual != expected or any(not isinstance(r.get("prediction"), str) for r in predictions):
        raise ValueError("Full ordered test IDs/references/predictions must match prepared data")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/personal_adapter_v1.json", type=Path)
    parser.add_argument("--condition", choices=("base", "adapter"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = validate_config(config)
    run = ROOT/config["output_dir"]
    output = run/"evaluation"/f"{args.condition}_metrics.json"
    prediction = output.with_suffix(".predictions.jsonl")
    receipt = output.with_suffix(".receipt.json")
    if any(p.exists() for p in (output, prediction, receipt)):
        raise FileExistsError("Evaluation evidence already exists; inspect it without overwriting")
    if not args.execute:
        print(json.dumps(dict(executable=False, test_records=len(rows["test"]), condition=args.condition,
                             next="--execute verifies training/base inputs and runs the complete test")))
        return 0
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0")
    with gpu_guard(ROOT):
        summary = json.loads((run/"training_summary.json").read_text(encoding="utf-8"))
        inputs = inputs_snapshot(config, args.config)
        if summary.get("status") != "completed_verified" or summary["inputs"] != inputs:
            raise ValueError("Training incomplete or inputs/config/code/model changed")
        verify_checkpoint(run/"checkpoints"/f"checkpoint-{config['max_steps']}", inputs)
        verify_checkpoint(Path(summary["best_checkpoint"]), inputs)
        adapter_file = run/"adapter/adapter_model.safetensors"
        if sha(adapter_file) != summary["adapter_sha256"]:
            raise ValueError("Exported adapter checksum mismatch")
        command = [sys.executable, "-u", str(ROOT/"scripts/evaluate_generation.py"),
            "--model", str(ROOT/config["model_name_or_path"]), "--data", str(ROOT/config["data_dir"]/"test.jsonl"),
            "--output", str(output), "--max-samples", str(len(rows["test"])), "--batch-size", "1",
            "--max-input-tokens", str(config["max_length"]), "--max-new-tokens", str(config["max_new_tokens"]),
            "--quantization", config["quantization"]]
        if args.condition == "adapter":
            command.extend(["--adapter", str(run/"adapter")])
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True)
        verify_predictions(rows["test"], read_rows(prediction))
        metrics = json.loads(output.read_text(encoding="utf-8"))
        if metrics["samples"] != len(rows["test"]) or inputs_snapshot(config, args.config) != inputs:
            raise ValueError("Incomplete test or inputs changed during evaluation")
        write_json(receipt, dict(condition=args.condition, samples=len(rows["test"]), command=command,
            inputs=inputs, adapter_sha256=summary["adapter_sha256"] if args.condition == "adapter" else None,
            evaluator_sha256=sha(ROOT/"scripts/evaluate_generation.py"),
            outputs_sha256={p.name: sha(p) for p in (output, prediction)},
            limitations="Reference similarity and surface proxies; not factual correctness, personal-style proof or a bootstrap result"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
