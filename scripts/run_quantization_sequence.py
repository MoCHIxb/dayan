"""One-shot sequential experiment batch; stops on failure and never restarts training.

Attach to an already verified Windows training PID, or resume evaluation of a
completed training run. Validate saved results, evaluate int8, then train/evaluate
NF4. This is not a scheduler.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/runs"
BF16 = RUNS / "qwen3-4b-lamp7-full-lora-r16"
DATA = ROOT / "data/processed/lamp7_bm25_k12/test.jsonl"
STATE = ROOT / "experiments/quantization_sequence_state.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def status(stage, **details):
    event = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage, **details}
    write_json(STATE, event)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def wait_for_training(pid):
    if os.name != "nt":
        raise RuntimeError("Attaching to an existing PID currently requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        status("waiting_for_existing_int8_training", training_pid=pid)
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def execute(script, arguments, log):
    status("running", script=script, arguments=[str(item) for item in arguments], log=str(log))
    # Exclusive creation preserves prior logs/results on accidental re-entry.
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script),
                        *map(str, arguments)], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_training(run, mode):
    summary = read_json(run / "training_summary.json")
    config = summary["config"]
    baseline = read_json(BF16 / "training_summary.json")["config"]
    omit = {"output_dir", "quantization"}
    if ({k: v for k, v in config.items() if k not in omit}
            != {k: v for k, v in baseline.items() if k not in omit}):
        raise ValueError("Quantization run changed a controlled training factor")
    if (summary["train_records"], summary["eval_records"]) != (10437, 500):
        raise ValueError("Training/dev record counts mismatch")
    if config["quantization"] != mode or summary["quantization"] != mode:
        raise ValueError("Wrong quantization mode")
    model_config = read_json(ROOT / config["model_name_or_path"] / "config.json")
    expected_modules = model_config["num_hidden_layers"] * len(config["target_modules"])
    if summary["quantized_linear_modules"] != expected_modules:
        raise ValueError("Actual quantized projection count mismatch")
    evals = [row for row in summary["log_history"] if "eval_loss" in row]
    if len(evals) != 2 or [row["epoch"] for row in evals] != [1.0, 2.0]:
        raise ValueError("Expected complete two-epoch evaluation")
    state = read_json(run / "checkpoints/trainer_state.json")
    if state["global_step"] != 1306:
        raise ValueError("Incomplete optimizer steps")
    best = min(evals, key=lambda row: row["eval_loss"])
    checkpoint = run / f"checkpoints/checkpoint-{best['step']}"
    if Path(state["best_model_checkpoint"]).name != checkpoint.name:
        raise ValueError("Trainer best checkpoint disagrees with minimum dev loss")
    final_hash = sha256(run / "adapter/adapter_model.safetensors")
    if final_hash != sha256(checkpoint / "adapter_model.safetensors"):
        raise ValueError("Final adapter differs from best checkpoint")
    audit = {"mode": mode, "train_records": 10437, "eval_records": 500,
             "optimizer_steps": state["global_step"],
             "dev_losses": [row["eval_loss"] for row in evals],
             "best_checkpoint": str(checkpoint), "best_epoch": best["epoch"],
             "adapter_sha256": final_hash,
             "quantized_linear_modules": summary["quantized_linear_modules"],
             "elapsed_seconds": summary["elapsed_seconds"],
             "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"]}
    write_json(run / "training_audit.json", audit)
    return audit


def validate_predictions(path):
    with DATA.open(encoding="utf-8") as stream:
        expected = {str(row["id"]): row["messages"][-1]["content"]
                    for line in stream if line.strip() for row in [json.loads(line)]}
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 1496 or len({str(row["id"]) for row in rows}) != 1496:
        raise ValueError("Expected 1496 unique test predictions")
    if {str(row["id"]): row["reference"] for row in rows} != expected:
        raise ValueError("Frozen test IDs/references mismatch")


def evaluate_condition(run, mode, name, resume=False):
    output = run / f"{name}_test_metrics.json"
    prediction = output.with_suffix(".predictions.jsonl")
    if output.exists() != prediction.exists():
        raise FileExistsError(f"Incomplete saved evaluation; inspect without overwriting: {output}")
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    if not output.exists():
        arguments = ["--model", "models/Qwen3-4B", "--data", DATA, "--output", output,
                     "--max-samples", 1496, "--batch-size", 4,
                     "--max-input-tokens", 768, "--max-new-tokens", 80,
                     "--quantization", mode]
        if name.startswith("adapter"):
            arguments += ["--adapter", run / "adapter"]
        if name.endswith("no_profile"):
            arguments += ["--drop-profile"]
        log = run / f"{name}.eval.log"
        if resume and log.exists():
            log = run / f"{name}.eval.resume-{time.time_ns()}.log"
        execute("evaluate_generation.py", arguments, log)
    validate_predictions(prediction)
    metric = read_json(output)
    expected_adapter = run / "adapter" if name.startswith("adapter") else None
    actual_adapter = metric["adapter"]
    adapter_matches = ((actual_adapter is None and expected_adapter is None)
                       or (actual_adapter is not None and expected_adapter is not None
                           and Path(actual_adapter).resolve() == expected_adapter.resolve()))
    if (metric["samples"] != 1496 or metric["quantization"] != mode
            or metric["do_sample"] or metric["drop_profile"] != name.endswith("no_profile")
            or metric["batch_size"] != 4 or metric["max_input_tokens"] != 768
            or metric["max_new_tokens"] != 80 or not adapter_matches
            or Path(metric["model"]).resolve() != (ROOT / "models/Qwen3-4B").resolve()):
        raise ValueError("Generation protocol mismatch")
    return prediction, metric


def evaluate(run, mode, resume=False):
    audit = audit_training(run, mode)
    paths = {}
    scores = {}
    for name in ("base_no_profile", "base_profile", "adapter_no_profile", "adapter_profile"):
        prediction, metric = evaluate_condition(run, mode, name, resume=resume)
        paths[name] = prediction
        scores[name] = metric
    shared_args = []
    for name, path in paths.items():
        shared_args += ["--" + name.replace("_", "-"), path]
    execute("bertscore_eval.py", [*shared_args, "--output", run / "bertscore.json",
            "--model", "roberta-large", "--batch-size", 32,
            "--bootstrap", 2000, "--seed", 20251206], run / "bertscore.log")
    semantic = read_json(run / "bertscore.json")
    if (semantic["samples"] != 1496 or semantic["model"] != "roberta-large"
            or not semantic["idf"] or not semantic["rescale_with_baseline"]):
        raise ValueError("BERTScore protocol mismatch")
    execute("analyze_predictions.py", [*shared_args, "--data", DATA,
            "--output", run / "error_analysis.json"], run / "error_analysis.log")
    paired_path = run / f"paired_{mode}_vs_bf16_profile_bootstrap.json"
    execute("paired_adapter_comparison.py", ["--reference-system",
            BF16 / "adapter_profile_test_metrics.predictions.jsonl",
            "--candidate-system", paths["adapter_profile"], "--reference-name", "bf16",
            "--candidate-name", mode, "--output", paired_path,
            "--samples", 2000, "--seed", 20251206], run / "paired_bootstrap.log")
    paired = read_json(paired_path)
    if paired["samples"] != 1496 or paired["bootstrap_resamples"] != 2000:
        raise ValueError("Bootstrap incomplete")
    result = {"training": audit, "conditions": scores, "bertscore": semantic,
              "paired_vs_bf16_profile": paired}
    write_json(run / "quantization_result.json", result)
    text = (f"\n\n## {mode} 全量对照完成（{time.strftime('%Y-%m-%d')}）\n\n"
            f"训练与评价均已完成核验。训练 {audit['train_records']} 条，两轮 dev loss 为 "
            f"{audit['dev_losses']}，最佳 epoch={audit['best_epoch']}，"
            f"峰值分配显存 {audit['peak_gpu_memory_bytes']} bytes，训练耗时 "
            f"{audit['elapsed_seconds']:.2f} 秒。量化 projection 数 "
            f"{audit['quantized_linear_modules']}；最终适配器 SHA256={audit['adapter_sha256']}。\n\n"
            "同精度基座/适配器×无画像/有画像四组均使用完整 1,496 条冻结测试集。\n\n"
            "| 条件 | BLEU | chrF | ROUGE-L | BERTScore F1 |\n|---|---:|---:|---:|---:|\n")
    for name, metric in scores.items():
        text += (f"| {name} | {metric['bleu']:.6f} | {metric['chrf']:.6f} | "
                 f"{metric['rougeL_f']:.6f} | {semantic['condition_scores'][name]['f1']:.6f} |\n")
    text += "\n有画像适配器相对 BF16 的原始语料级配对 bootstrap（2,000 次；候选−BF16）：\n"
    for name, contrast in paired["contrasts"].items():
        text += (f"\n- {name}: {contrast['observed']:.6f}，95% CI "
                 f"[{contrast['ci95_low']:.6f}, {contrast['ci95_high']:.6f}]。")
    text += (f"\n\n机器结果：`{run.relative_to(ROOT)}/quantization_result.json`。"
             "以上为单训练种子的名义逐项区间，未进行多重比较校正。"
             "量化配方包含 k-bit 准备函数的非量化参数转换，不能把差异完全归为位宽。\n")
    for document in ("13_quantization_experiment.md", "09_full_scale_qwen3_4b_experiment.md"):
        with (ROOT / "docs" / document).open("a", encoding="utf-8") as stream:
            stream.write(text)
    status("mode_complete", mode=mode, result=str(run / "quantization_result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--training-pid", type=int)
    entry.add_argument("--resume-evaluation", action="store_true",
                       help="Reuse audited, complete int8 predictions; rerun missing conditions only")
    args = parser.parse_args()
    lock = ROOT / "experiments/quantization_sequence.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    os.environ.update(PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        if args.training_pid is not None:
            wait_for_training(args.training_pid)
        evaluate(RUNS / "qwen3-4b-lamp7-full-lora-r16-a32-int8", "int8",
                 resume=args.resume_evaluation)
        nf4 = RUNS / "qwen3-4b-lamp7-full-lora-r16-a32-nf4"
        if nf4.exists() and any(nf4.iterdir()):
            raise FileExistsError("NF4 output exists; manual review required before training")
        nf4.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
            raise RuntimeError("Less than 20 GiB free; inspect disk before NF4")
        execute("train_qwen_lora.py", ["--config",
                "configs/qwen3_4b_lamp7_full_lora_r16_nf4.json"], nf4 / "train.combined.log")
        evaluate(nf4, "nf4")
        status("complete", completed_modes=["int8", "nf4"])
    except BaseException as error:
        status("failed", error=repr(error), traceback=traceback.format_exc(),
               instruction="Preserve artifacts. Do not automatically restart.")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
