"""Bounded one-shot execution of protocol v1. No scheduler, retries or old queue."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "experiments/refocused"
STATE = WORK / "execution_state.json"
BASE = ROOT / "experiments/runs/qwen3-4b-lamp7-full-lora-r16"
DATA = ROOT / "data/processed/lamp7_bm25_k12/test.jsonl"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def status(stage, **kwargs):
    state = dict(stage=stage, protocol_id="small_sample_v1", supervisor_pid=os.getpid(),
                 updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kwargs)
    write(STATE, state)
    print(json.dumps(state), flush=True)


def execute(script, args, log):
    status("running", script=script, arguments=list(map(str, args)), log=str(log))
    with Path(log).open("x", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, args)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def wait_existing(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        status("waiting_for_first_candidate", training_pid=pid)
        deadline = time.monotonic() + 7200
        while True:
            result = kernel.WaitForSingleObject(handle, 30000)
            if result == 0:
                return
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            if time.monotonic() > deadline:
                raise TimeoutError("Existing candidate exceeded two-hour attachment limit")
    finally:
        kernel.CloseHandle(handle)


def audit(config):
    run = ROOT / config["output_dir"]
    summary = read(run / "training_summary.json")
    n = len([line for line in (ROOT / config["train_file"]).read_text(encoding="utf-8").splitlines() if line.strip()])
    if summary["config"] != config or (summary["train_records"], summary["eval_records"]) != (n, 1500):
        raise ValueError("Training config or full-dev size mismatch")
    evals = [r for r in summary["log_history"] if "eval_loss" in r]
    if [r["step"] for r in evals] != list(range(16, 129, 16)) or summary["completed_optimizer_steps"] != 128:
        raise ValueError("Incomplete eight-evaluation / 128-step trajectory")
    for row in summary["log_history"]:
        for key in ("loss", "eval_loss", "grad_norm"):
            if key in row and not math.isfinite(row[key]):
                raise ValueError("Nonfinite training evidence")
    best = min(evals, key=lambda r: (r["eval_loss"], r["step"]))
    checkpoint = run / f"checkpoints/checkpoint-{best['step']}"
    adapter_hash = sha(run / "adapter/adapter_model.safetensors")
    if adapter_hash != sha(checkpoint / "adapter_model.safetensors"):
        raise ValueError("Adapter does not match best dev checkpoint")
    if Path(summary["best_model_checkpoint"]).name != checkpoint.name or summary["best_dev_loss"] != best["eval_loss"]:
        raise ValueError("Best checkpoint metadata mismatch")
    if not (run / "checkpoints/checkpoint-128/trainer_state.json").exists():
        raise ValueError("Final checkpoint missing")
    result = dict(run=config["output_dir"], config=config, dev_evaluations=evals,
                  best_step=best["step"], best_dev_loss=best["eval_loss"], adapter_sha256=adapter_hash,
                  elapsed_seconds=summary["elapsed_seconds"], peak_gpu_memory_bytes=summary["peak_gpu_memory_bytes"])
    write(run / "training_audit.json", result)
    return result


def train(config_path):
    config = read(config_path)
    run = ROOT / config["output_dir"]
    if not (run / "training_summary.json").exists():
        execute("train_qwen_lora.py", ["--config", config_path], WORK / "logs" / (config_path.stem + ".batch_train.log"))
    return audit(config)


def validate_predictions(path):
    def rows(p):
        return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {str(r["id"]): r["messages"][-1]["content"] for r in rows(DATA)}
    actual = rows(path)
    if len(actual) != 1496 or len({str(r["id"]) for r in actual}) != 1496:
        raise ValueError("Incomplete or duplicate test predictions")
    if {str(r["id"]): r["reference"] for r in actual} != expected:
        raise ValueError("Test IDs/references differ")


def task(script, args, output):
    """Only reuse outputs accompanied by a matching completed-command receipt."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    signature = dict(script=script, script_sha256=sha(ROOT / "scripts" / script), arguments=list(map(str, args)))
    if output.exists() or receipt.exists():
        record = read(receipt)
        if record["command"] != signature or record["sha256"] != sha(output):
            raise ValueError(f"Invalid completion receipt: {output}")
        return
    execute(script, args, output.with_suffix(output.suffix + ".log"))
    read(output)
    write(receipt, dict(command=signature, sha256=sha(output)))


def evaluate(run):
    paths = {name: WORK / "baseline" / f"{name}_test_metrics.predictions.jsonl"
             for name in ("base_no_profile", "base_profile")}
    for drop in (True, False):
        name = "adapter_no_profile" if drop else "adapter_profile"
        out = run / f"{name}_test_metrics.json"
        args = ["--model", "models/Qwen3-4B", "--adapter", run / "adapter", "--data", DATA,
                "--output", out, "--max-samples", 1496, "--batch-size", 4,
                "--max-input-tokens", 768, "--max-new-tokens", 80, "--quantization", "none"]
        if drop:
            args += ["--drop-profile"]
        task("evaluate_generation.py", args, out)
        paths[name] = out.with_suffix(".predictions.jsonl")
    for path in paths.values():
        validate_predictions(path)
    shared = [v for name, path in paths.items() for v in ("--" + name.replace('_', '-'), path)]
    full_dev = run / "full_dev.json"
    task("evaluate_full_dev.py", ["--adapter", run / "adapter", "--output", full_dev], full_dev)
    semantic = run / "bertscore.json"
    task("bertscore_eval.py", [*shared, "--output", semantic, "--model", "roberta-large",
         "--batch-size", 32, "--bootstrap", 2000, "--seed", 20251206], semantic)
    scores = read(semantic)
    if scores["samples"] != 1496 or not scores["idf"] or not scores["rescale_with_baseline"]:
        raise ValueError("BERTScore protocol mismatch")
    errors = run / "error_analysis.json"
    task("analyze_predictions.py", [*shared, "--data", DATA, "--output", errors], errors)
    # Independent descriptive copy diagnosis; full-test scores are never replaced.
    records = {str(r["id"]): r for r in (json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip())}
    diagnosis = {}
    for name, path in paths.items():
        groups = {"all": [], "reference_in_history": [], "reference_not_in_history": []}
        for line in path.read_text(encoding="utf-8").splitlines():
            p = json.loads(line)
            history = [h["text"] for h in records[str(p["id"])]["profile"]]
            copied = p["prediction"] in history
            groups["all"].append(copied)
            groups["reference_in_history" if p["reference"] in history else "reference_not_in_history"].append(copied)
        diagnosis[name] = {g: dict(samples=len(v), exact_history_count=sum(v), exact_history_rate=sum(v)/len(v)) for g,v in groups.items()}
    write(run / "copy_diagnostics.json", dict(scope="descriptive exact-copy diagnostic, not proof of style or factual error", conditions=diagnosis))
    return shared, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    args = parser.parse_args()
    lock = WORK / "small_sample_sequence.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    os.environ.update(PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        (WORK / "logs").mkdir(exist_ok=True)
        implementation = WORK / "small_sample_execution_manifest.json"
        if implementation.exists():
            raise FileExistsError("Execution manifest exists; inspect previous execution before restarting")
        scripts = ["run_small_sample_sequence.py", "evaluate_generation.py", "evaluate_full_dev.py",
                   "bertscore_eval.py", "analyze_predictions.py", "paired_bootstrap.py", "paired_adapter_comparison.py"]
        write(implementation, dict(protocol_manifest_sha256=sha(WORK / "small_sample_v1_manifest.json"),
            scripts={s: sha(ROOT / "scripts" / s) for s in scripts}, retries=0, recurring=False))
        execute("freeze_small_sample_protocol.py", [], WORK / "logs" / f"preflight-{time.time_ns()}.log")
        wait_existing(args.training_pid)
        configs = [ROOT / p for p in read(WORK / "small_sample_v1_manifest.json")["stage1_configs"]]
        candidates = [train(path) for path in configs]
        selected = []
        for n in (50, 100, 200, 500):
            pool = [r for r in candidates if r["config"]["train_file"].endswith(f"train_{n}.jsonl")]
            selected.append(min(pool, key=lambda r: (r["best_dev_loss"], r["config"]["learning_rate"], r["best_step"])))
        selection = dict(protocol_id="small_sample_v1", candidates=candidates, selected=selected)
        selection_path = WORK / "small_sample_selection.json"
        if selection_path.exists():
            raise FileExistsError("Selection already exists; do not overwrite")
        write(selection_path, selection)
        confirmation = []
        for r in selected:
            for seed in (20251207, 20251208):
                config = r["config"] | {"seed": seed, "output_dir": r["config"]["output_dir"].replace("20251206", str(seed))}
                path = ROOT / "configs/small_sample_v1_confirm" / (Path(config["output_dir"]).name + ".json")
                if path.exists():
                    raise FileExistsError(path)
                write(path, config)
                confirmation.append(path)
        write(WORK / "small_sample_confirmation_manifest.json", dict(selection_sha256=sha(selection_path),
              configs={str(p.relative_to(ROOT)): sha(p) for p in confirmation}))
        runs = selected + [train(path) for path in confirmation]
        # Legacy baseline metadata lacks model/decode/budget fields: rerun both fully.
        for name in ("base_no_profile", "base_profile"):
            out = WORK / "baseline" / f"{name}_test_metrics.json"
            arguments = ["--model", "models/Qwen3-4B", "--data", DATA, "--output", out,
                "--max-samples", 1496, "--batch-size", 4, "--max-input-tokens", 768,
                "--max-new-tokens", 80, "--quantization", "none"]
            if name == "base_no_profile":
                arguments += ["--drop-profile"]
            task("evaluate_generation.py", arguments, out)
            validate_predictions(out.with_suffix(".predictions.jsonl"))
        evaluated = []
        for r in runs:
            run = ROOT / r["run"]
            shared, paths = evaluate(run)
            evaluated.append((r, shared, paths))
        for r, shared, paths in evaluated:
            run = ROOT / r["run"]
            out = run / "paired_2x2_bootstrap.json"
            task("paired_bootstrap.py", [*shared, "--output", out, "--samples", 2000, "--seed", 20251206], out)
            if read(out)["bootstrap_resamples"] != 2000:
                raise ValueError("Incomplete original corpus bootstrap")
            n = Path(r["config"]["train_file"]).stem.split('_')[1]
            for label, reference in (("historical_same_n", ROOT / f"experiments/runs/qwen3-4b-lamp7-n{n}-lora-r16-a32/adapter_profile_test_metrics.predictions.jsonl"),
                                     ("historical_full", BASE / "adapter_profile_test_metrics.predictions.jsonl")):
                out = run / f"paired_vs_{label}.json"
                task("paired_adapter_comparison.py", ["--reference-system", reference, "--candidate-system", paths["adapter_profile"],
                     "--reference-name", label, "--candidate-name", "small_sample_v1", "--output", out, "--samples", 2000, "--seed", 20251206], out)
            write(run / "small_sample_result.json", dict(training=r, metrics={name: read(run / f"{name}_test_metrics.json") for name in ("adapter_profile", "adapter_no_profile")},
                  bertscore=read(run / "bertscore.json"), bootstrap=read(run / "paired_2x2_bootstrap.json")))
        report = {"protocol_id": "small_sample_v1", "runs": [read(ROOT / r["run"] / "small_sample_result.json") for r in runs]}
        report["aggregate"] = {}
        for n in (50,100,200,500):
            group = [r for r in report["runs"] if r["training"]["config"]["train_file"].endswith(f"train_{n}.jsonl")]
            report["aggregate"][str(n)] = {m: {"mean": statistics.mean(r["metrics"]["adapter_profile"][m] for r in group),
                "sample_sd": statistics.stdev(r["metrics"]["adapter_profile"][m] for r in group)} for m in ("bleu", "chrf", "rougeL_f")}
        write(WORK / "small_sample_results.json", report)
        lines = ["# 小样本协议v1结果", "", "所有正式条件均为1496条test；每档三种子，开发集选择规则见20号协议。", "",
                 "| 数据量 | 有画像BLEU均值 | 种子样本标准差 | chrF均值 | ROUGE-L均值 |", "|---|---:|---:|---:|---:|"]
        for n, values in report["aggregate"].items():
            lines.append(f"| {n} | {values['bleu']['mean']:.4f} | {values['bleu']['sample_sd']:.4f} | {values['chrf']['mean']:.4f} | {values['rougeL_f']['mean']:.4f} |")
        lines += ["", "本表仅作汇总；逐种子全部指标、BERTScore、复制诊断、完整dev NLL/PPL与2000次原始语料级配对bootstrap位于各run目录。置信区间为逐比较名义区间，未校正多重比较；不据区间跨0判等效。第一种子参与学习率选择，另两种子固定该学习率。固定子集的训练稳定性不包含数据抽样变异。", "",
                  "机器汇总：experiments/refocused/small_sample_results.json。尚需分析质量/资源权衡、制定小样本PEFT对照协议，并完成框架及指南。协议v1完成不等于整个课题完成。"]
        (ROOT / "docs/21_small_sample_results.md").write_text('\n'.join(lines)+'\n', encoding="utf-8")
        status("small_sample_v1_complete", result="experiments/refocused/small_sample_results.json",
               remaining_project_work=["interpret results", "small-sample PEFT protocol", "framework and guide delivery"])
    except BaseException:
        status("failed_no_automatic_retry", error=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
