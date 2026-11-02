"""Finite, single-GPU follow-up batch. Human ratings remain an external input."""
from __future__ import annotations

import argparse
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
OUT = ROOT / "experiments/confirmation"
RUNS = ROOT / "experiments/runs"
SEEDS = (20251206, 20251207, 20251208)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def status(stage, **details):
    data = {"stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details}
    write(OUT / "state.json", data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def job(name, script, arguments, result, expected_samples=None):
    result = ROOT / result
    if result.exists():
        data = read(result)
        if expected_samples is not None and data.get("samples") != expected_samples:
            raise ValueError(f"Saved sample count mismatch: {name}")
        return data
    log = OUT / "logs" / f"{name}-{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status("running", job=name, script=script, log=str(log), result=str(result))
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, arguments)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    if not result.exists():
        raise FileNotFoundError(f"Missing completed job result: {result}")
    data = read(result)
    if expected_samples is not None and data.get("samples") != expected_samples:
        raise ValueError(f"Job sample count mismatch: {name}")
    return data


def dev(name, adapter, mode="none"):
    path = Path("experiments/full_dev") / f"{name}.json"
    return job("dev-" + name, "evaluate_full_dev.py",
               ["--adapter", adapter, "--quantization", mode, "--output", path], path, 1500)


def generation(name, model, adapter, data, output, drop=False, reuse=None):
    output = Path(output)
    if reuse and not (ROOT / output).exists():
        source, source_adapter = reuse
        expected_hash = hashlib.sha256((Path(adapter)/"adapter_model.safetensors").read_bytes()).hexdigest() if adapter else None
        source_hash = hashlib.sha256((Path(source_adapter)/"adapter_model.safetensors").read_bytes()).hexdigest() if source_adapter else None
        if expected_hash == source_hash:
            metadata = read(source)
            metadata["reused_from"] = str(source)
            metadata["verified_adapter_sha256"] = expected_hash
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(source).with_suffix(".predictions.jsonl"), output.with_suffix(".predictions.jsonl"))
            write(output, metadata)
    arguments = ["--model", model, "--data", data, "--output", output, "--max-samples",
                 1500 if str(data).endswith("dev.jsonl") else 1496, "--batch-size", 4,
                 "--max-input-tokens", 768, "--max-new-tokens", 80]
    if adapter:
        arguments += ["--adapter", adapter]
    if drop:
        arguments += ["--drop-profile"]
    count = 1500 if str(data).endswith("dev.jsonl") else 1496
    metric = job(name, "evaluate_generation.py", arguments, output, count)
    prediction_path = output.with_suffix(".predictions.jsonl")
    with prediction_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    with Path(data).open(encoding="utf-8") as stream:
        expected = {str(r["id"]):r["messages"][-1]["content"] for line in stream if line.strip() for r in [json.loads(line)]}
    if len(rows) != count or {str(r['id']):r['reference'] for r in rows} != expected:
        raise ValueError("Generation IDs/references incomplete")
    return output, metric


def general(name, model, adapter=None):
    output = OUT / "general" / f"{name}.json"
    args = ["--model", model, "--output", output]
    if adapter:
        args += ["--adapter", adapter]
    result = job("general-" + name, "evaluate_general_abilities.py", args, output)
    expected = {"arc_challenge":1172, "gsm8k_cot_zeroshot":1319, "ifeval":541}
    if any(result["n-samples"][task]["effective"] != n for task,n in expected.items()):
        raise ValueError("Incomplete general ability benchmark")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    lock = OUT / "gpu_queue.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    os.environ.update(PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    try:
        baseline = RUNS / "qwen3-4b-lamp7-full-lora-r16"
        epochs = [dev(f"r16_a32_epoch{epoch}", baseline/f"checkpoints/checkpoint-{step}")
                  for epoch,step in ((1,653),(2,1306))
                  if (baseline/f"checkpoints/checkpoint-{step}/adapter_model.safetensors").exists()]
        selected = min(epochs, key=lambda x:x["mean_record_loss"])
        selected_adapter = Path(selected["adapter"])
        write(OUT / "historical_saved_adapter_recheck.json",
              {"criterion":"full1500_mean_record_loss_among_surviving_checkpoints",
               "all_epoch_weights_available":len(epochs)==2, **selected})
        for name, directory, mode in (
            ("rank8_best", "qwen3-4b-lamp7-full-lora-r8-a16", "none"),
            ("rank32_best", "qwen3-4b-lamp7-full-lora-r32-a64", "none"),
            ("alpha16_best", "qwen3-4b-lamp7-full-lora-r16-a16", "none"),
            ("alpha64_best", "qwen3-4b-lamp7-full-lora-r16-a64", "none"),
            ("int8_best", "qwen3-4b-lamp7-full-lora-r16-a32-int8", "int8"),
            ("nf4_best", "qwen3-4b-lamp7-full-lora-r16-a32-nf4", "nf4")):
            dev(name, RUNS/directory/"adapter", mode)
        # Candidate comparison above rechecks saved best adapters; missing inferior
        # checkpoints are not silently presented as full-dev epoch reselection.
        retrieval_manifest = ROOT / "data/processed/lamp7_retrieval_confirmation/manifest.json"
        job("prepare-retrieval", "prepare_retrieval_variants.py", [], retrieval_manifest)
        retrieval_scores = {}
        for name in read(retrieval_manifest)["variants"]:
            data = f"data/processed/lamp7_retrieval_confirmation/{name}/dev.jsonl"
            _, metric = generation("retrieval-dev-"+name, "models/Qwen3-4B", selected_adapter,
                                   data, OUT/"retrieval"/f"{name}_dev.json")
            retrieval_scores[name] = metric
        winner = max(retrieval_scores, key=lambda name:(retrieval_scores[name]["chrf"], name))
        write(OUT/"retrieval_selection.json", {"primary_metric":"dev_chrf", "winner":winner,
              "candidates":retrieval_scores, "scope":"inference_only; no test tuning"})
        for name in dict.fromkeys(("bm25_k12", winner)):
            data = f"data/processed/lamp7_retrieval_confirmation/{name}/test.jsonl"
            generation("retrieval-test-"+name, "models/Qwen3-4B", selected_adapter,
                       data, OUT/"retrieval"/f"{name}_test.json")
        if winner != "bm25_k12":
            result = OUT/"retrieval/winner_vs_budgeted_bm25_bootstrap.json"
            job("retrieval-bootstrap", "paired_adapter_comparison.py",
                ["--reference-system", OUT/"retrieval/bm25_k12_test.predictions.jsonl",
                 "--candidate-system", OUT/"retrieval"/f"{winner}_test.predictions.jsonl",
                 "--reference-name", "budgeted_bm25_k12", "--candidate-name", winner,
                 "--output", result, "--samples", 2000, "--seed", 20251206], result, 1496)
        cohort = []
        for size in ("4B", "1.7B"):
            model = f"models/Qwen3-{size}"
            base_paths = {}
            for profile in (False,True):
                condition = "profile" if profile else "no_profile"
                reuse = None
                if size == "4B":
                    source = baseline/("baseline_test_metrics.json" if profile else "baseline_no_profile_test_metrics.json")
                    reuse = (source,None)
                path,_ = generation(f"base-{size}-{condition}", model, None,
                                    "data/processed/lamp7_bm25_k12/test.jsonl",
                                    OUT/"cohort"/f"base_{size}_{condition}.json", not profile, reuse)
                base_paths[condition] = path.with_suffix(".predictions.jsonl")
            general(f"base_{size}", model)
            for seed in SEEDS:
                name = f"{size}_seed{seed}"
                config_path = f"configs/confirm_qwen3_{size.lower()}_seed{seed}.json"
                config = read(ROOT/config_path)
                run = ROOT/config["output_dir"]
                if not (run/"training_summary.json").exists() and run.exists() and any(run.iterdir()):
                    raise RuntimeError(f"Incomplete training requires manual recovery: {run}")
                job("train-"+name, "train_qwen_lora.py", ["--config",config_path], run/"training_summary.json")
                summary = read(run/"training_summary.json")
                if summary["train_records"] != 10437 or summary["eval_records"] != 1500 or summary["config"] != config:
                    raise ValueError("Strict model/seed training protocol mismatch")
                adapter = run/"adapter"
                state=read(run/"checkpoints/trainer_state.json")
                evaluations=[r for r in summary['log_history'] if 'eval_loss' in r]
                if len(evaluations)!=2 or [r['epoch'] for r in evaluations]!=[1.0,2.0] or state['global_step']!=1306:
                    raise ValueError("Incomplete fresh full-dev training")
                best=Path(state['best_model_checkpoint'])
                final_hash=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
                if final_hash!=hashlib.sha256((best/'adapter_model.safetensors').read_bytes()).hexdigest():
                    raise ValueError("Final adapter differs from full-dev best checkpoint")
                write(run/'confirmation_audit.json',{'seed':seed,'model':model,'eval_records':1500,
                      'train_records':10437,'best_checkpoint':str(best),'adapter_sha256':final_hash,
                      'dev_losses':[r['eval_loss'] for r in evaluations]})
                paths, metrics = {}, {}
                for profile in (False,True):
                    condition = "profile" if profile else "no_profile"
                    reuse = None
                    if size == "4B" and seed == 20251206:
                        reuse = (baseline/f"adapter_{condition}_test_metrics.json", baseline/"adapter")
                    path,metric = generation(f"adapter-{name}-{condition}", model, adapter,
                                            "data/processed/lamp7_bm25_k12/test.jsonl",
                                            OUT/"cohort"/f"adapter_{name}_{condition}.json", not profile, reuse)
                    paths[condition] = path.with_suffix(".predictions.jsonl")
                    metrics[condition] = metric
                bert_path = OUT/"cohort"/f"bertscore_{name}.json"
                job("bertscore-"+name, "bertscore_eval.py",
                    ["--base-no-profile",base_paths["no_profile"], "--base-profile",base_paths["profile"],
                     "--adapter-no-profile",paths["no_profile"], "--adapter-profile",paths["profile"],
                     "--output",bert_path,"--bootstrap",2000,"--seed",20251206], bert_path,1496)
                general(name,model,adapter)
                retention_path=OUT/"general"/f"retention_{name}.json"
                job("retention-"+name,"analyze_general_retention.py",
                    ["--base",OUT/"general"/f"base_{size}.json",
                     "--adapter",OUT/"general"/f"{name}.json",
                     "--output",retention_path],retention_path)
                cohort.append({"model_size":size,"seed":seed,"adapter":str(adapter),
                               "metrics":metrics,"bertscore":read(bert_path)})
                write(OUT/"cohort_progress.json",cohort)
        # Match training seeds when testing the model-size effect.
        for seed in SEEDS:
            result = OUT/"cohort"/f"paired_1.7b_vs_4b_seed{seed}.json"
            job(f"scale-bootstrap-{seed}","paired_adapter_comparison.py",
                ["--reference-system",OUT/"cohort"/f"adapter_4B_seed{seed}_profile.predictions.jsonl",
                 "--candidate-system",OUT/"cohort"/f"adapter_1.7B_seed{seed}_profile.predictions.jsonl",
                 "--reference-name","qwen3_4b","--candidate-name","qwen3_1.7b",
                 "--output",result,"--samples",2000,"--seed",20251206],result,1496)
        summary_path=OUT/"machine_summary.json"
        job("summarize-confirmation","summarize_confirmation.py",[],summary_path)
        status("machine_experiments_complete", human_ratings="awaiting_two_independent_raters",
               instruction="Do not mark the entire study complete before blind-rating analysis and consolidated review")
    except BaseException as error:
        status("failed", error=repr(error), traceback=traceback.format_exc(), instruction="Preserve outputs; do not auto-restart")
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
