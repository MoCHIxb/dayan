"""Formal research BANKING training, independent of frozen4B implementations."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

from personal_llm_data.personal_data import sha, verify_prepared
from personal_llm_data.personal_runtime import checkpoint_inventory, gpu_guard, write_json
from personal_llm_data.training_protocol import protect_output

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    from deploy_research_models import catalog
    key=config.get('model_key');recipe=config.get('recipe_id')
    if key not in catalog()['models'] or recipe not in ('A','B','C','D','E'):raise ValueError('Unknown registered research condition')
    canonical=ROOT/'configs/research_banking_v1'/f'{key}_{recipe}.json'
    # Config files are independently frozen by the campaign; reject wrong task,
    # output reuse or unannounced scientific edits even in direct invocation.
    expected=json.loads(canonical.read_text(encoding='utf-8'))
    if config!=expected or config.get('protocol_id')!='research_banking_v1':raise ValueError('Canonical research config required')
    modes={'A':('none',8,16),'B':('none',16,16),'C':('none',16,32),'D':('nf4',16,32),'E':('int8',16,32)}
    entry=catalog()['models'][key]
    if (config['quantization'],config['lora_r'],config['lora_alpha'])!=modes[recipe]:raise ValueError('Frozen recipe differs')
    if config['target_modules']!=entry['lora_projections'] or config['model_name_or_path']!=entry['directory']:raise ValueError('Model/projection identity differs')
    if config['output_dir']!=f'experiments/research_banking_v1/{key}_{recipe}_seed20261113/run' or config['data_dir']!='data/processed/banking77_8b_v1/n500':raise ValueError('Independent research paths required')
    fixed=dict(seed=20261113,max_steps=128,eval_steps=16,max_length=1536,max_new_tokens=64,learning_rate=1e-4,gradient_accumulation_steps=16,lora_dropout=.05)
    if any(config[k]!=v for k,v in fixed.items()):raise ValueError('Fixed first-seed research budget differs')
    if json.loads((ROOT/config['model_name_or_path']/'config.json').read_text())['model_type']!=entry['model_type']:raise ValueError('Architecture differs')
    rows=verify_prepared(ROOT/config['data_dir'])
    if {k:len(v) for k,v in rows.items()}!=dict(train=500,dev=100,test=400):raise ValueError('Complete task split required')
    return rows


def inputs_snapshot(config, config_path):
    model = ROOT/config["model_name_or_path"]
    weights = list(model.glob("*.safetensors"))
    if not weights:
        raise ValueError("Local model weights required; no automatic downloads")
    data = ROOT/config["data_dir"]
    from importlib.metadata import version
    return dict(runtime={p:version(p) for p in ("torch","transformers","peft","accelerate","bitsandbytes","sentencepiece","protobuf","tokenizers")},config_sha256=sha(config_path),
        data_sha256={p.name: sha(p) for p in (data/"manifest.json", data/"train.jsonl", data/"dev.jsonl", data/"test.jsonl")},
        model_sha256={p.name: sha(p) for p in sorted(model.iterdir()) if p.is_file()},
        code_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (
            Path(__file__), ROOT/"scripts/train_qwen_lora.py", ROOT/"src/personal_llm_data/model_loading.py",
            ROOT/"src/personal_llm_data/personal_data.py", ROOT/"src/personal_llm_data/personal_runtime.py",
            ROOT/"src/personal_llm_data/training_protocol.py", ROOT/"src/personal_llm_data/text_safety.py", ROOT/"src/personal_llm_data/research_chat.py",
            ROOT/"src/personal_llm_data/research_sft.py", ROOT/"configs/research_models.json", ROOT/"scripts/deploy_research_models.py")})


def verify_checkpoint(path, inputs):
    actual = checkpoint_inventory(path)
    receipt = json.loads((Path(path)/"personal_checkpoint_receipt.json").read_text(encoding="utf-8"))
    if receipt != dict(checkpoint=actual, inputs=inputs):
        raise ValueError("Checkpoint/input checksum mismatch; inspect without retrying")
    return actual


def validate_resume_position(output, step, max_steps):
    if step >= max_steps:
        raise ValueError("Saved budget already reached; audit finalization instead of retraining")
    for existing in (Path(output)/"checkpoints").glob("checkpoint-*"):
        suffix = existing.name.removeprefix("checkpoint-")
        if suffix.isdigit() and int(suffix) > step:
            raise ValueError("Later checkpoint evidence exists; inspect/archive it before resuming an earlier step")


def execute(config, rows, config_path, resume):
    output = ROOT/config["output_dir"]
    if (output/"training_summary.json").exists():
        raise FileExistsError("Completed/evidenced run exists; do not retrain or overwrite it")
    # Resolve once so the frozen output-protection helper uses the same directory.
    protected = dict(config, output_dir=str(output))
    protect_output(protected, resume)
    inputs = inputs_snapshot(config, config_path)
    if verify_prepared(ROOT/config["data_dir"]) != rows:
        raise ValueError("Prepared data changed while validating")
    if resume:
        checkpoint = verify_checkpoint(resume, inputs)
        validate_resume_position(output, checkpoint["step"], config["max_steps"])
        if json.loads((output/"run_inputs.json").read_text(encoding="utf-8")) != inputs:
            raise ValueError("Run inputs changed since checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(output/"run_config.json", protected)
        write_json(output/"run_inputs.json", inputs)
    pause = output/"pause.request"
    if resume and pause.exists():
        pause.rename(output/f"pause.consumed.{time.time_ns()}")
    write_json(output/"run_status.json", dict(stage="preflight", pid=os.getpid()))
    try:
        # GPU libraries and existing scientific components load only after opt-in.
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
        from train_qwen_lora import SFTCollator
        from personal_llm_data.research_sft import ChatSFTDataset,encode_row
        from personal_llm_data.model_loading import load_causal_model
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!="NVIDIA GeForce RTX 4090" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support required")
        set_seed(config["seed"])
        model_path = str(ROOT/config["model_name_or_path"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # The exact policy and target mask validated by CPU/GPU diagnostics.
        for records in rows.values():
            for row in records:encode_row(tokenizer,row,config['model_key'],config['max_length'],config['max_new_tokens'])
        quant = config["quantization"]
        model = load_causal_model(model_path, quant)
        model.config.use_cache = False
        if quant == "none":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                         gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=config["lora_r"],
                    lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                    target_modules=config["target_modules"], bias="none"))

        class ProgressAndPause(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if any(k in (logs or {}) and not math.isfinite(float(logs[k])) for k in ('loss','grad_norm','eval_loss')):raise ValueError('Nonfinite training/dev evidence')
                with (output/"progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=state.global_step, time=time.time(), logs=logs or {}))+"\n")

            def on_step_end(self, args, state, control, **kwargs):
                if (pause.exists() or (ROOT/"experiments/research_banking_v1/pause.request").exists()) and state.global_step % config["eval_steps"] == 0:
                    control.should_save = control.should_evaluate = control.should_training_stop = True
                return control

            def on_save(self, args, state, control, **kwargs):
                checkpoint = output/"checkpoints"/f"checkpoint-{state.global_step}"
                write_json(checkpoint/"personal_checkpoint_receipt.json",
                           dict(checkpoint=checkpoint_inventory(checkpoint), inputs=inputs))

        args = TrainingArguments(output_dir=str(output/"checkpoints"), max_steps=config["max_steps"],
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"],
            weight_decay=0.01, warmup_ratio=0.05, lr_scheduler_type="cosine", bf16=True, tf32=True,
            logging_steps=1, eval_strategy="steps", save_strategy="steps", eval_steps=config["eval_steps"],
            save_steps=config["eval_steps"], save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, optim="adamw_torch", report_to=[],
            remove_unused_columns=False, dataloader_num_workers=0, seed=config["seed"], data_seed=config["seed"])
        trainer = Trainer(model=model, args=args,
            train_dataset=ChatSFTDataset(rows["train"], tokenizer, config["max_length"],config["model_key"],config["max_new_tokens"]),
            eval_dataset=ChatSFTDataset(rows["dev"], tokenizer, config["max_length"],config["model_key"],config["max_new_tokens"]),
            data_collator=SFTCollator(tokenizer.pad_token_id), processing_class=tokenizer,
            callbacks=[ProgressAndPause()])
        write_json(output/"run_status.json", dict(stage="running", pid=os.getpid()))
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if inputs_snapshot(config, config_path) != inputs:
            raise ValueError("Inputs/code/model changed while training; preserve evidence and inspect")
        step = trainer.state.global_step
        final = output/"checkpoints"/f"checkpoint-{step}"
        checkpoint = verify_checkpoint(final, inputs)
        if step < config["max_steps"]:
            if not (pause.exists() or (ROOT/"experiments/research_banking_v1/pause.request").exists()):
                raise RuntimeError("Training ended early without an explicit pause")
            write_json(output/"run_status.json", dict(stage="paused_checkpoint_verified", pid=None,
                       checkpoint=str(final), evidence=checkpoint))
            print(f"Paused with verified full state at optimizer step {step}; checkpoint={final}")
            return
        evaluations = [r for r in trainer.state.log_history if "eval_loss" in r]
        scheduled = set(range(config["eval_steps"], config["max_steps"]+1, config["eval_steps"]))
        if scheduled != {r["step"] for r in evaluations} or len(evaluations)!=8 or any(not math.isfinite(r["eval_loss"]) for r in evaluations):
            raise ValueError("Missing a scheduled complete dev evaluation")
        best = min(evaluations, key=lambda r: (r["eval_loss"], r["step"]))
        best_path = Path(trainer.state.best_model_checkpoint)
        if best_path.name != f"checkpoint-{best['step']}":
            raise ValueError("Best checkpoint does not match earliest minimum dev loss")
        verify_checkpoint(best_path, inputs)
        adapter = output/"adapter"
        trainer.model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        if sha(adapter/"adapter_model.safetensors") != sha(best_path/"adapter_model.safetensors"):
            raise ValueError("Exported adapter differs from selected checkpoint")
        trainable, total = trainer.model.get_nb_trainable_parameters()
        summary = dict(status="completed_verified", counts={k: len(v) for k, v in rows.items()},
            completed_optimizer_steps=step, best_step=best["step"], best_dev_loss=best["eval_loss"],
            adapter_sha256=sha(adapter/"adapter_model.safetensors"), final_checkpoint=checkpoint,
            best_checkpoint=str(best_path), trainable_parameters=trainable, total_parameters=total,
            elapsed_seconds_this_session=time.time()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), device=torch.cuda.get_device_name(0),
            python=platform.python_version(), torch=torch.__version__, inputs=inputs,
            limitations="Formal first-seed classification training; full held-out generation/scoring audited separately; not style/forgetting proof")
        write_json(output/"training_summary.json", summary)
        write_json(output/"run_status.json", dict(stage="completed_verified", pid=None, step=step))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException:
        write_json(output/"run_status.json", dict(stage="interrupted_or_failed_needs_audit", pid=None))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--request-pause", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.resume_from_checkpoint and not args.execute:
        parser.error("Resume requires --execute and a verified checkpoint")
    if args.request_pause:
        if config.get("protocol_id") != "research_banking_v1":
            raise ValueError("Pause requests apply only to the independent personal example")
        output = ROOT/config["output_dir"]
        saved = json.loads((output/"run_config.json").read_text(encoding="utf-8"))
        if saved != dict(config, output_dir=str(output)):
            raise ValueError("Pause configuration differs from the active personal run")
        status = json.loads((output/"run_status.json").read_text(encoding="utf-8"))
        if status["stage"] not in ("preflight", "running"):
            raise ValueError("No running personal task recorded; inspect actual process if uncertain")
        with (output/"pause.request").open("x", encoding="utf-8") as stream:
            stream.write(str(time.time()))
        print("Pause requested; wait for paused_checkpoint_verified and verify process exit before shutdown")
        return 0
    rows = validate_config(config)
    if not args.execute:
        print(json.dumps(dict(status="validated_files_only", executable=False,
            counts={k: len(v) for k, v in rows.items()}, next="--execute explicitly opts into GPU; token/resource checks still required")))
        return 0
    if platform.system() != "Linux":raise RuntimeError("Formal experiments run on owner Linux4090 only")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("The personal example runner supports one GPU/process only")
    with gpu_guard(ROOT):
        execute(config, rows, args.config, args.resume_from_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
