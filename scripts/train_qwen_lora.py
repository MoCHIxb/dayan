from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

from personal_llm_data.model_loading import load_causal_model
from personal_llm_data.pipeline import read_jsonl
from personal_llm_data.training_protocol import budget_arguments, protect_output


class ProgressLogCallback(TrainerCallback):
    """Persist Trainer log events immediately, including before an epoch ends."""

    def __init__(self, path: Path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        event = {"time": time.time(), "step": state.global_step, "epoch": state.epoch,
                 "logs": logs or {}}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


class ChatSFTDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.records[index]["messages"]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        response_ids = self.tokenizer(
            messages[-1]["content"], add_special_tokens=False
        )["input_ids"] + [self.tokenizer.eos_token_id]
        if len(response_ids) >= self.max_length:
            response_ids = response_ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
            prompt_ids = []
        elif len(prompt_ids) + len(response_ids) > self.max_length:
            prompt_ids = prompt_ids[-(self.max_length - len(response_ids)) :]
        return {
            "input_ids": prompt_ids + response_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(response_ids)),
            "labels": [-100] * len(prompt_ids) + response_ids,
        }


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in features)
        width = ((width + 7) // 8) * 8
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            padding = width - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen with LoRA on processed LaMP-7")
    parser.add_argument("--config", default="configs/qwen3_1.7b_lamp7_lora.json")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint directory to resume optimizer, scheduler, RNG, and training state from",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    budget = budget_arguments(config)
    protect_output(config, args.resume_from_checkpoint)
    set_seed(int(config["seed"]))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this configured experiment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected experiment requires BF16 support")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.get("protocol_id") and args.resume_from_checkpoint is None:
        (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Loading tokenizer; config={args.config}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = config.get("quantization", "none")
    print(f"Loading model; quantization={quantization}", flush=True)
    model = load_causal_model(config["model_name_or_path"], quantization)
    model.config.use_cache = False
    if quantization == "none":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Model prepared; trainable={trainable}; total={total}", flush=True)

    train_records = read_jsonl(config["train_file"])
    eval_records = read_jsonl(config["eval_file"])[: int(config["eval_max_samples"])]
    if budget and (len(eval_records) != 1500 or len({str(r["id"]) for r in eval_records}) != 1500):
        raise ValueError("Explicit-budget experiments require 1500 unique dev records")
    train_dataset = ChatSFTDataset(train_records, tokenizer, int(config["max_length"]))
    eval_dataset = ChatSFTDataset(eval_records, tokenizer, int(config["max_length"]))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        logging_strategy="steps",
        logging_steps=int(config.get("logging_steps", 1)),
        **({"eval_strategy": "epoch", "save_strategy": "epoch"} | budget),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[ProgressLogCallback(output_dir / "training_progress.jsonl")],
    )

    started = time.time()
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        resume_checkpoint = str(checkpoint_path)
    print(f"Training started; train_records={len(train_records)}; "
          f"eval_records={len(eval_records)}; epochs={config['num_train_epochs']}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed = time.time() - started
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    summary = {
        "config": config,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100 * trainable / total,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "resumed_from_checkpoint": resume_checkpoint,
        "quantization": quantization,
        "quantized_linear_modules": sum(
            type(module).__module__.startswith("bitsandbytes.")
            and type(module).__name__ in ("Linear8bitLt", "Linear4bit")
            for module in model.modules()
        ),
        "bitsandbytes_version": version("bitsandbytes") if quantization != "none" else None,
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_loss": trainer.state.best_metric,
        "completed_optimizer_steps": trainer.state.global_step,
        "completed_epochs": trainer.state.epoch,
        "num_input_tokens_seen": trainer.state.num_input_tokens_seen,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "log_history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
