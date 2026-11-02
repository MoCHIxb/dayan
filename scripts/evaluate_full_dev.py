"""Assistant-only teacher-forced loss on every official dev record."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from personal_llm_data.model_loading import load_causal_model
from personal_llm_data.pipeline import read_jsonl
from train_qwen_lora import ChatSFTDataset, SFTCollator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--quantization", choices=("none", "int8", "nf4"), default="none")
    parser.add_argument("--data", default="data/processed/lamp7_bm25_k12/dev.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    rows_path = output.with_suffix(".per_item.jsonl")
    if output.exists() or rows_path.exists():
        raise FileExistsError("Dev results already exist; inspect instead of overwriting")
    records = read_jsonl(args.data)
    if len(records) != 1500 or len({str(r['id']) for r in records}) != 1500:
        raise ValueError("Expected all 1500 unique official dev records")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = ChatSFTDataset(records, tokenizer, 768)
    collator = SFTCollator(tokenizer.pad_token_id)
    model = load_causal_model(args.model, args.quantization)
    if args.quantization == "none":
        model = model.cuda()
    model = PeftModel.from_pretrained(model, args.adapter)
    model.config.use_cache = False
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    losses, tokens = [], []
    started = time.time()
    with rows_path.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for index, record in enumerate(records):
            batch = {k: v.cuda() for k, v in collator([dataset[index]]).items()}
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            if not count:
                raise ValueError(f"No supervised tokens: {record['id']}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = float(model(**batch).loss.float().item())
            if not math.isfinite(loss):
                raise ValueError("Non-finite dev loss")
            losses.append(loss)
            tokens.append(count)
            stream.write(json.dumps({"id": record["id"], "loss": loss,
                                     "supervised_tokens": count}) + "\n")
            if (index + 1) % 100 == 0:
                stream.flush()
                print(f"dev {index + 1}/1500 mean_loss={sum(losses)/len(losses):.7f}", flush=True)
    mean_loss = sum(losses) / len(losses)
    token_loss = sum(loss * n for loss, n in zip(losses, tokens)) / sum(tokens)
    result = {"samples": 1500, "model": args.model, "adapter": args.adapter,
              "quantization": args.quantization, "max_length": 768,
              "mean_record_loss": mean_loss,
              "first500_mean_record_loss": sum(losses[:500]) / 500,
              "token_weighted_nll": token_loss, "token_weighted_perplexity": math.exp(token_loss),
              "supervised_tokens": sum(tokens), "elapsed_seconds": time.time() - started,
              "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256((Path(args.adapter)/"adapter_model.safetensors").read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
