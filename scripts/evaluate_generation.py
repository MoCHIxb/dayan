from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoTokenizer

from personal_llm_data.model_loading import QUANTIZATION_MODES, load_causal_model
from personal_llm_data.pipeline import read_jsonl, write_jsonl


def style_features(text: str) -> dict[str, float]:
    length = max(len(text), 1)
    letters = [char for char in text if char.isalpha()]
    return {
        "length": float(len(text)),
        "exclamation_rate": text.count("!") / length,
        "question_rate": text.count("?") / length,
        "ellipsis_rate": len(re.findall(r"\.{2,}", text)) / length,
        "uppercase_rate": sum(char.isupper() for char in letters) / max(len(letters), 1),
    }


def without_profile(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    task = messages[-1]["content"].split("\n\nTask:\n", 1)[-1]
    return [
        {
            "role": "system",
            "content": "You are a rewriting assistant. Preserve meaning and return only the rewritten tweet without explanation.",
        },
        {"role": "user", "content": f"Task:\n{task}"},
    ]


def generate_batch(
    model: Any,
    tokenizer: Any,
    message_batches: list[list[dict[str, str]]],
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    rendered = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for messages in message_batches
    ]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prefix_width = encoded["input_ids"].shape[1]
    return [text.strip() for text in tokenizer.batch_decode(output[:, prefix_width:], skip_special_tokens=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base Qwen or a LoRA adapter on LaMP-7")
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--adapter")
    parser.add_argument("--data", default="data/processed/lamp7/dev.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--drop-profile", action="store_true")
    parser.add_argument("--quantization", choices=QUANTIZATION_MODES, default="none")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_model(args.model, args.quantization)
    if args.quantization == "none":
        model = model.cuda()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    records = read_jsonl(args.data)[: args.max_samples]
    predictions: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    started = time.time()
    for start in range(0, len(records), args.batch_size):
        rows = records[start : start + args.batch_size]
        prompts = [without_profile(row["messages"][:-1]) if args.drop_profile else row["messages"][:-1] for row in rows]
        generated = generate_batch(model, tokenizer, prompts, args.max_input_tokens, args.max_new_tokens)
        for record, prediction in zip(rows, generated):
            predictions.append(
                {"id": record["id"], "prediction": prediction, "reference": record["messages"][-1]["content"]}
            )
        completed = min(start + args.batch_size, len(records))
        if completed % 100 == 0 or completed == len(records):
            print(f"generated {completed}/{len(records)}")

    torch.cuda.synchronize()
    generation_elapsed = time.time() - started
    refs = [row["reference"] for row in predictions]
    hyps = [row["prediction"] for row in predictions]
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_values = [rouge.score(ref, hyp) for ref, hyp in zip(refs, hyps)]
    feature_errors: dict[str, list[float]] = {}
    for ref, hyp in zip(refs, hyps):
        ref_features, hyp_features = style_features(ref), style_features(hyp)
        for key in ref_features:
            feature_errors.setdefault(key, []).append(abs(ref_features[key] - hyp_features[key]))
    metrics = {
        "samples": len(predictions),
        "bleu": BLEU(effective_order=True).corpus_score(hyps, [refs]).score,
        "chrf": CHRF().corpus_score(hyps, [refs]).score,
        "rouge1_f": sum(row["rouge1"].fmeasure for row in rouge_values) / len(rouge_values),
        "rouge2_f": sum(row["rouge2"].fmeasure for row in rouge_values) / len(rouge_values),
        "rougeL_f": sum(row["rougeL"].fmeasure for row in rouge_values) / len(rouge_values),
        "exact_match": sum(hyp == ref for hyp, ref in zip(hyps, refs)) / len(refs),
        "empty_rate": sum(not hyp for hyp in hyps) / len(hyps),
        "style_feature_mae": {key: sum(values) / len(values) for key, values in feature_errors.items()},
        "elapsed_seconds": time.time() - started,
        "generation_elapsed_seconds": generation_elapsed,
        "generation_samples_per_second": len(predictions) / generation_elapsed,
        "adapter": args.adapter,
        "drop_profile": args.drop_profile,
        "batch_size": args.batch_size,
        "model": args.model,
        "quantization": args.quantization,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, output.with_suffix(".predictions.jsonl"))
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
