"""Local deterministic inference. Inputs are prompts, never training examples."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def load_prompts(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("Each input needs nonempty messages")
            if any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                   or not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
                raise ValueError("Invalid message role/content")
            if messages[-1]["role"] != "user":
                raise ValueError("Last message must be user; remove reference answers before inference")
            rows.append(dict(id=str(row.get("id", len(rows))), messages=messages))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Input must contain unique IDs and at least one prompt")
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3-4B')
    parser.add_argument('--adapter')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt')
    group.add_argument('--input', help='JSONL with id and messages, ending in user')
    parser.add_argument('--system', default='You are a helpful assistant.')
    parser.add_argument('--output', required=True, help='New JSONL path; refuses existing output/partial/receipt')
    parser.add_argument('--quantization', choices=('none', 'int8', 'nf4'), default='none')
    parser.add_argument('--max-input-tokens', type=int, default=768)
    parser.add_argument('--max-new-tokens', type=int, default=80)
    args = parser.parse_args()
    if min(args.max_input_tokens, args.max_new_tokens) < 1:
        parser.error('Token budgets must be positive')
    rows = load_prompts(args.input) if args.input else [dict(id='0', messages=[
        dict(role='system', content=args.system), dict(role='user', content=args.prompt)])]
    if args.prompt is not None and not args.prompt.strip():
        parser.error('Prompt cannot be empty')
    output = Path(args.output)
    partial = output.with_suffix(output.suffix + '.partial')
    receipt = output.with_suffix(output.suffix + '.receipt.json')
    if any(p.exists() for p in (output, partial, receipt)):
        raise FileExistsError('Existing inference output: inspect it or select a new path')
    from personal_llm_data.personal_runtime import gpu_guard
    with gpu_guard(Path(__file__).resolve().parents[1]):
        execute_inference(args, rows, output, partial, receipt)


def execute_inference(args, rows, output, partial, receipt):
    # Delay GPU imports/loading until CLI and input validation complete.
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from personal_llm_data.model_loading import load_causal_model
    from evaluate_generation import generate_batch
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # Free input rejects overlength prompts instead of silently truncating them.
    for row in rows:
        rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer.encode(rendered, add_special_tokens=False)) > args.max_input_tokens:
            raise ValueError(f"Prompt {row['id']} exceeds token budget; shorten it explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the destination before loading weights; interrupted work remains reviewable.
    with partial.open('x', encoding='utf-8') as stream:
        model = load_causal_model(args.model, args.quantization)
        if args.quantization == 'none':
            model = model.cuda()
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
        model.eval()
        started = time.time()
        for row in rows:
            prediction = generate_batch(model, tokenizer, [row['messages']],
                                        args.max_input_tokens, args.max_new_tokens)[0]
            stream.write(json.dumps(dict(id=row['id'], prediction=prediction), ensure_ascii=False)+'\n')
            stream.flush()
        torch.cuda.synchronize()
    partial.replace(output)
    receipt.write_text(json.dumps(dict(arguments=vars(args), samples=len(rows),
        input_sha256=digest(args.input) if args.input else None,
        adapter_sha256=digest(Path(args.adapter)/'adapter_model.safetensors') if args.adapter else None,
        output_sha256=digest(output), elapsed_seconds=time.time()-started,
        decoding='greedy; enable_thinking=False; overlength=error'), indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} predictions: {output}')


if __name__ == '__main__':
    main()
