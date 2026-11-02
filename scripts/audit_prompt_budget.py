"""CPU tokenizer audit of the ACTUAL historical train/generation truncation rules."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from personal_llm_data.pipeline import read_jsonl
from transformers import AutoTokenizer


def main():
    tok = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-4B", local_files_only=True)
    report = {"model": "Qwen3-4B", "budget": 768, "splits": {}}
    for split in ("train_50", "train_100", "train_200", "train_500", "dev", "test"):
        rows = read_jsonl(ROOT / f"data/processed/lamp7_bm25_k12/{split}.jsonl")
        results = []
        for row in rows:
            prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False)
            response = tok(row["messages"][-1]["content"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            # Train keeps the rightmost prompt tokens, generation historically keeps leftmost.
            retained_train = prompt[-max(768-len(response), 0):] if len(response) < 768 else []
            train_text = tok.decode(retained_train)
            generation_text = tok.decode(prompt[:768])
            task = row["messages"][-2]["content"].split("\n\nTask:\n", 1)[1]
            results.append({"id": str(row["id"]), "prompt_tokens": len(prompt), "response_tokens": len(response),
                "train_prompt_truncated": len(prompt)+len(response) > 768,
                "generation_prompt_truncated": len(prompt) > 768,
                "train_full_task_retained": task in train_text,
                "generation_full_task_retained": task in generation_text})
        report["splits"][split] = {"samples": len(rows), "max_prompt_tokens": max(r["prompt_tokens"] for r in results),
            **{k: sum(r[k] for r in results) for k in ("train_prompt_truncated", "generation_prompt_truncated",
                "train_full_task_retained", "generation_full_task_retained")}, "per_item": results}
        print(split, {k: v for k, v in report["splits"][split].items() if k != "per_item"}, flush=True)
    out = ROOT / "experiments/refocused/prompt_budget_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
