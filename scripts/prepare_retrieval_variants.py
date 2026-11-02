"""CPU dense/BM25/RRF retrieval with complete task retained under 768 tokens."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from personal_llm_data.lamp7 import SYSTEM_PROMPT, sanitize_text, _task_text, select_profile, build_user_message
from personal_llm_data.pipeline import write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "data/processed/lamp7_retrieval_confirmation"
    if (output / "manifest.json").exists():
        print("Frozen retrieval variants already exist", flush=True)
        return
    torch.set_num_threads(4)
    splits = {}
    texts = set()
    for split in ("dev", "test"):
        raw = ROOT / f"data/raw/lamp7/{split}"
        questions = json.loads((raw / f"{split}_questions.json").read_text(encoding="utf-8"))
        gold = json.loads((raw / f"{split}_outputs.json").read_text(encoding="utf-8"))["golds"]
        splits[split] = (questions, {str(r["id"]): sanitize_text(r["output"]) for r in gold})
        for question in questions:
            texts.add(_task_text(sanitize_text(question["input"])))
            texts.update(sanitize_text(p.get("text", "")) for p in question["profile"])
    ordered = sorted(texts)
    encoder_path = ROOT / "models/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    encoder = AutoModel.from_pretrained(encoder_path).cpu().eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(ordered), 64):
            batch = tokenizer(ordered[start:start+64], padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.extend(torch.nn.functional.normalize(mean, p=2, dim=1).numpy())
            if start % 6400 == 0:
                print(f"CPU embeddings {min(start+64,len(ordered))}/{len(ordered)}", flush=True)
    embedded = dict(zip(ordered, vectors))
    generation_tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-4B")
    variants = ["bm25_k4", "bm25_k8", "bm25_k12", "bm25_k20", "original_order_k12", "dense_k12", "hybrid_rrf_k12"]
    manifest = {"encoder": "all-MiniLM-L6-v2", "encoder_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "encoder_device": "cpu", "encoder_max_tokens": 256,
                "input_max_tokens": 768, "rrf_constant": 60, "variants": {},
                "scope": "inference_only_retrieval; original_order is NOT asserted chronological"}
    for split, (questions, outputs) in splits.items():
        rows = {name: [] for name in variants}
        pruned = dict.fromkeys(variants, 0)
        for question in questions:
            task = sanitize_text(question["input"])
            history = [{"id": str(p["id"]), "text": sanitize_text(p["text"])}
                       for p in question["profile"] if sanitize_text(p.get("text", ""))]
            bm25 = select_profile(question["profile"], task, len(history))
            bm_rank = {p["id"]: i+1 for i,p in enumerate(bm25)}
            query = embedded[_task_text(task)]
            similarity = np.asarray([embedded[p["text"]] @ query for p in history])
            dense_order = sorted(range(len(history)), key=lambda i: (-float(similarity[i]), i))
            dense = [history[i] for i in dense_order]
            dense_rank = {p["id"]: i+1 for i,p in enumerate(dense)}
            hybrid = sorted(history, key=lambda p: (-(1/(60+bm_rank[p["id"]])+1/(60+dense_rank[p["id"]])), bm_rank[p["id"]]))
            for name in variants:
                k = int(name.rsplit("k", 1)[1])
                ranked = bm25 if name.startswith("bm25") else dense if name.startswith("dense") else hybrid if name.startswith("hybrid") else history
                selected = ranked[:k]
                while True:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": build_user_message(task, selected)}]
                    count = len(generation_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
                    if count <= 768:
                        break
                    if not selected:
                        raise ValueError("Task alone exceeds the fixed input budget")
                    selected = selected[:-1]
                if len(selected) < min(k,len(history)):
                    pruned[name] += 1
                rows[name].append({"id": str(question["id"]), "official_split": split,
                                   "messages": messages + [{"role": "assistant", "content": outputs[str(question["id"])]}],
                                   "profile": selected, "profile_size_original": len(question["profile"]),
                                   "profile_size_selected": len(selected), "profile_retrieval": name,
                                   "rendered_input_tokens": count})
        expected = 1500 if split == "dev" else 1496
        for name in variants:
            path = output / name / f"{split}.jsonl"
            if len(rows[name]) != expected:
                raise ValueError("Retrieval changed the benchmark sample count")
            write_jsonl(rows[name], path)
            manifest["variants"].setdefault(name, {})[split] = {
                "samples": expected, "budget_pruned_records": pruned[name],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Seven full dev/test retrieval variants prepared on CPU", flush=True)


if __name__ == "__main__":
    main()
