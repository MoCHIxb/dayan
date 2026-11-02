"""Full pinned ARC-Challenge, GSM8K and IFEval via official lm-eval scorers."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["NLTK_DATA"] = str(ROOT / ".cache/nltk")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/hub")

import langdetect
from lm_eval import simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.tasks._yaml_loader import load_yaml
import lm_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiments/confirmation/manifest.json").read_text())
    specifications = [("arc/arc_challenge.yaml", "allenai/ai2_arc", 1172),
                      ("gsm8k/gsm8k-cot-zeroshot.yaml", "openai/gsm8k", 1319),
                      ("ifeval/ifeval.yaml", "google/IFEval", 541)]
    tasks, task_hashes = [], {}
    for relative, repository, expected in specifications:
        path = Path(lm_eval.__file__).parent / "tasks" / relative
        config = load_yaml(path)
        config["dataset_kwargs"] = {"revision": manifest["datasets"][repository]["revision"],
                                     "cache_dir": str(ROOT / ".cache/datasets")}
        if "gsm8k" in relative:
            config.setdefault("generation_kwargs", {})["max_gen_toks"] = 1024
        task = list(get_task_dict([config]).values())[0]
        docs = list(task.test_docs())
        if len(docs) != expected:
            raise ValueError(f"Wrong full benchmark count: {relative} {len(docs)}")
        tasks.append(task)
        task_hashes[task.config.task] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Prepared {task.config.task}: {len(docs)}", flush=True)
    if args.prepare_only:
        return
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from personal_llm_data.model_loading import load_causal_model
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    langdetect.DetectorFactory.seed = 20251206
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = load_causal_model(args.model).cuda()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    wrapper = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1,
                   max_length=4096, truncation=False, enable_thinking=False,
                   mixed_precision_dtype="bfloat16", device="cuda")
    result = simple_evaluate(model=wrapper, tasks=tasks, num_fewshot=0,
                             batch_size=1, device="cuda", apply_chat_template=True,
                             bootstrap_iters=2000, log_samples=True,
                             random_seed=20251206, numpy_random_seed=20251206,
                             torch_random_seed=20251206, fewshot_random_seed=20251206)
    for task in tasks:
        if result["n-samples"][task.config.task]["effective"] != len(task.test_docs()):
            raise ValueError("A benchmark was not completely evaluated")
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = result.pop("samples")
    for name, rows in samples.items():
        with output.with_name(output.stem + f".{name}.samples.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    result["protocol"] = {"model": args.model, "adapter": args.adapter,
                          "lm_eval": importlib.metadata.version("lm_eval"), "task_yaml_hashes": task_hashes,
                          "dataset_revisions": manifest["datasets"], "enable_thinking": False,
                          "max_length": 4096, "num_fewshot": 0, "limit": None,
                          "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
