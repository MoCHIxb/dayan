"""Freeze/check small-sample assets. Never starts a model or a queue."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from personal_llm_data.pipeline import read_jsonl
from personal_llm_data.training_protocol import budget_arguments

MANIFEST = ROOT / "experiments/refocused/small_sample_v1_manifest.json"
CONFIGS = ROOT / "configs/small_sample_v1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidates():
    base = json.loads((ROOT / "configs/qwen3_4b_lamp7_n50_lora_r16_a32.json").read_text(encoding="utf-8"))
    result = {}
    for n in (50, 100, 200, 500):
        for label, lr in (("lr1e4", 1e-4), ("lr5e5", 5e-5)):
            name = f"n{n}_{label}_seed20251206"
            result[f"{name}.json"] = base | {
                "protocol_id": "small_sample_v1", "train_file": f"data/processed/lamp7_bm25_k12/train_{n}.jsonl",
                "output_dir": f"experiments/runs/small-sample-v1-{name}", "eval_max_samples": 1500,
                "max_steps": 128, "eval_steps": 16, "save_steps": 16, "logging_steps": 1,
                "quantization": "none", "learning_rate": lr}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Create immutable first-stage configs/manifest once")
    args = parser.parse_args()
    planned = candidates()
    if args.freeze:
        if MANIFEST.exists() or any((CONFIGS / name).exists() for name in planned):
            raise FileExistsError("Protocol exists; check it rather than overwriting")
        CONFIGS.mkdir(parents=True, exist_ok=True)
        for name, config in planned.items():
            (CONFIGS / name).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        assets = [ROOT / "scripts/train_qwen_lora.py", ROOT / "src/personal_llm_data/training_protocol.py",
                  ROOT / "scripts/freeze_small_sample_protocol.py", ROOT / "docs/20_small_sample_protocol.md",
                  ROOT / "requirements-train.lock", ROOT / "requirements-eval.lock"]
        assets += [CONFIGS / name for name in planned]
        assets += [ROOT / f"data/processed/lamp7_bm25_k12/{name}.jsonl" for name in ("train_50", "train_100", "train_200", "train_500", "dev", "test")]
        assets += sorted(p for p in (ROOT / "models/Qwen3-4B").iterdir() if p.is_file())
        manifest = {"protocol_id": "small_sample_v1", "date": "2026-10-29", "status": "frozen_not_started",
            "stage1_configs": [str((CONFIGS / n).relative_to(ROOT)).replace('\\', '/') for n in planned],
            "sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in assets}}
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for rel, expected in manifest["sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"Frozen asset changed: {rel}")
    for name, expected in planned.items():
        config = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
        assert config == expected
        options = budget_arguments(config)
        # Exercise the installed HF argument validation on CPU, without loading weights.
        from transformers import TrainingArguments
        arguments = TrainingArguments(output_dir=config["output_dir"], use_cpu=True, report_to=[],
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, **options)
        assert arguments.max_steps == 128 and arguments.eval_steps == arguments.save_steps == 16
        train = read_jsonl(ROOT / config["train_file"])
        n = int(name.split('_')[0][1:])
        assert len(train) == len({str(r['id']) for r in train}) == n
    for split, n in (("dev", 1500), ("test", 1496)):
        rows = read_jsonl(ROOT / f"data/processed/lamp7_bm25_k12/{split}.jsonl")
        assert len(rows) == len({str(r['id']) for r in rows}) == n
    print(f"Verified {len(planned)} configs, full dev/test, and {len(manifest['sha256'])} frozen assets including model weights. No training started.")


if __name__ == "__main__":
    main()
