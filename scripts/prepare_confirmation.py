"""Freeze additional experiment configs and download public evaluation assets."""
import hashlib
import json
import io
from pathlib import Path
import zipfile

from datasets import load_dataset
from huggingface_hub import snapshot_download
import nltk
import requests


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
}
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main():
    target = ROOT / "experiments/confirmation"
    target.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / "configs/qwen3_4b_lamp7_full_lora_r16.json").read_text())
    configs = []
    for size in ("4B", "1.7B"):
        for seed in (20251206, 20251207, 20251208):
            config = dict(baseline, model_name_or_path=f"models/Qwen3-{size}", seed=seed,
                          eval_max_samples=1500,
                          output_dir=f"experiments/runs/confirm-qwen3-{size.lower()}-r16-a32-seed{seed}")
            path = ROOT / f"configs/confirm_qwen3_{size.lower()}_seed{seed}.json"
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError(f"Refusing to alter frozen config: {path}")
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            configs.append(str(path.relative_to(ROOT)))
    manifest = {"seeds": [20251206, 20251207, 20251208], "configs": configs,
                "full_dev_samples": 1500, "test_samples": 1496,
                "training_design": "same_recipe_rank16_alpha32; parameter_count_varies_by_model",
                "datasets": {}, "embedding_revision": EMBED_REVISION}
    for name, config, split, expected in (("allenai/ai2_arc", "ARC-Challenge", "test", 1172),
                                          ("openai/gsm8k", "main", "test", 1319),
                                          ("google/IFEval", None, "train", 541)):
        dataset = load_dataset(name, config, revision=REVISIONS[name], split=split,
                               cache_dir=str(ROOT / ".cache/datasets"))
        if len(dataset) != expected:
            raise ValueError(f"Unexpected benchmark size: {name} {len(dataset)}")
        path = target / (name.split("/")[-1] + ".jsonl")
        with path.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(dataset):
                stream.write(json.dumps({"row_index": index, **row}, ensure_ascii=False) + "\n")
        manifest["datasets"][name] = {"revision": REVISIONS[name], "config": config,
                                       "split": split, "samples": len(dataset),
                                       "fingerprint": dataset._fingerprint,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"Frozen {name} {len(dataset)} records", flush=True)
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2", revision=EMBED_REVISION,
                      local_dir=ROOT / "models/all-MiniLM-L6-v2", cache_dir=ROOT / ".cache/hub",
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
    nltk_dir = ROOT / ".cache/nltk"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    for resource in ("punkt", "punkt_tab"):
        # The downloader's URL guard rejects a loopback HTTP proxy on this host.
        # Fetch the exact official public resource and validate extraction paths.
        response = requests.get(f"https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{resource}.zip", timeout=120)
        response.raise_for_status()
        archive_dir = nltk_dir / "tokenizers"
        archive_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if any(not (archive_dir / entry.filename).resolve().is_relative_to(archive_dir.resolve())
                   for entry in archive.infolist()):
                raise ValueError("Unsafe archive member")
            archive.extractall(archive_dir)
        manifest.setdefault("nltk_resources", {})[resource] = hashlib.sha256(response.content).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Confirmation assets and configs frozen", flush=True)


if __name__ == "__main__":
    main()
