"""Read-only CPU diagnostics; never imports a model backend or starts a job."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


def diagnose(root: Path, config: Path | None = None) -> dict:
    root = Path(root).resolve()
    checks = []

    def add(name, ok, detail):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))

    add("python", sys.version_info >= (3, 10), platform.python_version())
    add("source_checkout", (root / "scripts/train_personal_adapter.py").is_file(),
        "Workflow training commands require the source checkout and scripts directory")
    packages = {}
    for package in ("torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes", "psutil"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    # Package presence is informational: CPU-only data tools need none of these.
    if config is not None:
        path = Path(config)
        if not path.is_absolute():
            path = root / path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Configuration must be a JSON object")
            add("protocol", value.get("protocol_id") == "personal_adapter_example_v1",
                "Only the independent developer example protocol is supported")
            for key in ("data_dir", "model_name_or_path", "output_dir"):
                if not isinstance(value.get(key), str) or not value[key].strip():
                    raise ValueError(f"Missing path field: {key}")
            from personal_llm_data.personal_data import verify_prepared
            try:
                rows = verify_prepared(root / value["data_dir"])
                add("prepared_data", True, {key: len(items) for key, items in rows.items()})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # Do not echo source text or a potentially sensitive parser message.
                add("prepared_data", False, f"{type(exc).__name__}: run prepare-personal verification")
            model = root / value["model_name_or_path"]
            model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            if not isinstance(model_config, dict):
                raise ValueError("Model configuration must be a JSON object")
            add("model_type", model_config.get("model_type") == "qwen3", "Current runner supports Qwen3")
            add("weight_files_present", any(model.glob("*.safetensors")),
                "Presence only; completeness and SHA are NOT verified by doctor")
            output = root / value["output_dir"]
            add("fresh_output", not output.exists(),
                "Existing output must be audited or explicitly resumed; never automatically rerun")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            add("configuration", False, f"{type(exc).__name__}: check config and local assets")
    return dict(schema_version=1, status="cpu_checks_passed" if all(c["passed"] for c in checks) else "needs_attention",
                checks=checks, installed_packages=packages, gpu_checked=False,
                training_readiness_certified=False,
                next_step="Use train-personal --config PATH without --execute for runner validation; GPU execution needs separate explicit authorization and preflight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional developer example config; no training or downloads")
    args = parser.parse_args(argv)
    report = diagnose(Path(__file__).resolve().parents[2], args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "cpu_checks_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
