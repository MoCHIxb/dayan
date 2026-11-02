"""CPU-only task planning boundary for the future customization frontend.

This module validates declared metadata, not dataset truth or GPU feasibility.
It never imports training libraries, allocates CUDA, or launches a process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = {
    "schema_version", "goal", "model", "language", "train_records",
    "dev_records", "test_records", "real_data_authorized",
    "paired_responses_available", "source_groups_disjoint", "gpu_memory_gib",
}


def plan_task(spec: dict) -> dict:
    """Return a non-executable proposal; declarations require later file audits."""
    if not isinstance(spec, dict) or set(spec) != FIELDS:
        raise ValueError("Task specification must contain exactly the documented fields")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Unsupported task schema_version")
    if spec["goal"] != "professional_expression":
        raise ValueError("Only professional_expression is planned in this first interface")
    if spec["model"] != "Qwen/Qwen3-4B":
        raise ValueError("Only Qwen/Qwen3-4B has local research evidence")
    if type(spec["language"]) is not str or spec["language"] not in {"zh", "en", "other"}:
        raise ValueError("language must be zh, en, or other")
    for name in ("train_records", "dev_records", "test_records"):
        if type(spec[name]) is not int or spec[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("real_data_authorized", "paired_responses_available", "source_groups_disjoint"):
        if type(spec[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    memory = spec["gpu_memory_gib"]
    if memory is not None and (type(memory) not in (int, float) or not 0 < memory <= 1024):
        raise ValueError("gpu_memory_gib must be null or a finite positive number <= 1024")

    blockers = []
    if not spec["real_data_authorized"]:
        blockers.append("Confirm ownership/authorization and real-data provenance")
    if not spec["paired_responses_available"]:
        blockers.append("Obtain authentic task-response pairs; do not synthesize answers")
    if not 50 <= spec["train_records"] <= 500:
        blockers.append("First-case training budget must be 50–500 real pairs")
    if not spec["dev_records"] or not spec["test_records"]:
        blockers.append("Reserve independent labeled dev and test outside the training budget")
    if not spec["source_groups_disjoint"]:
        blockers.append("Split source groups and audit near duplicates before training")
    if memory is None:
        blockers.append("Measure actual GPU and free memory; declared capacity is insufficient")
    elif memory < 16:
        blockers.append("Below the locally tested 16 GiB GPU capacity; feasibility is unvalidated")

    return {
        "schema_version": 1,
        "status": "needs_input" if blockers else "ready_for_data_and_hardware_audit",
        "executable": False,
        "declared_task": dict(spec),
        "total_declared_labeled_records": sum(spec[k] for k in ("train_records", "dev_records", "test_records")),
        "blockers": blockers,
        "candidate_recipes": [
            {"id": "bf16_r8_a16", "r": 8, "alpha": 16, "quantization": "none"},
            {"id": "nf4_r16_a32", "r": 16, "alpha": 32, "quantization": "nf4"},
        ],
        "required_next_checks": [
            "Verify actual files, authorization, group splits, duplicates and hashes",
            "Check tokenizer lengths and task-specific content/terminology acceptance",
            "Measure GPU availability and dry-run memory; capacity is not free memory",
            "Freeze a separate personal-case protocol before executing any training",
        ],
        "evidence_scope": "Candidates from English LaMP-7 shared-adapter research; professional personal adaptation remains unvalidated",
        "hardware_scope": "Windows RTX5070Ti 16GiB tested; Linux/RTX4090 not yet executed",
        "selection_rule": "Select using dev only; test is held out; no recipe is a universal winner",
        "research_protocols": "small_sample_v1 and peft_v1 remain frozen and complete",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate declared task metadata and propose review steps; never train")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        result = plan_task(json.loads(args.spec.read_text(encoding="utf-8-sig")))
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0 if not result["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
