"""Read-only inventory of the owner's VLM-safety project's upstream MMDS data."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

UPSTREAM = {
    "repo_id": "leost233/MMDS",
    "revision": "d335497ec3287607cfb5f64cf7f046298cda27b5",
    "file": "mmds.jsonl",
    "sha256": "6c09ce1a459b1d7c3ca7aeff9d9225361c0269851da04dad84eb8630aec5b0c7",
    "license": "CC-BY-NC-4.0",
    "url": "https://huggingface.co/datasets/leost233/MMDS",
}


def summarize(rows):
    ids=Counter(str(r.get("id")) for r in rows)
    result = dict(records=len(rows), split_counts=dict(Counter(r.get("set", "missing") for r in rows)),
        data_type_counts=dict(Counter(r.get("data_type", "missing") for r in rows)),
        user_rating_counts=dict(Counter(r.get("user_rating", "missing") for r in rows)),
        assistant_rating_counts=dict(Counter(r.get("assistant_rating", "missing") for r in rows)))
    result["source_id_audit"]=dict(unique_ids=len(ids), repeated_id_values=sum(n>1 for n in ids.values()),
        extra_rows_with_repeated_ids=sum(n-1 for n in ids.values()),
        record_key="Pinned source SHA256 plus physical record ordinal; raw id is not assumed unique")
    cross = Counter((r.get("user_rating", "missing"), r.get("assistant_rating", "missing")) for r in rows)
    result["user_assistant_rating_pairs"] = [dict(user_rating=u, assistant_rating=a, records=n)
        for (u,a),n in sorted(cross.items())]
    image_free = [r for r in rows if not any(t.get("image") for t in r["conversations"])]
    result["image_free"] = dict(records=len(image_free),
        data_type_counts=dict(Counter(r.get("data_type", "missing") for r in image_free)),
        split_counts=dict(Counter(r.get("set", "missing") for r in image_free)),
        assistant_only_records=sum(all(t.get("role") == "assistant" for t in r["conversations"]) for r in image_free),
        note="Empty image arrays do not prove textual self-sufficiency: masked/augmented records may still refer to images")
    result["records_with_images"] = len(rows)-len(image_free)
    result["source_attack_dimension_counts"] = dict(Counter(r.get("dimension") or "benign_or_unspecified" for r in rows))
    result["existing_supervised_fields"] = ["user_rating", "user_dimension", "user_rationale",
        "assistant_rating", "assistant_dimension", "assistant_rationale"]
    result["target_semantics"] = {
        "user_rating": "Risk of user-side request/history; a candidate attack-intent target",
        "assistant_rating": "Safety of source model replies; NOT the label of whether the user attacked",
        "rationales": "Existing source explanations; not newly authored safe answers or independent human-verified truth",
    }
    return result


def audit_source(project_root):
    root = Path(project_root)
    path = root/"data/raw/mmds/mmds.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM["sha256"]:
        raise ValueError("Local MMDS bytes differ from the verified upstream revision; inspect before proceeding")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    report = dict(schema_version=1, status="source_audited_modality_decision_pending", upstream=UPSTREAM,
                  local_source_sha256=digest, upstream_byte_match=True, inventory=summarize(rows))
    bank_path = root/"data/seeds/formal_v3_seed_bank.json"
    manifest_path = root/"data/processed/formal_v3_shards/manifest.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = bank["harmful"]+bank["benign"]
    source_ids=set()
    for seed in seeds:
        source=seed.get("source",{})
        if "MMDS" in source.get("datasets",[]) or source.get("dataset")=="MMDS":
            source_ids.update(str(value) for value in source.get("source_ids",[]) if value is not None)
    report["formal_v3"] = dict(trajectory_count=manifest["total_records"], seed_count=len(seeds),
        status=manifest["status"], known_mmds_source_ids=len(source_ids),
        decision="Do not use constructed/unreviewed trajectories as trusted SFT answers; preserve source project unchanged")
    report["selection_status"] = dict(training_records=0, dev_records=0, test_records=0,
        frozen=False, original_answers_generated=False, dataset_published=False, gpu_started=False,
        required_decision="Text-only Qwen3 versus multimodal Qwen-VL review")
    report["publication"] = dict(owner_authorized_selected_dataset_sync=True,
        upstream_license="CC-BY-NC-4.0; attribution/noncommercial/change notices required",
        scope="Only an explicitly selected/attributed example subset; no model weights or private predictions",
        code_license="Unassigned; dataset attribution does not license the project code")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args(argv)
    result=audit_source(args.project_root)
    text=json.dumps(result, ensure_ascii=False, indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(text)
        print(f"Wrote aggregate source audit: {args.output}; no training data selected")
    else:
        print(text,end="")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
