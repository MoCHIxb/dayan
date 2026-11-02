"""Build a 200-item stratified, real-data rating pack; never fabricate scores."""
import csv
import hashlib
import json
from pathlib import Path
import random

from personal_llm_data.pipeline import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "experiments/blind_review_200"
    if output.exists():
        raise FileExistsError("Blind pack exists; preserve the frozen sampling and mapping")
    records = read_jsonl(ROOT / "data/processed/lamp7_bm25_k12/test.jsonl")
    runs = ROOT / "experiments/runs"
    systems = {
        "base_profile": runs / "qwen3-4b-lamp7-full-lora-r16/baseline_test_metrics.predictions.jsonl",
        "bf16_profile": runs / "qwen3-4b-lamp7-full-lora-r16/adapter_profile_test_metrics.predictions.jsonl",
        "int8_profile": runs / "qwen3-4b-lamp7-full-lora-r16-a32-int8/adapter_profile_test_metrics.predictions.jsonl",
        "nf4_profile": runs / "qwen3-4b-lamp7-full-lora-r16-a32-nf4/adapter_profile_test_metrics.predictions.jsonl",
    }
    predictions = {name: {str(r['id']): r for r in read_jsonl(path)} for name, path in systems.items()}
    ids = {str(r["id"]) for r in records}
    for name, rows in predictions.items():
        if set(rows) != ids:
            raise ValueError(f"Prediction ID mismatch: {name}")
        if any(rows[str(r['id'])]['reference'] != r['messages'][-1]['content'] for r in records):
            raise ValueError("Reference mismatch")
    lengths = sorted(len(r["messages"][-1]["content"].split()) for r in records)
    edges = [lengths[round((len(lengths)-1)*q)] for q in (.25, .5, .75)]
    strata = {}
    for row in records:
        length = len(row["messages"][-1]["content"].split())
        size = row["profile_size_original"]
        group = (sum(length > edge for edge in edges), sum(size > edge for edge in (10, 20, 50)))
        strata.setdefault(group, []).append(row)
    allocation = {key: min(2, len(rows)) for key, rows in strata.items()}
    while sum(allocation.values()) < 200:
        key = max((key for key in strata if allocation[key] < len(strata[key])),
                  key=lambda key: (200*len(strata[key])/1496-allocation[key], key))
        allocation[key] += 1
    rng = random.Random(20251206)
    sampled = [row for key in sorted(strata) for row in rng.sample(strata[key], allocation[key])]
    rng.shuffle(sampled)
    delivery, private = output / "for_raters", output / "private_do_not_share"
    delivery.mkdir(parents=True)
    private.mkdir()
    mapping, pack, rating_rows = [], [], []
    for index, record in enumerate(sampled, 1):
        item = f"item-{index:03d}"
        names = list(systems)
        rng.shuffle(names)
        outputs = {}
        for label, name in zip("ABCD", names):
            outputs[label] = predictions[name][str(record["id"])]["prediction"]
            mapping.append({"item": item, "label": label, "system": name, "source_id": record["id"]})
            rating_rows.append({"item": item, "label": label,
                                "content_preservation": "", "style_similarity": "",
                                "fluency": "", "instruction_following": "", "error_tags": "", "notes": ""})
        pack.append({"item": item, "task": record["messages"][1]["content"].split("\n\nTask:\n", 1)[1],
                     "profile": [r["text"] for r in record["profile"]], "outputs": outputs})
    def dump_lines(path, rows):
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump_lines(delivery / "anonymous_items.jsonl", pack)
    dump_lines(private / "system_mapping.jsonl", mapping)
    for rater in (1, 2):
        with (delivery / f"rater_{rater}_scores.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rating_rows[0]))
            writer.writeheader()
            writer.writerows(rating_rows)
    markdown = ["# 匿名英文改写评价材料", "每题阅读 Task 和用户历史，然后独立评价 A/B/C/D；不猜模型身份。"]
    for row in pack:
        markdown += [f"\n## {row['item']}", "\nTask:\n\n"+row["task"], "\n用户历史："]
        markdown += [f"- {text}" for text in row["profile"]]
        markdown += [f"\n### 输出 {label}\n\n{text}" for label, text in row["outputs"].items()]
    (delivery / "anonymous_items.md").write_text("\n".join(markdown), encoding="utf-8")
    (delivery / "rating_instructions.md").write_text(
        "# 独立人工盲评说明\n\n两名评价者分别填写自己的 CSV，不先讨论。每题四个匿名输出，四维均填1–5整数；"
        "1=严重不符合，2=明显问题，3=部分符合，4=基本符合但有小问题，5=完全符合。\n\n"
        "- content_preservation：相对Task是否保留事实、实体、态度和约束。\n"
        "- style_similarity：相对用户历史的措辞、大小写、标点、句法与语气相似度，不能只看主题词。\n"
        "- fluency：通顺、自然、无不必要重复。\n"
        "- instruction_following：是否只给改写文本，无解释或额外内容。\n\n"
        "error_tags可填：语义改变、信息遗漏、实体错误、历史误复制、过度模仿、风格不足、"
        "长度异常、检索不相关、不流畅、格式错误。看不懂或无法判断的输出在notes注明，不猜分。"
        "评价中不要查看自动指标、参考答案或系统映射，不使用LLM代填。"
        "先保留两份独立原始评分，完成后再处理分歧。数据仅用于获准的研究评价，不公开转发原文。\n",
        encoding="utf-8")
    manifest = {"seed": 20251206, "samples": 200, "systems": 4, "ratings_per_rater": 800,
                "raters_required": 2, "scores_filled": False,
                "strata": {str(k): {"population": len(strata[k]), "sample": allocation[k]} for k in sorted(strata)},
                "sampling": "length_quartile_x_profile_count; min2_per_nonempty_stratum_then_proportional",
                "source_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in systems.items()}}
    (private / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created real-data pack: {len(pack)} items; 800 blank output ratings for each of two raters")


if __name__ == "__main__":
    main()
