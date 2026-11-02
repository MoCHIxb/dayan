from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    convert_record,
    dataset_stats,
    export_annotation,
    import_annotation,
    prepare_records,
    read_jsonl,
    split_records,
    validation_errors,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="个性化 LLM 小样本数据工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="清洗、分段和精确去重")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config", required=True)

    export = subparsers.add_parser("export-annotation", help="导出人工标注 CSV")
    export.add_argument("--input", required=True)
    export.add_argument("--output", required=True)

    import_cmd = subparsers.add_parser("import-annotation", help="导入人工标注 CSV")
    import_cmd.add_argument("--input", required=True)
    import_cmd.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate", help="校验标注数据")
    validate.add_argument("--input", required=True)

    split = subparsers.add_parser("split", help="按来源分组划分数据")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--train", type=float, default=0.8)
    split.add_argument("--validation", type=float, default=0.1)
    split.add_argument("--test", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=20251206)

    convert = subparsers.add_parser("convert", help="转换训练数据格式")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--format", choices=("messages", "alpaca", "prompt-completion"), required=True)

    stats = subparsers.add_parser("stats", help="输出数据统计")
    stats.add_argument("--input", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        records = prepare_records(args.input, config)
        write_jsonl(records, args.output)
        print(f"已生成 {len(records)} 个候选片段: {args.output}")
    elif args.command == "export-annotation":
        records = read_jsonl(args.input)
        export_annotation(records, args.output)
        print(f"已导出 {len(records)} 条标注记录: {args.output}")
    elif args.command == "import-annotation":
        records = import_annotation(args.input)
        write_jsonl(records, args.output)
        print(f"已导入 {len(records)} 条标注记录: {args.output}")
    elif args.command == "validate":
        records = read_jsonl(args.input)
        errors = validation_errors(records)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
        print(f"校验通过: {len(records)} 条记录")
    elif args.command == "split":
        records = read_jsonl(args.input)
        errors = validation_errors(records)
        if errors:
            print("数据校验失败，请先修复：")
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
        splits = split_records(records, (args.train, args.validation, args.test), args.seed)
        output_dir = Path(args.output_dir)
        for name, rows in splits.items():
            write_jsonl(rows, output_dir / f"{name}.jsonl")
            print(f"{name}: {len(rows)}")
    elif args.command == "convert":
        records = read_jsonl(args.input)
        converted = [convert_record(record, args.format) for record in records if bool(record.get("keep", True))]
        write_jsonl(converted, args.output)
        print(f"已转换 {len(converted)} 条记录: {args.output}")
    elif args.command == "stats":
        print(json.dumps(dataset_stats(read_jsonl(args.input)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

