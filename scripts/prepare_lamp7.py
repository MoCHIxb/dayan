from __future__ import annotations

import argparse
import json

from personal_llm_data.lamp7 import prepare_lamp7


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official LaMP-7 without changing its splits")
    parser.add_argument("--raw-dir", default="data/raw/lamp7")
    parser.add_argument("--output-dir", default="data/processed/lamp7")
    parser.add_argument("--profile-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20251206)
    args = parser.parse_args()
    report = prepare_lamp7(args.raw_dir, args.output_dir, args.profile_k, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
