import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_llm_data.pipeline import (
    clean_text,
    convert_record,
    near_duplicate_similarity,
    pii_flags,
    segment_text,
    split_records,
    validation_errors,
)


def _record(record_id: str, source: str) -> dict:
    return {
        "id": record_id,
        "source": source,
        "text": "这是一段完整且足够长的目标风格文本，用于测试数据管线。",
        "keep": True,
        "style_label": "narrative_calm",
        "topic_label": "fiction",
        "quality": 5,
        "instruction": "请改写。",
        "input": "中性内容",
    }


class PipelineTests(unittest.TestCase):
    def test_clean_and_segment_chinese_text(self):
        raw = "第一句。 第二句！\n\n版权声明：删除\n\n第三句很长，包含足够多的信息用于测试。"
        cleaned = clean_text(raw, [r"^版权声明.*$"])
        self.assertNotIn("版权声明", cleaned)
        self.assertTrue(segment_text(cleaned, min_chars=4, max_chars=20))

    def test_validation_accepts_complete_record(self):
        self.assertEqual(validation_errors([_record("a", "one.txt")]), [])

    def test_group_split_has_no_source_leakage(self):
        rows = [
            _record("a1", "a.txt#0"),
            _record("a2", "a.txt#1"),
            _record("b1", "b.txt#0"),
            _record("c1", "c.txt#0"),
        ]
        splits = split_records(rows, seed=7)
        locations = {}
        for split_name, split_rows in splits.items():
            for row in split_rows:
                source = row["source"].split("#", 1)[0]
                locations.setdefault(source, set()).add(split_name)
        self.assertTrue(all(len(value) == 1 for value in locations.values()))

    def test_messages_conversion(self):
        converted = convert_record(_record("a", "one.txt"), "messages")
        self.assertEqual(converted["messages"][0]["role"], "user")
        self.assertEqual(converted["messages"][1]["role"], "assistant")
        json.dumps(converted, ensure_ascii=False)

    def test_near_duplicate_and_pii_flags(self):
        left = "这是一段用于测试近重复检测的中文文本，内容基本保持一致。"
        right = "这是一段用于测试近重复检测的中文文本，内容几乎保持一致。"
        self.assertGreater(near_duplicate_similarity(left, right), 0.5)
        flags = pii_flags("联系方式 test@example.com，电话 13800138000。")
        self.assertIn("email", flags)
        self.assertIn("phone_cn", flags)


if __name__ == "__main__":
    unittest.main()

