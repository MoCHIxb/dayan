import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_llm_data.lamp7 import sanitize_text, select_profile


class Lamp7Tests(unittest.TestCase):
    def test_sanitize_redacts_identifiers_and_decodes_html(self):
        value = sanitize_text("Hi @alice &amp; mail a@example.com at https://example.com")
        self.assertEqual(value, "Hi <USER> & mail <EMAIL> at <URL>")

    def test_bm25_profile_selection_prefers_relevant_existing_text(self):
        profile = [
            {"id": "1", "text": "I like quiet mornings."},
            {"id": "2", "text": "This music album is excellent."},
        ]
        selected = select_profile(profile, "Rewrite this album review", 1)
        self.assertEqual(selected[0]["id"], "2")
        self.assertGreater(selected[0]["retrieval_score"], 0)


if __name__ == "__main__":
    unittest.main()
