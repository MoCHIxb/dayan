import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("ai_safety_audit", Path(__file__).resolve().parents[1]/"scripts/audit_ai_safety_source.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SafetySourceAuditTests(unittest.TestCase):
    def test_repeated_upstream_ids_are_reported_without_discarding_records(self):
        row=dict(id=1,set="train",data_type="default",conversations=[dict(role="assistant",image=[])])
        report=audit.summarize([row,dict(row,data_type="Augmentaion")])
        self.assertEqual(report["records"],2)
        self.assertEqual(report["source_id_audit"]["unique_ids"],1)
        self.assertEqual(report["source_id_audit"]["extra_rows_with_repeated_ids"],1)

    def test_attack_intent_is_distinct_from_source_assistant_compliance(self):
        # Metadata fixture only; no generated training pair.
        rows=[dict(id=1,set="train",data_type="default",user_rating="Unsafe",assistant_rating="Safe",
                   dimension="Privacy Violation",conversations=[dict(role="user", image=["image.jpg"])])]
        report=audit.summarize(rows)
        self.assertEqual(report["user_assistant_rating_pairs"],[dict(user_rating="Unsafe",assistant_rating="Safe",records=1)])
        self.assertEqual(report["records_with_images"],1)

    def test_image_free_augmentation_is_not_claimed_as_independent_text_task(self):
        rows=[dict(id=1,set="train",data_type="Augmentaion",user_rating="null",assistant_rating="Unsafe",
                   conversations=[dict(role="assistant", image=[])])]
        report=audit.summarize(rows)
        self.assertEqual(report["image_free"]["records"],1)
        self.assertEqual(report["image_free"]["assistant_only_records"],1)
        self.assertIn("do not prove",report["image_free"]["note"])
