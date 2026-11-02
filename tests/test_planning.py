import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from personal_llm_data import planning, workflow


def declared_spec():
    # Metadata fixture, never a training sample or claimed personal dataset.
    return dict(schema_version=1, goal="professional_expression", model="Qwen/Qwen3-4B",
                language="zh", train_records=50, dev_records=10, test_records=10,
                real_data_authorized=True, paired_responses_available=True,
                source_groups_disjoint=True, gpu_memory_gib=16)


class PlanningTest(unittest.TestCase):
    def test_valid_declaration_cannot_authorize_execution(self):
        spec = declared_spec()
        before = copy.deepcopy(spec)
        result = planning.plan_task(spec)
        self.assertEqual(spec, before)
        self.assertFalse(result["executable"])
        self.assertEqual(result["status"], "ready_for_data_and_hardware_audit")
        self.assertEqual(result["total_declared_labeled_records"], 70)
        self.assertIn("unvalidated", result["evidence_scope"])

    def test_blocked_missing_pairs_splits_and_budget(self):
        spec = declared_spec()
        spec.update(train_records=0, dev_records=0, test_records=0,
                    real_data_authorized=False, paired_responses_available=False,
                    source_groups_disjoint=False, gpu_memory_gib=None)
        result = planning.plan_task(spec)
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(len(result["blockers"]), 6)

    def test_strict_validation_and_unknown_fields(self):
        for key, value in [("train_records", True), ("dev_records", -1),
                           ("real_data_authorized", "yes"), ("gpu_memory_gib", float("nan")),
                           ("gpu_memory_gib", float("inf")), ("gpu_memory_gib", True),
                           ("goal", "knowledge_qa"), ("model", "arbitrary/model"),
                           ("extra", "ignored")]:
            with self.subTest(key=key, value=value):
                spec = declared_spec()
                spec[key] = value
                with self.assertRaises(ValueError):
                    planning.plan_task(spec)

    def test_insufficient_declared_memory_is_not_silently_supported(self):
        spec = declared_spec()
        spec["gpu_memory_gib"] = 8
        self.assertEqual(planning.plan_task(spec)["status"], "needs_input")

    def test_cli_no_subprocess_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"spec.json", Path(tmp)/"plan.json"
            source.write_text(json.dumps(declared_spec()), encoding="utf-8")
            with patch.object(workflow.subprocess, "run") as process:
                self.assertEqual(workflow.main(["plan", "--spec", str(source), "--output", str(output)]), 0)
                process.assert_not_called()
            previous = output.read_bytes()
            with self.assertRaises(SystemExit):
                planning.main(["--spec", str(source), "--output", str(output)])
            self.assertEqual(output.read_bytes(), previous)
