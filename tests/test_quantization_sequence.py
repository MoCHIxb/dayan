"""Failure containment for the expensive, one-shot experiment sequence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "quantization_sequence", Path(__file__).resolve().parents[1] / "scripts/run_quantization_sequence.py"
)
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)


class SequenceSafetyTests(unittest.TestCase):
    def test_resume_reuses_complete_predictions_without_running_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            run = Path(folder)
            output = run / "base_no_profile_test_metrics.json"
            metric = {"samples": 1496, "quantization": "int8", "do_sample": False,
                      "drop_profile": True, "batch_size": 4, "max_input_tokens": 768,
                      "max_new_tokens": 80, "adapter": None,
                      "model": str(sequence.ROOT / "models/Qwen3-4B")}
            output.write_text(json.dumps(metric), encoding="utf-8")
            prediction = output.with_suffix(".predictions.jsonl")
            prediction.touch()
            with (patch.object(sequence, "validate_predictions") as validate,
                  patch.object(sequence, "execute") as execute):
                sequence.evaluate_condition(run, "int8", "base_no_profile", resume=True)
                execute.assert_not_called()
                validate.assert_called_once_with(prediction)
                with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                    sequence.evaluate_condition(run, "nf4", "base_no_profile", resume=True)
                execute.assert_not_called()

    def test_resume_refuses_metrics_without_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            run = Path(folder)
            output = run / "base_profile_test_metrics.json"
            output.write_text("{}", encoding="utf-8")
            with patch.object(sequence, "execute") as execute:
                with self.assertRaisesRegex(FileExistsError, "Incomplete saved evaluation"):
                    sequence.evaluate_condition(run, "int8", "base_profile", resume=True)
                execute.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "{}")

    def test_failed_int8_audit_stops_before_nf4_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            (root / "experiments").mkdir()
            state = root / "experiments/state.json"
            with (patch.object(sequence, "ROOT", root),
                  patch.object(sequence, "STATE", state),
                  patch.object(sequence, "wait_for_training") as wait,
                  patch.object(sequence, "evaluate", side_effect=ValueError("incomplete summary")),
                  patch.object(sequence, "execute") as execute,
                  patch("builtins.print"),
                  patch.object(sequence.sys, "argv", ["sequence", "--training-pid", "123"])):
                with self.assertRaisesRegex(ValueError, "incomplete summary"):
                    sequence.main()
                wait.assert_called_once_with(123)
                execute.assert_not_called()
            self.assertEqual(json.loads(state.read_text())["stage"], "failed")
            self.assertFalse((root / "experiments/quantization_sequence.lock").exists())

    def test_existing_lock_prevents_second_sequence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            (root / "experiments").mkdir()
            lock = root / "experiments/quantization_sequence.lock"
            lock.write_text("999")
            with (patch.object(sequence, "ROOT", root),
                  patch.object(sequence, "wait_for_training") as wait,
                  patch.object(sequence.sys, "argv", ["sequence", "--training-pid", "123"])):
                with self.assertRaises(FileExistsError):
                    sequence.main()
                wait.assert_not_called()
            self.assertEqual(lock.read_text(), "999")

    def test_existing_log_prevents_command_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            log = Path(folder) / "prior.log"
            log.write_text("existing experiment evidence", encoding="utf-8")
            with patch.object(sequence, "status"), patch.object(sequence.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    sequence.execute("train_qwen_lora.py", [], log)
                run.assert_not_called()
            self.assertEqual(log.read_text(encoding="utf-8"), "existing experiment evidence")


if __name__ == "__main__":
    unittest.main()
