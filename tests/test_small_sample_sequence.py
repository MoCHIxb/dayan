import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("small_sequence", Path(__file__).resolve().parents[1] / "scripts/run_small_sample_sequence.py")
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)


class SmallSampleSequenceTests(unittest.TestCase):
    def test_incomplete_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.json"
            output.write_text('{}')
            with patch.object(sequence, "execute") as execute:
                with self.assertRaises(FileNotFoundError):
                    sequence.task("evaluate_generation.py", [], output)
                execute.assert_not_called()
            self.assertEqual(output.read_text(), '{}')

    def test_matching_receipt_reuses_result_but_changed_result_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.json"
            output.write_text('{}')
            command = dict(script="evaluate_generation.py", script_sha256=sequence.sha(sequence.ROOT / "scripts/evaluate_generation.py"), arguments=[])
            sequence.write(output.with_suffix('.json.receipt.json'), dict(command=command, sha256=sequence.sha(output)))
            with patch.object(sequence, "execute") as execute:
                sequence.task("evaluate_generation.py", [], output)
                output.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "receipt"):
                    sequence.task("evaluate_generation.py", [], output)
                execute.assert_not_called()

    def test_failed_child_never_gets_success_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.json"
            with patch.object(sequence, "execute", side_effect=RuntimeError("failed child")):
                with self.assertRaises(RuntimeError):
                    sequence.task("evaluate_generation.py", [], output)
            self.assertFalse(output.with_suffix('.json.receipt.json').exists())
