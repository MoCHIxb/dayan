"""Fixtures exercise guards only; no fixture is used to train a model."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from personal_llm_data import personal_data as data
from personal_llm_data.personal_runtime import checkpoint_inventory, gpu_guard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import train_personal_adapter as training
import evaluate_personal_adapter as evaluation


def records():
    return [dict(id=str(i), source_group=str(i), split="train" if i < 50 else ("dev" if i == 50 else "test"),
                 input=f"input fixture {i}", output=f"output fixture {i}") for i in range(52)]


class PersonalDataTests(unittest.TestCase):
    def test_real_pairs_required_and_no_files_written_on_bad_input(self):
        rows = records()
        rows[0].pop("output")
        with self.assertRaises(ValueError):
            data.validate_pairs(rows)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"not_created"
            with self.assertRaises(ValueError):
                data.prepare(Path(tmp)/"missing", out, authorized=False, near_duplicates_reviewed=True)
            self.assertFalse(out.exists())

    def test_group_and_pair_leakage_and_role_order_rejected(self):
        for change in ("group", "pair", "roles"):
            rows = records()
            if change == "group":
                rows[-1]["source_group"] = rows[0]["source_group"]
            elif change == "pair":
                rows[-1].update(input=rows[0]["input"], output=rows[0]["output"])
            else:
                rows[0]["messages"] = [dict(role="assistant", content="reference"), dict(role="user", content="prompt")]
            with self.subTest(change=change), self.assertRaises(ValueError):
                data.validate_pairs(rows)

    def test_full_round_trip_unicode_tamper_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/"pairs.jsonl", Path(tmp)/"prepared"
            rows = records()
            rows[0]["input"] = "a\u0085b"
            source.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
            report = data.prepare(source, out, authorized=True, near_duplicates_reviewed=True)
            self.assertEqual(report["counts"], dict(train=50, dev=1, test=1))
            self.assertEqual(data.verify_prepared(out)["train"][0]["messages"][0]["content"], "a\u0085b")
            with self.assertRaises(FileExistsError):
                data.prepare(source, out, authorized=True, near_duplicates_reviewed=True)
            with (out/"dev.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Changed prepared"):
                data.verify_prepared(out)

    def test_csv_quoting_and_missing_source_group(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"pairs.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "source_group", "split", "input", "output"])
                writer.writeheader()
                rows = records()
                rows[0]["input"] = "comma, and\nnewline"
                writer.writerows(rows)
            self.assertEqual(data.read_rows(source)[0]["input"], "comma, and\nnewline")
        rows = records()
        rows[0].pop("source_group")
        with self.assertRaises(ValueError):
            data.validate_pairs(rows)


class PersonalRunnerTests(unittest.TestCase):
    def test_bad_budget_numeric_value_or_unknown_field_rejected(self):
        path = Path(__file__).resolve().parents[1]/"configs/personal_adapter_v1.json"
        for key, value in [("max_steps", 127), ("eval_steps", True), ("learning_rate", float("nan")),
                           ("protocol_id", "peft_v1"), ("unknown", "ignored")]:
            config = json.loads(path.read_text())
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                training.validate_config(config)

    def test_completed_example_cannot_be_retrained_even_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output/"training_summary.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                training.execute(dict(output_dir=str(output)), {}, Path("missing"), Path("checkpoint-128"))

    def test_resume_cannot_overwrite_later_checkpoint_or_retrain_finished_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            later = output/"checkpoints/checkpoint-48"
            later.mkdir(parents=True)
            (later/"evidence").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "Later checkpoint"):
                training.validate_resume_position(output, 32, 128)
            self.assertEqual((later/"evidence").read_text(), "preserve")
            training.validate_resume_position(output, 48, 128)
            with self.assertRaisesRegex(ValueError, "already reached"):
                training.validate_resume_position(output, 128, 128)

    def test_default_and_help_do_not_execute_or_load_gpu(self):
        rows = {"train": [1]*50, "dev": [1], "test": [1]}
        config = Path(__file__).resolve().parents[1]/"configs/personal_adapter_v1.json"
        with (patch.object(training, "validate_config", return_value=rows),
              patch.object(training, "execute") as run, patch("builtins.print")):
            self.assertEqual(training.main(["--config", str(config)]), 0)
            run.assert_not_called()

    def test_missing_optimizer_rng_or_receipt_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (checkpoint/name).write_bytes(b"test-only fixture")
            (checkpoint/"trainer_state.json").write_text(json.dumps(dict(global_step=16)))
            inventory = checkpoint_inventory(checkpoint)
            self.assertEqual(inventory["step"], 16)
            with self.assertRaises(FileNotFoundError):
                training.verify_checkpoint(checkpoint, {})
            (checkpoint/"personal_checkpoint_receipt.json").write_text(json.dumps(dict(checkpoint=inventory, inputs={})))
            self.assertEqual(training.verify_checkpoint(checkpoint, {}), inventory)
            (checkpoint/"optimizer.pt").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                training.verify_checkpoint(checkpoint, {})

    def test_gpu_guard_rejects_existing_compute_and_releases_os_lock(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.check_output", return_value="123456\n"):
            with self.assertRaisesRegex(RuntimeError, "Another GPU"):
                with gpu_guard(tmp):
                    self.fail("Must not execute")
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.check_output", return_value=""):
            for _ in range(2):
                with gpu_guard(tmp):
                    pass

    def test_second_gpu_lock_is_rejected_while_first_is_held(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.check_output", return_value=""):
            with gpu_guard(tmp):
                with self.assertRaises(OSError):
                    with gpu_guard(tmp):
                        self.fail("Second GPU task must not start")

    def test_full_test_id_reference_order_is_enforced(self):
        rows = [dict(id="a", messages=[dict(content="prompt"), dict(content="reference")])]
        evaluation.verify_predictions(rows, [dict(id="a", reference="reference", prediction="output")])
        for predictions in ([], [dict(id="other", reference="reference", prediction="output")],
                            [dict(id="a", reference="wrong", prediction="output")]):
            with self.assertRaises(ValueError):
                evaluation.verify_predictions(rows, predictions)
