import json
from pathlib import Path
import tempfile
import unittest

from personal_llm_data.training_protocol import budget_arguments, protect_output


class TrainingProtocolTests(unittest.TestCase):
    def test_legacy_keeps_epoch_defaults(self):
        self.assertEqual(budget_arguments({"num_train_epochs": 2, "eval_max_samples": 500}), {})

    def test_rejects_incomplete_dev_or_unevaluated_endpoint(self):
        config = dict(max_steps=128, eval_steps=16, save_steps=16, eval_max_samples=1500)
        self.assertEqual(budget_arguments(config)["eval_strategy"], "steps")
        for changes in ({"eval_max_samples": 500}, {"save_steps": 32}, {"max_steps": 127}, {"eval_steps": 0}):
            with self.assertRaises(ValueError):
                budget_arguments(config | changes)

    def test_preserves_existing_run_and_checks_resume_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            config = {"protocol_id": "test", "output_dir": str(output)}
            protect_output(config, None)
            checkpoint = output / "checkpoints/checkpoint-16"
            checkpoint.mkdir(parents=True)
            (output / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                protect_output(config, None)
            protect_output(config, str(checkpoint))
            with self.assertRaises(ValueError):
                protect_output(config | {"seed": 7}, str(checkpoint))
            with self.assertRaises(ValueError):
                protect_output(config, temp)


if __name__ == "__main__":
    unittest.main()
