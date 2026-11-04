import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from personal_llm_data.doctor import diagnose
from personal_llm_data import workflow


class DoctorTest(unittest.TestCase):
    def test_cpu_only_does_not_launch_process_or_import_torch(self):
        import builtins
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft'}:
                raise AssertionError('Model backend imported')
            return original(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=guarded), patch('subprocess.run') as run:
            report = diagnose(Path(__file__).resolve().parents[1])
        self.assertEqual(report['status'], 'cpu_checks_passed')
        self.assertFalse(report['gpu_checked'])
        self.assertFalse(report['training_readiness_certified'])
        run.assert_not_called()

    def test_bad_config_reports_failure_without_echoing_contents_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'broken.json'
            config.write_text('{private raw text', encoding='utf-8')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            report = diagnose(root, config)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            self.assertEqual(report['status'], 'needs_attention')
            self.assertNotIn('private raw text', json.dumps(report))

    def test_existing_run_is_never_a_fresh_training_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            (root/'scripts/train_personal_adapter.py').touch()
            (root/'model').mkdir()
            (root/'model/config.json').write_text('{"model_type":"qwen3"}')
            (root/'model/shard.safetensors').touch()
            (root/'run').mkdir()
            config = root/'config.json'
            config.write_text(json.dumps(dict(protocol_id='personal_adapter_example_v1', data_dir='data',
                                             model_name_or_path='model', output_dir='run')))
            with patch('personal_llm_data.personal_data.verify_prepared', return_value={'train':[1], 'dev':[2], 'test':[3]}):
                report = diagnose(root, config)
            self.assertFalse(next(c for c in report['checks'] if c['name']=='fresh_output')['passed'])
            self.assertEqual(report['status'], 'needs_attention')

    def test_cli_routes_in_process(self):
        with patch('personal_llm_data.doctor.main', return_value=2) as main, patch.object(workflow.subprocess, 'run') as run:
            self.assertEqual(workflow.main(['doctor', '--config', 'literal path.json']), 2)
            main.assert_called_once_with(['--config', 'literal path.json'])
            run.assert_not_called()
