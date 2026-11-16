import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('research_deploy',ROOT/'scripts/deploy_research_models.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ResearchModelDeploymentTests(unittest.TestCase):
    def test_text_only_roster(self):
        self.assertEqual(m.selected(None),['qwen3_8b','mistral7b','phi4mini'])
        self.assertNotIn('llava15_7b',m.catalog()['models'])
        self.assertFalse(m.catalog()['training_ready'])
    def test_unknown_duplicate_and_empty_selection_rejected(self):
        for keys in ([],['llava15_7b'],['qwen3_8b','qwen3_8b']):
            with self.assertRaises(ValueError):m.selected(keys)
    def test_existing_qwen_lock_does_not_repin(self):
        import types
        with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)):
            p=Path(d)/'assets/qwen3_8b.json';p.parent.mkdir();p.write_text(json.dumps(dict(repo_id='Qwen/Qwen3-8B',revision='a'*40)))
            def forbidden(key):raise AssertionError('existing lock repinned')
            m.pin_once('qwen3_8b',types.SimpleNamespace(pin=forbidden))
            p.write_text(json.dumps(dict(repo_id='wrong')))
            with self.assertRaises(ValueError):m.pin_once('qwen3_8b',types.SimpleNamespace(pin=forbidden))
    def test_candidate_mapping_uses_exact_versions(self):
        backend=m.backend()
        self.assertIn('/mistralai/Mistral-7B-Instruct-v0.3/',backend.candidate_url('mistral7b','model.safetensors'))
        self.assertIn('/LLM-Research/Phi-4-mini-instruct/',backend.candidate_url('phi4mini','model.safetensors'))
    def test_failure_stops_before_next_model(self):
        from contextlib import nullcontext
        import types
        calls=[]
        def fail(key,download,source):calls.append(key);raise ValueError('hash mismatch')
        with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)),patch.object(m,'gpu_guard',return_value=nullcontext()),patch.object(m,'backend',return_value=types.SimpleNamespace(assets=fail)),patch.object(m,'pin_once'):
            with self.assertRaises(ValueError):m.deploy(['qwen3_8b','mistral7b'])
            self.assertEqual(calls,['qwen3_8b'])
            self.assertEqual(json.loads((Path(d)/'deployment_state.json').read_text())['status'],'failed_preserved_no_auto_retry')
