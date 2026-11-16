import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('gpu_diag',ROOT/'scripts/gpu_probe_research.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TokenizerFixture:
    def apply_chat_template(self,messages,**kwargs):return [1,2] if kwargs['add_generation_prompt'] else [1,2,3,4]
    def decode(self,ids,skip_special_tokens):return '{"label":"fixture"}'
class GPUDiagnosticContractTests(unittest.TestCase):
    # Test fixtures never enter experimental training data.
    def row(self):return dict(messages=[dict(role='system',content='instruction'),dict(role='user',content='request'),dict(role='assistant',content='{"label":"fixture"}')])
    def test_mask_only_original_target(self):
        ids,labels,prompt=m.supervised_ids(TokenizerFixture(),self.row(),'mistral7b')
        self.assertEqual((ids,labels,prompt),([1,2,3,4],[-100,-100,3,4],[1,2]))
    def test_target_mismatch_rejected(self):
        row=self.row();row['messages'][-1]['content']='changed'
        with self.assertRaises(ValueError):m.supervised_ids(TokenizerFixture(),row,'qwen3_8b')
    def test_boundary_mismatch_rejected(self):
        class Wrong(TokenizerFixture):
            def apply_chat_template(self,messages,**kwargs):return [1,2] if kwargs['add_generation_prompt'] else [3,4,5]
        with self.assertRaises(ValueError):m.supervised_ids(Wrong(),self.row(),'phi4mini')
    def test_diagnostic_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';m.save_new(p,{'step':1})
            with self.assertRaises(FileExistsError):m.save_new(p,{'step':2})
    def test_matrix_is_nine_finite_conditions(self):
        self.assertEqual(len(m.catalog()['models'])*len(m.MODES),9)
        self.assertEqual(m.MODES,('none','nf4','int8'))
