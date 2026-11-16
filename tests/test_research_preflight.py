import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('cpu_preflight',ROOT/'scripts/preflight_research_models.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TokenizerFixture:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
        assert enable_thinking is False
        return [10,11] if add_generation_prompt else [10,11,12,13]
    def decode(self,ids,skip_special_tokens):return '{"label":"fixture"}'

class ResearchPreflightTests(unittest.TestCase):
    # These contract fixtures are never stored as training data.
    def row(self):return dict(messages=[dict(role='user',content='fixture'),dict(role='assistant',content='{"label":"fixture"}')])
    def test_native_target_boundary(self):
        stats=m.encode_record(TokenizerFixture(),self.row())
        self.assertEqual(stats,dict(prompt_tokens=2,target_tokens=2,sequence_tokens=4,last_token_id=13))
    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):m.encode_record(TokenizerFixture(),self.row(),max_length=3)
        with self.assertRaises(ValueError):m.encode_record(TokenizerFixture(),self.row(),max_new=1)
    def test_changed_label_supervision_rejected(self):
        r=self.row();r['messages'][-1]['content']='different'
        with self.assertRaises(ValueError):m.encode_record(TokenizerFixture(),r)
    def test_different_prompt_prefix_rejected(self):
        class Wrong(TokenizerFixture):
            def apply_chat_template(self,messages,**kwargs):return [1,2] if kwargs['add_generation_prompt'] else [3,4,5]
        with self.assertRaises(ValueError):m.encode_record(Wrong(),self.row())
