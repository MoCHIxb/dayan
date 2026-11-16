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

    def test_failure_evidence_retains_prior_model_result(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'preflight.json';done={'qwen3_8b':{'status':'checked'}}
            p=m.failure_evidence(output,'mistral7b',done,ValueError('tokenizer load failed'))
            saved=json.loads(p.read_text())
            self.assertEqual(saved['active_model'],'mistral7b')
            self.assertEqual(saved['completed_models'],done)
            self.assertFalse(saved['training_ready'])
            with self.assertRaises(FileExistsError):m.failure_evidence(output,'mistral7b',{},ValueError('retry'))
    def test_missing_mistral_dependency_fails_explicitly(self):
        from unittest.mock import patch
        from importlib.metadata import PackageNotFoundError
        with patch('importlib.metadata.version',side_effect=PackageNotFoundError('sentencepiece')):
            with self.assertRaisesRegex(RuntimeError,'requirements-tokenizers.lock'):m.tokenizer_dependencies('mistral7b')
    def test_other_models_do_not_require_sentencepiece_conversion(self):
        self.assertEqual(m.tokenizer_dependencies('qwen3_8b'),{})

    def test_mistral_last_message_system_template_regression(self):
        class LastMessageTemplate(TokenizerFixture):
            # Reproduces the official conditional behavior, not model inference.
            def apply_chat_template(self,messages,**kwargs):
                has_system=messages[0]['role']=='system'
                request=messages[1]['content'] if has_system else messages[0]['content']
                rendered=((messages[0]['content']+'\n\n') if has_system and messages[-1]['role']=='user' else '')+request
                prefix=[ord(c) for c in rendered]
                return prefix if kwargs['add_generation_prompt'] else prefix+[999,1000]
        row=dict(messages=[dict(role='system',content='CATALOG'),dict(role='user',content='REQUEST'),dict(role='assistant',content='{"label":"fixture"}')])
        with self.assertRaises(ValueError):m.encode_record(LastMessageTemplate(),row)
        self.assertEqual(m.encode_record(LastMessageTemplate(),row,model_key='mistral7b')['target_tokens'],2)
    def test_mistral_does_not_relax_original_label_validation(self):
        row=dict(messages=[dict(role='system',content='CATALOG'),dict(role='user',content='REQUEST'),dict(role='assistant',content='WRONG')])
        with self.assertRaisesRegex(ValueError,'supervision differs'):m.encode_record(TokenizerFixture(),row,model_key='mistral7b')
