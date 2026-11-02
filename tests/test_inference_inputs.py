import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('inference', Path(__file__).resolve().parents[1]/'scripts/infer_qwen.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InferenceInputsTest(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.jsonl'
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
            return module.load_prompts(path)

    def test_unicode_separator_is_content_not_record_boundary(self):
        rows = self.load([dict(id='a', messages=[dict(role='user', content='a\u0085b')])])
        self.assertEqual(rows[0]['messages'][0]['content'], 'a\u0085b')

    def test_reference_answer_cannot_be_sent_as_last_message(self):
        with self.assertRaisesRegex(ValueError, 'Last message'):
            self.load([dict(messages=[dict(role='user', content='question'), dict(role='assistant', content='answer')])])

    def test_duplicate_id_is_rejected(self):
        row = dict(id='a', messages=[dict(role='user', content='prompt')])
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.load([row, row])

    def test_missing_messages_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load([dict(id='a')])

    def test_gpu_lock_precedes_model_loading(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'predictions.jsonl'
            with patch('sys.argv',['infer','--prompt','request','--output',str(output)]), \
                 patch('personal_llm_data.personal_runtime.gpu_guard',side_effect=RuntimeError('busy')), \
                 patch.object(module,'execute_inference') as execute:
                with self.assertRaisesRegex(RuntimeError,'busy'):
                    module.main()
                execute.assert_not_called()
            self.assertFalse(output.exists())
