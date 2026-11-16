import copy
import unittest
from personal_llm_data.research_chat import message_view,POLICIES

class ResearchChatPolicyTests(unittest.TestCase):
    # Transport/rendering fixtures only; never used as training examples.
    def messages(self):return [dict(role='system',content='CATALOG'),dict(role='user',content='REQUEST'),dict(role='assistant',content='TARGET')]
    def test_mistral_keeps_all_instruction_and_original_request(self):
        rows=self.messages();original=copy.deepcopy(rows);view=message_view(rows,'mistral7b')
        self.assertEqual(rows,original)
        self.assertEqual(view,[dict(role='user',content='CATALOG\n\nREQUEST'),dict(role='assistant',content='TARGET')])
    def test_generation_and_training_prompt_views_match(self):
        rows=self.messages()
        self.assertEqual(message_view(rows,'mistral7b')[:-1],message_view(rows[:-1],'mistral7b'))
    def test_other_model_content_stays_native(self):
        rows=self.messages()
        for key in ('qwen3_8b','phi4mini'):
            self.assertEqual(message_view(rows,key),rows)
            self.assertIsNot(message_view(rows,key)[0],rows[0])
    def test_multiturn_and_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):message_view(self.messages()*2,'mistral7b')
        with self.assertRaises(ValueError):message_view(self.messages(),'unknown')
    def test_policy_version_explicit(self):
        self.assertEqual(POLICIES['mistral7b'],'mistral_system_prefix_in_user_v1')
