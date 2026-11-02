import subprocess
import unittest
from unittest.mock import patch
from personal_llm_data import workflow


class WorkflowTest(unittest.TestCase):
    def test_forwarding_preserves_arguments_and_child_failure(self):
        with patch.object(workflow.subprocess,'run',return_value=subprocess.CompletedProcess([],9)) as run:
            result=workflow.main(['infer','--prompt','literal text with spaces','--output','result.jsonl'])
        self.assertEqual(result,9)
        argv=run.call_args.args[0]
        self.assertTrue(argv[2].endswith('infer_qwen.py'))
        self.assertEqual(argv[3:],['--prompt','literal text with spaces','--output','result.jsonl'])
        self.assertNotIn('shell',run.call_args.kwargs)

    def test_unknown_command_cannot_launch(self):
        with patch.object(workflow.subprocess,'run') as run:
            with self.assertRaises(SystemExit): workflow.main(['unknown'])
            run.assert_not_called()
