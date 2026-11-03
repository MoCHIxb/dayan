import importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class ClassificationAuditPathsTest(unittest.TestCase):
    def test_cli_relative_and_absolute_score_paths_have_same_public_key(self):
        spec=importlib.util.spec_from_file_location('classification_audit_paths',ROOT/'scripts/audit_classification_run.py')
        module=importlib.util.module_from_spec(spec)
        with patch.object(sys,'path',[str(ROOT/'scripts'),*sys.path]):spec.loader.exec_module(module)
        # Source-checkout CLI executes from repository root.
        self.assertEqual(module.public_relative_path(ROOT/'experiments/scores.json'),'experiments/scores.json')
        with patch('pathlib.Path.cwd',return_value=ROOT):
            self.assertEqual(module.public_relative_path(Path('experiments/scores.json')),'experiments/scores.json')
        with self.assertRaises(ValueError):module.public_relative_path(ROOT.parent/'outside.json')
