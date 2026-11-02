from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_objective_supplement as evidence


class ObjectiveSupplementTests(unittest.TestCase):
    def test_derangement_is_fixed_complete_and_never_self(self):
        a=evidence.derangement(1496)
        self.assertEqual(a,evidence.derangement(1496))
        self.assertEqual(sorted(a),list(range(1496)))
        self.assertTrue(all(i!=j for i,j in enumerate(a)))

    def test_style_features_handle_empty_and_curly_apostrophe(self):
        self.assertTrue(all(v==0 for v in evidence.features('').values()))
        self.assertEqual(evidence.features('I’m ready')['contraction_rate'],0.5)

    def test_numbers_and_negation_are_explicit_flags(self):
        self.assertEqual(evidence.numbers('cost 3.50, count 2'),{'3.50','2'})
        self.assertTrue(evidence.negated('I don’t agree'))
        self.assertFalse(evidence.negated('I agree'))

    def test_task_prefix_extraction_preserves_inner_colon(self):
        r=dict(messages=[dict(role='user',content='History\n\nTask:\nRewrite: time: noon'),dict(role='assistant',content='reference')])
        self.assertEqual(evidence.input_text(r),'time: noon')

    def test_prediction_to_metric_path(self):
        self.assertEqual(evidence.metrics_path(Path('run/adapter_profile_test_metrics.predictions.jsonl')),
                         Path('run/adapter_profile_test_metrics.json'))
