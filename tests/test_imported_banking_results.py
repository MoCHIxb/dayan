"""Aggregate-count fixtures only; never training examples."""
import importlib.util
import json
from pathlib import Path
import unittest

from personal_llm_data.banking77 import LABELS
from personal_llm_data.classification import score_labels

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('analyze_banking77_results',ROOT/'scripts/analyze_banking77_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    records=[];predictions=[]
    for label in LABELS:
        for number in range(40):
            key=f'{label}/{number}';target=json.dumps(dict(label=label))
            records.append(dict(id=key,messages=[dict(role='assistant',content=target)]))
            predictions.append(dict(id=key,reference=target,prediction=target))
    predictions[0]['prediction']='invalid'
    return score_labels(records,predictions,LABELS)


class ImportedBankingResultsTests(unittest.TestCase):
    def test_counts_recompute_with_invalid_in_denominator(self):
        result=module.metrics(fixture())
        self.assertEqual(result['correct_records'],399)
        self.assertEqual(result['accuracy'],399/400)
        self.assertEqual(result['invalid_rate'],1/400)
        self.assertFalse(result['all_absolute_gates_passed'])

    def test_claimed_high_score_without_matching_confusion_is_rejected(self):
        score=fixture();score['macro_f1']=1.0
        with self.assertRaises(ValueError):module.metrics(score)

    def test_incomplete_class_coverage_is_rejected(self):
        score=fixture();score['confusion'][LABELS[0]][LABELS[0]]-=1
        with self.assertRaises(ValueError):module.metrics(score)

    def test_negative_or_boolean_confusion_counts_are_rejected(self):
        for value in (-1,True):
            score=fixture();score['confusion'][LABELS[0]][LABELS[1]]=value
            with self.assertRaises(ValueError):module.metrics(score)

    def test_unknown_extra_confusion_class_is_rejected(self):
        score=fixture();score['confusion'][LABELS[0]]['outside_catalog']=0
        with self.assertRaises(ValueError):module.metrics(score)
