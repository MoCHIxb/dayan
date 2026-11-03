import json,unittest
from personal_llm_data.classification import parse_label,score_labels

def records():return [dict(id=str(i),messages=[dict(role='assistant',content=json.dumps(dict(label=label)))]) for i,label in enumerate(('ham','spam','spam','ham'))]
def predictions(guesses):return [dict(id=r['id'],reference=r['messages'][-1]['content'],prediction=g) for r,g in zip(records(),guesses)]

class ClassificationTests(unittest.TestCase):
    def test_perfect(self):
        rows=records();score=score_labels(rows,[dict(id=r['id'],reference=r['messages'][-1]['content'],prediction=r['messages'][-1]['content']) for r in rows],['ham','spam'])
        self.assertEqual(score['accuracy'],1);self.assertEqual(score['macro_f1'],1)
    def test_invalid_is_error_and_not_safe_fallback(self):
        p=predictions(['{"label":"ham"}','not json','{"label":"ham"}','{"label":"spam"}'])
        score=score_labels(records(),p,['ham','spam'])
        self.assertEqual(score['accuracy'],.25);self.assertEqual(score['invalid_rate'],.25)
        self.assertEqual(score['per_class']['spam']['recall'],0)
        self.assertEqual(score['confusion']['ham']['spam'],1)
    def test_duplicate_unknown_extra_and_wrong_type_rejected(self):
        for text in ('{"label":"ham","label":"spam"}','{"label":true}','{"label":"unknown"}','{"label":"ham","why":"x"}','```{"label":"ham"}```'):
            self.assertIsNone(parse_label(text,['ham','spam']))
    def test_coverage_reference_and_vocabulary_guards(self):
        p=predictions(['{"label":"ham"}']*4)
        for bad in (p[:-1],p[::-1]):
            with self.assertRaises(ValueError):score_labels(records(),bad,['ham','spam'])
        p[0]['reference']='{"label":"spam"}'
        with self.assertRaises(ValueError):score_labels(records(),p,['ham','spam'])
        with self.assertRaises(ValueError):score_labels(records(),p,['ham','ham'])
