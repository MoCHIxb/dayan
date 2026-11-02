"""Synthetic fixtures below validate code guards; none are training samples."""
import json
from pathlib import Path
import sys
import unittest
from personal_llm_data import text_safety as safety
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_text_safety_sets as builder

def beaver(safe):
    flags={key:False for key in safety.TAXONOMY}
    if not safe:flags['non_violent_unethical_behavior']=True
    return dict(prompt='financial fixture request',response='source response fixture',is_safe=safe,category=flags)

def row(key,prompt,safe):
    return dict(id=key,source_group='prompt-'+safety.text_hash(prompt),
        messages=[dict(role='user',content=json.dumps(dict(user_request=prompt,candidate_response='fixture'))),
                  dict(role='assistant',content=json.dumps(dict(is_safe=safe)))])

class TextSafetyTests(unittest.TestCase):
    def test_original_labels_map_without_new_answers_or_prompt_label_inference(self):
        source=beaver(True)
        source['prompt']='A request fixture that cannot determine response safety alone'
        result=safety.record_from_source(source,'financial','train',1,'pinned')
        self.assertEqual(json.loads(result['messages'][-1]['content']),dict(is_safe=True))
        self.assertEqual(json.loads(result['messages'][-2]['content'])['candidate_response'],source['response'])
        self.assertNotIn('is_safe',json.loads(result['messages'][-2]['content']))
        self.assertEqual(safety.existing_target(beaver(False)),dict(is_safe=False))

    def test_source_flag_category_inconsistency_is_not_relabelled(self):
        source=beaver(True);source['category']['privacy_violation']=True
        with self.assertRaises(ValueError):safety.existing_target(source)

    def test_medical_existing_labels_and_original_text_are_preserved(self):
        source=dict(context='medical question fixture',response='existing response fixture',category='Unauthorized Expertise',label='Unsafe')
        result=safety.record_from_diasafety(source,'train',1,'pinned')
        self.assertEqual(json.loads(result['messages'][-1]['content']),dict(is_safe=False))
        self.assertEqual(json.loads(result['messages'][-2]['content'])['user_request'],source['context'])
        self.assertEqual(result['provenance']['original_label'],'Unsafe')

    def test_domain_rules_reject_ambiguous_topic(self):
        self.assertEqual(safety.domain_for('financial investment'),'financial')
        self.assertIsNone(safety.domain_for('medical investment'))

    def test_automated_near_duplicate_audit_rejects_cross_split_copy(self):
        splits=dict(train=[row('a','a shared source question',True)],dev=[row('b','A shared source question',False)],test=[])
        with self.assertRaises(ValueError):safety.verify_automated_audit(splits,builder.AUDIT)

    def test_invalid_json_is_an_error_and_unsafe_false_negative(self):
        records=[row('a','first fixture',True),row('b','second fixture',False)]
        predictions=[dict(id='a',reference=records[0]['messages'][-1]['content'],prediction='{"is_safe":true}'),
                     dict(id='b',reference=records[1]['messages'][-1]['content'],prediction='not JSON')]
        result=safety.score_verdicts(records,predictions)
        self.assertEqual(result['invalid_rate'],0.5)
        self.assertEqual(result['accuracy'],0.5)
        self.assertEqual(result['per_class']['unsafe']['recall'],0)
        self.assertIsNone(safety.parse_verdict('{"is_safe":"true"}'))

    def test_incomplete_predictions_or_changed_gold_are_rejected(self):
        records=[row('a','fixture',True)]
        with self.assertRaises(ValueError):safety.score_verdicts(records,[])
        with self.assertRaises(ValueError):safety.score_verdicts(records,[dict(id='a',reference='changed',prediction='{}')])

    def test_balanced_selection_cannot_reuse_heldout_source_group(self):
        texts=['orchard apples harvest','ocean currents sailing','quantum optics photons','mountain hiking weather',
               'books poems literature','ceramic pottery kiln','orchestra violin concert','satellite orbital telemetry',
               'baking flour kitchen','botanical gardens flowers']
        rows=[row(str(i),text,bool(i%2)) for i,text in enumerate(texts)]
        heldout=builder.features(rows[:2])
        selected=builder.select(rows,4,1,heldout)
        self.assertEqual(len(selected),4)
        self.assertTrue(all(r['source_group'] not in heldout for r in selected))
