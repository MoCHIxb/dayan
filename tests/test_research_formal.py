"""CPU contract tests only; fixtures are never training data."""
import copy
from contextlib import nullcontext
import subprocess
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from personal_llm_data.research_sft import encode_row,ChatSFTDataset
from gpu_probe_research import supervised_ids
import run_research_banking as campaign
from audit_research_banking import scheduled_best,audit_evaluation

class TinyTokenizer:
    # Transport fixture: prefix token and two target tokens, no model or data.
    def apply_chat_template(self,messages,**kwargs):
        return [1,2] if kwargs['add_generation_prompt'] else [1,2,3,4]
    def decode(self,ids,**kwargs):return 'TARGET'

class ResearchFormalTests(unittest.TestCase):
    def row(self):return dict(id='fixture',messages=[dict(role='system',content='CATALOG'),dict(role='user',content='REQUEST'),dict(role='assistant',content='TARGET')])
    def test_native_mask_matches_diagnostic_for_all_models(self):
        row=self.row();before=copy.deepcopy(row)
        for key in campaign.KEYS:
            encoded,prompt=encode_row(TinyTokenizer(),row,key)
            full,labels,probe_prompt=supervised_ids(TinyTokenizer(),row,key)
            self.assertEqual((encoded['input_ids'],encoded['labels'],prompt),(full,labels,probe_prompt))
            self.assertEqual(labels,[-100,-100,3,4])
        self.assertEqual(row,before)
    def test_no_length_or_target_repair(self):
        with self.assertRaises(ValueError):encode_row(TinyTokenizer(),self.row(),'qwen3_8b',max_length=3)
        with self.assertRaises(ValueError):encode_row(TinyTokenizer(),self.row(),'qwen3_8b',max_new=1)
        bad=self.row();bad['messages'][-1]['content']='CHANGED'
        with self.assertRaises(ValueError):encode_row(TinyTokenizer(),bad,'qwen3_8b')
    def test_prefix_mismatch_rejected(self):
        class Broken(TinyTokenizer):
            def apply_chat_template(self,messages,**kwargs):return [1,2] if kwargs['add_generation_prompt'] else [1,5,3,4]
        with self.assertRaises(ValueError):encode_row(Broken(),self.row(),'mistral7b')
    def test_dataset_exposes_masked_records(self):
        data=ChatSFTDataset([self.row()],TinyTokenizer(),1536,'phi4mini')
        self.assertEqual(len(data),1);self.assertEqual(data[0]['labels'],[-100,-100,3,4])
    def test_finite_complete_plan_without_execution(self):
        plan=campaign.stages()
        self.assertEqual(len(plan),58)
        self.assertEqual(sum(s['kind']=='train' for s in plan),15)
        self.assertEqual(sum(s['kind'] in ('test','baseline') for s in plan),27)
        self.assertEqual(len({s['name'] for s in plan}),58)
        outputs=[p for s in plan for p in s['outputs']]
        self.assertEqual(len(outputs),len(set(outputs)))
        self.assertTrue(all('research_banking_v1' in p for p in outputs))
    def test_recipes_match_registry_and_are_independent(self):
        registry=json.loads((ROOT/'configs/research_models.json').read_text())
        for key in campaign.KEYS:
            for recipe in registry['recipes']:
                c=campaign.load(ROOT/campaign.config_path(key,recipe['id']))
                self.assertEqual((c['quantization'],c['lora_r'],c['lora_alpha']),(recipe['quantization'],recipe['rank'],recipe['alpha']))
                self.assertEqual(c['target_modules'],registry['models'][key]['lora_projections'])
                self.assertEqual((c['max_steps'],c['eval_steps'],c['seed']),(128,16,20261113))
                self.assertTrue(c['output_dir'].startswith('experiments/research_banking_v1/'))
    def history(self):return [dict(step=i,loss=.2) for i in range(1,129)]+[dict(step=i,eval_loss=.3 if i in (32,64) else .5) for i in range(16,129,16)]
    def test_earliest_minimum_selection(self):self.assertEqual(scheduled_best(self.history())['step'],32)
    def test_missing_duplicate_nonfinite_dev_rejected(self):
        history=self.history()
        for bad in (history[:-1],history+[dict(step=128,eval_loss=.1)],history+[dict(step=1,eval_loss=float('nan'))]):
            with self.assertRaises(ValueError):scheduled_best(bad)
        bad=copy.deepcopy(history);bad[-1]['eval_loss']=float('inf')
        with self.assertRaises(ValueError):scheduled_best(bad)
    def test_incomplete_training_log_rejected(self):
        with self.assertRaises(ValueError):scheduled_best(self.history()[1:])
    def test_receipt_rejects_command_or_output_changes(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'ROOT',Path(tmp)):
            (Path(tmp)/'a').write_text('ok');h=campaign.sha(Path(tmp)/'a')
            stage=dict(command=['python','safe'],outputs=['a']);receipt=dict(command=stage['command'],inputs={'fixed':1},outputs={'a':h})
            campaign.verify_receipt(stage,receipt,{'fixed':1})
            with self.assertRaises(ValueError):campaign.verify_receipt(stage,receipt,{'fixed':2})
            (Path(tmp)/'a').write_text('changed')
            with self.assertRaises(ValueError):campaign.verify_receipt(stage,receipt,{'fixed':1})
    def test_failed_worker_preserved_and_not_automatically_retried(self):
        stage=dict(name='fixture_stage',command=['python','fixture_only'],outputs=['output.json'],kind='cpu',config=None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'ROOT',Path(tmp)),patch.object(campaign,'OUT',Path(tmp)/'campaign'),patch.object(campaign,'lock',return_value=nullcontext()),patch.object(campaign,'check',return_value={'fixed':1}),patch.object(campaign,'stages',return_value=[stage]),patch.object(campaign.subprocess,'run') as launch:
            launch.side_effect=[None,subprocess.CalledProcessError(1,stage['command']),None]
            with self.assertRaises(subprocess.CalledProcessError):campaign.run()
            self.assertEqual(campaign.load(campaign.OUT/'execution_state.json')['stage'],'failed')
            self.assertTrue((campaign.OUT/'attempts/fixture_stage/1.stderr.log').exists())
            with self.assertRaisesRegex(ValueError,'no automatic retry'):campaign.run(resume=True)
            self.assertEqual(launch.call_count,3) # Two environment inventories, only one worker.
            self.assertFalse((campaign.OUT/'attempts/fixture_stage/2.started.json').exists())
    def test_between_stage_pause_does_not_start_worker(self):
        stage=dict(name='fixture_stage',command=['python','fixture_only'],outputs=['output.json'],kind='cpu',config=None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'ROOT',Path(tmp)),patch.object(campaign,'OUT',Path(tmp)/'campaign'),patch.object(campaign,'lock',return_value=nullcontext()),patch.object(campaign,'check',return_value={'fixed':1}),patch.object(campaign,'stages',return_value=[stage]),patch.object(campaign.subprocess,'run') as launch:
            campaign.OUT.mkdir();campaign.fresh(campaign.OUT/'pause.request',{'requested':True})
            campaign.run()
            self.assertEqual(campaign.load(campaign.OUT/'execution_state.json')['stage'],'paused_between_stages')
            self.assertEqual(launch.call_count,1) # Environment inventory only.
            self.assertFalse((campaign.OUT/'attempts').exists())
    def test_evaluation_auditor_rejects_stale_source_before_scoring(self):
        import audit_research_banking as audit
        with tempfile.TemporaryDirectory() as tmp,patch.object(audit,'ROOT',Path(tmp)),patch.object(audit,'validate_config'),patch.object(audit,'inputs_snapshot',return_value={'frozen':1}):
            root=Path(tmp);cfg=root/'config.json';cfg.write_text(json.dumps(dict(data_dir='data')))
            (root/'data').mkdir();(root/'data/test.jsonl').write_text('fixture only')
            out=root/'eval.json';out.write_text('{}');out.with_suffix('.receipt.json').write_text(json.dumps(dict(condition='zero',inputs={'frozen':1},source_sha256='wrong',config=str(cfg))))
            with self.assertRaisesRegex(ValueError,'inputs/condition'):audit_evaluation(cfg,'zero',out)

if __name__=='__main__':unittest.main()
