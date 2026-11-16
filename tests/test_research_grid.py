"""CPU-only full-grid protocol tests; transport/header fixtures never train models."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import research_grid as design
import run_research_grid as campaign
from train_research_grid import validate_config

class ResearchGridTests(unittest.TestCase):
    def test_extreme_finite_dev_loss_keeps_log_loss_without_invalid_json(self):
        from audit_research_grid import derived_perplexity
        self.assertEqual(derived_perplexity(0),dict(value=1.,status='finite'))
        overflow=derived_perplexity(1000)
        self.assertIsNone(overflow['value']);self.assertEqual(overflow['status'],'overflow_log_loss_retained')
        json.dumps(overflow,allow_nan=False)
        with self.assertRaises(ValueError):derived_perplexity(float('nan'))
    def test_full_factorial_has_all1125_unique_conditions(self):
        rows=list(design.configs())
        self.assertEqual(len(rows),1125)
        self.assertEqual(len({campaign.name(row) for row in rows}),1125)
        self.assertEqual(len({row['output_dir'] for row in rows}),1125)
        for model in design.spec()['models']:
            for mode in design.spec()['quantizations']:
                subset=[row for row in rows if row['model_key']==model and row['quantization']==mode]
                self.assertEqual(len(subset),125)
                self.assertEqual({(row['lora_r'],row['lora_alpha']) for row in subset},{(r,a) for r in (4,8,16,32,64) for a in (8,16,32,64,128)})
                self.assertTrue(all({row['seed'] for row in subset if row['lora_r']==r and row['lora_alpha']==a}==set(design.spec()['training_seeds']) for r in (4,8,16,32,64) for a in (8,16,32,64,128)))
    def test_block_and_full_counts_and_outputs(self):
        block=campaign.stages('qwen3_8b','nf4',4)
        self.assertEqual(len(block),77)
        self.assertEqual(sum(stage['kind']=='train' for stage in block),25)
        full=campaign.stages()
        self.assertEqual(len(full),3388)
        self.assertEqual(sum(stage['kind']=='train' for stage in full),1125)
        self.assertEqual(sum(stage['kind'] in ('test','baseline') for stage in full),1137)
        self.assertEqual(len({stage['name'] for stage in full}),3388)
        paths=[path for stage in full for path in stage['outputs']]
        self.assertEqual(len(paths),len(set(paths)))
        self.assertTrue(all('research_banking_v2' in path for path in paths))
    def test_schedule_is_deterministic_and_covers_whole_block(self):
        first=list(design.configs('mistral7b','int8',64))
        self.assertEqual(first,list(design.configs('mistral7b','int8',64)))
        self.assertEqual(len(first),25)
        self.assertNotEqual([(row['lora_alpha'],row['seed']) for row in first],sorted((row['lora_alpha'],row['seed']) for row in first))
    def test_out_of_grid_or_changed_scientific_config_rejected(self):
        with self.assertRaises(ValueError):design.config('qwen3_8b','nf4',128,32,20261113)
        condition=design.config('qwen3_8b','nf4',64,128,20261113)
        changed=copy.deepcopy(condition);changed['max_steps']=64
        with self.assertRaises(ValueError):validate_config(changed)
        changed=copy.deepcopy(condition);changed['output_dir']='experiments/research_banking_v1/overwrite'
        with self.assertRaises(ValueError):validate_config(changed)
    def test_five_seeds_have_sample_sd_and_incomplete_cells_no_mean(self):
        self.assertEqual(campaign.summary_statistics([1,2,3,4,5])['mean'],3)
        self.assertAlmostEqual(campaign.summary_statistics([1,2,3,4,5])['sample_sd'],2.5**.5)
        with self.assertRaises(ValueError):campaign.summary_statistics([1,2,3])
        records=[]
        for seed in design.spec()['training_seeds']:
            records.append(dict(model_key='qwen3_8b',quantization='nf4',rank=16,alpha=32,seed=seed,best_dev_loss=.2,quality=dict(macro_f1=.8,accuracy=.8,invalid_rate=0.,minimum_class_recall=.5)))
        complete=campaign.aggregate(records,{})[0]
        self.assertEqual(complete['status'],'complete5')
        self.assertEqual(complete['paired_seed_macro_f1_difference_vs_r16_a32']['mean'],0)
        partial=campaign.aggregate(records[:-1],{})[0]
        self.assertEqual(partial['status'],'incomplete_no_mean');self.assertNotIn('metrics',partial)
        with self.assertRaises(ValueError):campaign.aggregate(records+[records[0]],{})
    def test_pairwise_seed_differences_match_reference(self):
        records=[]
        for index,seed in enumerate(design.spec()['training_seeds']):
            for rank,alpha,offset in ((16,32,0),(64,128,.05)):
                records.append(dict(model_key='qwen3_8b',quantization='none',rank=rank,alpha=alpha,seed=seed,best_dev_loss=.2,quality=dict(macro_f1=.7+index*.01+offset,accuracy=.8,invalid_rate=0.,minimum_class_recall=.5)))
        cell=next(row for row in campaign.aggregate(records,{}) if row['rank']==64)
        self.assertAlmostEqual(cell['paired_seed_macro_f1_difference_vs_r16_a32']['mean'],.05)
        self.assertAlmostEqual(cell['paired_seed_macro_f1_difference_vs_r16_a32']['sample_sd'],0)
    def test_header_parameter_estimate_without_tensor_loading(self):
        header={'model.layers.0.q_proj.weight':dict(shape=[16,8]),'model.layers.0.v_proj.weight':dict(shape=[8,8]),'lm_head.weight':dict(shape=[128,8])}
        with tempfile.TemporaryDirectory() as directory:
            content=json.dumps(header).encode();Path(directory,'fixture.safetensors').write_bytes(struct.pack('<Q',len(content))+content)
            coefficient,counts=design.projection_coefficient(directory,['q_proj','v_proj'])
            self.assertEqual(coefficient,40);self.assertEqual(counts,dict(q_proj=1,v_proj=1))
            forecast=design.storage_forecast([dict(model_key='fixture',lora_r=64)],{'fixture':coefficient})
            self.assertEqual(forecast['estimated_retained_bytes'],28*40*64+64*1024**2)
            with self.assertRaises(ValueError):design.projection_coefficient(directory,['nonexistent'])
    def test_materialization_idempotent_and_does_not_overwrite(self):
        registry=(ROOT/'configs/research_models.json').read_text()
        with tempfile.TemporaryDirectory() as directory,patch.object(design,'ROOT',Path(directory)):
            path=Path(directory)/'configs/research_models.json';path.parent.mkdir(parents=True);path.write_text(registry)
            design.materialize();design.materialize()
            paths=list((Path(directory)/'experiments/research_banking_v2/configs').glob('*.json'))
            self.assertEqual(len(paths),1125)
            paths[0].write_text('{}')
            with self.assertRaises(ValueError):design.materialize()
            self.assertEqual(paths[0].read_text(),'{}')
    def test_disk_guard_blocks_without_deleting(self):
        from collections import namedtuple
        Usage=namedtuple('Usage','total used free')
        with tempfile.TemporaryDirectory() as directory,patch.object(campaign,'OUT',Path(directory)),patch.object(campaign.shutil,'disk_usage',return_value=Usage(100,99,1)):
            condition=design.config('qwen3_8b','nf4',64,128,20261113)
            evidence=Path(directory)/'keep';evidence.write_text('evidence')
            with self.assertRaisesRegex(RuntimeError,'Insufficient'):campaign.disk_guard([condition],{'qwen3_8b':100})
            self.assertEqual(evidence.read_text(),'evidence')
    def test_stage_receipt_detects_change(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(campaign,'ROOT',Path(directory)):
            path=Path(directory)/'result';path.write_text('original');stage=dict(command=['fixture'],outputs=['result'],config=None)
            receipt=dict(command=['fixture'],outputs={'result':campaign.sha(path)},inputs_sha256='frozen')
            campaign.verify_receipt(stage,receipt,'frozen')
            path.write_text('changed')
            with self.assertRaises(ValueError):campaign.verify_receipt(stage,receipt,'frozen')
    def test_partial_report_preserves_failed_state_and_writes_tsv(self):
        stage=dict(name='not_done',command=['fixture'],outputs=['not_done'],kind='cpu',config=None)
        with tempfile.TemporaryDirectory() as directory,patch.object(campaign,'ROOT',Path(directory)),patch.object(campaign,'OUT',Path(directory)/'campaign'),patch.object(campaign,'stages',return_value=[stage]):
            campaign.OUT.mkdir();current={'fixed':1}
            campaign.fresh(campaign.OUT/'manifest.json',dict(inputs=current,inputs_sha256=campaign.digest(current),stages=[stage]))
            campaign.write_json(campaign.OUT/'execution_state.json',dict(stage='failed'))
            campaign.report(current)
            self.assertEqual(campaign.load(campaign.OUT/'execution_state.json')['stage'],'failed')
            self.assertIn('\t',(campaign.OUT/'trends.tsv').read_text())
            self.assertFalse((campaign.OUT/'results.json').exists())
    def test_paused_training_sessions_accumulate_after_verified_resume(self):
        condition=design.config('qwen3_8b','nf4',4,8,20261113)
        cfg_path=design.config_path(condition)
        stage=dict(name=campaign.name(condition)+'_train',command=['python','fixture_train'],outputs=[condition['output_dir']+'/training_summary.json',condition['output_dir']+'/training_sessions.json'],kind='train',config=str(cfg_path))
        with tempfile.TemporaryDirectory() as directory,patch.object(campaign,'ROOT',Path(directory)),patch.object(campaign,'OUT',Path(directory)/'experiments/research_banking_v2'),patch.object(campaign,'lock',return_value=nullcontext()),patch.object(campaign,'check',return_value=({'fixed':1},{})),patch.object(campaign,'inputs',return_value={'fixed':1}),patch.object(campaign,'stages',return_value=[stage]),patch.object(design,'configs',return_value=[condition]),patch.object(campaign,'verify_large_probe'),patch.object(campaign,'disk_guard'),patch.object(campaign,'verify_checkpoint'),patch.object(campaign,'inputs_snapshot',return_value={}),patch.object(campaign,'report'),patch.object(campaign.subprocess,'run') as launch:
            root=Path(directory);run=root/condition['output_dir'];checkpoint=run/'checkpoints/checkpoint-16';checkpoint.mkdir(parents=True)
            config_file=root/cfg_path;config_file.parent.mkdir(parents=True);config_file.write_text(json.dumps(condition))
            probe_file=campaign.OUT/'rank64_probe/qwen3_8b_nf4/complete.json';probe_file.parent.mkdir(parents=True);probe_file.write_text('{}')
            def worker(command,**kwargs):
                if command[-1]=='check':return
                if '--resume-from-checkpoint' not in command:
                    campaign.write_json(run/'run_status.json',dict(stage='paused_checkpoint_verified',checkpoint=str(checkpoint),evidence=dict(step=16)))
                    campaign.write_json(run/'pause_metrics.16.json',dict(elapsed_seconds_this_session=1.,peak_allocated_bytes=100,peak_reserved_bytes=200))
                else:
                    campaign.write_json(run/'run_status.json',dict(stage='completed_verified'))
                    campaign.write_json(run/'training_summary.json',dict(elapsed_seconds_this_session=2.,peak_allocated_bytes=300,peak_reserved_bytes=400))
            launch.side_effect=worker
            campaign.run()
            self.assertEqual(campaign.load(campaign.OUT/'execution_state.json')['stage'],'paused_checkpoint_verified')
            campaign.run(resume=True,checkpoint=str(checkpoint))
            receipt=campaign.load(run/'training_sessions.json')
            self.assertEqual([session['status'] for session in receipt['sessions']],['paused','completed'])
            self.assertEqual(sum(session['metrics']['elapsed_seconds_this_session'] for session in receipt['sessions']),3.)
            for relative,expected in receipt['files_sha256'].items():self.assertEqual(campaign.sha(root/relative),expected)
    def test_failure_stops_and_resume_does_not_retry_worker(self):
        condition=design.config('qwen3_8b','nf4',4,8,20261113)
        stage=dict(name='fixture',command=['python','fixture_not_a_model'],outputs=['result'],kind='cpu',config=None)
        with tempfile.TemporaryDirectory() as directory,patch.object(campaign,'ROOT',Path(directory)),patch.object(campaign,'OUT',Path(directory)/'campaign'),patch.object(campaign,'lock',return_value=nullcontext()),patch.object(campaign,'check',return_value=({'fixed':1},{})),patch.object(campaign,'stages',return_value=[stage]),patch.object(design,'configs',return_value=[condition]),patch.object(campaign,'verify_large_probe'),patch.object(campaign,'disk_guard'),patch.object(campaign.subprocess,'run') as launch:
            launch.side_effect=[None,subprocess.CalledProcessError(1,stage['command']),None]
            with self.assertRaises(subprocess.CalledProcessError):campaign.run()
            self.assertEqual(campaign.load(campaign.OUT/'execution_state.json')['stage'],'failed')
            self.assertTrue((campaign.OUT/'attempts/fixture/1.failed.json').exists())
            with self.assertRaisesRegex(ValueError,'no automatic retry'):campaign.run(resume=True)
            self.assertEqual(launch.call_count,3)
            self.assertFalse((campaign.OUT/'attempts/fixture/2.started.json').exists())

if __name__=='__main__':unittest.main()
