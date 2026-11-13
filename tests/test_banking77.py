"""Small unit fixtures verify contracts only; they are never training samples."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from personal_llm_data.banking77 import LABELS, source_record, score_intents, SIZES

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.object(sys,'path',[str(ROOT/'scripts'),*sys.path]):spec.loader.exec_module(module)
    return module


class Banking77Tests(unittest.TestCase):
    def test_source_text_and_label_preserved(self):
        raw=dict(text='  fixture card request\n',category=LABELS[0])
        row=source_record(raw,'train',7)
        self.assertEqual(json.loads(row['messages'][-2]['content'])['user_request'],raw['text'])
        self.assertEqual(json.loads(row['messages'][-1]['content']),dict(label=raw['category']))
        self.assertEqual(row['provenance']['csv_data_row'],7)
        with self.assertRaises(ValueError):source_record(dict(raw,category='unsupported'),'train',7)

    def test_duplicate_json_keys_are_invalid_not_repaired(self):
        row=source_record(dict(text='fixture',category=LABELS[0]),'test',1)
        prediction=dict(id=row['id'],reference=row['messages'][-1]['content'],
                        prediction='{"label":"'+LABELS[0]+'","label":"'+LABELS[0]+'"}')
        scores=score_intents([row],[prediction])
        self.assertEqual(scores['invalid_rate'],1)
        self.assertEqual(scores['accuracy'],0)

    def test_centroid_fits_only_training_and_uses_supported_labels(self):
        module=load('intent_baselines')
        training=[source_record(dict(text='alpha alpha orchard',category=LABELS[0]),'train',1),
                  source_record(dict(text='zebra zebra ocean',category=LABELS[1]),'train',2)]
        records=[source_record(dict(text='alpha orchard',category=LABELS[0]),'test',1)]
        predictions=module.centroid_predictions(training,records)
        self.assertEqual(json.loads(predictions[0]['prediction'])['label'],LABELS[0])
        changed=json.loads(json.dumps(records));changed[0]['messages'][-1]['content']=json.dumps(dict(label=LABELS[1]))
        self.assertEqual(module.centroid_predictions(training,changed)[0]['prediction'],predictions[0]['prediction'])

    def test_pipeline_has_finite_three_training_budget_and_separate_outputs(self):
        pipeline=load('run_intenttune_3090');stages=pipeline.stages()
        self.assertEqual([s['name'] for s in stages if s['name'].startswith('train_')],['train_500','train_200','train_50'])
        self.assertEqual(len(stages),22)
        self.assertEqual(len({s['name'] for s in stages}),len(stages))
        for stage in stages:
            self.assertTrue(all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in stage['outputs']))
            self.assertNotIn('run_peft_sequence.py',' '.join(stage['command']))

    def test_stage_receipt_cannot_accept_changed_output_or_missing_inventory(self):
        pipeline=load('run_intenttune_3090')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'score.json';path.write_text('{}')
            stage=dict(command=['python','fixture.py'],outputs=['score.json'])
            receipt=dict(planned_command=stage['command'],inputs_sha256={'code':'fixed'},outputs_sha256={'score.json':hashlib.sha256(path.read_bytes()).hexdigest()})
            with patch.object(pipeline,'OUT',root):
                pipeline.verify_receipt(receipt,stage,{'code':'fixed'})
                path.write_text('{"changed":true}')
                with self.assertRaises(ValueError):pipeline.verify_receipt(receipt,stage,{'code':'fixed'})
                with self.assertRaises(ValueError):pipeline.verify_receipt(dict(receipt,outputs_sha256={}),stage,{'code':'fixed'})

    def test_configurations_differ_only_in_data_and_output_and_keep_budget(self):
        configs=[json.loads((ROOT/f'configs/intenttune_v1/qwen3_4b_n{size}.json').read_text()) for size in SIZES]
        for size,config in zip(SIZES,configs):
            self.assertEqual(config['max_steps'],128)
            self.assertEqual(config['eval_steps'],16)
            self.assertEqual(config['data_dir'],f'data/processed/banking77_v1/n{size}')
            self.assertTrue(config['output_dir'].startswith('experiments/intenttune_banking77_v1/'))
            self.assertEqual({k:v for k,v in config.items() if k not in ('data_dir','output_dir')},
                             {k:v for k,v in configs[0].items() if k not in ('data_dir','output_dir')})

    def test_coordinator_runs_cuda_preflight_in_child_process(self):
        # A parent CUDA context would block each child through the existing GPU guard.
        pipeline=load('run_intenttune_3090')
        with patch.object(pipeline.subprocess,'check_output',return_value='{"gpu_checked":true}') as run,patch.object(pipeline,'gpu_guard') as guard:
            self.assertTrue(pipeline.gpu_preflight()['gpu_checked'])
            self.assertEqual(run.call_args.args[0][-1],'--gpu-check')
            guard.assert_not_called()

    def test_gpu_execution_refuses_local_windows_before_lock_or_model_load(self):
        pipeline=load('run_intenttune_3090')
        with patch.object(pipeline.platform,'system',return_value='Windows'),patch.object(pipeline,'pipeline_lock') as lock:
            with self.assertRaises(ValueError):pipeline.run()
            lock.assert_not_called()

    def test_resume_reuses_only_unchanged_completed_stage_and_never_reexecutes_it(self):
        pipeline=load('run_intenttune_3090')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stage=dict(name='fixture',command=['fixture'],outputs=['score.json'])
            def execute(*args,**kwargs):(root/'score.json').write_text('{}')
            with patch.object(pipeline,'OUT',root),patch.object(pipeline.platform,'system',return_value='Linux'),\
                 patch.object(pipeline,'pipeline_lock',return_value=nullcontext()),\
                 patch.object(pipeline,'inputs',return_value={'code':'fixed'}),\
                 patch.object(pipeline,'stages',return_value=[stage]),\
                 patch.object(pipeline,'gpu_preflight',return_value={'gpu_checked':True}),\
                 patch.object(pipeline.subprocess,'check_output',return_value='fixture-commit'),\
                 patch.object(pipeline.subprocess,'run',side_effect=execute) as run,\
                 patch.object(pipeline,'summarize',return_value={'status':'fixture_complete'}):
                pipeline.run()
                pipeline.run(resume=True)
                self.assertEqual(run.call_count,1)
                (root/'score.json').write_text('{"tampered":true}')
                with self.assertRaises(ValueError):pipeline.run(resume=True)
                self.assertEqual(run.call_count,1)

    def test_failed_stage_is_not_automatically_retried_on_resume(self):
        import subprocess
        pipeline=load('run_intenttune_3090')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stage=dict(name='fixture',command=['fixture'],outputs=['score.json'])
            with patch.object(pipeline,'OUT',root),patch.object(pipeline.platform,'system',return_value='Linux'),\
                 patch.object(pipeline,'pipeline_lock',return_value=nullcontext()),\
                 patch.object(pipeline,'inputs',return_value={'code':'fixed'}),\
                 patch.object(pipeline,'stages',return_value=[stage]),\
                 patch.object(pipeline,'gpu_preflight',return_value={'gpu_checked':True}),\
                 patch.object(pipeline.subprocess,'check_output',return_value='fixture-commit'),\
                 patch.object(pipeline.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['fixture'])) as run:
                with self.assertRaises(subprocess.CalledProcessError):pipeline.run()
                with self.assertRaises(ValueError):pipeline.run(resume=True)
                self.assertEqual(run.call_count,1)
                self.assertTrue((root/'attempts/fixture/1.stderr.log').exists())
