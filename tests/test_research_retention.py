"""Temporary analysis-retention fixtures only; no model/data training or production cleanup."""
from contextlib import ExitStack,nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import compact_research_grid as retention
import run_research_analysis as controller
import research_grid as grid
from personal_llm_data.banking77 import LABELS,score_intents

class RetentionTests(unittest.TestCase):
    def fixture(self,directory,representative=False):
        root=Path(directory);config=grid.config('qwen3_8b','nf4',16 if representative else 4,32 if representative else 8,20261113)
        control=root/'experiments/research_banking_analysis_v1';run=root/config['output_dir'];run.mkdir(parents=True)
        cfg=root/grid.config_path(config);cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps(config))
        def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        records=[dict(id=f'fixture-{index}',messages=[dict(role='assistant',content=json.dumps(dict(label=LABELS[index%10])))]) for index in range(400)]
        predictions=[dict(id=row['id'],reference=row['messages'][-1]['content'],prediction=row['messages'][-1]['content']) for row in records]
        test=root/config['data_dir']/'test.jsonl';test.parent.mkdir(parents=True,exist_ok=True);test.write_text(''.join(json.dumps(row)+'\n' for row in records))
        prediction=run/'evaluation/adapter.predictions.jsonl';prediction.parent.mkdir();prediction.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
        result=dict(best_step=16,best_dev_loss=.2,quality=score_intents(records,predictions))
        save(run/'audit.json',result);save(run/'run_inputs.json',{})
        save(run/'training_summary.json',dict(counts=dict(train=500,dev=100,test=400)))
        save(run/'run_status.json',dict(stage='completed_verified'))
        for step in (16,128):
            cp=run/'checkpoints'/f'checkpoint-{step}';cp.mkdir(parents=True)
            for name in retention.BINARY_NAMES:(cp/name).write_bytes(b'fixture bytes only')
            state=dict(global_step=step,log_history=[dict(step=i,loss=.1) for i in range(1,129)]+[dict(step=i,eval_loss=.2 if i==16 else .3) for i in range(16,129,16)])
            save(cp/'trainer_state.json',state);save(cp/'personal_checkpoint_receipt.json',dict(fixture=True));save(cp/'adapter_config.json',dict(r=config['lora_r']))
        adapter=run/'adapter/adapter_model.safetensors';adapter.parent.mkdir();adapter.write_bytes(b'fixture exported adapter')
        save(run/'adapter/adapter_config.json',dict(r=config['lora_r']))
        session=control/'attempts/fixture/1.finished.json';save(session,dict(status='completed'))
        save(run/'training_sessions.json',dict(files_sha256={str(session.relative_to(root)):retention.sha(session)},sessions=[]))
        stages=[]
        for suffix,outputs in [('train',[run/'training_summary.json',adapter]),('test',[prediction]),('audit',[run/'audit.json'])]:
            name=retention.condition_name(config)+'_'+suffix;command=['fixture',suffix];outputs=[str(path.relative_to(root)) for path in outputs]
            stages.append(dict(name=name,command=command,outputs=outputs))
            save(control/'receipts'/(name+'.json'),dict(command=command,inputs_sha256='frozen',outputs={relative:retention.sha(root/relative) for relative in outputs}))
        save(control/'manifest.json',dict(inputs_sha256='frozen',stages=stages))
        save(control/'origins'/(retention.condition_name(config)+'.json'),dict(config_sha256=retention.sha(cfg),run_path=config['output_dir'],inputs_sha256='frozen'))
        return root,control,run,cfg,result
    def context(self,root,control,result):
        stack=ExitStack();stack.enter_context(patch.object(retention,'ROOT',root));stack.enter_context(patch.object(retention,'CONTROL',control));stack.enter_context(patch.object(retention,'validate_config'));stack.enter_context(patch.object(retention,'inputs_snapshot',return_value={}));stack.enter_context(patch.object(retention,'full_audit',return_value=result));return stack
    def test_plan_does_not_delete_and_execute_preserves_reanalysable_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                plan=retention.compact(cfg)
                self.assertFalse(plan['executable']);self.assertTrue((run/'adapter/adapter_model.safetensors').exists())
                self.assertEqual(retention.compact(cfg,execute=True),result)
                self.assertFalse((run/'adapter/adapter_model.safetensors').exists())
                self.assertTrue((run/'evaluation/adapter.predictions.jsonl').exists())
                self.assertTrue((run/'checkpoints/checkpoint-128/trainer_state.json').exists())
                self.assertEqual(retention.verify_retained(cfg),result)
                self.assertEqual(retention.compact(cfg,execute=True),result)
    def test_predeclared_representative_export_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory,True)
            with self.context(root,control,result):
                retention.compact(cfg,execute=True)
                self.assertTrue((run/'adapter/adapter_model.safetensors').exists())
                self.assertFalse((run/'checkpoints/checkpoint-128/optimizer.pt').exists())
    def test_changed_metrics_or_unrecorded_missing_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                retention.compact(cfg,execute=True)
                (run/'evaluation/adapter.predictions.jsonl').write_text('changed')
                with self.assertRaises(ValueError):retention.verify_retained(cfg)
    def test_unowned_old_paused_and_changed_binary_are_never_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                (control/'origins'/(retention.condition_name(grid.load(cfg))+'.json')).unlink()
                with self.assertRaises(FileNotFoundError):retention.compact(cfg,execute=True)
                self.assertTrue((run/'checkpoints/checkpoint-128/optimizer.pt').exists())
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                (run/'run_status.json').write_text(json.dumps(dict(stage='paused_checkpoint_verified')))
                with self.assertRaises(ValueError):retention.compact(cfg,execute=True)
                self.assertTrue((run/'adapter/adapter_model.safetensors').exists())
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                plan=retention.build_plan(cfg);(run/'checkpoints/checkpoint-128/optimizer.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError):retention.apply_plan(cfg,plan)
                self.assertTrue((run/'adapter/adapter_model.safetensors').exists())
    def test_path_escape_is_rejected_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                plan=retention.build_plan(cfg);outside=root/'keep.txt';outside.write_text('preserve')
                plan['removed']['../../../../keep.txt']=dict(sha256=retention.sha(outside),bytes=outside.stat().st_size)
                with self.assertRaises(ValueError):retention.apply_plan(cfg,plan)
                self.assertEqual(outside.read_text(),'preserve');self.assertTrue((run/'adapter/adapter_model.safetensors').exists())
    def test_cleanup_interruption_requires_explicit_finish_and_keeps_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                original_unlink=Path.unlink;counter=[0]
                def interrupted(path,*args,**kwargs):
                    counter[0]+=1
                    if counter[0]==2:raise OSError('fixture simulated interruption')
                    return original_unlink(path,*args,**kwargs)
                with patch.object(Path,'unlink',interrupted):
                    with self.assertRaises(OSError):retention.compact(cfg,execute=True)
                self.assertTrue((run/'retention/plan.json').exists());self.assertFalse((run/'retention/complete.json').exists())
                with self.assertRaises(ValueError):retention.compact(cfg,execute=True)
                self.assertEqual(retention.compact(cfg,execute=True,finish=True),result)
    def test_missing_pruned_output_needs_exact_completed_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root,control,run,cfg,result=self.fixture(directory)
            with self.context(root,control,result):
                adapter=run/'adapter/adapter_model.safetensors';expected=retention.sha(adapter);relative=str(adapter.relative_to(root))
                retention.compact(cfg,execute=True);retention.verify_deleted_output(cfg,relative,expected)
                with self.assertRaises(ValueError):retention.verify_deleted_output(cfg,relative,'wrong')
                with self.assertRaises(ValueError):retention.verify_deleted_output(cfg,str((run/'unknown.pt').relative_to(root)),'wrong')
    def test_analysis_stages_keep_scientific_grid_and_add_retention_after_each_audit(self):
        stages=controller.stages('qwen3_8b','nf4',4)
        self.assertEqual(len(stages),102);self.assertEqual(sum(stage['kind']=='train' for stage in stages),25)
        for index,stage in enumerate(stages):
            if stage['kind']=='retain':self.assertEqual(stages[index-1]['kind'],'audit')
        full=controller.stages();self.assertEqual(len(full),4513)
        self.assertEqual(sum(stage['kind']=='retain' for stage in full),1125)
    def test_analysis_mode_rejects_old_manifest_without_touching_it(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(controller,'RUNS',Path(directory)),patch.object(controller,'lock',return_value=nullcontext()),patch.object(controller,'check') as check:
            path=Path(directory)/'manifest.json';path.write_text('old evidence')
            with self.assertRaisesRegex(ValueError,'Old full-retention'):controller.run()
            check.assert_not_called();self.assertEqual(path.read_text(),'old evidence')

if __name__=='__main__':unittest.main()
