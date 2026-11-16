import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('qwen8first',ROOT/'scripts/run_qwen8b_banking500.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Qwen8FirstRunTests(unittest.TestCase):
    def test_finite_full_comparison_plan(self):
        stages=m.stages();self.assertEqual(len(stages),8)
        self.assertEqual([s['name'] for s in stages],['centroid','fewshot','train','evaluate_base','score_base','evaluate_adapter','score_adapter','audit'])
        self.assertEqual(len(set(p for s in stages for p in s['outputs'])),sum(len(s['outputs']) for s in stages))
        self.assertIn('run/checkpoints/checkpoint-128/personal_checkpoint_receipt.json',stages[2]['outputs'])
    def test_config_is_independent_and_full_budget(self):
        c=json.loads((ROOT/m.CONFIG).read_text())
        self.assertEqual((c['model_name_or_path'],c['max_steps'],c['eval_steps']),('models/Qwen3-8B',128,16))
        self.assertEqual(c['lora_r'],8);self.assertEqual(c['quantization'],'none')
        self.assertIn('cloud_8b_4090',c['output_dir'])
    def test_modified_stage_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)):
            p=Path(d)/'score.json';p.write_text('original')
            stage=dict(command=['fixed'],outputs=['score.json']);receipt=dict(command=['fixed'],inputs={'x':1},outputs={'score.json':m.sha(p)})
            m.verify_receipt(stage,receipt,{'x':1})
            p.write_text('changed')
            with self.assertRaises(ValueError):m.verify_receipt(stage,receipt,{'x':1})
    def test_receipt_does_not_allow_different_command_or_input(self):
        stage=dict(command=['fixed'],outputs=[])
        for receipt in (dict(command=['other'],inputs={},outputs={}),dict(command=['fixed'],inputs={'changed':1},outputs={})):
            with self.assertRaises(ValueError):m.verify_receipt(stage,receipt,{})
