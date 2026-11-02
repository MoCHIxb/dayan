import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_peft_sequence as study


class PeftAuditTest(unittest.TestCase):
    def fixture(self, root, quant='none'):
        run=root/'run'; run.mkdir()
        c=dict(output_dir='run',quantization=quant)
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        summary=dict(config=c,train_records=500,eval_records=1500,completed_optimizer_steps=128,
            log_history=[dict(step=s,eval_loss=1.0 if s in (32,48) else 2.0) for s in range(16,129,16)],
            best_dev_loss=1.0,best_model_checkpoint='checkpoint-32',quantization=quant,
            quantized_linear_modules=0,trainable_parameters=12,elapsed_seconds=100,
            peak_gpu_memory_bytes=1024,peak_gpu_reserved_bytes=2048)
        save(run/'training_summary.json',summary)
        save(run/'checkpoints/checkpoint-128/trainer_state.json',dict(global_step=128))
        for path in (run/'adapter/adapter_model.safetensors',run/'checkpoints/checkpoint-32/adapter_model.safetensors'):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'weight-fixture')
        return c,summary

    def test_tied_minimum_uses_earliest_and_checks_weights(self):
        with tempfile.TemporaryDirectory() as d, patch.object(study,'ROOT',Path(d)):
            c,_=self.fixture(Path(d))
            self.assertEqual(study.audit(c)['best_step'],32)
            (Path(d)/'run/adapter/adapter_model.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): study.audit(c)

    def test_nf4_without_quantized_modules_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(study,'ROOT',Path(d)):
            c,_=self.fixture(Path(d),'nf4')
            with self.assertRaisesRegex(ValueError,'Quantization'): study.audit(c)

    def test_missing_dev_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(study,'ROOT',Path(d)):
            c,s=self.fixture(Path(d))
            s['log_history'].pop()
            (Path(d)/'run/training_summary.json').write_text(json.dumps(s),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Eight full dev'): study.audit(c)
