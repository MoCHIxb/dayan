"""CPU fixtures for explicit maintenance recovery; no GPU resume or production mutations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import recover_research_maintenance as recovery

class MaintenanceTests(unittest.TestCase):
    def test_progress_rollback_preserves_checkpoint_dev_and_drops_only_unsaved_tail(self):
        rows=[dict(step=i,logs=dict(loss=.1)) for i in range(1,89)]
        for step in (16,32,48,64,80):rows.append(dict(step=step,logs=dict(eval_loss=.2)))
        raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
        kept,last=recovery.canonical_progress(raw,80,[dict(step=step,eval_loss=.2) for step in (16,32,48,64,80)])
        self.assertEqual(last,88);self.assertTrue(all(row['step']<=80 for row in map(json.loads,kept.splitlines())))
        self.assertEqual(sum('loss' in row['logs'] for row in map(json.loads,kept.splitlines())),80)
    def test_dev_history_mismatch_stops_recovery(self):
        raw=json.dumps(dict(step=16,logs=dict(eval_loss=.3))).encode()+b'\n'
        with self.assertRaises(ValueError):recovery.canonical_progress(raw,16,[dict(step=16,eval_loss=.2)])
    def test_only_terminal_torn_log_line_can_be_ignored(self):
        row=json.dumps(dict(step=1,logs={})).encode()+b'\n'
        kept,last=recovery.canonical_progress(row+b'{torn',1,[])
        self.assertEqual(kept,row);self.assertEqual(last,1)
        with self.assertRaises(ValueError):recovery.canonical_progress(b'{torn\n'+row,1,[])
    def test_highest_valid_checkpoint_selected_and_partial_newer_one_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);cpdir=run/'checkpoints';cpdir.mkdir()
            for step in (16,80,96):
                path=cpdir/f'checkpoint-{step}';path.mkdir();(path/'trainer_state.json').write_text(json.dumps(dict(best_model_checkpoint=str(cpdir/'checkpoint-16'))))
            def verify(path,inputs):
                step=int(path.name.split('-')[-1])
                if step==96:raise ValueError('fixture incomplete state')
                return dict(step=step)
            with patch.object(recovery,'verify_checkpoint',side_effect=verify):
                step,path,errors=recovery.choose_checkpoint(run,{})
            self.assertEqual(step,80);self.assertIn('checkpoint-96',errors);self.assertTrue((cpdir/'checkpoint-96').exists())
    def test_saved128_does_not_launch_more_training(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);path=run/'checkpoints/checkpoint-128';path.mkdir(parents=True);(path/'trainer_state.json').write_text(json.dumps(dict(best_model_checkpoint=str(path))))
            with patch.object(recovery,'verify_checkpoint',return_value=dict(step=128)):
                with self.assertRaisesRegex(ValueError,'finalization'):recovery.choose_checkpoint(run,{})
    def test_live_workers_prevent_recovery(self):
        with patch.object(recovery.subprocess,'check_output',return_value='123 1 00:10 python scripts/train_research_grid.py --config fixture'):
            with self.assertRaisesRegex(RuntimeError,'Existing workers'):recovery.assert_no_workers()

if __name__=='__main__':unittest.main()
