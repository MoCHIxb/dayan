import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cloud_comparison as cloud


class CloudComparisonTests(unittest.TestCase):
    def test_default_does_not_launch_any_task(self):
        with patch.object(cloud.sys,'argv',['runner']),patch.object(cloud,'MANIFEST',Path('missing-cloud-manifest.json')),patch.object(cloud,'gpu_phase') as gpu,patch.object(cloud,'statistics_phase') as stats,patch('builtins.print'):
            cloud.main(); gpu.assert_not_called(); stats.assert_not_called()

    def test_all_twelve_configs_match_local_reference(self):
        reference={(r['variant'],r['seed']):r for r in cloud.read(cloud.ROOT/'results/local_500_reference.json')['runs']}
        values=list(cloud.configurations()); self.assertEqual(len(values),12)
        for variant,seed,_,config in values:
            self.assertEqual({k:v for k,v in config.items() if k not in ('output_dir','protocol_id')},reference[variant,seed]['config'])
            self.assertTrue(config['output_dir'].startswith('experiments/cloud_4090_v1/runs/'))

    def test_incomplete_output_never_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'result.json'; out.write_text('{}')
            with patch.object(cloud,'command') as command:
                with self.assertRaises(FileNotFoundError): cloud.task('paired_bootstrap.py',[],out)
                command.assert_not_called()
            self.assertEqual(out.read_text(),'{}')

    def test_receipt_detects_changed_sidecar(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cloud,'ROOT',Path(d)):
            out=Path(d)/'result.json'; side=Path(d)/'result.predictions.jsonl'
            out.write_text('{}'); side.write_text('prediction')
            cloud.write(out.with_suffix('.json.receipt.json'),dict(outputs_sha256={out.name:cloud.sha(out),side.name:cloud.sha(side)},inputs_sha256={}))
            cloud.verify_receipt(out)
            side.write_text('changed')
            with self.assertRaisesRegex(ValueError,'output changed'): cloud.verify_receipt(out)

    def test_receipt_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cloud,'ROOT',Path(d)):
            out=Path(d)/'result.json'; data=Path(d)/'input.jsonl'
            out.write_text('{}'); data.write_text('input')
            cloud.write(out.with_suffix('.json.receipt.json'),dict(outputs_sha256={out.name:cloud.sha(out)},inputs_sha256={data.name:cloud.sha(data)}))
            data.write_text('changed')
            with self.assertRaisesRegex(ValueError,'input changed'): cloud.verify_receipt(out)

    def test_source_bytes_match_local_asset_reference(self):
        spec=cloud.read(cloud.ROOT/'repro/local_assets.json')
        for name,digest in spec['sha256'].items():
            if name.startswith(('src/','scripts/')):
                self.assertEqual(cloud.sha(cloud.ROOT/name),digest,name)
