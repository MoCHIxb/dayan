import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.object(sys,'path',[str(ROOT/'scripts'),*sys.path]):spec.loader.exec_module(module)
    return module

class CloudSmsPreparationTests(unittest.TestCase):
    def test_newline_transport_requires_same_metadata_and_frozen_digest(self):
        module=load('prepare_cloud_sms');metadata={'counts':{'train':500}}
        lf=(json.dumps(metadata,indent=2)+'\n').encode();crlf=lf.replace(b'\n',b'\r\n');expected=hashlib.sha256(crlf).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'manifest.json';p.write_bytes(lf)
            self.assertTrue(module.canonicalize_manifest(p,expected,metadata)['changed'])
            self.assertEqual(p.read_bytes(),crlf)
            self.assertFalse(module.canonicalize_manifest(p,expected,metadata)['changed'])
            p.write_bytes(lf)
            with self.assertRaises(ValueError):module.canonicalize_manifest(p,expected,{'counts':{'train':501}})
            self.assertEqual(p.read_bytes(),lf)
            with self.assertRaises(ValueError):module.canonicalize_manifest(p,'a'*64,metadata)
            self.assertEqual(p.read_bytes(),lf)

    def test_cloud_output_is_separate_but_scientific_settings_identical(self):
        module=load('check_cloud_sms_assets')
        local=json.loads((ROOT/'configs/sms_spam_v1/qwen3_4b.json').read_text())
        cloud=dict(local,output_dir='experiments/cloud_sms_spam_v1/run')
        module.validate_cloud_config(local,cloud)
        with self.assertRaises(ValueError):module.validate_cloud_config(local,dict(cloud,learning_rate=.0002))
        with self.assertRaises(ValueError):module.validate_cloud_config(local,dict(cloud,output_dir=local['output_dir']))

    def test_restore_rejects_escaping_and_overwrite(self):
        module=load('restore_workspace_archive')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('../escape','/absolute','C:/escape','models/../escape'):
                with self.assertRaises(ValueError):module.inside(root,name)
            destination=root/'data/processed/lamp7';destination.mkdir(parents=True)
            record=dict(path='data/processed/lamp7',kind='archived_retired_dataset',archive='missing.zip',archive_sha256='a'*64)
            with self.assertRaises(FileExistsError):module.restore_record(record,root)

    def test_log_restoration_preserves_original_bytes(self):
        import gzip
        module=load('restore_workspace_archive');payload=b'original\r\nlog\x00\n'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'private/archive.gz';archive.parent.mkdir()
            archive.write_bytes(gzip.compress(payload))
            record=dict(path='experiments/runs/example/train.log',kind='archived_closed_log',archive='private/archive.gz',
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),sha256=hashlib.sha256(payload).hexdigest())
            module.restore_record(record,root)
            self.assertEqual((root/record['path']).read_bytes(),payload)
            with self.assertRaises(FileExistsError):module.restore_record(record,root)
