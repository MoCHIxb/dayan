import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare8b',ROOT/'scripts/prepare_8b_4090.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Cloud8BPreparationTests(unittest.TestCase):
    def test_unsafe_filenames_rejected(self):
        for name in ('../config.json','a/b','a\\b','..',''):
            with self.assertRaises(ValueError):m.filename(name)
    def test_remote_lfs_digest_and_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture';p.write_bytes(b'asset')
            h=hashlib.sha256(b'asset').hexdigest()
            self.assertEqual(m.verify_file(p,dict(size=5,sha256=h)),h)
            with self.assertRaises(ValueError):m.verify_file(p,dict(size=5,sha256='0'*64))
            with self.assertRaises(ValueError):m.verify_file(p,dict(size=6,sha256=h))
            self.assertEqual(p.read_bytes(),b'asset')
    def test_git_blob_hash_is_not_plain_sha1(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config';p.write_bytes(b'abc')
            blob=hashlib.sha1(b'blob 3\0abc').hexdigest()
            self.assertEqual(m.verify_file(p,dict(size=3,sha256=None,blob_id=blob)),hashlib.sha256(b'abc').hexdigest())
            with self.assertRaises(ValueError):m.verify_file(p,dict(size=3,sha256=None,blob_id=hashlib.sha1(b'abc').hexdigest()))
    def test_lock_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock.json';m.save_new(p,dict(revision='original'))
            with self.assertRaises(FileExistsError):m.save_new(p,dict(revision='changed'))
            self.assertIn('original',p.read_text())
    def test_spec_is_preparation_only(self):
        s=m.read_spec()
        self.assertFalse(s['training_ready'])
        self.assertEqual(s['environment']['cuda'],'13.0')
        self.assertEqual(len(s['models']),2)
        self.assertEqual(len(s['recipes']),5)

    def test_candidate_mapping_and_path_validation(self):
        self.assertEqual(m.candidate_url('qwen3_8b','model.safetensors'),'https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model.safetensors')
        self.assertIn('LLM-Research/Meta-Llama-3-8B-Instruct',m.candidate_url('llama3_8b','part.safetensors'))
        for name in ('config.json','../model.safetensors'):
            with self.assertRaises(ValueError):m.candidate_url('qwen3_8b',name)
    def test_candidate_requires_official_sha(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.candidate_download('qwen3_8b','model.safetensors',Path(d)/'model.safetensors',dict(size=5,sha256=None))
    def test_candidate_transfer_bound_to_official_sha(self):
        import importlib.util
        from unittest.mock import patch
        import types
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.safetensors';expected=hashlib.sha256(b'asset').hexdigest();calls=[]
            def transfer(url,destination,sha):
                calls.append((url,sha));destination.write_bytes(b'asset')
            fake=types.SimpleNamespace(download_one=transfer)
            loader=types.SimpleNamespace(exec_module=lambda module:None)
            with patch.object(importlib.util,'spec_from_file_location',return_value=types.SimpleNamespace(loader=loader)),patch.object(importlib.util,'module_from_spec',return_value=fake):
                self.assertEqual(m.candidate_download('qwen3_8b',path.name,path,dict(size=5,sha256=expected)),expected)
            self.assertEqual(calls,[(m.candidate_url('qwen3_8b',path.name),expected)])
