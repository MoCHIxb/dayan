import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_model_assets',ROOT/'scripts/prepare_model_assets.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Response(io.BytesIO):
    def __init__(self,payload,status=206,content_range='bytes 0-5/6'):
        super().__init__(payload);self.status=status;self.headers={'Content-Range':content_range,'Content-Length':str(len(payload))}


class ModelAssetRecoveryTests(unittest.TestCase):
    def test_complete_file_skips_network(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'part.safetensors';p.write_bytes(b'abcdef')
            def forbidden(*args,**kwargs):raise AssertionError('Network called for complete asset')
            result=module.download_one('https://fixture',p,hashlib.sha256(b'abcdef').hexdigest(),forbidden)
            self.assertEqual(result['status'],'verified_existing')

    def test_explicit_resume_matches_range_and_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'part.safetensors';cache=p.parent/'.cache/intenttune_download';cache.mkdir(parents=True)
            partial=cache/(p.name+'.partial');partial.write_bytes(b'abc')
            def opener(request,timeout):
                self.assertEqual(request.headers['Range'],'bytes=3-')
                return Response(b'def',content_range='bytes 3-5/6')
            result=module.download_one('https://fixture',p,hashlib.sha256(b'abcdef').hexdigest(),opener)
            self.assertEqual(result['resumed_bytes'],3)
            self.assertEqual(p.read_bytes(),b'abcdef')

    def test_ignored_resume_does_not_append_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'part.safetensors';cache=p.parent/'.cache/intenttune_download';cache.mkdir(parents=True)
            partial=cache/(p.name+'.partial');partial.write_bytes(b'abc')
            with self.assertRaises(ValueError):module.download_one('https://fixture',p,'0'*64,lambda *a,**k:Response(b'abcdef',status=200))
            self.assertEqual(partial.read_bytes(),b'abc');self.assertFalse(p.exists())

    def test_bad_hash_keeps_partial_and_never_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'part.safetensors'
            with self.assertRaises(ValueError):module.download_one('https://fixture',p,'0'*64,lambda *a,**k:Response(b'abcdef'))
            self.assertFalse(p.exists());self.assertEqual((p.parent/'.cache/intenttune_download'/ (p.name+'.partial')).read_bytes(),b'abcdef')

    def test_short_read_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'part.safetensors';calls=[]
            def opener(*args,**kwargs):calls.append(1);return Response(b'abc',content_range='bytes 0-5/6')
            with self.assertRaises(ValueError):module.download_one('https://fixture',p,'0'*64,opener)
            self.assertEqual(len(calls),1);self.assertFalse(p.exists())

    def test_existing_wrong_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'part.safetensors';p.write_bytes(b'wrong')
            with self.assertRaises(ValueError):module.download_one('https://fixture',p,'0'*64)
            self.assertEqual(p.read_bytes(),b'wrong')

    def test_modelscope_is_candidate_for_weights_only(self):
        self.assertIn('modelscope.cn',module.source_url('model-00001-of-00003.safetensors','modelscope'))
        self.assertIn(module.REVISION,module.source_url('tokenizer.json','modelscope'))
        with self.assertRaises(ValueError):module.source_url('../escape','modelscope')
