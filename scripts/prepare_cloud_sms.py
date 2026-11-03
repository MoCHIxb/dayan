"""Explicit fixed-version assets for the single SMS cloud run; never trains."""
import argparse,hashlib,json
from pathlib import Path
from build_sms_spam import ROOT,fetch,build


def canonicalize_manifest(path,expected_sha256,expected_metadata):
    """Transport newline differences only, and only to the frozen exact digest."""
    path=Path(path);original=path.read_bytes();before=hashlib.sha256(original).hexdigest()
    if json.loads(original)!=expected_metadata:raise ValueError('Manifest semantics differ; no automatic repair')
    if before==expected_sha256:return dict(changed=False,before_sha256=before,after_sha256=before)
    lf=original.replace(b'\r\n',b'\n')
    for candidate in (lf,lf.replace(b'\n',b'\r\n')):
        if hashlib.sha256(candidate).hexdigest()==expected_sha256:
            temporary=path.with_suffix('.transport.partial')
            with temporary.open('xb') as stream:stream.write(candidate)
            temporary.replace(path)
            return dict(changed=True,before_sha256=before,after_sha256=expected_sha256,operation='Newline transport only; metadata and all sample bytes unchanged')
    raise ValueError('Manifest differs beyond a verified newline transport; stop')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/cloud_sms_spam_v1/qwen3_4b.json')
    p.add_argument('--download-model',action='store_true');p.add_argument('--download-data',action='store_true')
    p.add_argument('--build-data',action='store_true');p.add_argument('--canonicalize-manifest',action='store_true')
    a=p.parse_args(argv)
    reference=json.loads((ROOT/'results/sms_spam_run_audit.json').read_text(encoding='utf-8'))
    config=json.loads(a.config.read_text(encoding='utf-8'));data=ROOT/config['data_dir'];model=ROOT/config['model_name_or_path']
    if not any((a.download_model,a.download_data,a.build_data,a.canonicalize_manifest)):
        print(json.dumps(dict(executable=False,training=False,config=str(a.config),next='Explicit --download-model, --download-data, --build-data; verified manifest transport runs after build'),indent=2));return 0
    if a.download_model:
        from huggingface_hub import snapshot_download
        spec=json.loads((ROOT/'repro/local_assets.json').read_text(encoding='utf-8'))['model']
        snapshot_download(spec['repo_id'],revision=spec['revision'],local_dir=model,allow_patterns=list(reference['inputs']['model_sha256']))
    if a.download_data:fetch(ROOT/'data/raw/sms_spam_v1')
    if a.build_data:build(ROOT/'data/raw/sms_spam_v1/SMSSpamCollection',data,model)
    if a.build_data or a.canonicalize_manifest:
        path=data/'manifest.json'
        expected=reference['inputs']['data_sha256']['manifest.json']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected and (ROOT/config['output_dir']).exists():
            raise ValueError('Run evidence exists; never mutate inputs after training starts')
        receipt=canonicalize_manifest(path,expected,reference['data_manifest'])
        print(json.dumps(receipt))
    print('Asset preparation only; next check_cloud_sms_assets.py; no model training started')
    return 0


if __name__=='__main__':raise SystemExit(main())
