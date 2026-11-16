"""Import the exact existing LaMP transfer, never overwrite different data or trust tar paths."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[1]

def import_data(archive,root=ROOT):
    root=Path(root).resolve()
    reference=json.loads((root/'repro/local_assets.json').read_text(encoding='utf-8'))
    expected={k:v for k,v in reference['sha256'].items() if k.startswith('data/')}
    if not expected:raise ValueError('No frozen data inventory')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or set(names)!=set(expected)|{'private_transfer_manifest.json'}:raise ValueError('Unexpected or duplicate archive inventory')
        if any(not m.isfile() for m in members):raise ValueError('Only regular data files allowed')
        if any(m.size>128*1024*1024 or m.size<0 for m in members):raise ValueError('Unexpected member size')
        manifest=json.load(tar.extractfile('private_transfer_manifest.json'))
        if manifest!={'private':True,'files_sha256':expected}:raise ValueError('Transfer manifest differs from frozen real-data source')
        payloads={name:tar.extractfile(name).read() for name in expected}
    for name,payload in payloads.items():
        if hashlib.sha256(payload).hexdigest()!=expected[name]:raise ValueError('Input data SHA mismatch')
        target=root/name
        if not target.resolve().is_relative_to(root) or target.is_symlink():raise ValueError('Data path escapes workspace')
        if target.exists() and (not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected[name]):raise FileExistsError('Existing different data retained: '+name)
    for name,payload in payloads.items():
        target=root/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(payload)
    return dict(status='frozen_existing_lamp_data_imported',files_sha256=expected,model_loaded=False,gpu_started=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    print(json.dumps(import_data(a.archive)))
if __name__=='__main__':main()
