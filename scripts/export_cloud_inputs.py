"""Create a PRIVATE transfer archive of existing real data; never upload it to Git."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix+'.partial').exists(): raise FileExistsError('Archive exists')
    spec=json.loads((ROOT/'repro/local_assets.json').read_text(encoding='utf-8'))
    files=[name for name in spec['sha256'] if name.startswith('data/')]
    for name in files:
        if hashlib.sha256((args.local_root/name).read_bytes()).hexdigest()!=spec['sha256'][name]: raise ValueError(f'Data mismatch: {name}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_suffix(args.output.suffix+'.partial')
    with tarfile.open(partial,'x:gz') as archive:
        for name in files: archive.add(args.local_root/name,arcname=name,recursive=False)
        payload=json.dumps(dict(private=True,files_sha256={name:spec['sha256'][name] for name in files}),indent=2).encode()
        info=tarfile.TarInfo('private_transfer_manifest.json'); info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
    partial.rename(args.output)
    print(f'Private archive created: {args.output}. Keep it outside Git/public storage.')


if __name__=='__main__': main()
