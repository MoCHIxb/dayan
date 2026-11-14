"""CPU-only exact-SHA model recovery from explicit sources; no automatic retry."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REVISION='1cfa9a7208912126459214e8b04321603b3df60c'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def source_url(name, source):
    if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Expected a frozen top-level filename')
    if source=='modelscope' and name.endswith('.safetensors'):
        # ModelScope revision names are not the HF commit. Acceptance is by the
        # existing frozen full-file SHA, never by trusting the candidate branch.
        return f'https://modelscope.cn/models/Qwen/Qwen3-4B/resolve/master/{name}'
    endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
    if not endpoint.startswith('https://'):raise ValueError('HTTPS model endpoint required')
    return f'{endpoint}/Qwen/Qwen3-4B/resolve/{REVISION}/{name}'


@contextmanager
def transfer_lock(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'transfer.lock').open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def download_one(url, destination, expected, opener=urllib.request.urlopen):
    destination=Path(destination)
    if destination.exists():
        if digest(destination)!=expected:raise ValueError('Existing file differs; preserve and inspect: '+destination.name)
        return dict(status='verified_existing',sha256=expected)
    cache=destination.parent/'.cache/intenttune_download'
    cache.mkdir(parents=True,exist_ok=True)
    partial=cache/(destination.name+'.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    # An interrupted process may have received every byte before finalization.
    if offset and digest(partial)==expected:
        partial.rename(destination)
        return dict(status='verified_complete_partial',sha256=expected)
    request=urllib.request.Request(url,headers={'User-Agent':'IntentTune-fixed-asset-recovery','Range':f'bytes={offset}-','Accept-Encoding':'identity'})
    started=time.monotonic();last=started;received=0
    with opener(request,timeout=45) as response:
        if response.status==206:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if not match or int(match[1])!=offset:raise ValueError('Server returned an unexpected byte range; partial retained')
            end,total=int(match[2]),int(match[3])
            if end!=total-1 or offset>=total:raise ValueError('Server did not return the complete remaining range')
        elif response.status==200 and offset==0:
            length=response.headers.get('Content-Length')
            if not length or not length.isdigit():raise ValueError('Full transfer must declare Content-Length')
            total=int(length)
        else:
            raise ValueError('Server did not honor resume; partial retained without modification')
        with partial.open('ab' if partial.exists() else 'xb') as stream:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                stream.write(chunk);received+=len(chunk)
                now=time.monotonic()
                if now-last>=5:
                    stream.flush()
                    print(f'{destination.name}: {offset+received}/{total} bytes, {received/max(now-started,.001)/1048576:.2f} MiB/s',flush=True)
                    last=now
    if partial.stat().st_size!=total:raise ValueError('Incomplete transfer; partial retained for explicit resume')
    if digest(partial)!=expected:raise ValueError('Frozen SHA mismatch; keep partial for diagnosis, do not use this file')
    partial.rename(destination)
    return dict(status='downloaded_and_sha_verified',sha256=expected,bytes=total,resumed_bytes=offset)


def prepare(source='huggingface',download=False):
    reference=json.loads((ROOT/'repro/banking77_assets.json').read_text(encoding='utf-8'))
    model=ROOT/'models/Qwen3-4B'
    if not download:
        print(json.dumps(dict(executable=False,source=source,files=list(reference['model_sha256']),
                              next='--download explicitly fetches missing files; complete files are SHA-verified and skipped')))
        return
    if (ROOT/'experiments/intenttune_banking77_v1/manifest.json').exists():
        raise FileExistsError('Pipeline is frozen; inspect existing model bytes instead of running recovery')
    results={}
    with transfer_lock(model/'.cache/intenttune_download'):
        for name,expected in reference['model_sha256'].items():
            print('Checking '+name,flush=True)
            results[name]=download_one(source_url(name,source),model/name,expected)
    inventory={p.name for p in model.iterdir() if p.is_file()}
    if inventory!=set(reference['model_sha256']):raise ValueError('Unexpected model file inventory; keep files and inspect')
    print(json.dumps(dict(status='all_13_model_files_frozen_sha_verified',source=source,files=results,gpu_started=False)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=('huggingface','modelscope'),default='huggingface')
    p.add_argument('--download',action='store_true')
    args=p.parse_args(argv);prepare(args.source,args.download);return 0


if __name__=='__main__':raise SystemExit(main())
