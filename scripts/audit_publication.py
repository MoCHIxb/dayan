"""Audit the source snapshot and the owner's narrow public dataset exception."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_DATA={f'examples/text_safety_v1/{d}/{s}.jsonl' for d in ('medical','financial','public_safety','joint') for s in ('train','dev','test')}
PRIVATE={'data','models','experiments','outputs','checkpoints','.venv','.codex','.agents'}
SECRET_PATTERNS=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',
                 rb'github_pat_[A-Za-z0-9_]{40,}',rb'AKIA[A-Z0-9]{16}',rb'sk-[A-Za-z0-9_-]{32,}']

def audit(update=False):
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    added=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    paths=sorted(set(filter(None,tracked+added)))
    for name in paths:
        p=Path(name)
        if any(part in PRIVATE or part.startswith('.venv-') for part in p.parts):raise ValueError('Private publication path: '+name)
        if p.name.startswith('.env') and p.name!='.env.example':raise ValueError('Environment publication path: '+name)
        if p.suffix.lower()=='.jsonl' and name not in ALLOWED_DATA:raise ValueError('Unapproved data path: '+name)
        if p.suffix.lower() in {'.log','.safetensors','.pt','.pth','.bin','.zip','.gz','.xz','.parquet','.key','.pem','.partial'}:raise ValueError('Private/binary extension: '+name)
        value=(ROOT/name).read_bytes()
        if len(value)>2_000_000 or b'\0' in value:raise ValueError('Unexpected large/binary file: '+name)
        if any(re.search(pattern,value) for pattern in SECRET_PATTERNS):raise ValueError('Potential credential; inspect privately: '+name)
    if ALLOWED_DATA-set(paths):raise ValueError('Approved example dataset incomplete')
    sys.path.insert(0,str(ROOT/'src'))
    from personal_llm_data.personal_data import verify_prepared
    for domain in ('medical','financial','public_safety','joint'):verify_prepared(ROOT/'examples/text_safety_v1'/domain)
    file='repro/public_snapshot_manifest.json'
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths if name!=file}
    if update:
        report=dict(date='2026-11-02',files_sha256=hashes,excludes_self=True,
            note='Source/aggregate evidence plus exactly 12 owner-authorized attributed text-safety JSONL files; raw corpora, VLM data, model weights and predictions excluded')
        payload=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
        existing=(ROOT/file).read_bytes() if (ROOT/file).exists() else b''
        if b'\r\n' in existing and existing.count(b'\r\n')==existing.count(b'\n'):
            payload=payload.replace(b'\n',b'\r\n')
        (ROOT/file).write_bytes(payload)
    else:
        current=json.loads((ROOT/file).read_text())
        if current['files_sha256']!=hashes:raise ValueError('Public snapshot hash/inventory mismatch')
    print(f'Publication audit passed: {len(paths)} files; exactly {len(ALLOWED_DATA)} approved dataset files; no private assets detected')
    return len(paths)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--update-manifest',action='store_true')
    a=p.parse_args(argv);audit(a.update_manifest);return 0

if __name__=='__main__':raise SystemExit(main())
