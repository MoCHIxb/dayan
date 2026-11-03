"""List or explicitly restore a verified private maintenance archive; no downloads."""
import argparse,gzip,hashlib,json,shutil,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def inside(root,relative):
    root=Path(root).resolve();p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or any(':' in part or '\\' in part for part in p.parts):raise ValueError('Unsafe archive path')
    target=root.joinpath(*p.parts).resolve();target.relative_to(root)
    return target


def restore_record(record,root=ROOT):
    root=Path(root).resolve();name=record['path'];kind=record['kind']
    allowed=(name in ('data/processed/lamp7','data/processed/lamp7_retrieval_confirmation','models/Qwen3-1.7B','models/all-MiniLM-L6-v2') or
             name.startswith('experiments/runs/') and name.endswith('.log'))
    if not allowed or kind not in ('archived_closed_log','archived_retired_dataset','retired_model_directory'):raise ValueError('Not a restorable maintenance asset')
    target=inside(root,name);archive=inside(root,record['archive'])
    if target.exists():raise FileExistsError('Never overwrite an existing original asset')
    if digest(archive)!=record['archive_sha256']:raise ValueError('Archive SHA mismatch')
    temporary=target.with_name(target.name+'.restoring')
    if temporary.exists():raise FileExistsError('Previous incomplete restoration; inspect it first')
    target.parent.mkdir(parents=True,exist_ok=True)
    if kind=='archived_closed_log':
        with gzip.open(archive,'rb') as source,temporary.open('xb') as out:shutil.copyfileobj(source,out,1024*1024)
        if digest(temporary)!=record['sha256']:raise ValueError('Original log SHA mismatch')
    else:
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist())!=set(record['members_sha256']):raise ValueError('ZIP member inventory mismatch')
            temporary.mkdir()
            for member,expected in record['members_sha256'].items():
                output=inside(temporary,member);output.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as source,output.open('xb') as out:shutil.copyfileobj(source,out,1024*1024)
                if digest(output)!=expected:raise ValueError('ZIP member SHA mismatch')
    temporary.rename(target)
    return dict(restored=name,weights_downloaded=False,weights_to_refetch=record.get('weights',[]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'experiments/maintenance_20261103/cleanup_manifest.json')
    p.add_argument('--restore',help='One exact relative path from manifest; never bulk restores or trains')
    a=p.parse_args(argv);plan=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    records=[r for r in plan['records'] if 'archive' in r]
    if not a.restore:
        print(json.dumps([dict(path=r['path'],kind=r['kind'],archive=r['archive']) for r in records],indent=2));return 0
    matches=[r for r in records if r['path']==a.restore]
    if len(matches)!=1:raise ValueError('Choose exactly one archived path')
    print(json.dumps(restore_record(matches[0]),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
