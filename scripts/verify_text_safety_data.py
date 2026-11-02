"""Replay selected records against pinned local sources without model loading."""
import argparse,json,lzma
from pathlib import Path
from collections import Counter
from personal_llm_data.personal_data import read_rows,sha,verify_prepared
from personal_llm_data.text_safety import text_hash
from build_text_safety_sets import SOURCES

ROOT=Path(__file__).resolve().parents[1]
def verify(source,data):
    selected={}
    report={}
    for domain in ('medical','financial','public_safety','joint'):
        splits=verify_prepared(data/domain)
        report[domain]=dict(counts={s:len(rows) for s,rows in splits.items()},groups={s:len({r['source_group'] for r in rows}) for s,rows in splits.items()})
        for split in ('train','dev','test'):
            for row in read_rows(data/domain/f'{split}.jsonl'):
                if row['id'] in selected and selected[row['id']]!=row:raise ValueError('Joint copy differs from domain record')
                selected[row['id']]=row
    for split in ('train','dev','test'):
        domain_ids=[r['id'] for d in ('medical','financial','public_safety') for r in read_rows(data/d/f'{split}.jsonl')]
        if domain_ids!=[r['id'] for r in read_rows(data/'joint'/f'{split}.jsonl')]:raise ValueError('Incomplete joint union')
    index={}
    for row in selected.values():
        p=row['provenance'];key=('diasafety/' if p['repo_id']=='thu-coai/DiaSafety' else 'beavertails/')+('val' if p['source_file'].endswith('/val.json') else ('test' if '/test.' in p['source_file'] else 'train'))
        index.setdefault(key,{})[p['source_line']]=row
    checked=0
    for key,(filename,digest) in SOURCES.items():
        path=source/filename
        if sha(path)!=digest:raise ValueError('Changed upstream source')
        if path.suffix=='.xz':
            with lzma.open(path,'rt',encoding='utf-8') as stream:rows=((i,json.loads(line)) for i,line in enumerate(stream,1) if line.strip());checked+=check_rows(rows,index.get(key,{}),key)
        else:checked+=check_rows(enumerate(json.loads(path.read_text(encoding='utf-8')),1),index.get(key,{}),key)
    if checked!=len(selected):raise ValueError('Some selected rows not traced to source')
    return dict(status='all_selected_source_text_and_labels_verified',unique_qa_records=checked,
        datasets=report,vlm_project_used=False,gpu_started=False,manual_review=False)

def check_rows(rows,expected,key):
    checked=0
    for number,raw in rows:
        if number not in expected:continue
        row=expected[number];payload=json.loads(row['messages'][-2]['content']);gold=json.loads(row['messages'][-1]['content'])
        prompt=raw['context'] if key.startswith('diasafety') else raw['prompt']
        original_label=raw['label']=='Safe' if key.startswith('diasafety') else raw['is_safe']
        if payload!={'user_request':prompt,'candidate_response':raw['response']} or gold!={'is_safe':original_label}:
            raise ValueError('Selected text/label differs from source')
        if row['source_group']!='prompt-'+text_hash(prompt):raise ValueError('Source group differs')
        checked+=1
    return checked

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/raw/text_safety_sources')
    p.add_argument('--data-dir',type=Path,default=ROOT/'examples/text_safety_v1')
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv);report=verify(a.source_dir,a.data_dir)
    if a.output:
        with a.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
