"""Replay all selected real records against pinned BANKING77 sources; CPU only."""
import argparse
import json
from pathlib import Path
from build_banking77 import ROOT, rows_from_source, content
from personal_llm_data.banking77 import SOURCES, SIZES, LABELS, SYSTEM, source_record
from personal_llm_data.personal_data import read_rows, sha, verify_prepared


def verify(source, data):
    for name,digest in SOURCES.items():
        if sha(source/name)!=digest:raise ValueError('Source bytes differ: '+name)
    official={split:rows_from_source(source,split) for split in ('train','test')}
    views={size:{split:read_rows(data/f'n{size}/{split}.jsonl') for split in ('train','dev','test')} for size in SIZES}
    selected={}
    for size,splits in views.items():
        verify_prepared(data/f'n{size}')
        if {key:len(rows) for key,rows in splits.items()}!=dict(train=size,dev=100,test=400):raise ValueError('Incomplete coverage')
        if splits['train']!=views[500]['train'][:size] or splits['dev']!=views[500]['dev'] or splits['test']!=views[500]['test']:
            raise ValueError('Nested train or shared holdout differs')
        for split,rows in splits.items():
            for row in rows:
                provenance=row['provenance']
                source_split='test' if split=='test' else 'train'
                ordinal=provenance['csv_data_row']
                if type(ordinal) is not int or not 1<=ordinal<=len(official[source_split]):raise ValueError('Invalid 1-based source row')
                if row!=source_record(official[source_split][ordinal-1],source_split,ordinal):raise ValueError('Source text/label/provenance mismatch')
                if row['id'] in selected and selected[row['id']]!=row:raise ValueError('View copy differs')
                selected[row['id']]=row
    expected=[source_record(raw,'test',i) for i,raw in enumerate(official['test'],1) if raw['category'] in LABELS]
    if views[500]['test']!=expected:raise ValueError('Official selected test coverage changed')
    fewshot=read_rows(data/'fewshot_test.jsonl')
    examples=views[50]['train'][:10]
    demonstration='\n'.join(json.dumps(dict(customer_message=content(row),label=row['intent']),ensure_ascii=False) for row in examples)
    expected=[]
    for row in views[50]['test']:
        clone=json.loads(json.dumps(row));clone['messages'][0]['content']=SYSTEM+'\nExamples from the training set:\n'+demonstration
        expected.append(clone)
    if fewshot!=expected:raise ValueError('Few-shot inputs are not the fixed train-only demonstration baseline')
    return dict(status='all_selected_source_text_labels_and_nested_views_verified',unique_records=len(selected),
                counts={str(size):{split:len(rows) for split,rows in splits.items()} for size,splits in views.items()},
                test_records_per_intent=40,source_replay=True,fewshot_train_only=True,gpu_started=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/raw/banking77_v1')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/processed/banking77_v1')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv);report=verify(args.source_dir,args.data_dir)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report));return 0


if __name__=='__main__':raise SystemExit(main())
