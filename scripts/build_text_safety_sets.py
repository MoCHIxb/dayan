"""Build three single-turn review datasets from published source labels only."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,lzma,re
from pathlib import Path
from personal_llm_data.personal_data import sha,validate_pairs
from personal_llm_data.text_safety import (domain_for,existing_target,record_from_source,
    record_from_diasafety,text_hash,grams,similarity,verify_automated_audit)

ROOT=Path(__file__).resolve().parents[1]
BT_REV='8401fe609d288129cc684a9b3be6a93e41cfe678'
DIA_REV='5106a041b1c7e36f30e02a97c44897a7f2057278'
SOURCES={
 'beavertails/train':('beavertails/round0/330k/train.jsonl.xz','12f04c9200000929ebbc616c29cb53331f6df36e7341d73e4a12cffac0eb1c08'),
 'beavertails/test':('beavertails/round0/330k/test.jsonl.xz','1c726c662b85bf4761913041fe9e098f80e5206344864597dbde86bc9a1bfc96'),
 'diasafety/train':('diasafety/DiaSafety_dataset/train.json','c1faee9dc1f67b28e8cbc2d4d146be1af749a9ad8caef5d76c10203a4a0dc6ae'),
 'diasafety/val':('diasafety/DiaSafety_dataset/val.json','ec0e4e705224584627dd8a9ff9f0cbafc7b56932752c6038b1a1164f01e14609'),
 'diasafety/test':('diasafety/DiaSafety_dataset/test.json','3a816555d097effda2b3f788b0372fb64359f1d736d881440d0ae6f386d6716c')}
DOMAINS=('medical','financial','public_safety')
AUDIT=dict(method='source_prompt_character5_jaccard_v1',threshold=0.9)
BLOCK=re.compile(r'\b(?:biolog\w*|pathogen\w*|bacter\w*|viruse?s?|viral|virolog\w*|genom\w*|crispr|anthrax|smallpox|bioweapon\w*|toxin\w*|ricin|aerosol\w*|sars|plasmid\w*|spores?|prion\w*|inoculat\w*|incubat\w*|bomb\w*|explosiv\w*|detonat\w*|gunpowder|thermite|cyanide|poison\w*|weapons?|guns?|firearms?|ammunition|malware|ransomware|credential\w*|phish\w*)\b|```|https?://|[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\d[ -]?){13,19}',re.I)

def source_rows(path):
    if path.suffix=='.xz':
        with lzma.open(path,'rt',encoding='utf-8') as stream:
            yield from ((i,json.loads(line)) for i,line in enumerate(stream,1) if line.strip())
    else:yield from enumerate(json.loads(path.read_text(encoding='utf-8')),1)

def features(rows):
    return {r['source_group']:grams(json.loads(r['messages'][-2]['content'])['user_request']) for r in rows}

def select(rows,count,seed,forbidden=None):
    forbidden=forbidden or {};chosen=[];labels=Counter();groups=Counter()
    for r in sorted(rows,key=lambda r:hashlib.sha256(f'{seed}/{r["id"]}'.encode()).hexdigest()):
        label=json.loads(r['messages'][-1]['content'])['is_safe'];group=r['source_group']
        if labels[label]>=count//2 or groups[group]>=3 or group in forbidden:continue
        g=grams(json.loads(r['messages'][-2]['content'])['user_request'])
        if any(similarity(g,other)>=0.9 for other in forbidden.values()):continue
        chosen.append(r);labels[label]+=1;groups[group]+=1
        if len(chosen)==count:return chosen
    raise ValueError(f'Insufficient grouped rows: need {count}; got {len(chosen)}; labels={dict(labels)}')

def write_set(path,splits):
    validate_pairs([dict(r,split=s) for s,rows in splits.items() for r in rows])
    audit=verify_automated_audit(splits,AUDIT)
    path.mkdir(parents=True,exist_ok=False);files={}
    for split,rows in splits.items():
        p=path/f'{split}.jsonl'
        with p.open('x',encoding='utf-8',newline='\n') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        files[p.name]=sha(p)
    report=dict(schema_version=1,purpose='personal_adapter_example_v1',
        counts={s:len(rows) for s,rows in splits.items()},files_sha256=files,
        real_authorization_declared=True,manual_near_duplicate_review_declared=False,
        automated_near_duplicate_review=audit,
        independent_prompt_groups={s:len(features(rows)) for s,rows in splits.items()},
        label_counts={s:dict(Counter('safe' if json.loads(r['messages'][-1]['content'])['is_safe'] else 'unsafe' for r in rows)) for s,rows in splits.items()},
        source_sha256={k:v[1] for k,v in SOURCES.items()},
        max_length=1024,max_new_tokens=64,source_policies='DiaSafety Unauthorized Expertise for medical; BeaverTails harmlessness for finance/public safety',
        limitations='Existing response-safety labels; not request attack labels, clinical correctness or equivalent policies across sources. No manual review claimed.')
    (path/'manifest.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
    return report

def build(source,output,tokenizer_path):
    if output.exists():raise FileExistsError('Preserve existing dataset; choose a new destination')
    paths={key:source/value[0] for key,value in SOURCES.items()}
    for key,path in paths.items():
        if sha(path)!=SOURCES[key][1]:raise ValueError('Source checksum mismatch: '+key)
    test_groups=set();val_groups=set()
    for key in ('beavertails/test','diasafety/test','diasafety/val'):
        for _,r in source_rows(paths[key]):
            group='prompt-'+text_hash(r['context'] if key.startswith('diasafety') else r['prompt'])
            (val_groups if key.endswith('/val') else test_groups).add(group)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True)
    pools={d:{s:[] for s in ('train','dev','test')} for d in DOMAINS};stats=Counter();signatures=defaultdict(set)
    for key,path in paths.items():
        original_split=key.split('/')[1];split='dev' if original_split=='val' else original_split;seen=set()
        for number,r in source_rows(path):
            stats[key+'/source_records']+=1
            if key.startswith('diasafety'):
                if r.get('category')!='Unauthorized Expertise' or r.get('label') not in ('Safe','Unsafe'):continue
                domain='medical';record=record_from_diasafety(r,original_split,number,DIA_REV)
            else:
                domain=domain_for(r['prompt'])
                if domain not in ('financial','public_safety'):continue
                try:existing_target(r)
                except ValueError:stats['invalid_source_labels']+=1;continue
                if any(r['category'][c] for c in ('child_abuse','sexually_explicit,adult_content','self_harm')):stats['category_excluded']+=1;continue
                record=record_from_source(r,domain,split,number,BT_REV)
            group=record['source_group']
            if split!='test' and group in test_groups:stats['official_test_prompt_held_out']+=1;continue
            if domain=='medical' and split=='train' and group in val_groups:stats['official_medical_val_prompt_held_out']+=1;continue
            payload=json.loads(record['messages'][-2]['content'])
            if BLOCK.search(payload['user_request']+'\n'+payload['candidate_response']):stats['content_scope_excluded']+=1;continue
            pair=record['provenance']['original_pair_sha256'];signatures[pair].add(record['messages'][-1]['content'])
            if pair in seen:stats['duplicate_qa_pair']+=1;continue
            seen.add(pair)
            prompt=tokenizer.apply_chat_template(record['messages'][:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            answer=tokenizer(record['messages'][-1]['content'],add_special_tokens=False)['input_ids']
            if len(prompt)+len(answer)+1>1024 or len(answer)+1>64:stats['token_budget_excluded']+=1;continue
            pools[domain][split].append(record)
    conflicts={pair for pair,labels in signatures.items() if len(labels)>1};chosen={};pool_counts={}
    for d in DOMAINS:
        for s in pools[d]:pools[d][s]=[r for r in pools[d][s] if r['provenance']['original_pair_sha256'] not in conflicts]
        pool_counts[d]={s:dict(records=len(rows),groups=len(features(rows)),labels=dict(Counter('safe' if json.loads(r['messages'][-1]['content'])['is_safe'] else 'unsafe' for r in rows))) for s,rows in pools[d].items()}
        print('Eligible',d,json.dumps(pool_counts[d]),flush=True)
        test=select(pools[d]['test'],40,20251206);blocked=features(test)
        dev=select(pools[d]['dev' if d=='medical' else 'train'],20,20251207,blocked);blocked.update(features(dev))
        train=select(pools[d]['train'],50,20251208,blocked)
        chosen[d]=dict(train=train,dev=dev,test=test)
    joint={s:[r for d in DOMAINS for r in chosen[d][s]] for s in ('train','dev','test')}
    verify_automated_audit(joint,AUDIT)
    output.mkdir(parents=True,exist_ok=False)
    reports={d:write_set(output/d,joint if d=='joint' else chosen[d]) for d in (*DOMAINS,'joint')}
    report=dict(schema_version=1,status='constructed_cpu_validated_not_gpu_trained',
        sources=dict(beavertails=dict(repo='PKU-Alignment/BeaverTails',revision=BT_REV,license='CC-BY-NC-4.0'),
                     diasafety=dict(repo='thu-coai/DiaSafety',revision=DIA_REV,license='Apache-2.0 upstream repository')),
        counts={d:r['counts'] for d,r in reports.items()},independent_prompt_groups={d:r['independent_prompt_groups'] for d,r in reports.items()},
        filters=dict(stats),source_label_conflicting_pairs=len(conflicts),eligible_pools=pool_counts,
        previously_proposed_vlm_data_used=False,new_requests_or_answers_generated=False,manual_review=False,
        scope='Single-turn candidate-response safety; existing source policies differ; not clinical correctness or prompt attack detection',gpu_training='not_started')
    (output/'construction_report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(report['counts'],indent=2));return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/raw/text_safety_sources')
    p.add_argument('--output-dir',type=Path,default=ROOT/'examples/text_safety_v1')
    p.add_argument('--tokenizer',type=Path,default=ROOT/'models/Qwen3-4B')
    a=p.parse_args(argv);build(a.source_dir,a.output_dir,a.tokenizer);return 0

if __name__=='__main__':raise SystemExit(main())
