"""CPU classification scores from complete existing base/adapter predictions."""
import argparse,json
from pathlib import Path
from personal_llm_data.personal_data import read_rows,sha,verify_prepared
from personal_llm_data.text_safety import score_verdicts

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Preserve existing score output')
    verify_prepared(a.data_dir)
    records=read_rows(a.data_dir/'test.jsonl');predictions=read_rows(a.predictions)
    result=score_verdicts(records,predictions)
    result['by_domain']={}
    for domain in sorted({r['domain'] for r in records}):
        pairs=[(r,pred) for r,pred in zip(records,predictions) if r['domain']==domain]
        result['by_domain'][domain]=score_verdicts([r for r,_ in pairs],[pred for _,pred in pairs])
    result['inputs_sha256']={str(a.data_dir/'test.jsonl'):sha(a.data_dir/'test.jsonl'),str(a.predictions):sha(a.predictions)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(samples=result['samples'],accuracy=result['accuracy'],macro_f1=result['macro_f1'],invalid_rate=result['invalid_rate'])))
    return 0

if __name__=='__main__':raise SystemExit(main())
