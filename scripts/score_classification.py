"""Score a complete held-out existing-label task; refuses output overwrite."""
import argparse,json
from pathlib import Path
from personal_llm_data.personal_data import read_rows,sha,verify_prepared
from personal_llm_data.classification import score_labels


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Existing scores; preserve evidence')
    verify_prepared(a.data_dir)
    records=read_rows(a.data_dir/'test.jsonl')
    result=score_labels(records,read_rows(a.predictions),a.labels)
    result['inputs_sha256']={str(a.data_dir/'test.jsonl'):sha(a.data_dir/'test.jsonl'),str(a.predictions):sha(a.predictions)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('samples','accuracy','macro_f1','invalid_rate')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
