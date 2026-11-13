"""IntentTune strict-label metrics, class recall and confusion diagnostics."""
import argparse
import json
from pathlib import Path
from personal_llm_data.banking77 import score_intents
from personal_llm_data.personal_data import read_rows, sha, verify_prepared


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():raise FileExistsError('Keep previous scoring evidence')
    verify_prepared(args.data_dir)
    result=score_intents(read_rows(args.data_dir/'test.jsonl'),read_rows(args.predictions))
    result['inputs_sha256']={str(args.data_dir/'test.jsonl'):sha(args.data_dir/'test.jsonl'),str(args.predictions):sha(args.predictions)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('samples','accuracy','macro_f1','minimum_class_recall','invalid_rate')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
