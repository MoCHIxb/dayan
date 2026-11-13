"""Fixed character TF-IDF centroid baseline and explicit GPU few-shot baseline."""
import argparse
from collections import Counter, defaultdict
import json
import math
import platform
import re
from pathlib import Path
import subprocess
import sys

from personal_llm_data.banking77 import LABELS, score_intents, MAX_LENGTH, MAX_NEW_TOKENS
from personal_llm_data.personal_data import read_rows, sha, verify_prepared
from personal_llm_data.text_safety import normalized
from personal_llm_data.personal_runtime import gpu_guard

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/banking77_v1'
OUT=ROOT/'experiments/intenttune_banking77_v1'


def tokens(row):
    text=normalized(json.loads(row['messages'][-2]['content'])['user_request'])
    return Counter(text[i:i+3] for i in range(max(1,len(text)-2)))


def centroid_predictions(training, records):
    """Vocabulary/IDF/centroids use training only; no hyperparameter search."""
    documents=[tokens(row) for row in training];df=Counter()
    for document in documents:df.update(document.keys())
    idf={term:math.log((1+len(training))/(1+count))+1 for term,count in df.items()}

    def vector(document):
        values={term:(1+math.log(count))*idf[term] for term,count in document.items() if term in idf}
        norm=math.sqrt(sum(value*value for value in values.values()))
        return {key:value/norm for key,value in values.items()} if norm else {}

    centers={label:defaultdict(float) for label in LABELS}
    for row,document in zip(training,documents):
        label=json.loads(row['messages'][-1]['content'])['label']
        for term,value in vector(document).items():centers[label][term]+=value
    for label,center in centers.items():
        norm=math.sqrt(sum(value*value for value in center.values()))
        centers[label]={key:value/norm for key,value in center.items()} if norm else {}
    output=[]
    for row in records:
        values=vector(tokens(row))
        # Catalog order is the fixed tie-break, including an empty vocabulary match.
        label=max(LABELS,key=lambda label:sum(value*centers[label].get(key,0) for key,value in values.items()))
        output.append(dict(id=row['id'],reference=row['messages'][-1]['content'],prediction=json.dumps(dict(label=label),separators=(',',':'))))
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=('centroid','fewshot'),required=True)
    p.add_argument('--size',type=int,choices=(50,200,500),default=500)
    p.add_argument('--execute',action='store_true',help='Required for few-shot model generation; centroid is CPU only')
    args=p.parse_args(argv)
    if args.condition=='centroid':
        directory=DATA/f'n{args.size}';verify_prepared(directory)
        records=read_rows(directory/'test.jsonl');training=read_rows(directory/'train.jsonl')
        path=OUT/f'n{args.size}/centroid_scores.json'
        prediction=path.with_suffix('.predictions.jsonl')
        if path.exists() or prediction.exists():raise FileExistsError('Keep existing baseline evidence')
        result_predictions=centroid_predictions(training,records)
        result=score_intents(records,result_predictions)
        result.update(method='train-only character 3-gram TF-IDF cosine centroid; fixed no dev tuning',
                      inputs_sha256={str(directory/name):sha(directory/name) for name in ('train.jsonl','test.jsonl')})
        path.parent.mkdir(parents=True,exist_ok=True)
        with prediction.open('x',encoding='utf-8',newline='\n') as stream:
            for row in result_predictions:stream.write(json.dumps(row)+'\n')
        result['prediction_sha256']=sha(prediction)
        with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    else:
        records=read_rows(DATA/'fewshot_test.jsonl')
        if not args.execute:
            print(json.dumps(dict(executable=False,samples=len(records),demonstrations=10,next='--execute after RTX3090 preflight')));return 0
        if platform.system()!='Linux':raise ValueError('New model experiments run on the designated Linux RTX3090 server only')
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip().splitlines()
        if len(inventory)!=1 or not re.search(r'RTX\s+3090\b(?!\s*Ti)',inventory[0]):raise ValueError('One RTX3090 required')
        path=OUT/'fewshot/fewshot_metrics.json'
        predictions=path.with_suffix('.predictions.jsonl');receipt=path.with_suffix('.receipt.json')
        if any(p.exists() for p in (path,predictions,receipt)):raise FileExistsError('Keep few-shot evidence')
        command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_generation.py'),'--model',str(ROOT/'models/Qwen3-4B'),
                 '--data',str(DATA/'fewshot_test.jsonl'),'--output',str(path),'--max-samples','400','--batch-size','1',
                 '--max-input-tokens',str(MAX_LENGTH),'--max-new-tokens',str(MAX_NEW_TOKENS),'--quantization','none']
        before=sha(DATA/'fewshot_test.jsonl')
        with gpu_guard(ROOT):
            path.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(command,cwd=ROOT,check=True)
        result=score_intents(records,read_rows(predictions))
        if sha(DATA/'fewshot_test.jsonl')!=before or json.loads(path.read_text())['samples']!=400:raise ValueError('Incomplete or changed few-shot evaluation')
        with receipt.open('x',encoding='utf-8') as stream:
            json.dump(dict(command=command,input_sha256=before,evaluator_sha256=sha(ROOT/'scripts/evaluate_generation.py'),
                           outputs_sha256={p.name:sha(p) for p in (path,predictions)}),stream,indent=2)
        score=path.parent/'fewshot_scores.json'
        with score.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('samples','macro_f1','minimum_class_recall','invalid_rate')}));return 0


if __name__=='__main__':raise SystemExit(main())
