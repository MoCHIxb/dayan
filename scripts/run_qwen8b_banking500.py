"""Finite first Qwen8B BANKING500 experiment; plan by default, no auto retries."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from personal_llm_data.personal_data import sha,read_rows
from personal_llm_data.personal_runtime import gpu_guard,write_json
from personal_llm_data.banking77 import LABELS,score_intents
from build_banking77 import fetch,build
from verify_banking77 import verify
from prepare_8b_4090 import assets
from train_personal_adapter import validate_config,inputs_snapshot,verify_checkpoint
from evaluate_personal_adapter import verify_predictions
ROOT=Path(__file__).resolve().parents[1]
CONFIG=Path('configs/cloud_8b_4090/qwen3_8b_banking_n500_A.json')
DATA=Path('data/processed/banking77_8b_v1')
OUT=Path('experiments/cloud_8b_4090/banking_n500_A_seed20261113')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def fresh(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)

def data_check():
    reference=load('repro/banking77_assets.json')['data_sha256']
    names=[f'n{size}/{split}.jsonl' for size in (50,200,500) for split in ('train','dev','test')]+['fewshot_test.jsonl']
    for name in names:
        if sha(DATA/name)!=reference[name]:raise ValueError('8B build changed fixed real records: '+name)
    report=verify(ROOT/'data/raw/banking77_v1',ROOT/DATA)
    if report['unique_records']!=1000:raise ValueError('Incomplete source replay')
    config=load(CONFIG);validate_config(config)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/config['model_name_or_path']),local_files_only=True)
    for name in names:
        for row in read_rows(DATA/name):
            prompt=tok.apply_chat_template(row['messages'][:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            response=tok(row['messages'][-1]['content'],add_special_tokens=False)['input_ids']
            if len(prompt)+len(response)+1>1536 or len(response)+1>64:raise ValueError('8B token budget exceeded; no truncation permitted')
    return report

def prepare():
    if (OUT/'manifest.json').exists():raise ValueError('Frozen run blocks preparation')
    assets('qwen3_8b')
    source=ROOT/'data/raw/banking77_v1';fetch(source)
    if not DATA.exists():build(source,ROOT/DATA,ROOT/'models/Qwen3-8B')
    print(json.dumps(data_check()))

def inputs():
    config=load(CONFIG);snapshot=inputs_snapshot(config,ROOT/CONFIG)
    names=[CONFIG,Path('repro/banking77_assets.json'),Path('requirements-cloud8b.lock'),Path('configs/cloud_8b_4090/environment.json'),
           Path('experiments/cloud_8b_4090/assets/qwen3_8b.json'),Path('experiments/cloud_8b_4090/assets/qwen3_8b.verified.json')]
    names+=list(DATA.rglob('*.jsonl'))+list(DATA.rglob('*.json'))
    names+=[Path('scripts')/name for name in ('run_qwen8b_banking500.py','prepare_8b_4090.py','build_banking77.py','verify_banking77.py','intent_baselines.py','evaluate_generation.py','evaluate_personal_adapter.py','score_intents.py','audit_classification_run.py')]
    names+=list(Path('src/personal_llm_data').glob('*.py'))
    from importlib.metadata import version
    runtime={name:version(name) for name in ('torch','transformers','peft','accelerate','bitsandbytes','tokenizers','safetensors')}
    return dict(training=snapshot,runtime=runtime,files_sha256={str(p).replace('\\','/'):sha(p) for p in sorted(set(names))})

def stages():
    result=[]
    def add(name,command,outputs):result.append(dict(name=name,command=[sys.executable,'-u',*command],outputs=outputs))
    add('centroid',['scripts/run_qwen8b_banking500.py','centroid'],['centroid_scores.json','centroid_predictions.jsonl'])
    add('fewshot',['scripts/run_qwen8b_banking500.py','fewshot'],['fewshot/metrics.json','fewshot/metrics.predictions.jsonl','fewshot/receipt.json','fewshot_scores.json'])
    add('train',['scripts/train_personal_adapter.py','--config',str(CONFIG),'--execute'],['run/training_summary.json','run/progress.jsonl','run/run_inputs.json','run/adapter/adapter_model.safetensors','run/checkpoints/checkpoint-128/personal_checkpoint_receipt.json'])
    for condition in ('base','adapter'):
        add('evaluate_'+condition,['scripts/evaluate_personal_adapter.py','--config',str(CONFIG),'--condition',condition,'--execute'],['run/evaluation/'+condition+'_metrics'+s for s in ('.json','.predictions.jsonl','.receipt.json')])
        add('score_'+condition,['scripts/score_intents.py','--data-dir',str(DATA/'n500'),'--predictions',str(OUT/f'run/evaluation/{condition}_metrics.predictions.jsonl'),'--output',str(OUT/f'{condition}_scores.json')],[f'{condition}_scores.json'])
    add('audit',['scripts/audit_classification_run.py','--config',str(CONFIG),'--labels',*LABELS,'--score-directory',str(OUT),'--output',str(OUT/'run_audit.json')],['run_audit.json'])
    return result

@contextmanager
def lock():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'pipeline.lock').open('a+b') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def verify_receipt(stage,receipt,current):
    if receipt['command']!=stage['command'] or receipt['inputs']!=current or set(receipt['outputs'])!=set(stage['outputs']):raise ValueError('Stage command or input inventory mismatch')
    for name,h in receipt['outputs'].items():
        if sha(OUT/name)!=h:raise ValueError('Completed output changed')

def report(current=None):
    current=current or inputs();manifest=load(OUT/'manifest.json')
    if manifest!=dict(inputs=current,stages=stages()):raise ValueError('Frozen input/command changed')
    for stage in stages():verify_receipt(stage,load(OUT/'receipts'/f"{stage['name']}.json"),current)
    from audit_classification_run import audit
    audited=audit(ROOT/CONFIG,LABELS,ROOT/OUT)
    stored=load(OUT/'run_audit.json')
    if stored['quality']!=audited['quality'] or stored['adapter_sha256']!=audited['adapter_sha256']:raise ValueError('Audit recomputation differs')
    few=read_rows(DATA/'fewshot_test.jsonl');pred=read_rows(OUT/'fewshot/metrics.predictions.jsonl');verify_predictions(few,pred)
    if score_intents(few,pred)!=load(OUT/'fewshot_scores.json'):raise ValueError('Few-shot score changed')
    train=read_rows(DATA/'n500/train.jsonl');test=read_rows(DATA/'n500/test.jsonl')
    from intent_baselines import centroid_predictions
    cp=centroid_predictions(train,test)
    if cp!=read_rows(OUT/'centroid_predictions.jsonl') or score_intents(test,cp)!=load(OUT/'centroid_scores.json'):raise ValueError('Centroid recomputation differs')
    result=dict(status='complete_qwen8b_banking500_A_audited',completed_stages=len(stages()),audit=audited,fewshot=load(OUT/'fewshot_scores.json'),centroid=load(OUT/'centroid_scores.json'))
    if (OUT/'results.json').exists():
        if load(OUT/'results.json')!=result:raise ValueError('Existing summary differs')
    else:fresh(OUT/'results.json',result)
    write_json(OUT/'execution_state.json',dict(stage='complete_audited',completed_stages=len(stages())))
    print(json.dumps(dict(status=result['status'],completed_stages=len(stages()))))

def run(resume=False,checkpoint=None):
    with lock():
        assets('qwen3_8b');data_check()
        current=inputs();manifest=OUT/'manifest.json'
        if manifest.exists():
            if not resume or load(manifest)!=dict(inputs=current,stages=stages()):raise ValueError('Explicit resume with identical frozen inputs required')
        else:
            if resume:raise ValueError('No run to resume')
            if any((OUT/p).exists() for st in stages() for p in st['outputs']):raise FileExistsError('Unreceipted prior output')
            fresh(manifest,dict(inputs=current,stages=stages()))
        subprocess.run([sys.executable,'scripts/prepare_8b_4090.py','check'],check=True)
        previous=load(OUT/'execution_state.json') if (OUT/'execution_state.json').exists() else {}
        if resume and (OUT/'pause.request').exists():(OUT/'pause.request').rename(OUT/f'pause.consumed.{time.time_ns()}')
        for stage in stages():
            name=stage['name'];receipt=OUT/'receipts'/f'{name}.json'
            if receipt.exists():verify_receipt(stage,load(receipt),current);continue
            if (OUT/'pause.request').exists():write_json(OUT/'execution_state.json',dict(stage='paused_between_stages',next_stage=name));return
            attempts=OUT/'attempts'/name;command=list(stage['command'])
            if attempts.exists():
                if name!='train' or not checkpoint or previous.get('stage')!='paused_checkpoint_verified':raise ValueError('Prior incomplete/failed attempt; retain evidence and diagnose, no automatic retry')
                target=Path(checkpoint).resolve()
                if target.parent!=(ROOT/OUT/'run/checkpoints').resolve():raise ValueError('Wrong checkpoint directory')
                verify_checkpoint(target,current['training']);command+=['--resume-from-checkpoint',str(target)]
            elif any((OUT/p).exists() for p in stage['outputs']):raise FileExistsError('Partial stage output retained')
            attempts.mkdir(parents=True,exist_ok=True);number=len(list(attempts.glob('*.started.json')))+1
            fresh(attempts/f'{number}.started.json',dict(command=command))
            write_json(OUT/'execution_state.json',dict(stage='running',active_stage=name))
            try:
                with (attempts/f'{number}.stdout.log').open('xb') as out,(attempts/f'{number}.stderr.log').open('xb') as err:subprocess.run(command,check=True,stdout=out,stderr=err)
                if inputs()!=current:raise ValueError('Inputs changed during execution')
                if name=='train' and load(OUT/'run/run_status.json')['stage']=='paused_checkpoint_verified':
                    status=load(OUT/'run/run_status.json');verify_checkpoint(Path(status['checkpoint']),current['training']);write_json(OUT/'execution_state.json',dict(stage='paused_checkpoint_verified',checkpoint=status['checkpoint']));return
                fresh(receipt,dict(command=stage['command'],executed_command=command,inputs=current,outputs={p:sha(OUT/p) for p in stage['outputs']}))
            except Exception:
                write_json(OUT/'execution_state.json',dict(stage='failed',active_stage=name));raise
        report(current)

def baselines(kind):
    OUT.mkdir(parents=True,exist_ok=True);records=read_rows(DATA/('fewshot_test.jsonl' if kind=='fewshot' else 'n500/test.jsonl'))
    if kind=='centroid':
        from intent_baselines import centroid_predictions
        predictions=centroid_predictions(read_rows(DATA/'n500/train.jsonl'),records)
        with (OUT/'centroid_predictions.jsonl').open('x',encoding='utf-8') as f:
            for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    else:
        path=OUT/'fewshot/metrics.json';path.parent.mkdir(parents=True,exist_ok=True)
        if any(path.with_suffix(s).exists() for s in ('.json','.predictions.jsonl')):raise FileExistsError('Keep few-shot evidence')
        command=[sys.executable,'scripts/evaluate_generation.py','--model','models/Qwen3-8B','--data',str(DATA/'fewshot_test.jsonl'),'--output',str(path),'--max-samples','400','--batch-size','1','--max-input-tokens','1536','--max-new-tokens','64','--quantization','none']
        with gpu_guard(ROOT):subprocess.run(command,check=True)
        predictions=read_rows(path.with_suffix('.predictions.jsonl'));verify_predictions(records,predictions)
        if load(path)['samples']!=400:raise ValueError('Incomplete few-shot test')
        fresh(OUT/'fewshot/receipt.json',dict(command=command,inputs_sha256=sha(DATA/'fewshot_test.jsonl'),predictions_sha256=sha(path.with_suffix('.predictions.jsonl'))))
    fresh(OUT/f'{kind}_scores.json',score_intents(records,predictions))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','prepare','check','run','report','pause','centroid','fewshot'],default='plan',nargs='?');p.add_argument('--resume',action='store_true');p.add_argument('--checkpoint');a=p.parse_args();os.chdir(ROOT)
    if (a.resume or a.checkpoint) and a.action!='run':p.error('Resume flags require run')
    if a.checkpoint and not a.resume:p.error('Checkpoint requires explicit --resume')
    if a.action=='plan':print(json.dumps(dict(executable=False,config=str(CONFIG),stages=stages()),indent=2));return
    if platform.system()!='Linux':raise RuntimeError('New GPU/data preparation runs on owner Linux4090 only')
    if a.action=='prepare':prepare()
    elif a.action=='check':assets('qwen3_8b');print(json.dumps(data_check()));subprocess.run([sys.executable,'scripts/prepare_8b_4090.py','check'],check=True)
    elif a.action=='run':run(a.resume,a.checkpoint)
    elif a.action=='report':report()
    elif a.action=='pause':
        status=load(OUT/'execution_state.json')
        if status['stage']!='running':raise ValueError('No running pipeline recorded')
        fresh(OUT/'pause.request',dict(requested_unix=time.time()))
        if status['active_stage']=='train':subprocess.run([sys.executable,'scripts/train_personal_adapter.py','--config',str(CONFIG),'--request-pause'],check=True)
        print('Wait for verified paused state and actual process exit; do not power off yet')
    else:baselines(a.action)

if __name__=='__main__':main()
