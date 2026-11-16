"""Finite first-seed BANKING research: 15 trainings, 27 generations, no auto retry."""
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
from personal_llm_data.personal_runtime import write_json
from personal_llm_data.banking77 import score_intents
from train_research_adapter import ROOT,inputs_snapshot,validate_config,verify_checkpoint
from deploy_research_models import catalog,backend
from audit_research_banking import audit,audit_evaluation

OUT=ROOT/'experiments/research_banking_v1'
DATA=ROOT/'data/processed/banking77_8b_v1'
KEYS=('qwen3_8b','mistral7b','phi4mini')
RECIPES=('A','B','C','D','E')

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def fresh(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def config_path(key,recipe):return Path('configs/research_banking_v1')/f'{key}_{recipe}.json'
def run_dir(key,recipe):return Path(load(ROOT/config_path(key,recipe))['output_dir'])
def baseline_path(key,kind):return Path('experiments/research_banking_v1/baselines')/key/(kind+'.json')

def stages():
    result=[]
    def add(name,command,outputs,kind,config=None):
        result.append(dict(name=name,command=[sys.executable,'-u',*map(str,command)],outputs=list(map(str,outputs)),kind=kind,config=str(config) if config else None))
    add('centroid',['scripts/run_research_banking.py','centroid'],['experiments/research_banking_v1/centroid.json','experiments/research_banking_v1/centroid.predictions.jsonl'],'cpu')
    for key in KEYS:
        for mode,condition,recipe in [('none','zero','A'),('none','fewshot','A'),('nf4','zero','D'),('int8','zero','E')]:
            path=baseline_path(key,mode+'_'+condition);cfg=config_path(key,recipe)
            add(f'{key}_{mode}_{condition}',['scripts/evaluate_research_adapter.py','--config',cfg,'--condition',condition,'--output',path,'--execute'],[path,path.with_suffix('.predictions.jsonl'),path.with_suffix('.receipt.json')],'baseline',cfg)
        for recipe in RECIPES:
            cfg=config_path(key,recipe);run=run_dir(key,recipe);name=f'{key}_{recipe}'
            add(name+'_train',['scripts/train_research_adapter.py','--config',cfg,'--execute'],[run/'training_summary.json',run/'progress.jsonl',run/'run_inputs.json',run/'adapter/adapter_model.safetensors',run/'checkpoints/checkpoint-128/personal_checkpoint_receipt.json'],'train',cfg)
            path=run/'evaluation/adapter.json'
            add(name+'_test',['scripts/evaluate_research_adapter.py','--config',cfg,'--condition','adapter','--output',path,'--execute'],[path,path.with_suffix('.predictions.jsonl'),path.with_suffix('.receipt.json')],'test',cfg)
            add(name+'_audit',['scripts/audit_research_banking.py','--config',cfg,'--output',run/'audit.json'],[run/'audit.json'],'audit',cfg)
    return result

def diagnostics():
    import gpu_probe_research as probe
    current=probe.gate();files={}
    for key in KEYS:
        for mode in probe.MODES:
            directory=probe.BASE/(key+'_'+mode);complete=load(directory/'complete.json')
            if complete['status']!='diagnostic_pair_completed' or complete['inputs']!=current:raise ValueError('Missing/stale nine-condition diagnostic')
            for phase in ('train','restore'):
                p=directory/(phase+'_receipt.json')
                if sha(p)!=complete[phase+'_receipt_sha256']:raise ValueError('Diagnostic receipt changed')
                files[str(p.relative_to(ROOT))]=sha(p)
            for name,expected in load(directory/'train_receipt.json')['files_sha256'].items():
                if sha(directory/name)!=expected:raise ValueError('Diagnostic state changed')
            restored=load(directory/'restore_receipt.json')
            if restored['status']!='probe_restored_step2_and_generation_completed' or sha(directory/'generation_probe.json')!=restored['generation_sha256']:raise ValueError('Restore/generation not verified')
            files[str((directory/'complete.json').relative_to(ROOT))]=sha(directory/'complete.json')
    return dict(gate=current,receipts_sha256=files)

def inputs():
    paths=list((ROOT/'configs/research_banking_v1').glob('*.json'))+list(DATA.rglob('*.jsonl'))+list(DATA.rglob('*.json'))
    paths+=[ROOT/'docs/63_research_banking_formal_protocol.md',ROOT/'repro/banking77_assets.json',ROOT/'requirements-cloud8b.lock',ROOT/'requirements-tokenizers.lock']
    paths += [ROOT/'scripts'/name for name in ('run_research_banking.py','train_research_adapter.py','evaluate_research_adapter.py','audit_research_banking.py','intent_baselines.py','run_qwen8b_banking500.py','build_banking77.py','verify_banking77.py')]
    paths+=list((ROOT/'src/personal_llm_data').glob('*.py'))
    return dict(diagnostics=diagnostics(),models={key:inputs_snapshot(load(ROOT/config_path(key,'A')),config_path(key,'A')) for key in KEYS},files_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(paths))})

def check():
    # CPU checks only: tokenizer/bytes/source replay. No CUDA model is loaded.
    from run_qwen8b_banking500 import data_check
    from transformers import AutoTokenizer
    from personal_llm_data.research_sft import encode_row
    from gpu_probe_research import supervised_ids
    data_check()
    for key in KEYS:
        backend().assets(key)
        tok=AutoTokenizer.from_pretrained(str(ROOT/catalog()['models'][key]['directory']),local_files_only=True,trust_remote_code=False)
        for recipe in RECIPES:validate_config(load(ROOT/config_path(key,recipe)))
        for path in [DATA/'n500'/f'{s}.jsonl' for s in ('train','dev','test')]+[DATA/'fewshot_test.jsonl']:
            for row in read_rows(path):
                encoded,prompt=encode_row(tok,row,key);ids,labels,old_prompt=supervised_ids(tok,row,key)
                if encoded['input_ids']!=ids or encoded['labels']!=labels or prompt!=old_prompt:raise ValueError('Formal encoding differs from diagnostic')
    snapshot=inputs()
    print(json.dumps(dict(status='formal_inputs_and_9_diagnostics_verified_not_formal_training',models=list(KEYS),trainings=15,stages=len(stages()))),flush=True)
    return snapshot

@contextmanager
def lock():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'pipeline.lock').open('a+b') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def verify_receipt(stage,receipt,current):
    if receipt['command']!=stage['command'] or receipt['inputs']!=current or set(receipt['outputs'])!=set(stage['outputs']):raise ValueError('Stage command/input inventory differs')
    for name,h in receipt['outputs'].items():
        if sha(ROOT/name)!=h:raise ValueError('Completed stage output changed: '+name)

def centroid():
    from intent_baselines import centroid_predictions
    test=read_rows(DATA/'n500/test.jsonl');pred=centroid_predictions(read_rows(DATA/'n500/train.jsonl'),test)
    path=OUT/'centroid.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix('.predictions.jsonl').exists():raise FileExistsError('Keep existing baseline evidence')
    with path.with_suffix('.predictions.jsonl').open('x',encoding='utf-8') as f:
        for row in pred:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    fresh(path,score_intents(test,pred))

def report(current=None):
    current=current or check();manifest=load(OUT/'manifest.json')
    if manifest!=dict(inputs=current,stages=stages()):raise ValueError('Frozen campaign differs')
    for stage in stages():verify_receipt(stage,load(OUT/'receipts'/(stage['name']+'.json')),current)
    from intent_baselines import centroid_predictions
    test=read_rows(DATA/'n500/test.jsonl');pred=centroid_predictions(read_rows(DATA/'n500/train.jsonl'),test)
    if pred!=read_rows(OUT/'centroid.predictions.jsonl') or score_intents(test,pred)!=load(OUT/'centroid.json'):raise ValueError('CPU baseline recomputation differs')
    baselines={};runs={}
    for key in KEYS:
        baselines[key]={}
        for mode,condition,recipe in [('none','zero','A'),('none','fewshot','A'),('nf4','zero','D'),('int8','zero','E')]:
            baselines[key][mode+'_'+condition]=audit_evaluation(config_path(key,recipe),condition,ROOT/baseline_path(key,mode+'_'+condition))
        for recipe in RECIPES:
            result=audit(config_path(key,recipe))
            if result!=load(ROOT/run_dir(key,recipe)/'audit.json'):raise ValueError('Independent run audit differs')
            runs[key+'_'+recipe]=result
    result=dict(status='complete_research_banking15_first_seed_audited',completed_stages=len(stages()),runs=runs,baselines=baselines,centroid=load(OUT/'centroid.json'),limitations='Single-seed fixed card10 descriptive comparison, not full77/OOS, significant improvement, deep style or forgetting proof. Different backbone sizes and fused projections; not parameter-matched architecture causality. LaMP and additional seeds remain separate future experiments.')
    if (OUT/'results.json').exists():
        if load(OUT/'results.json')!=result:raise ValueError('Existing summary differs')
    else:fresh(OUT/'results.json',result)
    table=['# BANKING500 first-seed audited results','',result['limitations'],'','|Model / recipe|Macro-F1|Accuracy|Invalid rate|Best step|Train peak allocated GiB|','|---|---:|---:|---:|---:|---:|']
    for name,r in runs.items():
        q=r['quality'];table.append(f"|{name}|{q['macro_f1']:.4f}|{q['accuracy']:.4f}|{q['invalid_rate']:.4f}|{r['best_step']}|{r['training_resources']['peak_allocated_bytes']/2**30:.2f}|")
    text='\n'.join(table)+'\n';p=OUT/'results.md'
    if p.exists() and p.read_text()!=text:raise ValueError('Readable summary differs')
    if not p.exists():
        with p.open('x',encoding='utf-8') as f:f.write(text)
    write_json(OUT/'execution_state.json',dict(stage='complete_audited',completed_stages=len(stages()),summary='results.json'))
    print(json.dumps(dict(status=result['status'],completed_stages=len(stages()))))

def run(resume=False,checkpoint=None):
    with lock():
        current=check();manifest=OUT/'manifest.json';previous=load(OUT/'execution_state.json') if (OUT/'execution_state.json').exists() else {}
        if manifest.exists():
            if not resume or load(manifest)!=dict(inputs=current,stages=stages()):raise ValueError('Explicit resume with identical frozen inputs required')
        else:
            if resume:raise ValueError('No campaign to resume')
            if any((ROOT/p).exists() for st in stages() for p in st['outputs']):raise FileExistsError('Unreceipted prior output')
            fresh(manifest,dict(inputs=current,stages=stages()))
        subprocess.run([sys.executable,'scripts/prepare_8b_4090.py','check'],check=True)
        if resume and (OUT/'pause.request').exists():(OUT/'pause.request').rename(OUT/f'pause.consumed.{time.time_ns()}')
        for stage in stages():
            name=stage['name'];receipt=OUT/'receipts'/(name+'.json')
            if receipt.exists():verify_receipt(stage,load(receipt),current);continue
            if (OUT/'pause.request').exists():write_json(OUT/'execution_state.json',dict(stage='paused_between_stages',next_stage=name));return
            attempts=OUT/'attempts'/name;command=list(stage['command'])
            if attempts.exists():
                if stage['kind']!='train' or not checkpoint or previous.get('stage')!='paused_checkpoint_verified' or previous.get('active_stage')!=name:raise ValueError('Incomplete/failed attempt retained; diagnose, no automatic retry')
                target=Path(checkpoint).resolve();cfg=load(ROOT/stage['config']);runpath=ROOT/cfg['output_dir']
                if target.parent!=(runpath/'checkpoints').resolve() or target!=Path(previous['checkpoint']).resolve():raise ValueError('Wrong checkpoint')
                verify_checkpoint(target,inputs_snapshot(cfg,Path(stage['config'])));command+=['--resume-from-checkpoint',str(target)]
            elif any((ROOT/p).exists() for p in stage['outputs']):raise FileExistsError('Partial evidence retained')
            attempts.mkdir(parents=True,exist_ok=True);number=len(list(attempts.glob('*.started.json')))+1
            fresh(attempts/f'{number}.started.json',dict(command=command,started_unix=time.time()))
            write_json(OUT/'execution_state.json',dict(stage='running',active_stage=name,attempt=number))
            print(json.dumps(dict(stage='running',active_stage=name,attempt=number)),flush=True)
            try:
                with (attempts/f'{number}.stdout.log').open('xb') as out,(attempts/f'{number}.stderr.log').open('xb') as err:subprocess.run(command,check=True,stdout=out,stderr=err)
                if inputs()!=current:raise ValueError('Frozen input identity changed')
                if stage['kind']=='train':
                    cfg=load(ROOT/stage['config']);status=load(ROOT/cfg['output_dir']/'run_status.json')
                    if status['stage']=='paused_checkpoint_verified':
                        verify_checkpoint(Path(status['checkpoint']),inputs_snapshot(cfg,Path(stage['config'])))
                        write_json(OUT/'execution_state.json',dict(stage='paused_checkpoint_verified',active_stage=name,checkpoint=status['checkpoint']));return
                fresh(receipt,dict(command=stage['command'],executed_command=command,inputs=current,outputs={p:sha(ROOT/p) for p in stage['outputs']}))
            except BaseException:
                write_json(OUT/'execution_state.json',dict(stage='failed',active_stage=name));raise
        report(current)

def pause():
    status=load(OUT/'execution_state.json')
    if status['stage']!='running':raise ValueError('No running campaign recorded')
    fresh(OUT/'pause.request',dict(requested_unix=time.time()))
    print('Pause requested. Wait for paused state and process exit. Training saves at the next scheduled16-step boundary; generation finishes its current full400 stage.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',default='plan',choices=['plan','check','run','report','pause','centroid']);p.add_argument('--resume',action='store_true');p.add_argument('--checkpoint');a=p.parse_args();os.chdir(ROOT)
    if (a.resume or a.checkpoint) and a.action!='run':p.error('Resume flags require run')
    if a.checkpoint and not a.resume:p.error('Checkpoint requires --resume')
    if a.action=='plan':print(json.dumps(dict(executable=False,trainings=15,generations=27,stages=stages()),indent=2));return
    if platform.system()!='Linux':raise RuntimeError('Owner Linux4090 only; no new local experiments')
    if a.action=='check':check()
    elif a.action=='run':run(a.resume,a.checkpoint)
    elif a.action=='report':
        with lock():report()
    elif a.action=='pause':pause()
    else:centroid()

if __name__=='__main__':main()
