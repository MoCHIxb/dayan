"""Portable, opt-in RTX4090 reproduction. No task starts without --execute."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import run_peft_sequence as local

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/cloud_4090_v1'
MANIFEST=WORK/'manifest.json'
SEEDS=(20251206,20251207,20251208)
VARIANTS=('bf16_r16_a32','bf16_r8_a16','bf16_r16_a16','nf4_r16_a32')
read=local.seq.read
write=local.seq.write
sha=local.seq.sha
rows=local.rows
DATA=ROOT/'data/processed/lamp7_bm25_k12/test.jsonl'


def configurations():
    for variant in VARIANTS:
        for seed in SEEDS:
            path=ROOT/f'configs/cloud_4090_v1/{variant}_seed{seed}.json'
            yield variant,seed,path,read(path)


def asset_check(check_bert=True):
    spec=read(ROOT/'repro/local_assets.json')
    for name,digest in spec['sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Asset mismatch: {name}')
    for split,count in spec['records'].items():
        records=rows(ROOT/f'data/processed/lamp7_bm25_k12/{split}.jsonl')
        if len(records)!=count or len({str(r['id']) for r in records})!=count:
            raise ValueError(f'Incomplete split: {split}')
    if check_bert:
        from huggingface_hub import hf_hub_download
        bert=spec['bertscore_model']
        for name,digest in bert['files_sha256'].items():
            cached=Path(hf_hub_download(bert['repo_id'],name,local_files_only=True))
            if sha(cached)!=digest: raise ValueError(f'BERTScore cached model mismatch: {name}')
    reference={(r['variant'],r['seed']):r for r in read(ROOT/'results/local_500_reference.json')['runs']}
    for variant,seed,_,config in configurations():
        scientific={k:v for k,v in config.items() if k not in ('output_dir','protocol_id')}
        if scientific!=reference[variant,seed]['config']:
            raise ValueError(f'Cloud config differs from local scientific settings: {variant}/{seed}')
        expected=f'experiments/cloud_4090_v1/runs/{variant}_seed{seed}'
        if config['output_dir']!=expected or config['protocol_id']!='cloud_4090_v1':
            raise ValueError('Cloud output must be isolated from local results')


def verify_stack():
    expected={}
    for line in (ROOT/'requirements-train.lock').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name,version=line.split('==',1); expected[name]=version
    actual={name:metadata.version(name) for name in expected}
    mismatch={name:(expected[name],actual[name]) for name in expected if actual[name]!=expected[name]}
    if mismatch: raise ValueError(f'Pinned package mismatch; do not silently change stack: {mismatch}')
    if sys.version_info[:2]!=(3,12): raise ValueError('Use Python3.12 to match the local experiment')
    return actual


def freeze():
    if MANIFEST.exists(): raise FileExistsError('Cloud manifest already exists; never overwrite it')
    asset_check(); verify_stack()
    if any((ROOT/c['output_dir']).exists() for _,_,_,c in configurations()):
        raise FileExistsError('Cloud run directories must be new before freeze')
    names=[p for p in (ROOT/'scripts').glob('*.py')]
    names+=list((ROOT/'src/personal_llm_data').glob('*.py'))
    names+=[p for _,_,p,_ in configurations()]
    names+=[ROOT/'repro/local_assets.json',ROOT/'results/local_500_reference.json',ROOT/'requirements-train.lock',ROOT/'docs/27_cloud_4090_reproduction.md']
    write(MANIFEST,dict(protocol_id='cloud_4090_v1',files_sha256={str(p.relative_to(ROOT)):sha(p) for p in names},
        fixed_optimizer_steps=128,dev_records=1500,test_records=1496,bootstrap_resamples=2000,
        retries=0,note='Cross-platform reproduction, not a GPU-only causal comparison'))
    print('Cloud assets and 12 configurations frozen. No training started.')


def preflight():
    manifest=read(MANIFEST)
    for name,digest in manifest['files_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Frozen cloud implementation changed: {name}')


def status(stage,**kwargs):
    write(WORK/'execution_state.json',dict(protocol_id='cloud_4090_v1',stage=stage,pid=os.getpid(),
        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kwargs))
    print(stage,kwargs,flush=True)


def command(script,args,log):
    status('running',script=script,arguments=list(map(str,args)),log=str(log))
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def verify_receipt(output):
    record=read(output.with_suffix(output.suffix+'.receipt.json'))
    for relative,digest in record['outputs_sha256'].items():
        if sha(ROOT/relative)!=digest: raise ValueError(f'Completed output changed: {relative}')
    for relative,digest in record['inputs_sha256'].items():
        if sha(ROOT/relative)!=digest: raise ValueError(f'Task input changed: {relative}')
    return record


def task(script,args,output,inputs=()):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    receipt=output.with_suffix(output.suffix+'.receipt.json')
    signature=dict(script=script,script_sha256=sha(ROOT/'scripts'/script),arguments=list(map(str,args)))
    if output.exists() or receipt.exists():
        record=verify_receipt(output)
        if record['command']!=signature: raise ValueError(f'Command changed: {output}')
        return
    sidecar=output.with_suffix('.predictions.jsonl') if script=='evaluate_generation.py' else output.with_suffix('.per_item.jsonl') if script=='evaluate_full_dev.py' else None
    if sidecar is not None and sidecar.exists(): raise FileExistsError('Partial sidecar exists; inspect before retry')
    command(script,args,output.with_suffix(output.suffix+'.log'))
    read(output)
    produced=[output]+([sidecar] if sidecar is not None else [])
    write(receipt,dict(command=signature,outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in produced},
        inputs_sha256={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in inputs}))


def predictions(path):
    actual=rows(path); expected={str(r['id']):r['messages'][-1]['content'] for r in rows(DATA)}
    if len(actual)!=1496 or len({str(r['id']) for r in actual})!=1496 or {str(r['id']):r['reference'] for r in actual}!=expected:
        raise ValueError(f'Incomplete/mismatched test predictions: {path}')


def generation(run,quant,drop,base=False):
    name=('base' if base else 'adapter')+('_no_profile' if drop else '_profile')
    output=run/f'{name}_test_metrics.json'
    args=['--model','models/Qwen3-4B','--data',DATA,'--output',output,'--max-samples',1496,
          '--batch-size',4,'--max-input-tokens',768,'--max-new-tokens',80,'--quantization',quant]
    inputs=[DATA]
    if not base:
        args+=['--adapter',run/'adapter']; inputs+=[run/'adapter/adapter_model.safetensors']
    if drop: args+=['--drop-profile']
    task('evaluate_generation.py',args,output,inputs)
    predictions(output.with_suffix('.predictions.jsonl'))
    if read(output)['samples']!=1496: raise ValueError('Incomplete generation')


def paths_for(run,quant):
    base=WORK/'baseline'/quant
    return {name:(base if name.startswith('base_') else run)/f'{name}_test_metrics.predictions.jsonl'
            for name in ('base_no_profile','base_profile','adapter_no_profile','adapter_profile')}


def gpu_phase():
    asset_check(); packages=verify_stack()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    other_pids=[p.strip() for p in inventory.splitlines() if p.strip() and p.strip()!=str(os.getpid())]
    if other_pids: raise RuntimeError('Another GPU compute process is present; inspect nvidia-smi first')
    if '4090' not in torch.cuda.get_device_name(0): raise RuntimeError('This round requires the planned RTX4090; use a separate protocol for other GPUs')
    if (WORK/'gpu_completion.json').exists(): raise FileExistsError('GPU phase already complete; use the statistics phase')
    environment=dict(platform=platform.platform(),python=platform.python_version(),packages=packages,
        gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write(WORK/f'environment_{time.time_ns()}.json',environment)
    for variant,seed,path,config in configurations():
        run=ROOT/config['output_dir']
        if not (run/'training_summary.json').exists():
            command('train_qwen_lora.py',['--config',path],WORK/'logs'/f'{variant}_seed{seed}.train.log')
        result=local.audit(config)
        write(run/'cloud_training_audit.json',result)
    for quant in ('none','nf4'):
        for drop in (True,False): generation(WORK/'baseline'/quant,quant,drop,base=True)
    records={str(r['id']):r for r in rows(DATA)}
    for _,_,_,config in configurations():
        run=ROOT/config['output_dir']; quant=config['quantization']
        for drop in (True,False): generation(run,quant,drop)
        paths=paths_for(run,quant)
        shared=[v for name,path in paths.items() for v in ('--'+name.replace('_','-'),path)]
        out=run/'full_dev.json'
        task('evaluate_full_dev.py',['--adapter',run/'adapter','--quantization',quant,'--output',out],out,
             [ROOT/config['eval_file'],run/'adapter/adapter_model.safetensors'])
        if read(out)['samples']!=1500: raise ValueError('Incomplete full dev')
        out=run/'bertscore.json'
        task('bertscore_eval.py',[*shared,'--output',out,'--model','roberta-large','--batch-size',32,
             '--bootstrap',2000,'--seed',20251206],out,paths.values())
        scores=read(out)
        if scores['samples']!=1496 or not scores['idf'] or not scores['rescale_with_baseline']: raise ValueError('BERTScore mismatch')
        out=run/'error_analysis.json'
        task('analyze_predictions.py',[*shared,'--data',DATA,'--output',out],out,[DATA,*paths.values()])
        diagnosis={}
        for name,path in paths.items():
            groups={'all':[],'reference_in_history':[],'reference_not_in_history':[]}
            for p in rows(path):
                history=[h['text'] for h in records[str(p['id'])]['profile']]; copied=p['prediction'] in history
                groups['all'].append(copied)
                groups['reference_in_history' if p['reference'] in history else 'reference_not_in_history'].append(copied)
            diagnosis[name]={g:dict(samples=len(v),exact_history_count=sum(v),exact_history_rate=sum(v)/len(v)) for g,v in groups.items()}
        out=run/'copy_diagnostics.json'; value=dict(conditions=diagnosis,scope='descriptive exact-copy proxy')
        if out.exists() and read(out)!=value: raise ValueError('Copy diagnosis changed')
        if not out.exists(): write(out,value)
    # Freeze the exact inputs needed for CPU-only statistics, portable across roots.
    complete={}
    for path in WORK.rglob('*.json'):
        if path.name.endswith('.receipt.json'):
            verify_receipt(Path(str(path)[:-len('.receipt.json')]))
            complete.update(read(path)['outputs_sha256'])
    for _,_,_,config in configurations():
        run=ROOT/config['output_dir']
        for name in ('training_summary.json','cloud_training_audit.json','copy_diagnostics.json'):
            complete[str((run/name).relative_to(ROOT))]=sha(run/name)
    write(WORK/'gpu_completion.json',dict(files_sha256=complete,trainings=12,generation_groups=28,auxiliary_sets=12))
    status('gpu_complete_statistics_pending',trainings=12,generation_groups=28,auxiliary_sets=12)


def check_gpu_completion():
    value=read(WORK/'gpu_completion.json')
    if (value['trainings'],value['generation_groups'],value['auxiliary_sets'])!=(12,28,12): raise ValueError('Incomplete GPU phase')
    for name,digest in value['files_sha256'].items():
        if sha(ROOT/name)!=digest: raise ValueError(f'GPU output modified: {name}')


def statistics_phase():
    verify_stack(); check_gpu_completion()
    for variant,seed,_,config in configurations():
        run=ROOT/config['output_dir']; paths=paths_for(run,config['quantization'])
        for path in paths.values(): predictions(path)
        shared=[v for name,path in paths.items() for v in ('--'+name.replace('_','-'),path)]
        out=run/'paired_2x2_bootstrap.json'
        task('paired_bootstrap.py',[*shared,'--output',out,'--samples',2000,'--seed',20251206],out,paths.values())
        if variant!='bf16_r16_a32':
            reference=WORK/f'runs/bf16_r16_a32_seed{seed}/adapter_profile_test_metrics.predictions.jsonl'
            out=run/'paired_vs_bf16_r16_a32.json'
            task('paired_adapter_comparison.py',['--reference-system',reference,'--candidate-system',paths['adapter_profile'],
                 '--reference-name','bf16_r16_a32','--candidate-name',variant,'--output',out,'--samples',2000,'--seed',20251206],out,[reference,paths['adapter_profile']])
    bootstrap=list((WORK/'runs').glob('*/paired*.json'))
    if len(bootstrap)!=21: raise ValueError('Expected 21 cloud original-bootstrap tasks')
    for out in bootstrap:
        verify_receipt(out); value=read(out)
        if value['samples']!=1496 or value['bootstrap_resamples']!=2000: raise ValueError('Incomplete bootstrap')
    report()
    status('cloud_round_complete',result='experiments/cloud_4090_v1/results.json',original_bootstrap_tasks=21)


def report():
    reference={(r['variant'],r['seed']):r for r in read(ROOT/'results/local_500_reference.json')['runs']}
    result=[]
    for variant,seed,_,config in configurations():
        run=ROOT/config['output_dir']; summary=read(run/'training_summary.json'); metrics=read(run/'adapter_profile_test_metrics.json')
        old=reference[variant,seed]
        result.append(dict(variant=variant,seed=seed,metrics=metrics,
            training={k:summary[k] for k in ('elapsed_seconds','peak_gpu_memory_bytes','peak_gpu_reserved_bytes','trainable_parameters','best_dev_loss','device')},
            local_reference=old,descriptive_delta=dict(bleu=metrics['bleu']-old['metrics_profile']['bleu'],
                training_seconds=summary['elapsed_seconds']-old['training']['elapsed_seconds'])))
    write(WORK/'results.json',dict(scope='Linux4090 vs Windows5070Ti platform comparison; not isolated GPU effect',runs=result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--freeze',action='store_true')
    group.add_argument('--check-assets',action='store_true')
    p.add_argument('--phase',choices=('gpu','statistics'))
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.freeze: freeze(); return
    if args.check_assets: asset_check(); print('Local-matching data, model, source and BERTScore assets verified.'); return
    if not args.execute:
        if MANIFEST.exists(): preflight()
        print('No task started. Plan: 12 trainings, 28 full-test generations, 12 auxiliary sets, 21 original bootstrap tasks.'); return
    if args.phase is None: p.error('--execute requires --phase')
    preflight(); WORK.mkdir(parents=True,exist_ok=True)
    state=WORK/'execution_state.json'
    if state.exists() and not args.resume: raise FileExistsError('State exists; inspect before explicit --resume')
    os.environ.update(PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    lock=WORK/'sequence.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        gpu_phase() if args.phase=='gpu' else statistics_phase()
    except BaseException:
        status('failed_no_automatic_retry',error=traceback.format_exc()); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__': main()
