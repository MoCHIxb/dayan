"""Frozen, bounded small-sample PEFT study; default is read-only verification."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import run_small_sample_sequence as seq

ROOT = seq.ROOT
WORK = ROOT/'experiments/peft_v1'
MANIFEST = WORK/'manifest.json'
SEEDS = (20251206, 20251207, 20251208)
VARIANTS = {'bf16_r8_a16': (8,16,'none'), 'bf16_r16_a16': (16,16,'none'), 'nf4_r16_a32': (16,32,'nf4')}


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def baseline(seed):
    return ROOT/f'experiments/runs/small-sample-v1-n500_lr1e4_seed{seed}'


def configurations():
    for variant, (rank, alpha, quant) in VARIANTS.items():
        for seed in SEEDS:
            yield variant, seed, ROOT/f'configs/peft_v1/{variant}_seed{seed}.json'


def status(stage, **kwargs):
    value = dict(stage=stage, protocol_id='peft_v1', supervisor_pid=os.getpid(),
                 updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs)
    seq.write(WORK/'execution_state.json', value)
    print(json.dumps(value), flush=True)


def audit(config, write=False):
    run = ROOT/config['output_dir']
    summary = seq.read(run/'training_summary.json')
    ev = [r for r in summary['log_history'] if 'eval_loss' in r]
    if summary['config'] != config or (summary['train_records'], summary['eval_records'], summary['completed_optimizer_steps']) != (500,1500,128):
        raise ValueError(f'Incomplete training: {run}')
    if [r['step'] for r in ev] != list(range(16,129,16)):
        raise ValueError('Eight full dev evaluations required')
    for r in summary['log_history']:
        if any(k in r and not math.isfinite(r[k]) for k in ('loss','eval_loss','grad_norm')):
            raise ValueError('Nonfinite training value')
    best = min(ev, key=lambda r: (r['eval_loss'],r['step']))
    if summary['best_dev_loss'] != best['eval_loss'] or Path(summary['best_model_checkpoint']).name != f"checkpoint-{best['step']}":
        raise ValueError('Best checkpoint metadata mismatch')
    digest = seq.sha(run/'adapter/adapter_model.safetensors')
    if digest != seq.sha(run/f"checkpoints/checkpoint-{best['step']}/adapter_model.safetensors"):
        raise ValueError('Adapter hash mismatch')
    if seq.read(run/'checkpoints/checkpoint-128/trainer_state.json')['global_step'] != 128:
        raise ValueError('Final checkpoint incomplete')
    if summary['quantization'] != config['quantization'] or (config['quantization']=='nf4' and summary['quantized_linear_modules'] <= 0):
        raise ValueError('Quantization recipe mismatch')
    result = dict(run=config['output_dir'], config=config, best_step=best['step'],
        best_dev_loss=best['eval_loss'], adapter_sha256=digest,
        trainable_parameters=summary['trainable_parameters'], elapsed_seconds=summary['elapsed_seconds'],
        peak_gpu_memory_bytes=summary['peak_gpu_memory_bytes'], peak_gpu_reserved_bytes=summary['peak_gpu_reserved_bytes'])
    if write:
        seq.write(run/'peft_training_audit.json', result)
    return result


def freeze():
    if MANIFEST.exists():
        raise FileExistsError('Manifest already frozen')
    # Verify original frozen dependencies before inheriting them.
    old = seq.read(ROOT/'experiments/refocused/small_sample_v1_manifest.json')['sha256']
    for name, digest in old.items():
        if seq.sha(ROOT/name) != digest:
            raise ValueError(f'Original asset changed: {name}')
    assets = dict(old)
    for variant, seed, path in configurations():
        c = seq.read(path)
        expected = seq.read(baseline(seed)/'run_config.json')
        rank, alpha, quant = VARIANTS[variant]
        expected.update(protocol_id='peft_v1', output_dir=f'experiments/runs/peft-v1-{variant}_seed{seed}',
                        lora_r=rank, lora_alpha=alpha, quantization=quant)
        if c != expected:
            raise ValueError(f'Unexpected experimental change: {path}')
        if (ROOT/c['output_dir']).exists():
            raise FileExistsError('New run path must not exist before freeze')
        assets[str(path.relative_to(ROOT))] = seq.sha(path)
    for seed in SEEDS:
        run = baseline(seed)
        audit(seq.read(run/'run_config.json'))
        for path in run.iterdir():
            if path.is_file() and (path.suffix in ('.json','.jsonl') or path.name.endswith('.receipt.json')):
                assets[str(path.relative_to(ROOT))] = seq.sha(path)
        for path in (run/'adapter').iterdir():
            if path.is_file():
                assets[str(path.relative_to(ROOT))] = seq.sha(path)
    for path in (ROOT/'experiments/refocused/baseline').iterdir():
        if path.is_file() and path.suffix in ('.json','.jsonl'):
            assets[str(path.relative_to(ROOT))] = seq.sha(path)
    for name in ('run_peft_sequence.py','run_small_sample_sequence.py','evaluate_generation.py','evaluate_full_dev.py',
                 'bertscore_eval.py','analyze_predictions.py','paired_bootstrap.py','paired_adapter_comparison.py'):
        assets['scripts/'+name] = seq.sha(ROOT/'scripts'/name)
    for name in ('src/personal_llm_data/model_loading.py','src/personal_llm_data/pipeline.py','docs/24_small_sample_peft_protocol.md'):
        assets[name] = seq.sha(ROOT/name)
    seq.write(MANIFEST, dict(protocol_id='peft_v1', assets=assets, retries=0, recurring=False,
        configs=[str(p.relative_to(ROOT)) for _,_,p in configurations()]))
    print(f'Frozen {len(assets)} assets; no GPU task started.')


def preflight():
    manifest = seq.read(MANIFEST)
    for name, digest in manifest['assets'].items():
        if seq.sha(ROOT/name) != digest:
            raise ValueError(f'Frozen asset changed: {name}')
    for split, count in (('train_500',500),('dev',1500),('test',1496)):
        records = rows(ROOT/f'data/processed/lamp7_bm25_k12/{split}.jsonl')
        if len(records) != count or len({str(r['id']) for r in records}) != count:
            raise ValueError(f'Incomplete {split}')
    for seed in SEEDS:
        audit(seq.read(baseline(seed)/'run_config.json'))
    print('Frozen assets and three complete BF16 controls verified.', flush=True)


def check_processes():
    # Host-visible process inventory, run under the same approved host context as supervisor.
    code = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?(\\.exe)?$' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    value = subprocess.check_output(['powershell','-NoProfile','-Command',code], text=True).strip()
    records = json.loads(value) if value else []
    if isinstance(records,dict): records=[records]
    # Windows venv launcher + child may represent this one invocation.
    related = {os.getpid(), os.getppid()}
    for r in records:
        if int(r['ProcessId']) in related or (int(r['ParentProcessId']) in related and 'run_peft_sequence.py' in (r['CommandLine'] or '')):
            continue
        raise RuntimeError(f"Another Python process is present; inspect PID {r['ProcessId']} before GPU work")
    if (ROOT/'experiments/refocused/small_sample_sequence.lock').exists():
        raise FileExistsError('v1 lock present; inspect before starting')


def task(script, args, output):
    # No overwrites; frozen helper checks command/hash receipt on reuse.
    seq.task(script, args, output)


def generation(run, quant, drop, base=False):
    name = ('base' if base else 'adapter') + ('_no_profile' if drop else '_profile')
    out = run/f'{name}_test_metrics.json'
    args = ['--model','models/Qwen3-4B','--data',seq.DATA,'--output',out,'--max-samples',1496,
            '--batch-size',4,'--max-input-tokens',768,'--max-new-tokens',80,'--quantization',quant]
    if not base: args += ['--adapter',run/'adapter']
    if drop: args += ['--drop-profile']
    task('evaluate_generation.py',args,out)
    seq.validate_predictions(out.with_suffix('.predictions.jsonl'))
    if seq.read(out)['samples'] != 1496: raise ValueError('Incomplete generation')
    return out.with_suffix('.predictions.jsonl')


def evaluate(result):
    run = ROOT/result['run']; quant = result['config']['quantization']
    base_dir = ROOT/'experiments/refocused/baseline' if quant=='none' else WORK/'nf4_baseline'
    paths = {name:base_dir/f'{name}_test_metrics.predictions.jsonl' for name in ('base_no_profile','base_profile')}
    for drop in (True,False):
        name = 'adapter_no_profile' if drop else 'adapter_profile'
        paths[name] = generation(run,quant,drop)
    for path in paths.values(): seq.validate_predictions(path)
    shared = [v for name,path in paths.items() for v in ('--'+name.replace('_','-'),path)]
    out = run/'full_dev.json'
    task('evaluate_full_dev.py',['--adapter',run/'adapter','--quantization',quant,'--output',out],out)
    dev = seq.read(out)
    if dev['samples'] != 1500 or dev['adapter_sha256'] != result['adapter_sha256']:
        raise ValueError('Invalid complete dev evidence')
    out = run/'bertscore.json'
    task('bertscore_eval.py',[*shared,'--output',out,'--model','roberta-large','--batch-size',32,
                            '--bootstrap',2000,'--seed',20251206],out)
    scores = seq.read(out)
    if scores['samples'] != 1496 or not scores['idf'] or not scores['rescale_with_baseline']:
        raise ValueError('Invalid BERTScore recipe')
    out = run/'error_analysis.json'
    task('analyze_predictions.py',[*shared,'--data',seq.DATA,'--output',out],out)
    records = {str(r['id']):r for r in rows(seq.DATA)}
    diagnosis = {}
    for name,path in paths.items():
        groups = {'all':[], 'reference_in_history':[], 'reference_not_in_history':[]}
        for p in rows(path):
            history = [h['text'] for h in records[str(p['id'])]['profile']]
            copied = p['prediction'] in history
            groups['all'].append(copied)
            groups['reference_in_history' if p['reference'] in history else 'reference_not_in_history'].append(copied)
        diagnosis[name] = {g:dict(samples=len(v),exact_history_count=sum(v),exact_history_rate=sum(v)/len(v)) for g,v in groups.items()}
    out = run/'copy_diagnostics.json'
    value = dict(scope='descriptive exact-copy diagnostic; not proof of deep style or factual error',conditions=diagnosis)
    if out.exists() and seq.read(out) != value: raise ValueError('Copy diagnosis changed')
    if not out.exists(): seq.write(out,value)
    return shared,paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--freeze',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if args.freeze: freeze(); return
    preflight()
    if not (args.run or args.resume): return
    check_processes()
    if args.run and (WORK/'execution_state.json').exists():
        raise FileExistsError('Previous execution state exists; inspect before explicit resume')
    if args.resume and seq.read(WORK/'execution_state.json')['stage'] == 'peft_v1_complete':
        raise ValueError('Study already complete')
    WORK.mkdir(exist_ok=True)
    lock = WORK/'sequence.lock'
    with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))
    seq.WORK = WORK
    seq.status = status
    os.environ.update(PYTHONPATH=str(ROOT/'src'), PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    try:
        (WORK/'logs').mkdir(exist_ok=True)
        trained=[]
        for variant,seed,path in configurations():
            config=seq.read(path); run=ROOT/config['output_dir']
            if not (run/'training_summary.json').exists():
                seq.execute('train_qwen_lora.py',['--config',path],WORK/'logs'/f'{path.stem}.train.log')
            result=audit(config,write=True)
            result.update(variant=variant,seed=seed)
            trained.append(result)
        for drop in (True,False): generation(WORK/'nf4_baseline','nf4',drop,base=True)
        evaluated=[(r,*evaluate(r)) for r in trained]
        status('gpu_stage_complete_cpu_bootstrap',new_trainings=9,adapter_generations=18,nf4_base_generations=2)
        for r,shared,paths in evaluated:
            run=ROOT/r['run']; out=run/'paired_2x2_bootstrap.json'
            task('paired_bootstrap.py',[*shared,'--output',out,'--samples',2000,'--seed',20251206],out)
            out2=run/'paired_vs_bf16_r16_a32.json'
            task('paired_adapter_comparison.py',['--reference-system',baseline(r['seed'])/'adapter_profile_test_metrics.predictions.jsonl',
                '--candidate-system',paths['adapter_profile'],'--reference-name','bf16_r16_a32',
                '--candidate-name',r['variant'],'--output',out2,'--samples',2000,'--seed',20251206],out2)
            for item in (out,out2):
                if seq.read(item)['samples'] != 1496 or seq.read(item)['bootstrap_resamples'] != 2000:
                    raise ValueError('Incomplete original corpus bootstrap')
        all_results=[]
        for r in trained + [dict(audit(seq.read(baseline(s)/'run_config.json')),variant='bf16_r16_a32',seed=s) for s in SEEDS]:
            run=ROOT/r['run']
            all_results.append(dict(training=r,metrics=seq.read(run/'adapter_profile_test_metrics.json'),
                bertscore=seq.read(run/'bertscore.json')['condition_scores']['adapter_profile'],
                copy=seq.read(run/'copy_diagnostics.json')['conditions']['adapter_profile']['all']))
        report=dict(protocol_id='peft_v1',runs=all_results,aggregate={})
        for variant in ('bf16_r16_a32',*VARIANTS):
            group=[r for r in all_results if r['training']['variant']==variant]
            report['aggregate'][variant]={m:dict(mean=statistics.mean(r['metrics'][m] for r in group),sample_sd=statistics.stdev(r['metrics'][m] for r in group)) for m in ('bleu','chrf','rougeL_f')}
        seq.write(WORK/'results.json',report)
        lines=['# 小样本PEFT结果','','全部三种子、1496条test，完整配方见24号协议。','','| 条件 | BLEU均值 | 样本SD | chrF均值 |','|---|---:|---:|---:|']
        for name,v in report['aggregate'].items():
            lines.append(f"| {name} | {v['bleu']['mean']:.4f} | {v['bleu']['sample_sd']:.4f} | {v['chrf']['mean']:.4f} |")
        lines+=['','原始配对bootstrap、资源、语义及复制诊断见各run目录；区间未校正多重比较，不据跨0判等效。NF4为整体配方对照。仍需解释结果、新环境复现、代码许可证确定与正式发布。']
        (ROOT/'docs/25_small_sample_peft_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status('peft_v1_complete',result='experiments/peft_v1/results.json',remaining=['interpret PEFT results','clean environment reproduction','license and release'])
    except BaseException:
        status('failed_no_automatic_retry',error=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__': main()
