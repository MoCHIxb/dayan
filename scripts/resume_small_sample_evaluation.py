"""Resume only evaluation of the frozen, already-trained small-sample v1 matrix."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import run_small_sample_sequence as seq


def preflight():
    selection_path = seq.WORK / 'small_sample_selection.json'
    selection = seq.read(selection_path)
    manifest = seq.read(seq.WORK / 'small_sample_confirmation_manifest.json')
    if manifest['selection_sha256'] != seq.sha(selection_path):
        raise ValueError('Selection hash changed')
    implementation = seq.read(seq.WORK / 'small_sample_execution_manifest.json')
    for name, digest in implementation['scripts'].items():
        if seq.sha(seq.ROOT / 'scripts' / name) != digest:
            raise ValueError(f'Original evaluation implementation changed: {name}')
    configs = [seq.ROOT / p for p in seq.read(seq.WORK / 'small_sample_v1_manifest.json')['stage1_configs']]
    for name, digest in manifest['configs'].items():
        if seq.sha(seq.ROOT / name) != digest:
            raise ValueError('Confirmation configuration changed')
        configs.append(seq.ROOT / name)
    audits = {}
    for path in configs:
        config = seq.read(path)
        run = seq.ROOT / config['output_dir']
        summary = seq.read(run / 'training_summary.json')
        evaluations = [r for r in summary['log_history'] if 'eval_loss' in r]
        if summary['config'] != config or summary['eval_records'] != 1500 or summary['completed_optimizer_steps'] != 128:
            raise ValueError('Training summary is not complete')
        if [r['step'] for r in evaluations] != list(range(16,129,16)) or not all(math.isfinite(r['eval_loss']) for r in evaluations):
            raise ValueError('Dev trajectory is not complete')
        best = min(evaluations, key=lambda r: (r['eval_loss'], r['step']))
        digest = seq.sha(run / 'adapter/adapter_model.safetensors')
        if digest != seq.sha(run / f"checkpoints/checkpoint-{best['step']}/adapter_model.safetensors"):
            raise ValueError('Adapter differs from best checkpoint')
        audit = seq.read(run / 'training_audit.json')
        if audit['config'] != config or audit['adapter_sha256'] != digest or audit['best_step'] != best['step']:
            raise ValueError('Training audit mismatch')
        audits[config['output_dir']] = audit
    if len(audits) != 16:
        raise ValueError('Expected 16 complete trainings')
    runs = [audits[r['run']] for r in selection['selected']]
    runs += [audits[seq.read(seq.ROOT / name)['output_dir']] for name in manifest['configs']]
    if len(runs) != 12 or len({r['run'] for r in runs}) != 12:
        raise ValueError('Expected 12 selected adapters')
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    os.environ.update(PYTHONPATH=str(seq.ROOT/'src'), PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    runs = preflight()
    print('Verified 16 completed trainings and 12 selected adapters; no training will be started.', flush=True)
    if args.check_only:
        return
    lock = seq.WORK / 'small_sample_sequence.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        manifest_path = seq.WORK/'evaluation_resume_manifest.json'
        if manifest_path.exists():
            raise FileExistsError('Resume manifest exists; inspect before re-entry')
        seq.write(manifest_path, dict(resume_script_sha256=seq.sha(Path(__file__)),
            selection_sha256=seq.sha(seq.WORK/'small_sample_selection.json'),
            original_execution_manifest_sha256=seq.sha(seq.WORK/'small_sample_execution_manifest.json'),
            retries=0, runs=[r['run'] for r in runs]))
        # Reuse only original complete baseline receipts with exactly the original arguments.
        for name in ('base_no_profile', 'base_profile'):
            output=seq.WORK/'baseline'/f'{name}_test_metrics.json'
            arguments=['--model','models/Qwen3-4B','--data',seq.DATA,'--output',output,'--max-samples',1496,
                       '--batch-size',4,'--max-input-tokens',768,'--max-new-tokens',80,'--quantization','none']
            if name=='base_no_profile': arguments+=['--drop-profile']
            seq.task('evaluate_generation.py',arguments,output)
            seq.validate_predictions(output.with_suffix('.predictions.jsonl'))
        evaluated=[]
        for result in runs:
            run=seq.ROOT/result['run']
            shared,paths=seq.evaluate(run)
            evaluated.append((result,shared,paths))
        for result,shared,paths in evaluated:
            run=seq.ROOT/result['run']; output=run/'paired_2x2_bootstrap.json'
            seq.task('paired_bootstrap.py',[*shared,'--output',output,'--samples',2000,'--seed',20251206],output)
            if seq.read(output)['bootstrap_resamples']!=2000: raise ValueError('Incomplete bootstrap')
            n=Path(result['config']['train_file']).stem.split('_')[1]
            for label,reference in (
                ('historical_same_n',seq.ROOT/f'experiments/runs/qwen3-4b-lamp7-n{n}-lora-r16-a32/adapter_profile_test_metrics.predictions.jsonl'),
                ('historical_full',seq.BASE/'adapter_profile_test_metrics.predictions.jsonl')):
                output=run/f'paired_vs_{label}.json'
                seq.task('paired_adapter_comparison.py',['--reference-system',reference,'--candidate-system',paths['adapter_profile'],
                    '--reference-name',label,'--candidate-name','small_sample_v1','--output',output,'--samples',2000,'--seed',20251206],output)
                if seq.read(output)['bootstrap_resamples']!=2000: raise ValueError('Incomplete historical comparison')
            seq.write(run/'small_sample_result.json',dict(training=result,
                metrics={name:seq.read(run/f'{name}_test_metrics.json') for name in ('adapter_profile','adapter_no_profile')},
                bertscore=seq.read(run/'bertscore.json'),bootstrap=seq.read(run/'paired_2x2_bootstrap.json')))
        report={'protocol_id':'small_sample_v1','runs':[seq.read(seq.ROOT/r['run']/'small_sample_result.json') for r in runs],'aggregate':{}}
        for n in (50,100,200,500):
            group=[r for r in report['runs'] if r['training']['config']['train_file'].endswith(f'train_{n}.jsonl')]
            report['aggregate'][str(n)]={metric:{'mean':statistics.mean(r['metrics']['adapter_profile'][metric] for r in group),
                'sample_sd':statistics.stdev(r['metrics']['adapter_profile'][metric] for r in group)} for metric in ('bleu','chrf','rougeL_f')}
        seq.write(seq.WORK/'small_sample_results.json',report)
        lines=['# 小样本协议v1结果','','全部正式条件均使用1496条test，每档3种子。逐种子指标及完整评价见机器汇总和各run目录。','',
            '| 数据量 | BLEU均值 | 种子样本标准差 | chrF均值 | ROUGE-L均值 |','|---|---:|---:|---:|---:|']
        for n,v in report['aggregate'].items():
            lines.append(f"| {n} | {v['bleu']['mean']:.4f} | {v['bleu']['sample_sd']:.4f} | {v['chrf']['mean']:.4f} | {v['rougeL_f']['mean']:.4f} |")
        lines+=['','统计沿用2000次原始corpus_score配对bootstrap；名义区间未做多重比较校正。第一种子参与学习率选择，其余种子固定所选学习率。尚需PEFT对照、框架与指南交付，协议v1完成不等于整个课题完成。']
        (seq.ROOT/'docs/21_small_sample_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        seq.status('small_sample_v1_complete',result='experiments/refocused/small_sample_results.json',remaining_project_work=['interpret results','small-sample PEFT protocol','framework and guide delivery'])
    except BaseException:
        seq.status('failed_no_automatic_retry',error=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
