"""Audit local PEFT evidence and write quality/resource conclusions; no GPU work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import run_peft_sequence as study

ROOT=study.ROOT
WORK=study.WORK
read=study.seq.read
write=study.seq.write
sha=study.seq.sha
VARIANTS=('bf16_r16_a32','bf16_r8_a16','bf16_r16_a16','nf4_r16_a32')
SEEDS=study.SEEDS


def location(variant,seed):
    return study.baseline(seed) if variant=='bf16_r16_a32' else ROOT/f'experiments/runs/peft-v1-{variant}_seed{seed}'


def receipt(path,script=None,args=None):
    saved=read(path.with_suffix(path.suffix+'.receipt.json'))
    if saved['sha256']!=sha(path): raise ValueError(f'Result changed: {path}')
    command=saved['command']
    if command['script_sha256']!=sha(ROOT/'scripts'/command['script']): raise ValueError('Original statistic implementation changed')
    if script is not None and command!=dict(script=script,script_sha256=sha(ROOT/'scripts'/script),arguments=list(map(str,args))):
        raise ValueError('Completed statistic command differs from frozen protocol')
    return read(path)


def summarize(final=False):
    study.preflight()
    results=[]; sources={}; available=[]; missing=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            run=location(variant,seed); training=read(run/'training_summary.json')
            audit=study.audit(training['config'])
            scores={}
            for name in ('adapter_profile','adapter_no_profile'):
                out=run/f'{name}_test_metrics.json'; scores[name]=receipt(out)
                study.seq.validate_predictions(out.with_suffix('.predictions.jsonl'))
                sources[str(out.relative_to(ROOT))]=sha(out)
                sources[str(out.with_suffix('.predictions.jsonl').relative_to(ROOT))]=sha(out.with_suffix('.predictions.jsonl'))
            dev=receipt(run/'full_dev.json'); bert=receipt(run/'bertscore.json'); errors=receipt(run/'error_analysis.json')
            copy=read(run/'copy_diagnostics.json')
            if dev['samples']!=1500 or dev['adapter_sha256']!=audit['adapter_sha256']: raise ValueError('Incomplete dev')
            items=study.rows(run/'full_dev.per_item.jsonl')
            if len(items)!=1500 or len({str(r['id']) for r in items})!=1500: raise ValueError('Dev item IDs incomplete')
            if bert['samples']!=1496 or not bert['idf'] or not bert['rescale_with_baseline']: raise ValueError('BERTScore recipe mismatch')
            sources[str((run/'training_summary.json').relative_to(ROOT))]=sha(run/'training_summary.json')
            for name in ('full_dev.json','bertscore.json','error_analysis.json','copy_diagnostics.json'):
                sources[str((run/name).relative_to(ROOT))]=sha(run/name)
            entry=dict(variant=variant,seed=seed,best_step=audit['best_step'],best_dev_loss=audit['best_dev_loss'],
                bleu=scores['adapter_profile']['bleu'],chrf=scores['adapter_profile']['chrf'],rougeL=scores['adapter_profile']['rougeL_f'],
                bert_f1=bert['condition_scores']['adapter_profile']['f1'],
                profile_delta_bleu=scores['adapter_profile']['bleu']-scores['adapter_no_profile']['bleu'],
                copy_rate=copy['conditions']['adapter_profile']['all']['exact_history_rate'],
                trainable_parameters=training['trainable_parameters'],training_minutes=training['elapsed_seconds']/60,
                allocated_gib=training['peak_gpu_memory_bytes']/2**30,reserved_gib=training['peak_gpu_reserved_bytes']/2**30,
                generation_samples_per_second=scores['adapter_profile']['generation_samples_per_second'])
            if variant!='bf16_r16_a32':
                base=ROOT/'experiments/refocused/baseline' if variant.startswith('bf16') else WORK/'nf4_baseline'
                paths={name:(base if name.startswith('base') else run)/f'{name}_test_metrics.predictions.jsonl'
                    for name in ('base_no_profile','base_profile','adapter_no_profile','adapter_profile')}
                for path in paths.values(): study.seq.validate_predictions(path)
                shared=[v for name,path in paths.items() for v in ('--'+name.replace('_','-'),path)]
                one=run/'paired_2x2_bootstrap.json'; two=run/'paired_vs_bf16_r16_a32.json'
                plan=[('paired_bootstrap.py',[*shared,'--output',one,'--samples',2000,'--seed',20251206],one),
                    ('paired_adapter_comparison.py',['--reference-system',study.baseline(seed)/'adapter_profile_test_metrics.predictions.jsonl',
                     '--candidate-system',paths['adapter_profile'],'--reference-name','bf16_r16_a32','--candidate-name',variant,
                     '--output',two,'--samples',2000,'--seed',20251206],two)]
                for script,args,path in plan:
                    if not path.exists() or not path.with_suffix(path.suffix+'.receipt.json').exists():
                        missing.append(str(path.relative_to(ROOT))); continue
                    statistic=receipt(path,script,args)
                    if statistic['samples']!=1496 or statistic['bootstrap_resamples']!=2000 or statistic['seed']!=20251206:
                        raise ValueError('Original bootstrap scope changed')
                    available.append(str(path.relative_to(ROOT))); sources[str(path.relative_to(ROOT))]=sha(path)
                    if path==two:
                        expected=scores['adapter_profile']['bleu']-read(study.baseline(seed)/'adapter_profile_test_metrics.json')['bleu']
                        if abs(statistic['contrasts']['bleu']['observed']-expected)>1e-8: raise ValueError('Observed paired difference mismatch')
                        entry['paired_vs_bf16']=statistic['contrasts']
                    else:
                        for name,path0 in paths.items():
                            score=read(path0.with_name(path0.name.replace('.predictions.jsonl','.json')))
                            if abs(statistic['condition_scores'][name]['bleu']-score['bleu'])>1e-8: raise ValueError('2x2 full score mismatch')
            results.append(entry)
    if final and (missing or len(available)!=18): raise ValueError('Final analysis requires all 18 completed original-bootstrap tasks')
    metrics=('bleu','chrf','rougeL','bert_f1','profile_delta_bleu','copy_rate','trainable_parameters','training_minutes','allocated_gib','reserved_gib','generation_samples_per_second')
    aggregate={}
    for variant in VARIANTS:
        group=[r for r in results if r['variant']==variant]
        aggregate[variant]={k:dict(mean=statistics.mean(r[k] for r in group),sample_sd=statistics.stdev(r[k] for r in group)) for k in metrics}
    baseline=aggregate['bf16_r16_a32']
    for variant in VARIANTS:
        aggregate[variant]['relative_to_bf16_r16_a32']={k:100*(aggregate[variant][k]['mean']/baseline[k]['mean']-1)
            for k in ('trainable_parameters','training_minutes','allocated_gib')}
    report=dict(status='complete_original_statistics_audited' if final else 'descriptive_draft_statistics_pending',
        new_trainings_verified=9,reused_control_trainings_verified=3,complete_test_records=1496,full_dev_records=1500,
        completed_original_bootstrap_tasks=len(available),planned_original_bootstrap_tasks=18,missing_original_bootstrap=missing,
        aggregate=aggregate,runs=results,sources_sha256=sources,
        limitations=['Nominal per-comparison intervals, no multiplicity correction','Crossing zero does not demonstrate equivalence or noninferiority',
                    'Fixed training subset: seed variation does not include data/user sampling','500 training plus 1500 labeled development records',
                    'NF4 is a complete training/inference recipe, not a pure bit-width factor','One Qwen3-4B model on RTX5070Ti, no cloud4090 results yet'])
    write(WORK/('interpretation_final.json' if final else 'interpretation_draft.json'),report)
    write(ROOT/'results/peft_quality_cost_analysis.json',report)
    lines=['# 小样本PEFT质量与资源分析','',
        ('状态：18/18原始配对bootstrap及独立证据审计已完成。' if final else f'状态：描述性草稿，原始配对bootstrap已核验{len(available)}/18；当前不能提供最终显著性判断。'),
        '', '固定500条训练、1500条有标签dev、1496条test；各条件三种子。完整协议见24号文档。', '',
        '| 条件 | BLEU均值±SD | BERTScore F1 | 可训练参数(百万) | allocated峰值GiB | 训练分钟 | 历史整句复制率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,a in aggregate.items():
        mean=lambda key:a[key]['mean']
        lines.append(f"| {name} | {mean('bleu'):.4f} ± {a['bleu']['sample_sd']:.4f} | {mean('bert_f1'):.4f} | {mean('trainable_parameters')/1e6:.3f} | {mean('allocated_gib'):.3f} | {mean('training_minutes'):.2f} | {100*mean('copy_rate'):.2f}% |")
    rank=aggregate['bf16_r8_a16']['relative_to_bf16_r16_a32']; quant=aggregate['nf4_r16_a32']['relative_to_bf16_r16_a32']
    lines+=['',f"降低rank在本轮使可训练参数变化{rank['trainable_parameters']:+.1f}%，allocated峰值变化{rank['allocated_gib']:+.1f}%。主干常驻显存占比较大，不能把参数减半写成显存减半。r8与alpha16的观测生成均值较高，但不能凭均值宣布普遍最优或跨数据量有效。",
        '',f"NF4整体配方的allocated峰值变化{quant['allocated_gib']:+.1f}%，训练耗时变化{quant['training_minutes']:+.1f}%；观测BLEU均值接近共同对照。这不是无损或非劣效结论，不能根据区间跨0判等效。",
        '', '训练时间含固定128步与8次完整dev，不含全部下载、加载和正式评价；allocated/reserved是进程指标，不能和桌面nvidia-smi整机占用混用。不同最佳checkpoint步数由同一dev选择规则产生，不能用test另选。',
        '', '## 同种子直接配对证据','','| 候选 | 种子 | BLEU差值 | 名义95%区间 |','|---|---|---:|---|']
    for r in results:
        if r['variant']=='bf16_r16_a32': continue
        if 'paired_vs_bf16' in r:
            d=r['paired_vs_bf16']['bleu']
            lines.append(f"| {r['variant']} | {r['seed']} | {d['observed']:+.4f} | [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] |")
        else: lines.append(f"| {r['variant']} | {r['seed']} | 待核验 | 待完成原始统计 |")
    lines+=['', '置信区间是逐种子、逐比较名义区间，未做多重比较校正；不将种子预测池化，不据跨0宣称等效。三种子固定相同子集，不代表用户/数据抽样变异。',
        '', '## 结论边界与后续工作','', '当前贡献是已有PEFT方法在固定小样本条件下的实证权衡及可复现工程，不是新的LoRA算法。50/100条训练策略收益来自多个因素同时变化，若要提出新机制，需要另立受控消融。',
        '', '客观风格与内容辅助分析见29/30号文档，属于探索性代理证据，不证明深层个人风格、事实正确或通用能力无遗忘。人工盲评、付费强模型和大规模通用矩阵仍暂缓。',
        '', 'Linux/RTX4090复现尚未执行；当前消费级硬件实测为Windows/RTX5070Ti。代码已公开但所有者暂未指定许可证，不能称正式开源发布已完成。',
        '', '完整机器证据results/peft_quality_cost_analysis.json；私有来源哈希与逐run证据保留在experiments目录。']
    supplement=ROOT/'results/style_content_supplement.json'
    if supplement.exists():
        evidence=read(supplement)
        if evidence['conditions_count']!=28: raise ValueError('Objective supplement scope mismatch')
        controls={r['condition']:r['groups']['all'] for r in evidence['conditions']}
        lines+=['', '## 表层风格的无画像与基座控制','',
            '| 条件 | 错配减真实画像距离 | 对真实画像距离 | 对参考答案距离 |',
            '|---|---:|---:|---:|']
        for name in ('bf16_base_no_profile','bf16_base_profile','nf4_base_no_profile','nf4_base_profile'):
            g=controls[name]
            lines.append(f"| {name} | {g['mismatch_minus_true_style_distance']:+.4f} | {g['style_to_true_profile']:.4f} | {g['style_to_reference']:.4f} |")
        for variant in VARIANTS:
            for condition in ('adapter_no_profile','adapter_profile'):
                group=[r['groups']['all'] for r in evidence['conditions'] if r['condition'].startswith(variant+'_seed') and r['condition'].endswith('_'+condition)]
                if len(group)!=3: raise ValueError('Missing surface-style seed')
                avg=lambda k:statistics.mean(r[k] for r in group)
                lines.append(f"| {variant} {condition} | {avg('mismatch_minus_true_style_distance'):+.4f} | {avg('style_to_true_profile'):.4f} | {avg('style_to_reference'):.4f} |")
        lines+=['', '无画像基座也有正的错配差值，说明正值本身不能归因为画像风格迁移。画像条件下相对差值增大只是探索性关联；例如r16/a32对真实画像的绝对距离并未优于有画像BF16基座，不能仅挑错配差值证明所有适配器都学到了更好的风格。参考距离和真实画像距离的排序也不完全相同。',
            '', '数字与否定变化只作风险筛查：参考答案本身也存在这些变化。没有人工核验或事实标签时，不将规则标记率写成事实错误率，也不把词汇Jaccard当作语义保真分数。']
    (ROOT/'docs/31_peft_quality_cost_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],completed_original_bootstrap_tasks=len(available),new_trainings_verified=9,reused_controls_verified=3),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--final',action='store_true'); args=p.parse_args()
    summarize(args.final)


if __name__=='__main__': main()
