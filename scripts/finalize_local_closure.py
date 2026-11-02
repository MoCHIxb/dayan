"""Finalize audited local PEFT closure; no training, generation or new statistics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/peft_v1'


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); temp.replace(path)
def replace(path,old,new):
    text=path.read_text(encoding='utf-8')
    path.write_text(text.replace(old,new),encoding='utf-8')


def finalize():
    if read(WORK/'execution_state.json')['stage']!='peft_v1_complete':
        raise ValueError('Original sequence is not complete')
    subprocess.run([sys.executable,'scripts/summarize_peft_evidence.py','--final'],cwd=ROOT,check=True)
    analysis=read(ROOT/'results/peft_quality_cost_analysis.json')
    if analysis['status']!='complete_original_statistics_audited' or analysis['completed_original_bootstrap_tasks']!=18:
        raise ValueError('Independent final audit incomplete')
    original=read(WORK/'results.json')
    if len(original['runs'])!=12 or len(original['aggregate'])!=4: raise ValueError('Original final result scope mismatch')
    markdown=(ROOT/'docs/25_small_sample_peft_results.md').read_text(encoding='utf-8')
    for variant,group in original['aggregate'].items():
        records=[r for r in original['runs'] if r['training']['variant']==variant]
        if len(records)!=3 or {r['training']['seed'] for r in records}!={20251206,20251207,20251208}:
            raise ValueError('Original seed coverage mismatch')
        for metric in ('bleu','chrf','rougeL_f'):
            values=[r['metrics'][metric] for r in records]
            if abs(statistics.mean(values)-group[metric]['mean'])>1e-12 or abs(statistics.stdev(values)-group[metric]['sample_sd'])>1e-12:
                raise ValueError('Original aggregate mismatch')
        line=f"| {variant} | {group['bleu']['mean']:.4f} | {group['bleu']['sample_sd']:.4f} | {group['chrf']['mean']:.4f} |"
        if line not in markdown: raise ValueError('Original readable summary table differs')
    sources={}; statistic_count=0
    for variant in ('bf16_r8_a16','bf16_r16_a16','nf4_r16_a32'):
        for seed in (20251206,20251207,20251208):
            run=ROOT/f'experiments/runs/peft-v1-{variant}_seed{seed}'
            baseline=ROOT/f'experiments/runs/small-sample-v1-n500_lr1e4_seed{seed}'
            candidate=read(run/'adapter_profile_test_metrics.json'); control=read(baseline/'adapter_profile_test_metrics.json')
            for filename in ('paired_2x2_bootstrap.json','paired_vs_bf16_r16_a32.json'):
                path=run/filename; value=read(path)
                receipt=read(path.with_suffix('.json.receipt.json'))
                if receipt['sha256']!=sha(path) or (value['samples'],value['bootstrap_resamples'],value['seed'])!=(1496,2000,20251206):
                    raise ValueError('Statistic receipt/scope mismatch')
                for metric,target in (('bleu','bleu'),('chrf','chrf'),('rougeL','rougeL_f')):
                    if filename.startswith('paired_vs'):
                        contrast=value['contrasts'][metric]
                        if abs(contrast['observed']-(candidate[target]-control[target]))>1e-8: raise ValueError('Observed metric difference mismatch')
                        candidates=[contrast]
                    else: candidates=list(value['contrasts'][metric].values())
                    for c in candidates:
                        if not all(math.isfinite(c[k]) for k in ('observed','ci95_low','ci95_high','bootstrap_probability_gt_zero')):
                            raise ValueError('Nonfinite statistic')
                        if c['ci95_low']>c['ci95_high'] or not 0<=c['bootstrap_probability_gt_zero']<=1: raise ValueError('Invalid interval')
                sources[str(path.relative_to(ROOT))]=sha(path); statistic_count+=1
    if statistic_count!=18: raise ValueError('Expected 18 original statistics')
    findings={}
    for variant in ('bf16_r8_a16','bf16_r16_a16','nf4_r16_a32'):
        contrasts=[r['paired_vs_bf16']['bleu'] for r in analysis['runs'] if r['variant']==variant]
        findings[variant]=dict(positive_nominal_ci=sum(c['ci95_low']>0 for c in contrasts),
            negative_nominal_ci=sum(c['ci95_high']<0 for c in contrasts),
            ci_includes_zero=sum(c['ci95_low']<=0<=c['ci95_high'] for c in contrasts),seeds=3,
            all_seeds_positive_nominal_ci=all(c['ci95_low']>0 for c in contrasts))
    analysis['bleu_same_seed_interval_summary']=findings
    analysis['interpretation']='No candidate has nominal BLEU improvement in all three seed comparisons; NF4 intervals crossing zero do not establish equivalence/noninferiority.'
    write(ROOT/'results/peft_quality_cost_analysis.json',analysis)
    write(WORK/'interpretation_final.json',analysis)
    doc=ROOT/'docs/31_peft_quality_cost_analysis.md'
    text=doc.read_text(encoding='utf-8')
    text+='''
## 本轮可落地结论

- r8/alpha16的BLEU均值比共同对照高约0.63分；同种子比较中1/3个名义区间在0以上，另外2个跨0。它减少一半可训练参数，但本轮没有三种子一致的名义显著质量优势，不能宣布普遍最优。
- r16/alpha16的BLEU均值高约0.58分；2/3个名义区间在0以上，另1个跨0。秩和显存没有下降，不应当作进一步压缩方案。
- NF4相对BF16的三个BLEU差值区间全部跨0。观测均值接近、训练allocated峰值约少38.3%、耗时约多14.5%，支持把它作为显存优先的候选配方；未设计非劣效界限，不能宣称无损或等效。

以上为单模型、500条固定子集、三个训练种子的本轮权衡。所有区间均未校正多重比较；没有任何候选在三个种子上均有名义BLEU优势。参考相似度和表层风格指标也不支持把本项目包装为已验证的新PEFT算法。

本地本轮训练、完整评价、原始统计及证据解释已完成。Linux/4090实机复现、代码许可证、正式发布验收、深层风格/事实/通用能力证据仍未完成或按用户要求暂缓。
'''
    doc.write_text(text,encoding='utf-8')
    replace(ROOT/'docs/25_small_sample_peft_results.md','仍需解释结果、新环境复现、代码许可证确定与正式发布。',
            '质量与资源解释及独立最终审计已完成，见31号报告；仍需Linux/4090实机复现、代码许可证确定与正式发布验收。')
    replace(ROOT/'README.md','2026-11-02用户明确恢复本地收尾，18项原始bootstrap正由原冻结执行器串行计算；已中断首项按原种子从0重算，不重训或重生成。**不提供尚未形成的PEFT最终置信区间。**',
            '2026-11-02已完成18/18项原始2000次配对bootstrap，完整收据、原始命令、样本/哈希及最终汇总均通过独立审计。最终[结果汇总](docs/25_small_sample_peft_results.md)和[质量/资源解释](docs/31_peft_quality_cost_analysis.md)已形成；没有候选在三个种子上均有名义BLEU优势，不据跨0判等效。')
    replace(ROOT/'README.md','后者在统计完成前明确标为描述性草稿，不提供虚构置信区间。',
            '后者现为18/18原始统计核验后的最终本轮报告，保留多重比较及适用范围限制。')
    replace(ROOT/'AGENTS.md','The original peft_v1 CPU bootstrap is running and its monitor is active. Training and GPU evaluation are already complete and must be reused.',
            'The local peft_v1 training, full evaluation, 18 original bootstrap tasks and independent final interpretation are complete and verified. Its finished monitor is being removed. Do not restart completed local work.')
    replace(ROOT/'docs/26_framework_delivery_status.md','| 小样本PEFT结论 | 已启动，未完成 |',
            '| 小样本PEFT结论 | 本轮统计、独立审计与解释已完成 |')
    replace(ROOT/'docs/26_framework_delivery_status.md','质量与资源草稿见31号文档，原始统计完成前不得当最终显著性结论。',
            '质量与资源最终本轮分析见31号文档，18项原始统计已核验，保留名义区间与外推边界。')
    replace(ROOT/'docs/28_next_work_plan.md','2026-11-02用户明确要求完成本地未完成工作，原冻结CPU统计已恢复，半小时监控已恢复；训练和GPU评价继续复用，不运行云端实验。原暂停记录保留为历史。',
            '2026-11-02本地本轮收尾完成：复用9次新训练及3次对照，完整评价、18项原始统计、辅助证据和最终解释均已核验；不再运行本轮训练/统计，结束监控。云端未启动，原暂停记录保留为历史。')
    p=ROOT/'docs/28_next_work_plan.md'; text=p.read_text(encoding='utf-8')
    text+='\n本轮最终状态：18/18原始统计已完成，--final独立审计及三指标观测差值核验通过，31号报告已更新为最终本轮分析。上面的本地统计计划和草稿说明保留为过程记录，不代表仍待运行。后续优先Linux/4090复现、论文论证边界与正式发布验收；当前不启动新实验。\n'
    p.write_text(text,encoding='utf-8')
    progress=dict(date='2026-11-02',stage='local_peft_closure_complete',completed_new_trainings=9,reused_control_trainings=3,
        completed_test_generations=20,completed_auxiliary_sets=9,completed_original_bootstrap_tasks=18,planned_original_bootstrap_tasks=18,
        objective_supplement_conditions=28,independent_final_audit='passed',monitor='completion verified; removal pending',
        gpu_work='complete; not restarted',cloud_experiment='not started',
        remaining=['Linux/RTX4090 end-to-end reproduction','code license decision','formal release acceptance','method-specific causal evidence if claimed'])
    write(ROOT/'results/local_progress.json',progress)
    reference=read(ROOT/'results/local_500_reference.json')
    reference['status']['peft_original_bootstrap']='18/18 completed and independently audited'
    reference['status']['note']='Final per-seed intervals and interpretation: results/peft_quality_cost_analysis.json; no deep-style, equivalence or no-forgetting claim.'
    write(ROOT/'results/local_500_reference.json',reference)
    audit=dict(completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage='local_peft_closure_complete',
        new_trainings=9,reused_controls=3,generation_groups=20,auxiliary_sets=9,original_bootstrap_tasks=18,
        samples_per_test=1496,dev_records=1500,resamples_per_original_task=2000,
        original_aggregate_recomputed=True,original_readable_table_verified=True,three_observed_metrics_verified=True,
        direct_interval_summary=findings,sources_sha256=sources,
        report_sha256={str(p.relative_to(ROOT)):sha(p) for p in (WORK/'results.json',ROOT/'docs/25_small_sample_peft_results.md',
            ROOT/'results/peft_quality_cost_analysis.json',ROOT/'docs/31_peft_quality_cost_analysis.md')},
        remaining=progress['remaining'])
    write(WORK/'closure_audit.json',audit)
    print(json.dumps(dict(stage=audit['stage'],original_bootstrap_tasks=18,findings=findings),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); args=p.parse_args()
    if not args.run: print('No action; --run requires the original sequence to be complete.'); return
    finalize()


if __name__=='__main__': main()
