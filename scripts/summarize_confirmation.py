"""Consolidate actual completed three-seed results without filling human scores."""
import json
from pathlib import Path
import statistics


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/confirmation"


def main():
    cohort=json.loads((OUT/"cohort_progress.json").read_text(encoding="utf-8"))
    if len(cohort)!=6 or {(r['model_size'],r['seed']) for r in cohort}!={(m,s) for m in ('4B','1.7B') for s in (20251206,20251207,20251208)}:
        raise ValueError("Expected complete 2-model x 3-seed cohort")
    summary={"models":{},"training_seeds":[20251206,20251207,20251208],
             "human_blind_review":"pending independent completed rating sheets",
             "llm_judge":"not run; user declined paid API",
             "claim_scope":"follow-up on an already examined benchmark, not an untouched external replication"}
    lines=["# 完整 dev、多种子与模型规模补充实验", "", "机器实验已完成；人工盲评未返回前，不宣称已验证独立风格结论。", "",
           "下表为三个独立训练种子的均值±样本标准差，不能把三个种子当作充分的大样本推断。", "",
           "| 模型 | BLEU | chrF | ROUGE-L | BERTScore F1 |", "|---|---:|---:|---:|---:|"]
    for size in ('4B','1.7B'):
        rows=[r for r in cohort if r['model_size']==size]
        aggregate={}
        for key in ('bleu','chrf','rougeL_f','bertscore_f1'):
            values=[r['bertscore']['condition_scores']['adapter_profile']['f1'] if key=='bertscore_f1' else r['metrics']['profile'][key] for r in rows]
            aggregate[key]={"mean":statistics.mean(values),"sample_sd":statistics.stdev(values),"values":values}
        summary['models'][size]=aggregate
        lines.append('| '+size+' | '+' | '.join(f"{aggregate[key]['mean']:.5f} ± {aggregate[key]['sample_sd']:.5f}" for key in aggregate)+' |')
    summary['retention']={}
    for row in cohort:
        name=f"{row['model_size']}_seed{row['seed']}"
        summary['retention'][name]=json.loads((OUT/'general'/f'retention_{name}.json').read_text())
    summary['retrieval']=json.loads((OUT/'retrieval_selection.json').read_text())
    lines += ["", "通用能力完整基准为 ARC-Challenge、GSM8K、IFEval；保存基座与适配器逐条结果和配对区间。",
              "检索仅在完整dev按chrF选择，随后仅对选中策略及预算对齐的BM25-k12评价test。",
              "模型规模比较保持训练配方、数据、种子和验证协议一致；不同模型可训练参数数目不同，不能解释为严格等参数或等计算预算实验。",
              "复用历史测试集形成后续分析，不能声称新测试集从未被查看。具体逐种子区间及错误样本见实验目录。"]
    path=OUT/'machine_summary.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'docs/16_confirmation_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Six complete cohort members summarized; human conclusions remain pending',flush=True)


if __name__=='__main__':
    main()
