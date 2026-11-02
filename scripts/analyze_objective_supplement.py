"""Exploratory surface-style/content-risk evidence from existing predictions only."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/peft_v1/objective_supplement'
SEEDS=(20251206,20251207,20251208)
VARIANTS=('bf16_r16_a32','bf16_r8_a16','bf16_r16_a16','nf4_r16_a32')
FIRST={'i','me','my','mine','myself','we','us','our','ours','ourselves'}
SECOND={'you','your','yours','yourself','yourselves'}
FUNCTION=set('a an the and or but if so as at by for from in into of on to with is am are was were be been being do does did have has had it its this that these those there here i me my we us our you your he him his she her they them their'.split())


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f: return [json.loads(line) for line in f if line.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)
def words(text): return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text.replace('’',"'"))


def features(text):
    tokens=words(text); lower=[t.lower() for t in tokens]; n=max(len(tokens),1)
    letters=[c for c in text if c.isalpha()]
    fragments=[part for part in re.split(r'[.!?]+',text) if words(part)]
    return dict(word_count=float(len(tokens)),mean_word_length=sum(map(len,tokens))/n,
        words_per_terminal_fragment=len(tokens)/max(len(fragments),1),
        first_person_rate=sum(t in FIRST for t in lower)/n,
        second_person_rate=sum(t in SECOND for t in lower)/n,
        contraction_rate=sum(bool(re.search(r"(?:n't|'re|'ve|'ll|'d|'m|'s)$",t)) for t in lower)/n,
        uppercase_rate=sum(c.isupper() for c in letters)/max(len(letters),1),
        exclamations_per100words=100*text.count('!')/n,questions_per100words=100*text.count('?')/n,
        commas_per100words=100*text.count(',')/n)


def distance(a,b,scales):
    return statistics.mean(abs(a[k]-b[k])/scales[k] for k in scales)


def derangement(n,seed=20251206):
    if n<2: raise ValueError('At least two records needed for mismatched-profile control')
    rng=random.Random(seed); indices=list(range(n))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(i!=j for i,j in enumerate(indices)): return indices
    raise RuntimeError('Could not construct fixed derangement')


def input_text(record):
    task=record['messages'][-2]['content'].split('\n\nTask:\n',1)[-1]
    return task.partition(':')[2].strip() if ':' in task else task
def numbers(text): return set(re.findall(r'\b\d+(?:\.\d+)?\b',text))
def negated(text): return bool(re.search(r"\b(?:no|not|never|neither|nor)\b|n't\b",text.lower().replace('’',"'")))
def content(text): return {t.lower() for t in words(text) if t.lower() not in FUNCTION}
def ngrams(text):
    tokens=re.findall(r"<\w+>|\w+(?:'\w+)?|[^\w\s]",text.lower())
    return {tuple(tokens[i:i+4]) for i in range(len(tokens)-3)}
def metrics_path(path):
    return Path(str(path).removesuffix('.predictions.jsonl')+'.json')


def conditions():
    for variant in VARIANTS:
        for seed in SEEDS:
            run=ROOT/(f'experiments/runs/small-sample-v1-n500_lr1e4_seed{seed}' if variant=='bf16_r16_a32' else f'experiments/runs/peft-v1-{variant}_seed{seed}')
            for name in ('adapter_profile','adapter_no_profile'):
                yield f'{variant}_seed{seed}_{name}',run/f'{name}_test_metrics.predictions.jsonl',run/'bertscore.json',name
    for quant,folder in (('bf16',ROOT/'experiments/refocused/baseline'),('nf4',ROOT/'experiments/peft_v1/nf4_baseline')):
        for name in ('base_profile','base_no_profile'):
            yield f'{quant}_{name}',folder/f'{name}_test_metrics.predictions.jsonl',None,name


def run():
    if WORK.exists(): raise FileExistsError('Supplement directory exists; inspect instead of overwriting')
    public=ROOT/'results/style_content_supplement.json'; doc=ROOT/'docs/30_objective_supplement_results.md'
    if public.exists() or doc.exists(): raise FileExistsError('Supplement public outputs already exist')
    train_path=ROOT/'data/processed/lamp7_bm25_k12/train_500.jsonl'; test_path=ROOT/'data/processed/lamp7_bm25_k12/test.jsonl'
    data={str(r['id']):r for r in rows(test_path)}; train=rows(train_path)
    if len(data)!=1496 or len(train)!=500: raise ValueError('Wrong data scope')
    selected=list(conditions()); sources=[Path(__file__),ROOT/'docs/29_objective_supplement_protocol.md',train_path,test_path]
    for _,path,semantic,_ in selected:
        sources.append(path)
        if semantic: sources.append(semantic)
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(sources)}
    write(WORK/'input_manifest.json',dict(protocol='objective_supplement_v1',scope='exploratory, not confirmatory',sources_sha256=source_hashes))
    fit=[features(r['messages'][-1]['content']) for r in train]
    scales={k:statistics.pstdev(r[k] for r in fit) or 1.0 for k in fit[0]}
    ids=sorted(data); permutation=derangement(len(ids))
    histories={i:[h['text'] for h in data[i]['profile']] for i in ids}
    history_features={i:[features(t) for t in histories[i]] for i in ids}
    centroids={i:{k:statistics.mean(f[k] for f in history_features[i]) for k in scales} for i in ids}
    history_ngrams={i:set().union(*(ngrams(t) for t in histories[i])) for i in ids}
    train_targets={r['messages'][-1]['content'] for r in train}
    input_cache={i:input_text(data[i]) for i in ids}
    reference_features={i:features(data[i]['messages'][-1]['content']) for i in ids}
    wrong={i:ids[permutation[j]] for j,i in enumerate(ids)}
    from sacrebleu.metrics import BLEU,CHRF
    bleu=BLEU(effective_order=True); chrf=CHRF(); results=[]
    for label,path,semantic,semantic_name in selected:
        predictions={str(r['id']):r for r in rows(path)}
        if len(predictions)!=1496 or set(predictions)!=set(ids): raise ValueError('Prediction IDs mismatch')
        values=[]; texts={g:[] for g in ('all','reference_in_history','reference_not_in_history')}
        for i in ids:
            pred=predictions[i]; hyp=pred['prediction']; ref=data[i]['messages'][-1]['content']; inp=input_cache[i]
            if pred['reference']!=ref: raise ValueError('Reference mismatch')
            hf=features(hyp); true_distance=distance(hf,centroids[i],scales); ng=ngrams(hyp)
            vocabulary=content(inp)|content(hyp)
            row=dict(id=i,reference_in_history=ref in histories[i],style_to_reference=distance(hf,reference_features[i],scales),
                style_to_true_profile=true_distance,style_to_mismatched_profile=distance(hf,centroids[wrong[i]],scales),
                number_set_changed=numbers(inp)!=numbers(hyp),reference_number_set_changed=numbers(inp)!=numbers(ref),
                negation_marker_changed=negated(inp)!=negated(hyp),reference_negation_marker_changed=negated(inp)!=negated(ref),
                content_lexical_jaccard=len(content(inp)&content(hyp))/max(len(vocabulary),1),
                exact_input_copy=hyp==inp,exact_history_copy=hyp in histories[i],exact_train500_target_copy=hyp in train_targets,
                history_4gram_coverage=len(ng&history_ngrams[i])/len(ng) if ng else 0.0)
            row['mismatch_minus_true_style_distance']=row['style_to_mismatched_profile']-true_distance
            values.append(row)
            texts['all'].append((hyp,ref)); texts['reference_in_history' if ref in histories[i] else 'reference_not_in_history'].append((hyp,ref))
        private=WORK/'per_item'/f'{label}.jsonl'; private.parent.mkdir(parents=True,exist_ok=True)
        with private.open('x',encoding='utf-8') as f:
            for row in values: f.write(json.dumps(row)+'\n')
        groups={}
        for group in texts:
            group_rows=values if group=='all' else [r for r in values if r['reference_in_history']==(group=='reference_in_history')]
            hyps,refs=zip(*texts[group])
            keys=[k for k in values[0] if k not in ('id','reference_in_history')]
            groups[group]=dict(samples=len(group_rows),**{k:statistics.mean(r[k] for r in group_rows) for k in keys},
                bleu=bleu.corpus_score(hyps,[refs]).score,chrf=chrf.corpus_score(hyps,[refs]).score)
        if (groups['reference_in_history']['samples'],groups['reference_not_in_history']['samples'])!=(43,1453): raise ValueError('Wrong subgroup counts')
        score=read(metrics_path(path))
        if abs(groups['all']['bleu']-score['bleu'])>1e-8 or abs(groups['all']['chrf']-score['chrf'])>1e-8: raise ValueError('Full-test corpus score changed')
        entry=dict(condition=label,predictions_sha256=sha(path),groups=groups)
        if semantic: entry['bertscore_f1']=read(semantic)['condition_scores'][semantic_name]['f1']
        results.append(entry); print(f'Analyzed {label}: 1496 records',flush=True)
    report=dict(protocol='objective_supplement_v1',scope='exploratory surface-style and content-risk proxies, not deep style or factual accuracy',
        conditions=results,conditions_count=28,feature_scales_from_train500=scales,
        input_manifest_sha256=sha(WORK/'input_manifest.json'),
        limitations=['No new generation under mismatched profiles: observational comparison only','Question profiles are not verified user IDs',
                    'Numeric/negation changes can be benign; not factual error rates','History is original selected BM25-k12, not guaranteed fully retained prompt',
                    'No new confidence intervals; complete primary evaluation and corpus bootstrap unchanged'])
    write(WORK/'aggregate.json',report); write(public,report)
    lines=['# 客观风格与内容辅助证据','','本分析为29号协议下的探索性补充，不替代完整主评价或既定统计。28组×1496条；分组43/1453。','',
        '| 有画像条件 | 参考不在历史1453条BLEU | 历史整句复制率 | 数字变化标记率 | 否定变化标记率 | 错配减真实画像风格距离 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        group=[r for r in results if r['condition'].startswith(variant+'_seed') and r['condition'].endswith('_adapter_profile')]
        avg=lambda subgroup,key:statistics.mean(r['groups'][subgroup][key] for r in group)
        lines.append(f"| {variant} | {avg('reference_not_in_history','bleu'):.4f} | {100*avg('all','exact_history_copy'):.2f}% | {100*avg('all','number_set_changed'):.2f}% | {100*avg('all','negation_marker_changed'):.2f}% | {avg('all','mismatch_minus_true_style_distance'):+.4f} |")
    ref=results[0]['groups']['all']
    lines+=['',f"参考答案本身相对输入的数字变化标记率为{100*ref['reference_number_set_changed']:.2f}%，否定变化标记率为{100*ref['reference_negation_marker_changed']:.2f}%。标记不等于错误，不能忽略合理改写或提取规则误差。",
        '', '风格距离仅来自预定表层特征与train_500尺度。正的错配差值表示现有输出在该距离下更接近真实画像，而不是证明因果画像遵循、独立个人身份或深层句法学习。不同条件的数值必须同时对照无画像及基座结果，不能只挑有利表格。',
        '', '公开汇总results/style_content_supplement.json含全部28条件及所有分组，不含原文/逐条ID；逐条数值和输入哈希manifest留在私有experiments目录。新辅助证据不提供额外显著性结论，不用于测试集调参。']
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); args=p.parse_args()
    if not args.run: print('Plan only: 28 existing complete prediction groups, no training/generation, exploratory CPU analysis.'); return
    run()


if __name__=='__main__': main()
