"""Validate imported aggregate evidence and publish a text-free BANKING77 analysis.

This recomputes metrics from aggregate confusion counts, not individual predictions.
It does not load models, train, bootstrap, or certify unavailable cloud artifacts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from personal_llm_data.banking77 import LABELS, SIZES, SEED, MAX_LENGTH, MAX_NEW_TOKENS

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def equal(actual,expected,label):
    if isinstance(expected,float):
        if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Metric differs from confusion counts: '+label)
    elif actual!=expected:raise ValueError('Evidence mismatch: '+label)


def metrics(score):
    equal(score['labels'],list(LABELS),'label inventory')
    matrix=score['confusion']
    if set(matrix)!=set(LABELS):raise ValueError('Incomplete confusion rows')
    for label,row in matrix.items():
        if set(row)!=set(LABELS)|{'invalid'}:raise ValueError('Incomplete confusion columns')
        if any(type(value) is not int or value<0 for value in row.values()):raise ValueError('Invalid confusion count')
        equal(sum(row.values()),40,'per-intent test support')
    n=sum(sum(row.values()) for row in matrix.values());equal(n,400,'full test')
    classes={}
    for label in LABELS:
        tp=matrix[label][label];support=sum(matrix[label].values());guessed=sum(matrix[gold][label] for gold in LABELS)
        classes[label]=dict(support=support,precision=tp/guessed if guessed else 0,recall=tp/support,
                            f1=2*tp/(support+guessed) if support+guessed else 0)
    invalid=sum(row['invalid'] for row in matrix.values())
    result=dict(samples=n,labels=list(LABELS),accuracy=sum(matrix[label][label] for label in LABELS)/n,
                macro_f1=sum(c['f1'] for c in classes.values())/len(LABELS),per_class=classes,
                invalid_outputs=invalid,invalid_rate=invalid/n,confusion=matrix)
    for key,value in result.items():
        if key=='per_class':
            for label,values in value.items():
                for metric,number in values.items():equal(score[key][label][metric],number,label+'/'+metric)
        elif key not in ('confusion','labels'):equal(score[key],value,key)
    result['minimum_class_recall']=min(c['recall'] for c in classes.values())
    if 'minimum_class_recall' in score:equal(score['minimum_class_recall'],result['minimum_class_recall'],'minimum recall')
    result['top_confusions']=sorted([dict(reference=gold,predicted=guess,count=matrix[gold][guess])
        for gold in LABELS for guess in LABELS if gold!=guess and matrix[gold][guess]],
        key=lambda row:(-row['count'],row['reference'],row['predicted']))[:10]
    if 'top_confusions' in score:equal(score['top_confusions'],result['top_confusions'],'confusion diagnostic')
    result['acceptance_gates']=dict(macro_f1_at_least_085=result['macro_f1']>=.85,
        minimum_class_recall_at_least_060=result['minimum_class_recall']>=.60,zero_invalid=invalid==0)
    result['all_absolute_gates_passed']=all(result['acceptance_gates'].values())
    result['correct_records']=sum(matrix[label][label] for label in LABELS)
    return result


def analyze(path,root=ROOT):
    root=Path(root)
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    equal(data['status'],'complete_banking77_card10_pipeline_audited','reported completion')
    equal(data['training_sizes'],list(SIZES),'training sizes');equal(data['dev'],100,'dev count');equal(data['test'],400,'test count')
    equal(set(data['results']),{str(size) for size in SIZES},'run inventory')
    lock=json.loads((root/'repro/banking77_assets.json').read_text(encoding='utf-8'))
    for name,expected in lock['model_sha256'].items():equal(sha(root/'models/Qwen3-4B'/name),expected,'local model '+name)
    fewshot=metrics(data['fewshot']);runs={};hash_checks=0;distinct_bases=[]
    digest_pattern=lambda value:isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    for size in SIZES:
        row=data['results'][str(size)];audit=row['audit'];prefix=f'n{size}'
        config_name=f'configs/intenttune_v1/qwen3_4b_n{size}.json'
        config=json.loads((root/config_name).read_text(encoding='utf-8'))
        equal(config['seed'],SEED,'training seed');equal(config['max_length'],MAX_LENGTH,'input budget')
        equal(config['max_new_tokens'],MAX_NEW_TOKENS,'output budget')
        equal(audit['config'],config_name,'config path');equal(audit['status'],'complete_classification_run_audited','cloud audit status')
        equal(audit['counts'],dict(train=size,dev=100,test=400),'split counts')
        equal(audit['inputs']['config_sha256'],sha(root/config_name),'config SHA');hash_checks+=1
        equal(audit['inputs']['config_sha256'],lock['config_sha256'][config_name],'frozen config SHA')
        expected_data={name:lock['data_sha256'][prefix+'/'+name] for name in ('train.jsonl','dev.jsonl','test.jsonl','manifest.json')}
        equal(audit['inputs']['data_sha256'],expected_data,'data SHA inventory');hash_checks+=len(expected_data)
        equal(audit['inputs']['model_sha256'],lock['model_sha256'],'model SHA inventory');hash_checks+=len(lock['model_sha256'])
        equal(audit['inputs']['code_sha256'],lock['core_code_sha256'],'core source SHA inventory')
        for name,digest in audit['inputs']['code_sha256'].items():equal(digest,sha(root/name),'local source '+name);hash_checks+=1
        equal(audit['audit_code_sha256'],sha(root/'scripts/audit_classification_run.py'),'audit implementation SHA');hash_checks+=1
        manifest=json.loads((root/config['data_dir']/'manifest.json').read_text(encoding='utf-8'))
        equal(audit['data_manifest'],manifest,'local prepared manifest')
        for name,digest in expected_data.items():equal(digest,sha(root/config['data_dir']/name),'local prepared '+name)
        evaluations=audit['evaluations']
        equal([e['step'] for e in evaluations],list(range(16,129,16)),'eight dev evaluation steps')
        if any(type(e['loss']) not in (int,float) or not math.isfinite(e['loss']) for e in evaluations):raise ValueError('Nonfinite dev loss')
        best=min(evaluations,key=lambda item:(item['loss'],item['step']))
        equal(best,dict(step=audit['best_step'],loss=audit['best_dev_loss']),'dev-only checkpoint choice')
        checkpoints=audit['checkpoint_evidence']
        required={'adapter_config.json','adapter_model.safetensors','optimizer.pt','rng_state.pth','scheduler.pt','trainer_state.json'}
        for condition,step in (('best',best['step']),('final',128)):
            equal(checkpoints[condition]['step'],step,'checkpoint step')
            equal(set(checkpoints[condition]['files_sha256']),required,'checkpoint file inventory')
            if not all(digest_pattern(value) for value in checkpoints[condition]['files_sha256'].values()):raise ValueError('Invalid checkpoint digest')
        equal(audit['adapter_sha256'],checkpoints['best']['files_sha256']['adapter_model.safetensors'],'best/export hash declaration')
        if not all(digest_pattern(value) for value in audit['evidence_sha256'].values()):raise ValueError('Invalid cloud artifact digest')
        equal(len(audit['evidence_sha256']),11,'cloud run evidence declarations')
        quality={condition:metrics(row['quality'][condition]) for condition in ('base','adapter')}
        for condition in ('base','adapter'):
            nested=metrics(audit['quality'][condition])
            equal(nested,quality[condition],'duplicated aggregate score consistency')
        distinct_bases.append(quality['base'])
        centroid=metrics(row['centroid'])
        centroid_inputs={key.rsplit('/',1)[-1]:value for key,value in row['centroid']['inputs_sha256'].items()}
        equal(centroid_inputs,{key:expected_data[key] for key in ('train.jsonl','test.jsonl')},'train-only centroid input identity')
        resources=audit['resources']
        equal(resources['device'],'NVIDIA GeForce RTX 3090','reported device')
        for field in ('peak_allocated_bytes','peak_reserved_bytes','elapsed_seconds_this_session','trainable_parameters','total_parameters'):
            if type(resources[field]) not in (int,float) or not math.isfinite(resources[field]) or resources[field]<=0:raise ValueError('Invalid resource report')
        costs={key:resources[key] for key in ('device','trainable_parameters','total_parameters','python','torch')}
        costs.update(allocated_gib=resources['peak_allocated_bytes']/2**30,reserved_gib=resources['peak_reserved_bytes']/2**30,
                     session_minutes=resources['elapsed_seconds_this_session']/60)
        adapter=quality['adapter']
        runs[str(size)]=dict(counts=audit['counts'],quality=quality,centroid=centroid,best_step=best['step'],best_dev_loss=best['loss'],
            evaluations=evaluations,resources=costs,adapter_sha256=audit['adapter_sha256'],checkpoint_evidence=checkpoints,
            recorded_cloud_evidence_sha256=audit['evidence_sha256'],
            macro_f1_differences=dict(vs_base=adapter['macro_f1']-quality['base']['macro_f1'],
                vs_fewshot=adapter['macro_f1']-fewshot['macro_f1'],
                vs_centroid=adapter['macro_f1']-centroid['macro_f1']))
    equal(distinct_bases[0],distinct_bases[1],'same baseline aggregate across n50/n200')
    equal(distinct_bases[0],distinct_bases[2],'same baseline aggregate across n50/n500')
    return dict(schema_version=1,date='2026-11-14',status='cloud_summary_imported_aggregate_metrics_and_local_assets_verified',
        framework='IntentTune',task='BANKING77 card10 closed-set',seed=SEED,training_sizes=list(SIZES),dev=100,test=400,
        cloud_reported_status=data['status'],imported_summary_sha256=sha(path),fewshot=fewshot,runs=runs,
        verification=dict(confusion_metric_sets_recomputed=10,declared_hashes_matched_to_local_assets=hash_checks,
            scheduled_dev_evaluations_checked=24,best_final_export_declarations_checked=3,
            cloud_reported_stages=22,local_prediction_files_available=False,local_checkpoint_files_available=False,
            local_stage_receipts_available=False,independent_cloud_file_hash_recheck=False),
        limitations=['Single seed and fixed ten-intent subset; not full BANKING77 or OOS validation',
            '400 test records are shared across all comparisons; repeated base scores are not independent tests',
            'Only the aggregate cloud result was imported. Metric arithmetic and recorded local asset identities were checked; individual predictions/checkpoints/22 receipts remain on the cloud',
            'No new significance/noninferiority/bootstrap analysis; do not claim universal improvements',
            'Training label counts exclude the additional 100 dev/400 test labels; larger sample size is not a monotonic effect',
            'CUDA allocator peaks and per-process session time are not whole-machine or end-to-end cost'])


def render(report):
    base=report['runs']['50']['quality']['base'];few=report['fewshot'];runs=report['runs']
    a50=runs['50']['quality']['adapter'];a200=runs['200']['quality']['adapter'];a500=runs['500']['quality']['adapter']
    c500=runs['500']['centroid'];c200=runs['200']['centroid']
    regressions=[label for label in LABELS if a500['per_class'][label]['recall']<few['per_class'][label]['recall']]
    rebound=[str(size) for size in SIZES if runs[str(size)]['evaluations'][-1]['loss']>runs[str(size)]['best_dev_loss']]
    lines=['# IntentTune首轮RTX3090结果：BANKING77 card10','',
        '2026-11-14。用户从云端导入聚合results.json。已独立从混淆矩阵复算指标、核对本地固定资产和dev选模声明；原始预测、checkpoint及22份阶段收据仍在云端，本地未逐文件复核。',
        '', '## 主要结果','',
        '| 条件 | 训练条数 | 正确/400 | Accuracy | macro-F1 | 最低类别召回 | 无效输出 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f'| 零示例基座 | 0 | {base["correct_records"]} | {base["accuracy"]:.4f} | {base["macro_f1"]:.4f} | {base["minimum_class_recall"]:.3f} | {base["invalid_outputs"]} |',
        f'| 固定10示例基座 | 10示例，无参数训练 | {few["correct_records"]} | {few["accuracy"]:.4f} | {few["macro_f1"]:.4f} | {few["minimum_class_recall"]:.3f} | {few["invalid_outputs"]} |']
    for size in SIZES:
        for name,score in (('TF-IDF质心',runs[str(size)]['centroid']),('LoRA',runs[str(size)]['quality']['adapter'])):
            lines.append(f'| {name} | {size} | {score["correct_records"]} | {score["accuracy"]:.4f} | {score["macro_f1"]:.4f} | {score["minimum_class_recall"]:.3f} | {score["invalid_outputs"]} |')
    lines+=['','各条件使用同一400条官方测试记录；每类40条。三次零示例基座的聚合结果相同，表中只列一次，不能当作三个独立重复实验。',
        '', '## LoRA相对收益','',
        '| 训练条数 | 相对零示例macro-F1百分点 | 相对10示例百分点 | 相对同档轻量模型百分点 |',
        '|---|---:|---:|---:|']
    for size in SIZES:
        delta=runs[str(size)]['macro_f1_differences']
        lines.append(f'| {size} | {delta["vs_base"]*100:+.2f} | {delta["vs_fewshot"]*100:+.2f} | {delta["vs_centroid"]*100:+.2f} |')
    lines+=['', f'500条LoRA为{a500["correct_records"]}/400正确，零示例基座为{base["correct_records"]}/400；相对10示例和同档轻量模型的正确总数差分别为{a500["correct_records"]-few["correct_records"]:+d}、{a500["correct_records"]-c500["correct_records"]:+d}。这些不是逐题配对胜负数量，也没有显著性结论。',
        '', f'50与200条LoRA的macro-F1分别为{a50["macro_f1"]:.4f}和{a200["macro_f1"]:.4f}。固定128步使数据重复曝光不同；只有一种子，不能归因为某一种机制或证明样本规模的普遍单调效果。200条相对10示例的macro-F1差为{a200["macro_f1"]-few["macro_f1"]:+.5f}，accuracy差为{a200["accuracy"]-few["accuracy"]:+.4f}，不应包装为有实际意义的明确优势；最低类别召回分别为LoRA {a200["minimum_class_recall"]:.3f}、同档轻量模型{c200["minimum_class_recall"]:.3f}。',
        '', '## 每类召回与混淆','',
        '| 原标签 | 零示例 | 10示例 | LoRA 50 | LoRA 200 | LoRA 500 |', '|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        values=[base['per_class'][label]['recall'],few['per_class'][label]['recall']]+[runs[str(n)]['quality']['adapter']['per_class'][label]['recall'] for n in SIZES]
        lines.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['', '零示例主要混淆：'+ '; '.join(f'{r["reference"]} → {r["predicted"]}: {r["count"]}' for r in base['top_confusions'][:3])+ '。500条LoRA主要混淆：'+ '; '.join(f'{r["reference"]} → {r["predicted"]}: {r["count"]}' for r in a500['top_confusions'][:3])+ '。相对10示例提示召回回退的类别：'+(', '.join(regressions) or '本轮未观察到')+'。具体召回见表；固定测试集的单次观测不能外推所有业务场景。',
        '', '## 开发集选择与资源','',
        '| 训练条数 | 最佳步 | 最低dev loss | allocated GiB | reserved GiB | 本次训练会话分钟 |',
        '|---|---:|---:|---:|---:|---:|']
    for size in SIZES:
        row=runs[str(size)];cost=row['resources']
        lines.append(f'| {size} | {row["best_step"]} | {row["best_dev_loss"]:.6f} | {cost["allocated_gib"]:.3f} | {cost["reserved_gib"]:.3f} | {cost["session_minutes"]:.2f} |')
    lines+=['', '三次训练均按报告完成128步、8次完整100dev；选择最小dev loss，非test选模。终点dev loss高于所选最佳的档位：'+(', '.join(rebound) or '无')+'。保存较早checkpoint可避免直接采用较差的终点权重；并未进行early stopping，不能将这种选择写成新的早停算法。三个导出SHA均与所报最佳checkpoint匹配，最终128步状态声明另保留。',
        '', '设备为RTX3090，Qwen3-4B BF16 LoRA r8/alpha16，可训练参数16,515,072（约总参数0.409%）。allocated为进程训练峰值，不是整卡显存；会话时间包括训练、保存与末尾核验，不含全部模型加载、下载、提示基线和测试。没有轻量模型完整耗时/部署数据，不能据此称LoRA成本最优。',
        '', '## 预设门槛与可支持结论','',
        '46号预设macro-F1≥0.85、最低类别召回≥0.60、零无效输出。三档LoRA均达到绝对门槛；10示例基座、200/500条轻量模型也达到门槛。因此达标与微调增益必须分开，是否值得微调取决于所需精度和可接受成本。',
        '', '本轮支持：（1）所测10意图任务在消费级3090上可完成有完整dev/test的少样本LoRA适配；（2）500条时对三个参照均有观测增益；（3）标签目录、数据隔离、提示/轻量对照、dev选模和完整审计可形成可执行的客服适配流程。没有无框架训练的受控消融，不能将全部增益归因于框架本身，更不能声称新算法、所有客服领域有效或可直接上线。',
        '', '限制：一个模型、一种子、十类固定子集；不是77类benchmark或OOS/真实个人用户测试。训练50–500之外另有100dev和400test标签；公开历史语料存在预训练重叠可能。无新的bootstrap、显著性或非劣效分析。后续其他模型使用独立配置/输出及同一留出集，另立协议并在看其test之前固定，不据本轮test反复调参。',
        '', '## 证据与归档','',
        '- 原始导入文件保持字节不变，私有归档：experiments/intenttune_banking77_v1/imported_cloud/results.json。',
        '- 公开聚合与校验范围：[banking77_card10_results.json](../results/banking77_card10_results.json)。',
        f'- 原始导入SHA256：`{report["imported_summary_sha256"]}`。',
        '- 本地独立核验：10组混淆矩阵的全指标重算、三份配置/模型/数据/六个核心源码与当前冻结资产对照、24次dev评价和最早最小loss选模、最佳/最终/导出哈希声明内部一致性。',
        '- 云端执行器报告22阶段收据与原文件通过；本地仅收到聚合JSON，不宣称已重新读取云端逐条预测、checkpoint或阶段收据。',
        '- 不上传客户原文、权重、逐条预测或环境密钥。没有重跑模型、改变46号协议或旧LaMP统计。', '']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists() or args.report.exists():raise FileExistsError('Preserve published evidence; choose new files')
    result=analyze(args.input)
    for path in (args.output,args.report):path.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with args.report.open('x',encoding='utf-8',newline='\n') as stream:stream.write(render(result))
    print(json.dumps(dict(status=result['status'],verification=result['verification'])));return 0


if __name__=='__main__':raise SystemExit(main())
