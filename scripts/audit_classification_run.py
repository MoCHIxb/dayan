"""Read-only CPU audit of a completed developer classification run and receipts."""
import argparse,json,os
from pathlib import Path
from personal_llm_data.personal_data import read_rows,sha,verify_prepared
from personal_llm_data.classification import score_labels
from train_personal_adapter import ROOT,inputs_snapshot,validate_config,verify_checkpoint
from evaluate_personal_adapter import verify_predictions


def public_relative_path(path,root=ROOT):
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace('\\','/')


def audit(config_path,labels,score_directory=None):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    data=ROOT/config['data_dir'];run=ROOT/config['output_dir']
    records={split:read_rows(data/(split+'.jsonl')) for split in ('train','dev','test')}
    inputs=inputs_snapshot(config,config_path)
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    if summary.get('status')!='completed_verified' or summary['inputs']!=inputs:
        raise ValueError('Training incomplete or current config/code/model/data differ from recorded run')
    if summary['counts']!={key:len(rows) for key,rows in records.items()} or summary['completed_optimizer_steps']!=config['max_steps']:
        raise ValueError('Training count or step budget mismatch')
    final=verify_checkpoint(run/'checkpoints'/f"checkpoint-{config['max_steps']}",inputs)
    best=verify_checkpoint(Path(summary['best_checkpoint']),inputs)
    if final['step']!=config['max_steps'] or summary['final_checkpoint']!=final or best['step']!=summary['best_step']:
        raise ValueError('Final/best checkpoint step inventory differs from training summary')
    adapter=run/'adapter/adapter_model.safetensors'
    if summary['adapter_sha256']!=sha(adapter) or sha(adapter)!=best['files_sha256']['adapter_model.safetensors']:
        raise ValueError('Selected checkpoint/export hash mismatch')
    progress=read_rows(run/'progress.jsonl')
    trained={row['step'] for row in progress if 'loss' in row['logs']}
    if trained!=set(range(1,config['max_steps']+1)):raise ValueError('Optimizer progress coverage incomplete')
    evaluations=[dict(step=row['step'],loss=row['logs']['eval_loss']) for row in progress if 'eval_loss' in row['logs']]
    scheduled=set(range(config['eval_steps'],config['max_steps']+1,config['eval_steps']))
    if not scheduled.issubset({row['step'] for row in evaluations}):raise ValueError('Scheduled dev coverage missing')
    minimum=min(evaluations,key=lambda row:(row['loss'],row['step']))
    if minimum!=dict(step=summary['best_step'],loss=summary['best_dev_loss']):raise ValueError('Earliest minimum dev selection mismatch')
    scores={};evidence=[run/'training_summary.json',run/'run_inputs.json',run/'progress.jsonl']
    for condition in ('base','adapter'):
        metrics=run/f'evaluation/{condition}_metrics.json'
        receipt_path=metrics.with_suffix('.receipt.json');prediction_path=metrics.with_suffix('.predictions.jsonl')
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        command=receipt['command']
        if receipt['condition']!=condition:raise ValueError('Receipt condition mismatch')
        for flag,value in (('--max-samples',len(records['test'])),('--max-input-tokens',config['max_length']),('--max-new-tokens',config['max_new_tokens']),('--quantization',config['quantization']),('--batch-size',1)):
            if flag not in command or command[command.index(flag)+1]!=str(value):raise ValueError('Full-test generation budget differs')
        if ('--adapter' in command)!=(condition=='adapter'):raise ValueError('Base/adapter command identity mismatch')
        for flag,target in (('--model',ROOT/config['model_name_or_path']),('--data',data/'test.jsonl'),('--output',metrics)):
            if flag not in command or Path(command[command.index(flag)+1]).resolve()!=target.resolve():raise ValueError('Evaluation command paths differ from frozen task')
        if condition=='adapter' and Path(command[command.index('--adapter')+1]).resolve()!=(run/'adapter').resolve():raise ValueError('Evaluation adapter directory differs')
        predictions=read_rows(prediction_path);verify_predictions(records['test'],predictions)
        if receipt['samples']!=len(records['test']) or receipt['inputs']!=inputs:raise ValueError('Full test receipt/input mismatch')
        if receipt['adapter_sha256']!=(summary['adapter_sha256'] if condition=='adapter' else None):raise ValueError('Evaluation adapter identity mismatch')
        if receipt['evaluator_sha256']!=sha(ROOT/'scripts/evaluate_generation.py'):raise ValueError('Evaluator changed')
        if json.loads(metrics.read_text(encoding='utf-8'))['samples']!=len(records['test']):raise ValueError('Metrics sample count mismatch')
        if set(receipt['outputs_sha256'])!={metrics.name,prediction_path.name}:raise ValueError('Incomplete evaluation output inventory')
        for name,digest in receipt['outputs_sha256'].items():
            if sha(metrics.parent/name)!=digest:raise ValueError('Evaluation output hash mismatch')
        scores[condition]=score_labels(records['test'],predictions,labels)
        if score_directory:
            saved=score_directory/f'{condition}_scores.json'
            result=json.loads(saved.read_text(encoding='utf-8'))
            if {key:result[key] for key in scores[condition]}!=scores[condition]:raise ValueError('Saved classification score mismatch')
            for file,digest in result['inputs_sha256'].items():
                if sha(Path(file))!=digest:raise ValueError('Saved score input hash mismatch')
            evidence.append(saved)
        evidence.extend([metrics,receipt_path,prediction_path])
    baseline=[dict(id=row['id'],reference=row['messages'][-1]['content'],prediction=json.dumps(dict(label=labels[0]))) for row in records['test']]
    return dict(status='complete_classification_run_audited',config=str(config_path),counts=summary['counts'],inputs=inputs,
        evaluations=evaluations,best_step=summary['best_step'],best_dev_loss=summary['best_dev_loss'],adapter_sha256=summary['adapter_sha256'],
        checkpoint_evidence=dict(best=best,final=final),quality=scores,
        constant_first_label_baseline=score_labels(records['test'],baseline,labels),
        resources={key:summary[key] for key in ('device','trainable_parameters','total_parameters','peak_allocated_bytes','peak_reserved_bytes','elapsed_seconds_this_session','python','torch')},
        data_manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8')),
        evidence_sha256={public_relative_path(path):sha(path) for path in evidence},
        audit_code_sha256=sha(Path(__file__)),
        limitations='One fixed run; metrics are held-out task agreement, not statistical significance, arbitrary-task generality or domain correctness. CUDA allocator/session resources are not whole-machine cost.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--score-directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);os.chdir(ROOT)
    if a.output.exists():raise FileExistsError('Preserve existing audit; choose new output')
    result=audit(a.config,a.labels,a.score_directory)
    # Relative public paths; no original text or individual predictions in report.
    def relative(value):
        if isinstance(value,dict):return {relative(k):relative(v) for k,v in value.items()}
        if isinstance(value,list):return [relative(v) for v in value]
        if isinstance(value,str):return value.replace(str(ROOT)+'\\','').replace(str(ROOT).replace('\\','/')+'/','').replace('\\','/')
        return value
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(relative(result),stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],best_step=result['best_step'],quality={c:{k:s[k] for k in ('accuracy','macro_f1','invalid_rate')} for c,s in result['quality'].items()})))
    return 0


if __name__=='__main__':raise SystemExit(main())
