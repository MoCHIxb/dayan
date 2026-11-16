"""Read-only full-state and full400 scoring audit for research_banking_v1."""
import argparse
import json
import math
from pathlib import Path
from personal_llm_data.personal_data import sha,read_rows
from personal_llm_data.banking77 import score_intents
from train_research_adapter import ROOT,validate_config,inputs_snapshot,verify_checkpoint

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def scheduled_best(history):
    evaluations=[r for r in history if 'eval_loss' in r]
    if len(evaluations)!=8 or [r['step'] for r in evaluations]!=list(range(16,129,16)) or any(not math.isfinite(r['eval_loss']) for r in evaluations):
        raise ValueError('Eight ordered finite complete dev evaluations required')
    losses=[r for r in history if 'loss' in r]
    if {r['step'] for r in losses}!=set(range(1,129)) or any(not math.isfinite(r['loss']) for r in losses):raise ValueError('Missing/nonfinite step loss evidence')
    return min(evaluations,key=lambda r:(r['eval_loss'],r['step']))

def audit_evaluation(config_path,condition,output,inputs=None):
    config=load(config_path);validate_config(config);inputs=inputs or inputs_snapshot(config,config_path)
    receipt=load(output.with_suffix('.receipt.json'));value=load(output)
    source=ROOT/'data/processed/banking77_8b_v1/fewshot_test.jsonl' if condition=='fewshot' else ROOT/config['data_dir']/'test.jsonl'
    pred=output.with_suffix('.predictions.jsonl')
    if receipt['condition']!=condition or receipt['inputs']!=inputs or receipt['source_sha256']!=sha(source) or receipt['config']!=str(config_path):raise ValueError('Evaluation inputs/condition differ')
    if receipt['predictions_sha256']!=sha(pred) or receipt['metrics_sha256']!=sha(output) or receipt['evaluator_sha256']!=sha(ROOT/'scripts/evaluate_research_adapter.py'):raise ValueError('Evaluation evidence changed')
    adapter=sha(ROOT/config['output_dir']/'adapter/adapter_model.safetensors') if condition=='adapter' else None
    if receipt['adapter_sha256']!=adapter:raise ValueError('Wrong adapter identity')
    rows=read_rows(source)
    if len(rows)!=400 or len({r['id'] for r in rows})!=400:raise ValueError('Full400 required')
    score=score_intents(rows,read_rows(pred))
    if value['status']!='complete_full400_scored' or value['condition']!=condition or value['quality']!=score:raise ValueError('Recomputed quality differs')
    if any(not math.isfinite(v) or v<0 for v in value['resources'].values()):raise ValueError('Invalid generation resources')
    return value

def audit(config_path):
    config=load(config_path);rows=validate_config(config);inputs=inputs_snapshot(config,config_path);run=ROOT/config['output_dir']
    summary=load(run/'training_summary.json')
    if summary['status']!='completed_verified' or summary['inputs']!=inputs or summary['counts']!={k:len(v) for k,v in rows.items()} or summary['completed_optimizer_steps']!=128:raise ValueError('Training budget/inputs incomplete')
    if load(run/'run_inputs.json')!=inputs or load(run/'run_config.json')!=dict(config,output_dir=str(run)):raise ValueError('Run configuration changed')
    final=run/'checkpoints/checkpoint-128';inventory=verify_checkpoint(final,inputs)
    state=load(final/'trainer_state.json')
    if state['global_step']!=128 or summary['final_checkpoint']!=inventory:raise ValueError('Final full state differs')
    best=scheduled_best(state['log_history'])
    if summary['best_step']!=best['step'] or summary['best_dev_loss']!=best['eval_loss']:raise ValueError('Dev selection differs')
    best_path=run/'checkpoints'/f"checkpoint-{best['step']}"
    if Path(summary['best_checkpoint']).resolve()!=best_path.resolve() or Path(state['best_model_checkpoint']).resolve()!=best_path.resolve():raise ValueError('Wrong selected checkpoint')
    verify_checkpoint(best_path,inputs)
    adapter=sha(run/'adapter/adapter_model.safetensors')
    if adapter!=summary['adapter_sha256'] or adapter!=sha(best_path/'adapter_model.safetensors'):raise ValueError('Export differs from selected adapter')
    actual_logs=read_rows(run/'progress.jsonl')
    logged=[r for r in actual_logs if 'eval_loss' in r.get('logs',{})]
    if [(r['step'],r['logs']['eval_loss']) for r in logged]!=[(r['step'],r['eval_loss']) for r in state['log_history'] if 'eval_loss' in r]:raise ValueError('Progress/dev state mismatch')
    evaluation=audit_evaluation(config_path,'adapter',run/'evaluation/adapter.json',inputs)
    return dict(status='research_training_and_full_test_audited',model_key=config['model_key'],recipe_id=config['recipe_id'],seed=config['seed'],train_records=500,dev_records=100,test_records=400,optimizer_steps=128,dev_evaluations=8,best_step=best['step'],best_dev_loss=best['eval_loss'],dev_perplexity=math.exp(best['eval_loss']),adapter_sha256=adapter,training_summary_sha256=sha(run/'training_summary.json'),quality=evaluation['quality'],generation_resources=evaluation['resources'],training_resources={k:summary[k] for k in ('trainable_parameters','total_parameters','elapsed_seconds_this_session','peak_allocated_bytes','peak_reserved_bytes','device')},cost_limitations='Training wall time/peak describe final session; resumed runs must also consult retained attempt logs. Allocated/reserved are process measures, not total board memory.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.config);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],model=result['model_key'],recipe=result['recipe_id'])))

if __name__=='__main__':main()
