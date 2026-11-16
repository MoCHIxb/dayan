"""Analysis-only retention for newly owned runs, after actual full-state/test audit."""
import argparse
import json
from pathlib import Path
import platform
from personal_llm_data.personal_data import sha,read_rows
from personal_llm_data.banking77 import score_intents
from train_research_grid import ROOT,validate_config,inputs_snapshot
from audit_research_grid import audit as full_audit,scheduled_best
from research_grid import load,identity
CONTROL=ROOT/'experiments/research_banking_analysis_v1'
POLICY=ROOT/'configs/research_banking_v2/analysis_retention.json'
BINARY_NAMES={'adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth'}

def fresh(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def condition_name(config):return identity(config['model_key'],config['quantization'],config['lora_r'],config['lora_alpha'],config['seed'])
def representative(config):
    keep=load(POLICY)['keep_representative_export']
    return (config['lora_r'],config['lora_alpha'],config['seed'])==(keep['rank'],keep['alpha'],keep['seed'])
def safe_run(config):
    expected=ROOT/'experiments/research_banking_v2'/condition_name(config)/'run'
    run=ROOT/config['output_dir']
    if run.absolute()!=expected.absolute() or not run.resolve().is_relative_to((ROOT/'experiments/research_banking_v2').resolve()):raise ValueError('Out-of-scope retention directory')
    for path in [run,*run.parents]:
        if path==ROOT.parent:break
        if path.is_symlink():raise ValueError('Symlink not allowed in retention scope')
    return run

def candidate(run,relative,keep_export):
    rel=Path(relative)
    checkpoint=(len(rel.parts)==3 and rel.parts[0]=='checkpoints' and rel.parts[1].startswith('checkpoint-') and rel.parts[1][11:].isdigit() and rel.name in BINARY_NAMES)
    export=(rel.as_posix()=='adapter/adapter_model.safetensors' and not keep_export)
    target=run/rel
    if not (checkpoint or export) or rel.is_absolute() or not target.resolve().is_relative_to(run.resolve()) or target.is_symlink():raise ValueError('Unapproved deletion target')
    for parent in target.parents:
        if parent==run:break
        if parent.is_symlink():raise ValueError('Symlink parent in deletion scope')
    return target

def verify_origin(config,config_path):
    manifest=load(CONTROL/'manifest.json');origin=load(CONTROL/'origins'/(condition_name(config)+'.json'))
    if origin!=dict(config_sha256=sha(config_path),run_path=config['output_dir'],inputs_sha256=manifest['inputs_sha256']):raise ValueError('Run was not created by analysis-retention controller')
    return manifest

def owned(config,config_path):
    manifest=verify_origin(config,config_path)
    stage_map={stage['name']:stage for stage in manifest['stages']}
    for suffix in ('train','test','audit'):
        label=condition_name(config)+'_'+suffix;stage=stage_map[label];receipt=load(CONTROL/'receipts'/(label+'.json'))
        if receipt['command']!=stage['command'] or receipt['inputs_sha256']!=manifest['inputs_sha256'] or set(receipt['outputs'])!=set(stage['outputs']):raise ValueError('Pre-retention stage identity differs')
        for relative,expected in receipt['outputs'].items():
            if sha(ROOT/relative)!=expected:raise ValueError('Pre-retention stage output changed')
    return manifest

def build_plan(config_path):
    config=load(config_path);validate_config(config);run=safe_run(config);owned(config,config_path)
    if load(run/'run_status.json')['stage']!='completed_verified':raise ValueError('Do not compact active, paused or failed work')
    result=full_audit(config_path)
    if result!=load(run/'audit.json'):raise ValueError('Full pre-deletion re-audit differs')
    keep_export=representative(config);removed={}
    paths=sorted(path for path in run.rglob('*') if path.is_file() and 'retention' not in path.relative_to(run).parts)
    for path in paths:
        if path.is_symlink():raise ValueError('Unexpected symlink in run')
        relative=path.relative_to(run).as_posix()
        if (path.parent.name.startswith('checkpoint-') and path.name in BINARY_NAMES) or (relative=='adapter/adapter_model.safetensors' and not keep_export):
            candidate(run,relative,keep_export);removed[relative]=dict(sha256=sha(path),bytes=path.stat().st_size)
    kept={path.relative_to(run).as_posix():sha(path) for path in paths if path.relative_to(run).as_posix() not in removed}
    if not removed:raise ValueError('No approved completed-run binaries to compact')
    return dict(config_sha256=sha(config_path),policy_sha256=sha(POLICY),run_path=config['output_dir'],keep_export=keep_export,removed=removed,kept_sha256=kept,original_audit=result,binary_audit_status='performed_on_original_files_before_pruning',limit='Deleted binary states cannot be replayed or freshly rehashed; only retained analysis evidence can be re-audited')

def verify_kept(run,plan):
    if plan['policy_sha256']!=sha(POLICY):raise ValueError('Retention policy changed')
    for relative,expected in plan['kept_sha256'].items():
        target=run/relative
        if not target.resolve().is_relative_to(run.resolve()) or target.is_symlink() or sha(target)!=expected:raise ValueError('Retained analysis evidence changed')

def apply_plan(config_path,plan):
    config=load(config_path);validate_config(config);run=safe_run(config)
    if plan['config_sha256']!=sha(config_path) or plan['run_path']!=config['output_dir'] or plan['keep_export']!=representative(config):raise ValueError('Cleanup plan/config differs')
    verify_kept(run,plan)
    # All targets are checked before any removal; only exact files, never recursive deletion.
    for relative,record in plan['removed'].items():
        target=candidate(run,relative,plan['keep_export'])
        if target.exists() and (sha(target)!=record['sha256'] or target.stat().st_size!=record['bytes']):raise ValueError('Planned binary changed; preserve it')
    for relative,record in plan['removed'].items():
        target=candidate(run,relative,plan['keep_export'])
        if target.exists():
            if sha(target)!=record['sha256']:raise ValueError('Target changed during cleanup')
            target.unlink()
    verify_kept(run,plan)
    return sum(record['bytes'] for record in plan['removed'].values())

def compact(config_path,execute=False,finish=False):
    config=load(config_path);run=safe_run(config);directory=run/'retention';plan_path=directory/'plan.json';receipt_path=directory/'complete.json'
    if receipt_path.exists():return verify_retained(config_path)
    if plan_path.exists():
        if not finish:raise ValueError('Prior cleanup intent exists; inspect then explicitly use --finish-cleanup --execute')
        verify_origin(config,config_path);plan=load(plan_path)
    else:
        if finish:raise ValueError('No recorded cleanup intent')
        plan=build_plan(config_path)
    if not execute:return dict(executable=False,targets=plan['removed'],bytes=sum(record['bytes'] for record in plan['removed'].values()),kept_files=len(plan['kept_sha256']),keep_export=plan['keep_export'])
    if not plan_path.exists():fresh(plan_path,plan)
    removed_bytes=apply_plan(config_path,plan)
    fresh(receipt_path,dict(status='analysis_evidence_retained_binaries_pruned_after_full_audit',plan_sha256=sha(plan_path),removed_bytes=removed_bytes,tool_sha256=sha(Path(__file__))))
    return verify_retained(config_path)

def verify_retained(config_path):
    config=load(config_path);validate_config(config);verify_origin(config,config_path);run=safe_run(config);plan_path=run/'retention/plan.json';plan=load(plan_path);receipt=load(run/'retention/complete.json')
    if receipt['status']!='analysis_evidence_retained_binaries_pruned_after_full_audit' or receipt['plan_sha256']!=sha(plan_path) or receipt['tool_sha256']!=sha(Path(__file__)):raise ValueError('Retention receipt changed')
    if plan['config_sha256']!=sha(config_path) or plan['run_path']!=config['output_dir']:raise ValueError('Retention config differs')
    verify_kept(run,plan)
    for relative in plan['removed']:
        if candidate(run,relative,plan['keep_export']).exists():raise ValueError('Pruned binary unexpectedly present')
    if load(run/'run_inputs.json')!=inputs_snapshot(config,config_path):raise ValueError('Scientific inputs changed')
    result=plan['original_audit'];summary=load(run/'training_summary.json');state=load(run/'checkpoints/checkpoint-128/trainer_state.json');best=scheduled_best(state['log_history'])
    if state['global_step']!=128 or summary['counts']!=dict(train=500,dev=100,test=400) or best['step']!=result['best_step'] or best['eval_loss']!=result['best_dev_loss']:raise ValueError('Retained training/selection evidence differs')
    sessions=load(run/'training_sessions.json')
    for relative,expected in sessions['files_sha256'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Retained session record changed')
    records=read_rows(ROOT/config['data_dir']/'test.jsonl');predictions=read_rows(run/'evaluation/adapter.predictions.jsonl')
    if len(records)!=400 or score_intents(records,predictions)!=result['quality']:raise ValueError('Retained full-test score differs')
    if load(run/'audit.json')!=result:raise ValueError('Original audit snapshot changed')
    return result

def audit_retained_or_live(config_path):
    config=load(config_path);run=safe_run(config)
    if (run/'retention/complete.json').exists():return verify_retained(config_path)
    if (run/'retention/plan.json').exists():raise ValueError('Incomplete cleanup requires explicit diagnosis/finish; no full-audit claim')
    return full_audit(config_path)

def verify_deleted_output(config_path,relative,expected):
    config=load(config_path);verify_origin(config,config_path);run=safe_run(config);plan_path=run/'retention/plan.json';receipt=load(run/'retention/complete.json');plan=load(plan_path)
    if receipt['plan_sha256']!=sha(plan_path) or receipt['tool_sha256']!=sha(Path(__file__)):raise ValueError('Invalid retention receipt')
    target=ROOT/relative
    if not target.resolve().is_relative_to(run.resolve()):raise ValueError('Missing output outside compacted run')
    local=target.relative_to(run).as_posix();record=plan['removed'].get(local)
    if not record or record['sha256']!=expected or target.exists():raise ValueError('Unrecorded missing output')
    candidate(run,local,plan['keep_export'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--finish-cleanup',action='store_true');args=parser.parse_args()
    if args.finish_cleanup and not args.execute:parser.error('--finish-cleanup requires --execute')
    if args.execute and platform.system()!='Linux':raise RuntimeError('Production compaction only on owner cloud; local tests use temporary fixtures')
    print(json.dumps(compact(args.config,args.execute,args.finish_cleanup),ensure_ascii=False))

if __name__=='__main__':main()
