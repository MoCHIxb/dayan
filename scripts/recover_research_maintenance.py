"""Explicit one-time recovery of an interrupted training stage after owner-confirmed maintenance."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import run_research_analysis as campaign
from train_research_grid import validate_config,inputs_snapshot,verify_checkpoint
from audit_research_grid import scheduled_best
from personal_llm_data.personal_data import sha
from personal_llm_data.personal_runtime import gpu_guard,write_json
ROOT=campaign.ROOT

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def assert_no_workers():
    names=('run_research_analysis.py','train_research_grid.py','evaluate_research_grid.py','audit_research_grid.py','compact_research_grid.py')
    processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True)
    matches=[line for line in processes.splitlines() if any(name in line for name in names)]
    if matches:raise RuntimeError('Existing workers; do not duplicate GPU work:\n'+'\n'.join(matches))

def choose_checkpoint(run,inputs):
    valid=[];errors={}
    for path in (run/'checkpoints').glob('checkpoint-*'):
        if not path.name.removeprefix('checkpoint-').isdigit() or path.is_symlink():raise ValueError('Unexpected checkpoint path')
        try:
            inventory=verify_checkpoint(path,inputs);state=load(path/'trainer_state.json')
            best=Path(state['best_model_checkpoint'])
            if best.parent.resolve()!=(run/'checkpoints').resolve():raise ValueError('Selected checkpoint outside run')
            verify_checkpoint(best,inputs)
            valid.append((inventory['step'],path))
        except Exception as error:errors[path.name]=repr(error)
    if not valid:raise ValueError('No valid full-state checkpoint. No training restarted: '+str(errors))
    step,path=max(valid,key=lambda item:item[0])
    if step>=128:raise ValueError('Budget already saved at128; needs finalization audit, not another training restart')
    return step,path,errors

def canonical_progress(raw,step,history):
    kept=[];last_step=0;observed=[]
    lines=raw.splitlines(keepends=True)
    for index,line in enumerate(lines):
        try:record=json.loads(line)
        except ValueError:
            if index!=len(lines)-1:raise ValueError('Malformed nonterminal progress row')
            continue
        last_step=max(last_step,int(record['step']))
        if record['step']<=step:
            kept.append(line if line.endswith(b'\n') else line+b'\n')
            if 'eval_loss' in record.get('logs',{}):observed.append((record['step'],record['logs']['eval_loss']))
    expected=[(row['step'],row['eval_loss']) for row in history if 'eval_loss' in row]
    if observed!=expected:raise ValueError('Progress history up to checkpoint does not match its saved dev history')
    return b''.join(kept),last_step

def inspect():
    state=load(campaign.OUT/'execution_state.json');manifest=load(campaign.OUT/'manifest.json')
    current=campaign.inputs()
    if manifest!=dict(inputs=current,inputs_sha256=campaign.digest(current),stages=campaign.stages()):raise ValueError('Frozen runtime/code/data differs; preserve evidence')
    active=state.get('active_stage');stage=next((item for item in manifest['stages'] if item['name']==active),None)
    if not stage or stage['kind']!='train':raise ValueError('This recovery supports interrupted train stages only')
    if (campaign.OUT/'receipts'/(active+'.json')).exists():raise ValueError('Training already receipted; use normal pipeline resume instead')
    cfg_path=Path(stage['config']);config=load(ROOT/cfg_path);validate_config(config);inputs=inputs_snapshot(config,cfg_path)
    run=ROOT/config['output_dir']
    if load(run/'run_inputs.json')!=inputs:raise ValueError('Training inputs/model/runtime changed')
    origin=load(campaign.OUT/'origins'/(campaign.name(config)+'.json'))
    if origin!=dict(config_sha256=sha(cfg_path),run_path=config['output_dir'],inputs_sha256=manifest['inputs_sha256']):raise ValueError('Run origin differs')
    if (run/'maintenance_recovery.json').exists():raise ValueError('Recovery evidence already exists; inspect it without automatic retry')
    if (run/'training_summary.json').exists():raise ValueError('Training summary exists; requires finalization-only audit, not retraining')
    step,checkpoint,errors=choose_checkpoint(run,inputs)
    canonical,last_step=canonical_progress((run/'progress.jsonl').read_bytes(),step,load(checkpoint/'trainer_state.json')['log_history'])
    return state,manifest,stage,cfg_path,config,inputs,run,step,checkpoint,errors,canonical,last_step

def execute():
    with campaign.lock():
        assert_no_workers()
        values=inspect();state,manifest,stage,cfg_path,config,inputs,run,step,checkpoint,errors,canonical,last_step=values
        proof=campaign.verify_large_probe(config['model_key'],config['quantization'])
        with gpu_guard(ROOT):pass  # Release before native child acquires the same guard.
        largest=max((sum(path.stat().st_size for path in directory.rglob('*') if path.is_file()) for directory in (run/'checkpoints').glob('checkpoint-*')),default=0)
        if shutil.disk_usage(run).free<10*1024**3+3*largest:raise RuntimeError('Insufficient conservative recovery save space; no training restarted')
        recovery=campaign.OUT/'recoveries'/stage['name']
        if recovery.exists():raise ValueError('Prior recovery attempt retained; no automatic retry')
        recovery.mkdir(parents=True)
        progress=run/'progress.jsonl';shutil.copy2(progress,recovery/'progress.before_recovery.jsonl')
        plan=dict(reason='owner_confirmed_server_maintenance_emergency_shutdown',stage=stage['name'],original_execution_state=state,restore_checkpoint=str(checkpoint),restored_step=step,last_logged_step=last_step,logged_steps_to_recompute=max(0,last_step-step),checkpoint_failures=errors,later_checkpoint_names_to_preserve=[path.name for path in (run/'checkpoints').glob('checkpoint-*') if int(path.name.removeprefix('checkpoint-'))>step],original_progress_sha256=sha(recovery/'progress.before_recovery.jsonl'),canonical_progress_sha256=__import__('hashlib').sha256(canonical).hexdigest(),inputs_sha256=manifest['inputs_sha256'],cost_limit='Interrupted-session end time/peak unavailable; do not treat recovered-session cumulative timing as full-run cost',tool_sha256=sha(Path(__file__)))
        campaign.fresh(recovery/'plan.json',plan)
        campaign.fresh(recovery/'cost_caveat.json',dict(condition=campaign.name(config),full_training_time_and_peak_incomplete=True,quality_recovery_requires_original128_steps_and_full_dev_test_audit=True,resource_analysis_instruction='Exclude interrupted condition from full-run timing and peak comparisons or report as incomplete; do not use recovered-only elapsed time as total'))
        for path in sorted((run/'checkpoints').glob('checkpoint-*')):
            if int(path.name.removeprefix('checkpoint-'))>step:
                destination=recovery/'later_checkpoint_evidence'/path.name
                if path.is_symlink() or not path.resolve().is_relative_to((run/'checkpoints').resolve()) or not destination.resolve().is_relative_to(recovery.resolve()):raise ValueError('Invalid evidence move')
                destination.parent.mkdir(parents=True,exist_ok=True);path.rename(destination)
        temporary=progress.with_suffix('.recovery.tmp');temporary.write_bytes(canonical);temporary.replace(progress)
        for request in (campaign.OUT/'pause.request',campaign.RUNS/'pause.request'):
            if request.exists():request.rename(recovery/(request.parent.name+'.pause.request.before_recovery'))
        command=list(stage['command'])+['--resume-from-checkpoint',str(checkpoint)]
        campaign.fresh(recovery/'started.json',dict(command=command,started_unix=time.time(),plan_sha256=sha(recovery/'plan.json')))
        write_json(campaign.OUT/'execution_state.json',dict(stage='maintenance_recovery_running',active_stage=stage['name'],restored_step=step))
        print(json.dumps(dict(status='verified_maintenance_recovery_starting',stage=stage['name'],restore_step=step,last_logged_step=last_step,command=command)),flush=True)
        started=time.monotonic()
        try:
            with (recovery/'stdout.log').open('xb') as stdout,(recovery/'stderr.log').open('xb') as stderr:subprocess.run(command,check=True,stdout=stdout,stderr=stderr,cwd=ROOT)
            if campaign.inputs()!=manifest['inputs']:raise ValueError('Frozen inputs changed during recovery')
            summary=load(run/'training_summary.json')
            if summary['status']!='completed_verified' or summary['inputs']!=inputs or summary['completed_optimizer_steps']!=128:raise ValueError('Recovery did not complete original budget')
            verify_checkpoint(run/'checkpoints/checkpoint-128',inputs);best=scheduled_best(load(run/'checkpoints/checkpoint-128/trainer_state.json')['log_history'])
            if best['step']!=summary['best_step'] or best['eval_loss']!=summary['best_dev_loss']:raise ValueError('Recovered selection differs')
            verify_checkpoint(Path(summary['best_checkpoint']),inputs)
            if sha(run/'adapter/adapter_model.safetensors')!=summary['adapter_sha256']:raise ValueError('Export differs')
            event=dict(status='completed',worker_seconds=time.monotonic()-started,metrics={key:summary[key] for key in ('elapsed_seconds_this_session','peak_allocated_bytes','peak_reserved_bytes')},inputs_sha256=manifest['inputs_sha256'])
            campaign.fresh(recovery/'finished.json',event)
            earlier=sorted((campaign.OUT/'attempts'/stage['name']).glob('*.finished.json'),key=lambda path:int(path.name.split('.')[0]))
            if any(load(path)['status']!='paused' for path in earlier):raise ValueError('Unexpected earlier finished session')
            sessions=earlier+[recovery/'finished.json']
            campaign.fresh(run/'training_sessions.json',dict(files_sha256={str(path.relative_to(ROOT)):sha(path) for path in sessions},sessions=[load(path) for path in sessions],maintenance_interruption=dict(plan_sha256=sha(recovery/'plan.json'),interrupted_session_not_measured=True,full_training_cost_incomplete=True)))
            campaign.fresh(run/'maintenance_recovery.json',dict(plan=plan,recovery_finished_sha256=sha(recovery/'finished.json'),analysis_note='Quality and full dev/test audit remain required. Interrupted-session cost is incomplete; exclude this condition from full-run timing/peak comparisons unless separately qualified.'))
            receipt=dict(command=stage['command'],executed_command=command,inputs_sha256=manifest['inputs_sha256'],rank64_probe_sha256=proof,elapsed_seconds_this_attempt=event['worker_seconds'],outputs={relative:sha(ROOT/relative) for relative in stage['outputs']},maintenance_recovery_plan_sha256=sha(recovery/'plan.json'),full_training_cost_incomplete=True)
            campaign.fresh(campaign.OUT/'receipts'/(stage['name']+'.json'),receipt)
            write_json(campaign.OUT/'execution_state.json',dict(stage='maintenance_training_recovered',active_stage=stage['name'],step=128,next='normal run --resume; full test and independent audit still pending'))
            print('Recovered original training to128. Now use the original analysis entry run --resume. Full-run timing for this interrupted condition is incomplete.',flush=True)
        except BaseException:
            write_json(campaign.OUT/'execution_state.json',dict(stage='maintenance_recovery_failed_preserved',active_stage=stage['name']))
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args();os.chdir(ROOT)
    if platform.system()!='Linux':raise RuntimeError('Owner cloud only; no local model runs')
    if args.execute:execute()
    else:
        assert_no_workers();values=inspect();print(json.dumps(dict(executable=False,active_stage=values[2]['name'],restore_checkpoint=str(values[8]),restore_step=values[7],last_logged_step=values[11],checkpoint_failures=values[9],next='--execute after read-only verification'),indent=2))

if __name__=='__main__':main()
