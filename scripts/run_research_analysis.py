"""Finite full-factorial BANKING campaign, with explicit blocks and immutable evidence."""
import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from personal_llm_data.personal_data import sha,read_rows
from personal_llm_data.personal_runtime import write_json
from personal_llm_data.banking77 import score_intents
import research_grid as grid
from research_grid import ROOT,load
from deploy_research_models import catalog,backend
from train_research_grid import validate_config,inputs_snapshot,verify_checkpoint
from audit_research_grid import audit_evaluation
from compact_research_grid import audit_retained_or_live as audit,verify_deleted_output

OUT=ROOT/'experiments/research_banking_analysis_v1'
RUNS=ROOT/'experiments/research_banking_v2'
PROBES=RUNS/'rank64_probe'
DATA=ROOT/'data/processed/banking77_8b_v1'

def fresh(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def name(condition):return grid.identity(condition['model_key'],condition['quantization'],condition['lora_r'],condition['lora_alpha'],condition['seed'])
def representative(key,mode):return grid.config(key,mode,16,32,grid.spec()['training_seeds'][0])
def baseline_path(key,mode,condition):return Path('experiments/research_banking_analysis_v1/baselines')/key/(mode+'_'+condition+'.json')

def stages(model=None,mode=None,rank=None):
    conditions=list(grid.configs(model,mode,rank));result=[]
    def add(label,command,outputs,kind,condition=None):
        result.append(dict(name=label,command=[sys.executable,'-u',*map(str,command)],outputs=list(map(str,outputs)),kind=kind,config=str(grid.config_path(condition)) if condition else None))
    add('centroid',['scripts/run_research_analysis.py','centroid'],['experiments/research_banking_analysis_v1/centroid.json','experiments/research_banking_analysis_v1/centroid.predictions.jsonl'],'cpu')
    pairs=list(dict.fromkeys((condition['model_key'],condition['quantization']) for condition in conditions))
    for key,precision in pairs:
        conditions_to_generate=['zero','fewshot'] if precision=='none' else ['zero']
        for baseline in conditions_to_generate:
            config=representative(key,precision);path=baseline_path(key,precision,baseline)
            add(f'baseline_{key}_{precision}_{baseline}',['scripts/evaluate_research_grid.py','--config',grid.config_path(config),'--condition',baseline,'--output',path,'--execute'],[path,path.with_suffix('.predictions.jsonl'),path.with_suffix('.receipt.json')],'baseline',config)
    for condition in conditions:
        label=name(condition);cfg=grid.config_path(condition);run=Path(condition['output_dir']);path=run/'evaluation/adapter.json'
        add(label+'_train',['scripts/train_research_grid.py','--config',cfg,'--execute'],[run/'training_summary.json',run/'training_sessions.json',run/'progress.jsonl',run/'run_inputs.json',run/'adapter/adapter_model.safetensors',run/'checkpoints/checkpoint-128/personal_checkpoint_receipt.json'],'train',condition)
        add(label+'_test',['scripts/evaluate_research_grid.py','--config',cfg,'--condition','adapter','--output',path,'--execute'],[path,path.with_suffix('.predictions.jsonl'),path.with_suffix('.receipt.json')],'test',condition)
        add(label+'_audit',['scripts/audit_research_grid.py','--config',cfg,'--output',run/'audit.json'],[run/'audit.json'],'audit',condition)
        add(label+'_retain',['scripts/compact_research_grid.py','--config',cfg,'--execute'],[run/'retention/plan.json',run/'retention/complete.json'],'retain',condition)
    return result

def verify_large_probe(key,mode):
    import gpu_probe_grid as probe
    directory=probe.BASE/(key+'_'+mode);complete=load(directory/'complete.json')
    if complete['status']!='diagnostic_pair_completed' or complete['inputs']!=probe.gate():raise ValueError('Missing/stale independent rank64 probe')
    for phase in ('train','restore'):
        if sha(directory/(phase+'_receipt.json'))!=complete[phase+'_receipt_sha256']:raise ValueError('rank64 probe receipt changed')
    for relative,expected in load(directory/'train_receipt.json')['files_sha256'].items():
        if sha(directory/relative)!=expected:raise ValueError('rank64 saved state changed')
    restored=load(directory/'restore_receipt.json')
    if restored['status']!='probe_restored_step2_and_generation_completed' or sha(directory/'generation_probe.json')!=restored['generation_sha256']:raise ValueError('rank64 restore/generation incomplete')
    return sha(directory/'complete.json')

def inputs():
    """Frequent lightweight hashes; model bytes are fully hashed in check and each worker."""
    paths=list((RUNS/'configs').glob('*.json'))+list(DATA.rglob('*.jsonl'))+list(DATA.rglob('*.json'))+list((ROOT/'src/personal_llm_data').glob('*.py'))
    paths += [ROOT/path for path in ('configs/research_banking_v2/grid.json','configs/research_models.json','docs/65_full_factorial_banking_protocol.md','repro/banking77_assets.json','requirements-cloud8b.lock','requirements-tokenizers.lock','scripts/cloud/run_research_grid.sh','scripts/cloud/run_research_analysis.sh','scripts/run_research_analysis.py','scripts/compact_research_grid.py','configs/research_banking_v2/analysis_retention.json','docs/66_analysis_evidence_retention.md')]
    paths += [ROOT/'scripts'/file for file in ('research_grid.py','run_research_grid.py','train_research_grid.py','evaluate_research_grid.py','audit_research_grid.py','gpu_probe_grid.py','gpu_probe_research.py','run_research_banking.py','train_qwen_lora.py','intent_baselines.py','run_qwen8b_banking500.py','build_banking77.py','verify_banking77.py','deploy_research_models.py','prepare_8b_4090.py','train_personal_adapter.py')]
    paths += list((ROOT/'experiments/cloud_8b_4090/assets').glob('*.json'))
    paths += [ROOT/'experiments/cloud_8b_4090/compatibility_preflight_chat_v3.json']
    for key in grid.spec()['models']:
        for mode in grid.spec()['quantizations']:
            directory=ROOT/'experiments/cloud_8b_4090/gpu_probe_v1'/(key+'_'+mode)
            paths += [directory/file for file in ('complete.json','train_receipt.json','restore_receipt.json')]
    if len(list((RUNS/'configs').glob('*.json')))!=1125:raise ValueError('prepare must materialize the entire1125-config grid first')
    return dict(python=sys.version,packages={package:version(package) for package in ('torch','transformers','peft','accelerate','bitsandbytes','sentencepiece','protobuf','safetensors','numpy','tokenizers')},files_sha256={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in sorted(set(paths))})

def check():
    from run_qwen8b_banking500 import data_check
    from run_research_banking import diagnostics
    from transformers import AutoTokenizer
    from personal_llm_data.research_sft import encode_row
    from gpu_probe_research import supervised_ids
    data_check();diagnostics();coefficients={}
    for key in grid.spec()['models']:
        backend().assets(key);entry=catalog()['models'][key]
        tokenizer=AutoTokenizer.from_pretrained(str(ROOT/entry['directory']),local_files_only=True,trust_remote_code=False)
        validate_config(representative(key,'none'))
        for path in [DATA/'n500'/f'{split}.jsonl' for split in ('train','dev','test')]+[DATA/'fewshot_test.jsonl']:
            for row in read_rows(path):
                encoded,prompt=encode_row(tokenizer,row,key);ids,labels,previous=supervised_ids(tokenizer,row,key)
                if (encoded['input_ids'],encoded['labels'],prompt)!=(ids,labels,previous):raise ValueError('Grid native encoding differs from verified diagnostic')
        coefficients[key],counts=grid.projection_coefficient(ROOT/entry['directory'],entry['lora_projections'])
        cpu=load(ROOT/'experiments/cloud_8b_4090/compatibility_preflight_chat_v3.json')['models'][key]['projection_module_counts']
        if any(counts[leaf]!=cpu[leaf] for leaf in counts):raise ValueError('Header projections differ from CPU module inventory')
    for condition in grid.configs():
        if load(ROOT/grid.config_path(condition))!=condition:raise ValueError('Generated configuration changed')
    current=inputs();print(json.dumps(dict(status='grid_cpu_inputs_verified_formal_gpu_unexecuted_by_this_check',trainings=1125,full_stages=len(stages()),retention='analysis_after_full_audit')),flush=True)
    return current,coefficients

@contextmanager
def lock():
    from run_research_grid import lock as full_grid_lock
    with full_grid_lock():
        OUT.mkdir(parents=True,exist_ok=True)
        yield


def verify_receipt(stage,receipt,current_digest):
    if receipt['command']!=stage['command'] or receipt['inputs_sha256']!=current_digest or set(receipt['outputs'])!=set(stage['outputs']):raise ValueError('Stage command/frozen inputs changed')
    for relative,expected in receipt['outputs'].items():
        if (ROOT/relative).exists():
            if sha(ROOT/relative)!=expected:raise ValueError('Completed output changed: '+relative)
        elif stage['config']:verify_deleted_output(Path(stage['config']),relative,expected)
        else:raise ValueError('Unrecorded missing output: '+relative)
    if stage['config']:
        condition=load(ROOT/stage['config'])
        complete=PROBES/(condition['model_key']+'_'+condition['quantization'])/'complete.json'
        if receipt['rank64_probe_sha256']!=sha(complete):raise ValueError('Stage prerequisite changed')

def centroid():
    from intent_baselines import centroid_predictions
    records=read_rows(DATA/'n500/test.jsonl');predictions=centroid_predictions(read_rows(DATA/'n500/train.jsonl'),records)
    path=OUT/'centroid.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix('.predictions.jsonl').exists():raise FileExistsError('Keep baseline evidence')
    with path.with_suffix('.predictions.jsonl').open('x',encoding='utf-8') as stream:
        for row in predictions:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    fresh(path,score_intents(records,predictions))

def summary_statistics(values):
    if len(values)!=5:raise ValueError('All five predeclared seeds required for cell statistics')
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),minimum=min(values),maximum=max(values))

def aggregate(runs,baselines):
    groups={}
    for record in runs:
        key=(record['model_key'],record['quantization'],record['rank'],record['alpha'])
        if key not in groups:groups[key]={}
        if record['seed'] in groups[key]:raise ValueError('Duplicate seed result')
        groups[key][record['seed']]=record
    cells=[]
    for (key,mode,rank,alpha),records in sorted(groups.items()):
        seeds=grid.spec()['training_seeds'];complete=set(records)==set(seeds)
        cell=dict(model=key,quantization=mode,rank=rank,alpha=alpha,alpha_over_rank=alpha/rank,seeds=sorted(records),status='complete5' if complete else 'incomplete_no_mean')
        if complete:
            cell['metrics']={metric:summary_statistics([records[seed]['quality'][metric] for seed in seeds]) for metric in ('macro_f1','accuracy','invalid_rate','minimum_class_recall')}
            cell['dev_loss']=summary_statistics([records[seed]['best_dev_loss'] for seed in seeds])
            if all('cumulative_training_resources' in records[seed] for seed in seeds):
                cell['resources']={metric:summary_statistics([records[seed]['cumulative_training_resources'][metric] for seed in seeds]) for metric in ('worker_seconds_all_sessions','peak_allocated_bytes_all_sessions','peak_reserved_bytes_all_sessions')}
                parameters={records[seed]['training_resources']['trainable_parameters'] for seed in seeds}
                if len(parameters)!=1:raise ValueError('Same cell has inconsistent LoRA parameter counts')
                cell['trainable_parameters']=next(iter(parameters))
            baseline=baselines.get(key,{}).get(mode+'_zero')
            if baseline:cell['macro_f1_gain_over_same_precision_zero']=summary_statistics([records[seed]['quality']['macro_f1']-baseline['quality']['macro_f1'] for seed in seeds])
            reference=groups.get((key,mode,16,32),{})
            if set(reference)==set(seeds):cell['paired_seed_macro_f1_difference_vs_r16_a32']=summary_statistics([records[seed]['quality']['macro_f1']-reference[seed]['quality']['macro_f1'] for seed in seeds])
        cells.append(cell)
    return cells

def report(current=None,selected=None):
    if current is None:current,_=check()
    current_digest=digest(current);manifest=load(OUT/'manifest.json')
    if manifest!=dict(inputs=current,inputs_sha256=current_digest,stages=stages()):raise ValueError('Campaign manifest changed')
    all_stages=stages();done=[]
    for stage in all_stages:
        path=OUT/'receipts'/(stage['name']+'.json')
        if path.exists():verify_receipt(stage,load(path),current_digest);done.append(stage)
    if selected:
        for stage in selected:
            if stage not in done:raise ValueError('Selected block incomplete')
    baselines={};runs=[]
    for stage in done:
        if stage['kind']=='baseline':
            condition=load(ROOT/stage['config']);key=condition['model_key'];mode=condition['quantization'];variant='fewshot' if stage['name'].endswith('_fewshot') else 'zero'
            baselines.setdefault(key,{})[mode+'_'+variant]=audit_evaluation(Path(stage['config']),variant,ROOT/baseline_path(key,mode,variant))
        elif stage['kind']=='audit':
            result=audit(Path(stage['config']))
            if result!=load(ROOT/stage['outputs'][0]):raise ValueError('Recomputed full-state/test audit differs')
            runs.append(result)
    if any(stage['kind']=='cpu' for stage in done):
        from intent_baselines import centroid_predictions
        records=read_rows(DATA/'n500/test.jsonl');predictions=centroid_predictions(read_rows(DATA/'n500/train.jsonl'),records)
        if predictions!=read_rows(OUT/'centroid.predictions.jsonl') or score_intents(records,predictions)!=load(OUT/'centroid.json'):raise ValueError('CPU baseline differs')
    complete=len(done)==len(all_stages)
    if complete:
        for key in grid.spec()['models']:
            for mode in grid.spec()['quantizations']:verify_large_probe(key,mode)
    result=dict(status='complete_grid1125_analysis_evidence_verified' if complete else 'partial_grid_progress',completed_stages=len(done),total_stages=len(all_stages),audited_trainings=len(runs),total_trainings=1125,evidence_level='original full-state audit before pruning; retained metadata/predictions/metrics re-audited afterwards; deleted binaries cannot be replayed',runs=runs,baselines=baselines,cells=aggregate(runs,baselines),limitations='Fixed card10 and training data, five optimization seeds, not full77 or data-sampling robustness; fixed128 steps/lr, not each model individually optimized; no automatic significance or universal optimal Rank/Alpha claim. Test seen historically; no test-driven pruning. Different model sizes/projections/tokenizers.')
    retained_stages=[stage for stage in done if stage['kind']=='retain']
    retained_plans=[load(ROOT/stage['outputs'][0]) for stage in retained_stages]
    retained_receipts=[load(ROOT/stage['outputs'][1]) for stage in retained_stages]
    result['retention_summary']=dict(pruned_conditions=len(retained_stages),kept_representative_exports=sum(plan['keep_export'] for plan in retained_plans),released_binary_bytes=sum(receipt['removed_bytes'] for receipt in retained_receipts),limits='Binary hashes were verified before pruning; retained analytical files and predictions remain re-auditable')
    write_json(OUT/'progress_summary.json',result)
    text=['# Full-factorial BANKING grid progress','',result['status'],f"Audited trainings: {len(runs)}/1125",'',result['limitations'],'','|Model|Precision|r|alpha|Seeds|Macro-F1 mean ± sample SD|','|---|---|---:|---:|---:|---|']
    for cell in result['cells']:
        metric=cell.get('metrics',{}).get('macro_f1');value=f"{metric['mean']:.4f} ± {metric['sample_sd']:.4f}" if metric else 'incomplete'
        text.append(f"|{cell['model']}|{cell['quantization']}|{cell['rank']}|{cell['alpha']}|{len(cell['seeds'])}|{value}|")
    temporary=OUT/'progress.md.tmp';temporary.write_text('\n'.join(text)+'\n',encoding='utf-8');temporary.replace(OUT/'progress.md')
    import csv
    with (OUT/'trends.tsv.tmp').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['model','precision','rank','alpha','alpha_over_rank','seed_count','status','macro_f1_mean','macro_f1_sample_sd','accuracy_mean','invalid_rate_mean','trainable_parameters','train_worker_seconds_mean','train_peak_allocated_bytes_mean'])
        for cell in result['cells']:
            metrics=cell.get('metrics',{});f1=metrics.get('macro_f1',{})
            writer.writerow([cell['model'],cell['quantization'],cell['rank'],cell['alpha'],cell['alpha_over_rank'],len(cell['seeds']),cell['status'],f1.get('mean',''),f1.get('sample_sd',''),metrics.get('accuracy',{}).get('mean',''),metrics.get('invalid_rate',{}).get('mean',''),cell.get('trainable_parameters',''),cell.get('resources',{}).get('worker_seconds_all_sessions',{}).get('mean',''),cell.get('resources',{}).get('peak_allocated_bytes_all_sessions',{}).get('mean','')])
    (OUT/'trends.tsv.tmp').replace(OUT/'trends.tsv')
    if complete:
        if (OUT/'results.json').exists():
            if load(OUT/'results.json')!=result:raise ValueError('Existing final summary differs')
        else:fresh(OUT/'results.json',result)
    if complete or selected is not None:
        write_json(OUT/'execution_state.json',dict(stage='complete_audited' if complete else 'selected_block_complete_global_incomplete',audited_trainings=len(runs),completed_stages=len(done),total_stages=len(all_stages)))
    print(json.dumps(dict(status=result['status'],audited_trainings=len(runs),total_trainings=1125)),flush=True)

def disk_guard(conditions,coefficients):
    pending=[condition for condition in conditions if not (OUT/'receipts'/(name(condition)+'_retain.json')).exists()]
    maximum=max((coefficients[condition['model_key']]*condition['lora_r'] for condition in pending),default=0)
    pairs=set((condition['model_key'],condition['quantization']) for condition in pending)
    keep_exports=sum(4*coefficients[key]*16 for key,mode in pairs)
    evidence=len(pending)*64*1024**2
    forecast=dict(pending_trainings=len(pending),estimated_analysis_evidence_bytes=evidence,representative_adapter_allowance_bytes=keep_exports,one_active_run_and_reserve_bytes=52*maximum+10*1024**3,required_free_bytes=evidence+keep_exports+52*maximum+10*1024**3,estimate_not_guarantee=True)
    free=shutil.disk_usage(OUT).free;print(json.dumps(dict(disk_free_bytes=free,forecast=forecast)),flush=True)
    if free<forecast['required_free_bytes']:raise RuntimeError('Insufficient estimated space for analysis evidence and one active run; no old files are deleted')
    return forecast


def run(model=None,mode=None,rank=None,resume=False,checkpoint=None,finish_cleanup=False):
    with lock():
        if (RUNS/'manifest.json').exists():raise ValueError('Old full-retention campaign exists; preserve it. This entry does not silently migrate or prune old runs.')
        current,coefficients=check();current_digest=digest(current);selected=stages(model,mode,rank);conditions=list(grid.configs(model,mode,rank))
        for key,precision in dict.fromkeys((condition['model_key'],condition['quantization']) for condition in conditions):verify_large_probe(key,precision)
        disk_guard(conditions,coefficients)
        expected=dict(inputs=current,inputs_sha256=current_digest,stages=stages());manifest=OUT/'manifest.json';previous=load(OUT/'execution_state.json') if (OUT/'execution_state.json').exists() else {}
        if manifest.exists():
            if not resume or load(manifest)!=expected:raise ValueError('Existing global grid requires --resume and identical frozen inputs')
        else:
            if resume:raise ValueError('No grid manifest to resume')
            if any((ROOT/path).exists() for stage in stages() for path in stage['outputs']):raise FileExistsError('Prior unreceipted output retained')
            fresh(manifest,expected)
        if finish_cleanup:
            recovery=[stage for stage in selected if stage['kind']=='retain' and stage['name']==previous.get('active_stage')]
            if previous.get('stage')!='failed' or len(recovery)!=1:raise ValueError('Explicit cleanup completion requires exactly the previously failed retention stage')
            selected=recovery+[stage for stage in selected if stage!=recovery[0]]
        subprocess.run([sys.executable,'scripts/prepare_8b_4090.py','check'],check=True)
        if resume and (OUT/'pause.request').exists():(OUT/'pause.request').rename(OUT/f'pause.consumed.{time.time_ns()}')
        for stage in selected:
            path=OUT/'receipts'/(stage['name']+'.json')
            if path.exists():verify_receipt(stage,load(path),current_digest);continue
            if (OUT/'pause.request').exists():write_json(OUT/'execution_state.json',dict(stage='paused_between_stages',next_stage=stage['name']));return
            attempts=OUT/'attempts'/stage['name'];command=list(stage['command'])
            if stage['kind']=='train':
                condition=load(ROOT/stage['config']);origin=OUT/'origins'/(name(condition)+'.json');expected_origin=dict(config_sha256=sha(ROOT/stage['config']),run_path=condition['output_dir'],inputs_sha256=current_digest)
                if not origin.exists():
                    target=ROOT/condition['output_dir']
                    if target.exists() and any(target.iterdir()):raise ValueError('Unowned existing run; preserve it without automatic adoption')
                    fresh(origin,expected_origin)
                elif load(origin)!=expected_origin:raise ValueError('Run ownership/input mismatch')
            if attempts.exists():
                if stage['kind']=='retain' and finish_cleanup and previous.get('stage')=='failed' and previous.get('active_stage')==stage['name']:
                    command+=['--finish-cleanup']
                elif stage['kind']!='train' or not checkpoint or previous.get('stage')!='paused_checkpoint_verified' or previous.get('active_stage')!=stage['name']:raise ValueError('Prior incomplete/failed attempt retained; no automatic retry')
                if stage['kind']=='train':
                    condition=load(ROOT/stage['config']);target=Path(checkpoint).resolve()
                    if target.parent!=(ROOT/condition['output_dir']/'checkpoints').resolve() or target!=Path(previous['checkpoint']).resolve():raise ValueError('Wrong paused checkpoint')
                    verify_checkpoint(target,inputs_snapshot(condition,Path(stage['config'])));command+=['--resume-from-checkpoint',str(target)]
            elif any((ROOT/relative).exists() for relative in stage['outputs']):raise FileExistsError('Unreceipted partial evidence retained')
            if stage['kind']=='train':
                condition=load(ROOT/stage['config']);disk_guard([condition],coefficients)
            attempts.mkdir(parents=True,exist_ok=True);attempt=len(list(attempts.glob('*.started.json')))+1
            started=time.monotonic();fresh(attempts/f'{attempt}.started.json',dict(command=command,started_unix=time.time()))
            write_json(OUT/'execution_state.json',dict(stage='running',active_stage=stage['name'],attempt=attempt,selected_trainings=len(conditions)))
            print(json.dumps(dict(stage='running',active_stage=stage['name'],attempt=attempt)),flush=True)
            try:
                with (attempts/f'{attempt}.stdout.log').open('xb') as out,(attempts/f'{attempt}.stderr.log').open('xb') as err:subprocess.run(command,check=True,stdout=out,stderr=err)
                worker_seconds=time.monotonic()-started
                if digest(inputs())!=current_digest:raise ValueError('Frozen scientific inputs/code changed')
                if stage['kind']=='train':
                    condition=load(ROOT/stage['config']);status=load(ROOT/condition['output_dir']/'run_status.json')
                    if status['stage']=='paused_checkpoint_verified':
                        verify_checkpoint(Path(status['checkpoint']),inputs_snapshot(condition,Path(stage['config'])))
                        step=status['evidence']['step'];metrics=load(ROOT/condition['output_dir']/f'pause_metrics.{step}.json')
                        fresh(attempts/f'{attempt}.finished.json',dict(status='paused',worker_seconds=worker_seconds,metrics=metrics,inputs_sha256=current_digest))
                        write_json(OUT/'execution_state.json',dict(stage='paused_checkpoint_verified',active_stage=stage['name'],checkpoint=status['checkpoint']));return
                    metrics=load(ROOT/condition['output_dir']/'training_summary.json')
                    fresh(attempts/f'{attempt}.finished.json',dict(status='completed',worker_seconds=worker_seconds,metrics={key:metrics[key] for key in ('elapsed_seconds_this_session','peak_allocated_bytes','peak_reserved_bytes')},inputs_sha256=current_digest))
                    session_paths=sorted(attempts.glob('*.finished.json'),key=lambda path:int(path.name.split('.')[0]))
                    if len(session_paths)!=len(list(attempts.glob('*.started.json'))):raise ValueError('Some training attempts lack end-of-session evidence')
                    sessions=[load(path) for path in session_paths]
                    if any(item['inputs_sha256']!=current_digest or item['status'] not in ('paused','completed') for item in sessions):raise ValueError('Invalid session history')
                    fresh(ROOT/condition['output_dir']/'training_sessions.json',dict(files_sha256={str(path.relative_to(ROOT)):sha(path) for path in session_paths},sessions=sessions))
                prerequisite=None
                if stage['config']:
                    condition=load(ROOT/stage['config']);prerequisite=sha(PROBES/(condition['model_key']+'_'+condition['quantization'])/'complete.json')
                fresh(path,dict(command=stage['command'],executed_command=command,inputs_sha256=current_digest,rank64_probe_sha256=prerequisite,elapsed_seconds_this_attempt=time.monotonic()-started,outputs={relative:sha(ROOT/relative) for relative in stage['outputs']}))
            except BaseException:
                failed_end=attempts/f'{attempt}.failed.json'
                if not failed_end.exists():fresh(failed_end,dict(status='failed_preserved',elapsed_seconds_this_attempt=time.monotonic()-started))
                write_json(OUT/'execution_state.json',dict(stage='failed',active_stage=stage['name'],attempt=attempt));raise
        report(current,selected)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',nargs='?',default='plan',choices=['plan','prepare','check','probe','run','report','pause','centroid']);parser.add_argument('--model',choices=grid.spec()['models']);parser.add_argument('--mode',choices=grid.spec()['quantizations']);parser.add_argument('--rank',type=int,choices=grid.spec()['ranks']);parser.add_argument('--resume',action='store_true');parser.add_argument('--checkpoint');parser.add_argument('--finish-cleanup',action='store_true');args=parser.parse_args();os.chdir(ROOT)
    if (args.resume or args.checkpoint or args.finish_cleanup) and args.action!='run':parser.error('Resume flags require run')
    if (args.checkpoint or args.finish_cleanup) and not args.resume:parser.error('Checkpoint/finish-cleanup requires --resume')
    if args.action=='plan':
        conditions=list(grid.configs(args.model,args.mode,args.rank));print(json.dumps(dict(executable=False,design=grid.spec(),selected_trainings=len(conditions),full_trainings=1125,selected_stages=len(stages(args.model,args.mode,args.rank)),full_stages=len(stages()),selection=dict(model=args.model,mode=args.mode,rank=args.rank)),indent=2));return
    if platform.system()!='Linux':raise RuntimeError('New model/data executions only on owner Linux4090')
    if args.action=='prepare':
        with lock():
            grid.materialize();_,coefficients=check();conditions=list(grid.configs(args.model,args.mode,args.rank));disk_guard(conditions,coefficients)
    elif args.action=='check':check()
    elif args.action=='probe':
        with lock():
            check()
            for key,mode in dict.fromkeys((condition['model_key'],condition['quantization']) for condition in grid.configs(args.model,args.mode,args.rank)):
                directory=PROBES/(key+'_'+mode)
                if directory.exists():verify_large_probe(key,mode);continue
                subprocess.run([sys.executable,'-u','scripts/gpu_probe_grid.py','--execute','--model',key,'--mode',mode],check=True);verify_large_probe(key,mode)
    elif args.action=='run':run(args.model,args.mode,args.rank,args.resume,args.checkpoint,args.finish_cleanup)
    elif args.action=='report':
        with lock():report()
    elif args.action=='pause':
        status=load(OUT/'execution_state.json')
        if status['stage']!='running':raise ValueError('No running v2 stage recorded')
        fresh(OUT/'pause.request',dict(requested_unix=time.time()))
        active=next(stage for stage in stages() if stage['name']==status['active_stage'])
        if active['kind']=='train':
            condition=load(ROOT/active['config']);run_path=ROOT/condition['output_dir']
            if (run_path/'run_config.json').exists() and not (run_path/'pause.request').exists():
                with (run_path/'pause.request').open('x',encoding='utf-8') as stream:stream.write(str(time.time()))
        print('Wait for verified pause and process exit. Training pauses at scheduled16-step boundary once initialized; a startup race may finish the current train stage. Generation finishes full400; cleanup finishes its recorded transaction.')
    else:centroid()

if __name__=='__main__':main()
