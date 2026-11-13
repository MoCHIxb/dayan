"""IntentTune's finite serial BANKING77 pipeline. Plan by default; no auto retry."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from personal_llm_data.banking77 import LABELS, SIZES, score_intents
from personal_llm_data.personal_data import sha, verify_prepared, read_rows
from personal_llm_data.personal_runtime import gpu_guard
from build_banking77 import ROOT, fetch, build, write_json
from verify_banking77 import verify

OUT=ROOT/'experiments/intenttune_banking77_v1'
DATA=ROOT/'data/processed/banking77_v1'
LOCK=ROOT/'repro/banking77_assets.json'


def step(name, command, outputs):
    return dict(name=name,command=[sys.executable,'-u',*command],outputs=outputs)


def stages():
    result=[step('fewshot',['scripts/intent_baselines.py','--condition','fewshot','--execute'],
                 ['fewshot/fewshot_metrics.json','fewshot/fewshot_metrics.predictions.jsonl','fewshot/fewshot_metrics.receipt.json','fewshot/fewshot_scores.json'])]
    for size in (500,200,50):
        config=f'configs/intenttune_v1/qwen3_4b_n{size}.json'
        prefix=f'n{size}'
        result.append(step(f'centroid_{size}',['scripts/intent_baselines.py','--condition','centroid','--size',str(size)],
                           [prefix+'/centroid_scores.json',prefix+'/centroid_scores.predictions.jsonl']))
        result.append(step(f'train_{size}',['-m','personal_llm_data.workflow','train-personal','--config',config,'--execute'],
                           [prefix+'/run/training_summary.json',prefix+'/run/run_inputs.json',prefix+'/run/progress.jsonl',
                            prefix+'/run/adapter/adapter_model.safetensors',prefix+'/run/checkpoints/checkpoint-128/personal_checkpoint_receipt.json']))
        for condition in ('base','adapter'):
            result.append(step(f'evaluate_{size}_{condition}',
                ['-m','personal_llm_data.workflow','evaluate-personal','--config',config,'--condition',condition,'--execute'],
                [f'{prefix}/run/evaluation/{condition}_metrics'+suffix for suffix in ('.json','.predictions.jsonl','.receipt.json')]))
            result.append(step(f'score_{size}_{condition}',
                ['scripts/score_intents.py','--data-dir',f'data/processed/banking77_v1/n{size}',
                 '--predictions',f'experiments/intenttune_banking77_v1/{prefix}/run/evaluation/{condition}_metrics.predictions.jsonl',
                 '--output',f'experiments/intenttune_banking77_v1/{prefix}/{condition}_scores.json'],[f'{prefix}/{condition}_scores.json']))
        result.append(step(f'audit_{size}',
            ['scripts/audit_classification_run.py','--config',config,'--labels',*LABELS,
             '--score-directory',f'experiments/intenttune_banking77_v1/{prefix}',
             '--output',f'experiments/intenttune_banking77_v1/{prefix}/run_audit.json'],[f'{prefix}/run_audit.json']))
    return result


def assets(check_gpu=False):
    reference=json.loads(LOCK.read_text(encoding='utf-8'))
    for name,digest in reference['data_sha256'].items():
        if sha(DATA/name)!=digest:raise ValueError('Prepared data changed: '+name)
    for name,digest in reference['config_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Scientific config changed: '+name)
    if {p.name for p in (ROOT/'models/Qwen3-4B').iterdir() if p.is_file()}!=set(reference['model_sha256']):
        raise ValueError('Unexpected model file inventory')
    for name,digest in reference['model_sha256'].items():
        if sha(ROOT/'models/Qwen3-4B'/name)!=digest:raise ValueError('Model changed: '+name)
    for name,digest in reference['core_code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen existing runtime changed: '+name)
    for size in SIZES:verify_prepared(DATA/f'n{size}')
    source_audit=verify(ROOT/'data/raw/banking77_v1',DATA)
    packages={};mismatches={}
    for line in (ROOT/'requirements-train.lock').read_text().splitlines():
        if not line or line.startswith(('#','--')):continue
        package,expected=line.split('==',1)
        try:actual=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:actual=None
        packages[package]=actual
        if actual!=expected:mismatches[package]=dict(expected=expected,actual=actual)
    if mismatches:raise ValueError('Locked environment differs: '+json.dumps(mismatches))
    result=dict(status='cpu_assets_verified_gpu_unexecuted',gpu_checked=False,source_audit=source_audit,packages=packages)
    if check_gpu:
        if platform.system()!='Linux':raise ValueError('GPU execution is restricted to the owner-designated Linux RTX3090 server')
        with gpu_guard(ROOT):
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
                raise ValueError('Exactly one visible CUDA/BF16 GPU required')
            device=torch.cuda.get_device_name(0)
            if not re.search(r'RTX\s+3090\b(?!\s*Ti)',device):raise ValueError('Expected RTX3090, not another platform')
            result.update(status='linux_rtx3090_preflight_passed',gpu_checked=True,
                          gpu=dict(name=device,total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                                   torch=torch.__version__,cuda=torch.version.cuda),
                          platform=dict(system=platform.system(),release=platform.release(),python=platform.python_version()))
    return result


def inputs():
    files=[LOCK,ROOT/'requirements-train.lock']
    reference=json.loads(LOCK.read_text(encoding='utf-8'))
    # Bind real asset bytes, not only the file that declares their expected hashes.
    files+=[DATA/name for name in reference['data_sha256']]
    files+=[ROOT/'models/Qwen3-4B'/name for name in reference['model_sha256']]
    files+=list((ROOT/'configs/intenttune_v1').glob('*.json'))
    files+=[ROOT/'src/personal_llm_data'/name for name in ('banking77.py','classification.py','personal_data.py',
            'personal_runtime.py','text_safety.py','model_loading.py','workflow.py','training_protocol.py')]
    files+=[ROOT/'scripts'/name for name in ('build_banking77.py','verify_banking77.py','intent_baselines.py',
            'score_intents.py','run_intenttune_3090.py','train_personal_adapter.py','train_qwen_lora.py',
            'evaluate_personal_adapter.py','evaluate_generation.py','audit_classification_run.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}


def verify_receipt(receipt, stage, frozen_inputs):
    if receipt.get('planned_command')!=stage['command'] or receipt.get('inputs_sha256')!=frozen_inputs:
        raise ValueError('Stage command/input identity mismatch')
    if set(receipt['outputs_sha256'])!=set(stage['outputs']):raise ValueError('Stage output inventory mismatch')
    for name,digest in receipt['outputs_sha256'].items():
        if sha(OUT/name)!=digest:raise ValueError('Stage output changed: '+name)


@contextmanager
def pipeline_lock():
    # Execution is Linux-only. Keep the pipeline lock separate from each child GPU lock.
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'pipeline.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def state(value):
    path=OUT/'execution_state.json';temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def gpu_preflight():
    """The short-lived child owns any CUDA context, never the coordinator."""
    return json.loads(subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--gpu-check'],cwd=ROOT,text=True))


def run(resume=False, acknowledge_failure=None, checkpoint=None):
    if platform.system()!='Linux':raise ValueError('Run all new GPU experiments on Linux/RTX3090 only')
    with pipeline_lock():
        manifest=OUT/'manifest.json';current=inputs();plan=stages()
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            if not resume:raise FileExistsError('Pipeline exists; inspect state and use explicit --resume')
            if saved['inputs_sha256']!=current or saved['stages']!=plan:raise ValueError('Frozen pipeline inputs/commands changed')
        else:
            if resume:raise ValueError('No frozen pipeline to resume')
            if any((OUT/p).exists() for stage in plan for p in stage['outputs']):raise FileExistsError('Pre-existing outputs lack a pipeline manifest')
            write_json(manifest,dict(inputs_sha256=current,stages=plan,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        # A preflight failure is not retried automatically; no model is loaded by the coordinator.
        # Run CUDA inspection in a child that exits before training; a coordinator
        # CUDA context would otherwise be detected as a competing GPU process.
        preflight=gpu_preflight()
        previous_state=json.loads((OUT/'execution_state.json').read_text()) if (OUT/'execution_state.json').exists() else {}
        if resume and (OUT/'pause.pipeline.request').exists():
            (OUT/'pause.pipeline.request').rename(OUT/f'pause.pipeline.consumed.{time.time_ns()}')
        for stage in plan:
            name=stage['name'];receipt=OUT/f'receipts/{name}.json'
            if receipt.exists():
                verify_receipt(json.loads(receipt.read_text()),stage,current);continue
            if (OUT/'pause.pipeline.request').exists():
                state(dict(stage='paused_between_stages',next_stage=name))
                print('Paused between stages. Remove the explicit request only when the owner resumes.');return
            attempts=OUT/'attempts'/name
            previous=list(attempts.glob('*.started.json')) if attempts.exists() else []
            paused_resume=bool(checkpoint and previous_state.get('stage')=='paused_checkpoint_verified' and previous_state.get('active_stage')==name)
            if previous and acknowledge_failure!=name and not paused_resume:
                raise ValueError('Incomplete prior attempt: preserve evidence and explicitly acknowledge '+name)
            training_resume=bool(checkpoint and name.startswith('train_'))
            if any((OUT/path).exists() for path in stage['outputs']) and not training_resume:
                raise FileExistsError('Partial stage outputs exist; inspect/archive explicitly before a retry')
            command=list(stage['command'])
            if checkpoint is not None and name.startswith('train_'):
                size=int(name.split('_')[1]);target=OUT/f'n{size}/run/checkpoints'
                resolved=Path(checkpoint).resolve()
                if resolved.parent!=target.resolve():
                    if previous:raise ValueError('Checkpoint does not belong to the interrupted stage')
                else:
                    command+=['--resume-from-checkpoint',str(resolved)];checkpoint=None
            attempts.mkdir(parents=True,exist_ok=True);index=len(previous)+1
            write_json(attempts/f'{index}.started.json',dict(planned_command=stage['command'],executed_command=command,started_unix=time.time()))
            state(dict(stage='running',active_stage=name,attempt=index))
            try:
                with (attempts/f'{index}.stdout.log').open('xb') as stdout,(attempts/f'{index}.stderr.log').open('xb') as stderr:
                    subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,check=True)
                if inputs()!=current:raise ValueError('Inputs changed during a stage')
                if name.startswith('train_'):
                    size=int(name.split('_')[1]);status_path=OUT/f'n{size}/run/run_status.json'
                    status=json.loads(status_path.read_text())
                    if status.get('stage')=='paused_checkpoint_verified':
                        from train_personal_adapter import inputs_snapshot, verify_checkpoint
                        config_path=ROOT/f'configs/intenttune_v1/qwen3_4b_n{size}.json'
                        evidence=verify_checkpoint(Path(status['checkpoint']),inputs_snapshot(json.loads(config_path.read_text()),config_path))
                        state(dict(stage='paused_checkpoint_verified',active_stage=name,checkpoint=status['checkpoint'],evidence=evidence))
                        print('Training paused with verified full state; pipeline stopped.');return
                hashes={path:sha(OUT/path) for path in stage['outputs']}
                receipt.parent.mkdir(parents=True,exist_ok=True)
                write_json(receipt,dict(planned_command=stage['command'],executed_command=command,
                                       inputs_sha256=current,outputs_sha256=hashes))
            except BaseException as exc:
                state(dict(stage='stopped_needs_inspection',active_stage=name,attempt=index,error_type=type(exc).__name__))
                raise
        if checkpoint is not None:raise ValueError('Resume checkpoint was not used')
        summary=summarize()
        state(dict(stage='complete_audited',completed_stages=len(plan),summary='results.json',gpu=preflight.get('gpu')))
        print(json.dumps(dict(status=summary['status'],completed_stages=len(plan))))


def summarize():
    frozen=json.loads((OUT/'manifest.json').read_text())
    current=inputs()
    if frozen['inputs_sha256']!=current:raise ValueError('Current inputs differ from the pipeline')
    for stage in stages():verify_receipt(json.loads((OUT/f'receipts/{stage["name"]}.json').read_text()),stage,current)
    result=dict(status='complete_banking77_card10_pipeline_audited',task='BANKING77 card10, not full 77-way benchmark',
                training_sizes=list(SIZES),dev=100,test=400,results={},fewshot=json.loads((OUT/'fewshot/fewshot_scores.json').read_text()),
                limitations='Single training seed; no significance, arbitrary-domain, OOS, safety or personal-writing claims')
    for size in SIZES:
        audit=json.loads((OUT/f'n{size}/run_audit.json').read_text())
        detailed={}
        for condition in ('base','adapter'):
            scores=score_intents(read_rows(DATA/f'n{size}/test.jsonl'),read_rows(OUT/f'n{size}/run/evaluation/{condition}_metrics.predictions.jsonl'))
            saved=json.loads((OUT/f'n{size}/{condition}_scores.json').read_text())
            if {key:saved[key] for key in scores}!=scores:raise ValueError('Intent diagnostic recomputation differs')
            detailed[condition]=scores
        result['results'][str(size)]=dict(audit=audit,quality=detailed,centroid=json.loads((OUT/f'n{size}/centroid_scores.json').read_text()))
    output=OUT/'results.json'
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError('Existing summary differs; do not overwrite')
    else:write_json(output,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--check',action='store_true')
    group.add_argument('--execute',action='store_true')
    group.add_argument('--report',action='store_true')
    group.add_argument('--gpu-check',action='store_true',help=argparse.SUPPRESS)
    group.add_argument('--pause',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--acknowledge-failure',choices=[stage['name'] for stage in stages()])
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args(argv);os.chdir(ROOT)
    if (args.resume or args.acknowledge_failure or args.checkpoint) and not args.execute:p.error('Resume options require --execute')
    if (args.acknowledge_failure or args.checkpoint) and not args.resume:p.error('Recovery/checkpoint options require explicit --resume')
    if args.prepare:
        # Reuse only its pinned 13-file model download, never its SMS data builder.
        subprocess.run([sys.executable,'scripts/prepare_cloud_sms.py','--download-model'],cwd=ROOT,check=True)
        fetch(ROOT/'data/raw/banking77_v1')
        if not DATA.exists():build(ROOT/'data/raw/banking77_v1',DATA,ROOT/'models/Qwen3-4B')
        print(json.dumps(assets()))
    elif args.check:print(json.dumps(assets()))
    elif args.gpu_check:print(json.dumps(assets(check_gpu=True)))
    elif args.pause:
        status=json.loads((OUT/'execution_state.json').read_text())
        if status.get('stage')!='running':raise ValueError('No running pipeline recorded; verify actual host processes')
        with (OUT/'pause.pipeline.request').open('x') as stream:stream.write(str(time.time()))
        name=status['active_stage']
        if name.startswith('train_'):
            size=int(name.split('_')[1])
            subprocess.run([sys.executable,'-m','personal_llm_data.workflow','train-personal','--config',f'configs/intenttune_v1/qwen3_4b_n{size}.json','--request-pause'],cwd=ROOT,check=True)
        print('Pause requested; wait for verified checkpoint/between-stage stop and actual process exit')
    elif args.execute:run(args.resume,args.acknowledge_failure,args.checkpoint)
    elif args.report:print(json.dumps(summarize()))
    else:print(json.dumps(dict(executable=False,task='IntentTune BANKING77 card10',stages=stages(),
                              next='--prepare; --check; --execute on Linux/RTX3090; no old experiments in this queue'),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
