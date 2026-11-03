"""Read-only frozen SMS/model/code checks; explicit Linux4090 preflight optional."""
import argparse,importlib.metadata,json,os,platform,re,subprocess
from pathlib import Path
from personal_llm_data.personal_data import sha,verify_prepared
from personal_llm_data.personal_runtime import gpu_guard
ROOT=Path(__file__).resolve().parents[1]


def validate_cloud_config(local,cloud):
    if set(local)!=set(cloud) or any(local[k]!=cloud[k] for k in local if k!='output_dir'):
        raise ValueError('Cloud scientific settings must exactly match local SMS config')
    output=cloud.get('output_dir','').replace('\\','/')
    if not output.startswith('experiments/cloud_sms_spam_v1/') or '..' in Path(output).parts:
        raise ValueError('Separate cloud_sms_spam_v1 output required')


def check(config_path,strict_environment=False,check_gpu=False):
    reference=json.loads((ROOT/'results/sms_spam_run_audit.json').read_text(encoding='utf-8'))
    local=json.loads((ROOT/'configs/sms_spam_v1/qwen3_4b.json').read_text(encoding='utf-8'))
    cloud=json.loads(Path(config_path).read_text(encoding='utf-8'));validate_cloud_config(local,cloud)
    if (ROOT/cloud['output_dir']).exists():raise FileExistsError('Cloud run exists; inspect/audit without starting another run')
    data=ROOT/cloud['data_dir'];model=ROOT/cloud['model_name_or_path']
    for name,digest in reference['inputs']['data_sha256'].items():
        if sha(data/name)!=digest:raise ValueError('Frozen SMS data/manifest hash mismatch: '+name)
    if {p.name for p in model.iterdir() if p.is_file()}!=set(reference['inputs']['model_sha256']):raise ValueError('Unexpected model file inventory')
    for name,digest in reference['inputs']['model_sha256'].items():
        if sha(model/name)!=digest:raise ValueError('Frozen model mismatch: '+name)
    for name,digest in reference['inputs']['code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen core implementation mismatch: '+name)
    rows=verify_prepared(data)
    counts={k:len(v) for k,v in rows.items()}
    if counts!=reference['counts']:raise ValueError('Complete split coverage differs')
    packages={};mismatches={}
    for line in (ROOT/'requirements-train.lock').read_text().splitlines():
        if not line or line.startswith(('#','--')):continue
        name,version=line.split('==',1)
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        packages[name]=actual
        if actual!=version:mismatches[name]=dict(expected=version,actual=actual)
    if strict_environment and mismatches:raise ValueError('Locked package versions differ: '+json.dumps(mismatches))
    result=dict(status='assets_verified_gpu_unexecuted',training_started=False,counts=counts,
        cloud_config=str(config_path),cloud_config_sha256=sha(config_path),
        verified_data_sha256=reference['inputs']['data_sha256'],verified_model_sha256=reference['inputs']['model_sha256'],verified_core_code_sha256=reference['inputs']['code_sha256'],
        platform=dict(system=platform.system(),release=platform.release(),python=platform.python_version()),packages=packages,package_mismatches=mismatches,gpu_checked=False)
    if check_gpu:
        if platform.system()!='Linux':raise ValueError('Actual cloud GPU preflight requires Linux; local CPU checks are not cloud validation')
        with gpu_guard(ROOT):
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise ValueError('One CUDA/BF16 GPU required')
            name=torch.cuda.get_device_name(0)
            if not re.search(r'RTX\s+4090\b(?!\s*D)',name):raise ValueError('Expected RTX4090; record other hardware under a new platform plan')
            result.update(status='linux_rtx4090_preflight_passed_training_unexecuted',gpu_checked=True,
                gpu=dict(name=name,total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,torch=torch.__version__,cuda=torch.version.cuda,
                    nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/cloud_sms_spam_v1/qwen3_4b.json')
    p.add_argument('--strict-environment',action='store_true');p.add_argument('--check-gpu',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv);os.chdir(ROOT)
    if a.output and a.output.exists():raise FileExistsError('Keep previous preflight evidence; choose another output')
    result=check(a.config,a.strict_environment,a.check_gpu)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],counts=result['counts'],platform=result['platform'],gpu_checked=result['gpu_checked'],package_mismatches=result['package_mismatches'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
