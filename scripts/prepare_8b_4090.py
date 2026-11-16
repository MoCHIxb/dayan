"""4090 cloud preparation only. No model loading, training, or automatic retries."""
import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'configs/cloud_8b_4090/environment.json'

def read_spec():
    return json.loads(SPEC.read_text(encoding='utf-8'))

def digest(path, algorithm='sha256', git=False):
    path=Path(path); h=hashlib.new(algorithm)
    if git:h.update(('blob '+str(path.stat().st_size)+'\0').encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def filename(name):
    if not isinstance(name,str) or not name or '/' in name or '\\' in name or name in ('.','..'):
        raise ValueError('Top-level model filename required')
    return name

def save_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)

def verify_file(path,entry):
    if path.is_symlink() or not path.is_file():raise ValueError('Missing/nonregular asset: '+path.name)
    if path.stat().st_size!=entry['size']:raise ValueError('Asset size mismatch: '+path.name)
    if entry.get('sha256'):
        actual=digest(path); expected=entry['sha256']
    else:
        actual=digest(path,'sha1',True);expected=entry['blob_id']
    if actual!=expected:raise ValueError('Asset hash mismatch; preserve file: '+path.name)
    return digest(path)

def pin(key):
    from huggingface_hub import HfApi
    model=read_spec()['models'][key];path=ROOT/'experiments/cloud_8b_4090/assets'/f'{key}.json'
    if path.exists():raise FileExistsError('Existing model revision retained; use verify/download')
    info=HfApi(endpoint='https://huggingface.co').model_info(model['repo_id'],files_metadata=True)
    if not re.fullmatch('[0-9a-f]{40}',info.sha):raise ValueError('Full commit required')
    files={}
    allowed={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','merges.txt','vocab.json','tokenizer.model','LICENSE','LICENSE.txt','USE_POLICY.md','README.md','model.safetensors.index.json'}
    for item in info.siblings:
        name=item.rfilename
        if name not in allowed and not (name.endswith('.safetensors') and '/' not in name):continue
        filename(name);lfs=item.lfs
        checksum=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
        entry=dict(size=item.size,sha256=checksum,blob_id=item.blob_id)
        if not isinstance(item.size,int) or not re.fullmatch('[0-9a-f]{64}' if checksum else '[0-9a-f]{40}',checksum or item.blob_id or ''):
            raise ValueError('Missing remote checksum metadata')
        files[name]=entry
    if not {'config.json','tokenizer_config.json','tokenizer.json'}<=set(files) or not any(n.endswith('.safetensors') for n in files):raise ValueError('Incomplete model inventory')
    save_new(path,dict(repo_id=model['repo_id'],revision=info.sha,files=files))
    print(json.dumps(dict(model=key,revision=info.sha,files=len(files),bytes=sum(x['size'] for x in files.values()))))

def assets(key,download=False):
    model=read_spec()['models'][key];lock=ROOT/'experiments/cloud_8b_4090/assets'/f'{key}.json'
    value=json.loads(lock.read_text(encoding='utf-8'))
    if value['repo_id']!=model['repo_id'] or not re.fullmatch('[0-9a-f]{40}',value['revision']):raise ValueError('Asset lock identity mismatch')
    directory=ROOT/model['directory'];hashes={}
    for name,entry in value['files'].items():
        path=directory/filename(name)
        if not path.exists() and download:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=value['repo_id'],filename=name,revision=value['revision'],local_dir=directory,endpoint='https://huggingface.co')
        hashes[name]=verify_file(path,entry)
    cfg=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if cfg['model_type']!=model['model_type']:raise ValueError('Model architecture mismatch')
    index=directory/'model.safetensors.index.json'
    if index.exists() and not set(json.loads(index.read_text())['weight_map'].values())<=set(hashes):raise ValueError('Missing weight shard')
    receipt=lock.with_name(key+'.verified.json');result=dict(repo_id=value['repo_id'],revision=value['revision'],files_sha256=hashes,lock_sha256=digest(lock))
    if receipt.exists():
        if json.loads(receipt.read_text())!=result:raise ValueError('Prior asset receipt differs')
    else:save_new(receipt,result)
    print(json.dumps(dict(model=key,status='asset_bytes_verified_not_training_ready',files=len(hashes))))

def check_environment():
    if platform.system()!='Linux' or platform.python_version_tuple()[:2]!=('3','12'):raise RuntimeError('Expected Linux Python3.12')
    import torch
    import bitsandbytes
    spec=read_spec()['environment']
    if not torch.__version__.startswith(spec['torch_prefix']) or torch.version.cuda!=spec['cuda']:raise RuntimeError('Image torch/CUDA differs; do not silently upgrade/downgrade')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise RuntimeError('Expected one BF16 CUDA GPU')
    from personal_llm_data.personal_runtime import gpu_guard
    with gpu_guard(ROOT):
        device=torch.cuda.get_device_properties(0)
        if device.name!=spec['gpu'] or device.total_memory<spec['minimum_memory_gib']*2**30:raise RuntimeError('Expected24GB RTX4090')
        result=dict(status='environment_inventory_only_not_model_fit',torch=torch.__version__,cuda=torch.version.cuda,gpu=device.name,memory_bytes=device.total_memory,bitsandbytes=bitsandbytes.__version__,training_ready=False)
    print(json.dumps(result))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','check','pin','download','verify']);p.add_argument('--model',choices=list(read_spec()['models']));a=p.parse_args()
    if a.action=='plan':print(json.dumps(read_spec(),indent=2));return
    if platform.system()!='Linux':raise RuntimeError('New assets/preflight run on designated Linux cloud only')
    if a.action=='check':check_environment();return
    if not a.model:p.error('--model required')
    if a.action=='pin':pin(a.model)
    elif a.action=='download':
        from personal_llm_data.personal_runtime import gpu_guard
        with gpu_guard(ROOT):assets(a.model,True)
    else:assets(a.model)

if __name__=='__main__':main()
