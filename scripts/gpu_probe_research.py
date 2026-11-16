"""Opt-in GPU diagnostics on real training rows; never a formal score or training queue."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from personal_llm_data.personal_data import read_rows,sha,verify_prepared
from personal_llm_data.personal_runtime import gpu_guard,write_json
from personal_llm_data.research_chat import message_view,POLICIES
from deploy_research_models import ROOT,catalog,backend

BASE=ROOT/'experiments/cloud_8b_4090/gpu_probe_v1'
CPU=ROOT/'experiments/cloud_8b_4090/compatibility_preflight_chat_v3.json'
DATA=ROOT/'data/processed/banking77_8b_v1/n500/train.jsonl'
MODES=('none','nf4','int8')

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save_new(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)

def supervised_ids(tokenizer,row,key):
    messages=message_view(row['messages'],key)
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
    if full[:len(prompt)]!=prompt or len(full)<=len(prompt):raise ValueError('Prompt/target boundary differs')
    if tokenizer.decode(full[len(prompt):],skip_special_tokens=True).strip()!=row['messages'][-1]['content'].strip():raise ValueError('Original target changed')
    if len(full)>1536 or len(full)-len(prompt)>64:raise ValueError('No truncation allowed')
    return full,[-100]*len(prompt)+full[len(prompt):],prompt

def gate():
    verify_prepared(DATA.parent)
    r=load(CPU)
    if r['status']!='cpu_contracts_passed_gpu_unverified' or r['registry_sha256']!=sha(ROOT/'configs/research_models.json') or r['preflight_code_sha256']!=sha(ROOT/'scripts/preflight_research_models.py'):raise ValueError('CPU preflight missing/stale')
    if r['data_manifest_sha256']!=sha(DATA.parent/'manifest.json'):raise ValueError('Data manifest changed')
    if set(r['models'])!=set(catalog()['models']):raise ValueError('CPU roster incomplete')
    for key,model in r['models'].items():
        if model['issues'] or model['chat_policy']!=POLICIES[key] or model['chat_adapter_sha256']!=sha(ROOT/'src/personal_llm_data/research_chat.py'):raise ValueError('Chat policy incomplete/stale')
        for split,count in (('train',500),('dev',100),('test',400),('fewshot_test',400)):
            if model['splits'][split]['checked']!=count:raise ValueError('Full CPU input coverage missing')
        for package,recorded in model.get('tokenizer_dependency_versions',{}).items():
            if version(package)!=recorded:raise ValueError('Tokenizer dependency differs from CPU preflight')
        for field,suffix in (('asset_lock_sha256','.json'),('asset_receipt_sha256','.verified.json')):
            if model[field]!=sha(ROOT/'experiments/cloud_8b_4090/assets'/(key+suffix)):raise ValueError('Model lock changed')
    paths=[CPU,DATA,DATA.parent/'manifest.json',ROOT/'configs/research_models.json',ROOT/'scripts/gpu_probe_research.py',ROOT/'scripts/prepare_8b_4090.py',ROOT/'src/personal_llm_data/research_chat.py',ROOT/'src/personal_llm_data/model_loading.py',ROOT/'src/personal_llm_data/personal_runtime.py']
    return dict(files_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},packages={p:version(p) for p in ('torch','transformers','peft','accelerate','bitsandbytes','sentencepiece','protobuf')})

@contextmanager
def batch_lock():
    import fcntl
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'probe.lock').open('a+b') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def worker(key,mode,phase):
    directory=BASE/(key+'_'+mode);frozen=load(directory/'inputs.json')
    if gate()!=frozen:raise ValueError('Frozen probe inputs differ')
    backend().assets(key)
    # GPU imports only in the worker, released when this process exits.
    import torch
    import numpy as np
    import random
    from transformers import AutoTokenizer,set_seed
    from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training
    from personal_llm_data.model_loading import load_causal_model
    with gpu_guard(ROOT):
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 4090' or not torch.cuda.is_bf16_supported():raise RuntimeError('Expected single BF16 RTX4090')
        torch.cuda.reset_peak_memory_stats();set_seed(20261116)
        spec=catalog()['models'][key];model_path=ROOT/spec['directory']
        tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False)
        rows=read_rows(DATA)[:2]
        if len(rows)!=2:raise ValueError('Real training rows required')
        encodings=[supervised_ids(tokenizer,row,key) for row in rows]
        base=load_causal_model(str(model_path),mode);base.config.use_cache=False
        if mode=='none':
            base.to('cuda');base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        adapter=directory/'adapter'
        if phase=='train':
            names=[n for n,_ in base.named_modules()]
            if not all(any(n.rsplit('.',1)[-1]==leaf for n in names) for leaf in spec['lora_projections']):raise ValueError('Missing projection')
            model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0.05,bias='none',target_modules=spec['lora_projections']))
        else:
            receipt=load(directory/'train_receipt.json')
            for name,h in receipt['files_sha256'].items():
                if sha(directory/name)!=h:raise ValueError('Saved probe state changed')
            model=PeftModel.from_pretrained(base,str(adapter),is_trainable=True)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0)
        if phase=='restore':
            optimizer.load_state_dict(torch.load(directory/'optimizer.pt',map_location='cuda',weights_only=True));scheduler.load_state_dict(torch.load(directory/'scheduler.pt',weights_only=True))
            if len(optimizer.state)==0 or any(int(state['step'].item())!=1 for state in optimizer.state.values()):raise ValueError('Optimizer step1 state missing')
            rng=torch.load(directory/'rng.pt',weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
            state=load(directory/'python_numpy_rng.json')
            def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,list) else v
            random.setstate(tuples(state['python']))
            ns=state['numpy'];np.random.set_state((ns[0],np.array(ns[1],dtype='uint32'),ns[2],ns[3],ns[4]))
        model.train();optimizer.zero_grad();losses=[]
        for ids,labels,_ in encodings:
            batch=dict(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device='cuda'),labels=torch.tensor([labels],device='cuda'))
            loss=model(**batch).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite real-data loss')
            losses.append(float(loss.detach()));(loss/len(encodings)).backward()
        if not any(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in params):raise ValueError('No adapter gradient')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise ValueError('Nonfinite gradient')
        optimizer.step();scheduler.step();optimizer.zero_grad();torch.cuda.synchronize()
        if phase=='train':
            model.save_pretrained(adapter,safe_serialization=True)
            torch.save(optimizer.state_dict(),directory/'optimizer.pt');torch.save(scheduler.state_dict(),directory/'scheduler.pt')
            torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),directory/'rng.pt')
            ns=np.random.get_state();save_new(directory/'python_numpy_rng.json',dict(python=random.getstate(),numpy=[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]]))
            save_new(directory/'trainer_state.json',dict(global_step=1,diagnostic_only=True,source_ids=[r['id'] for r in rows]))
            files=list(adapter.glob('*'))+[directory/n for n in ('optimizer.pt','scheduler.pt','rng.pt','python_numpy_rng.json','trainer_state.json')]
            save_new(directory/'train_receipt.json',dict(status='probe_step1_saved',losses=losses,files_sha256={str(p.relative_to(directory)):sha(p) for p in files if p.is_file()},trainable_parameters=sum(p.numel() for p in params),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        else:
            if any(int(s['step'].item())!=2 for s in optimizer.state.values()):raise ValueError('Restored optimizer did not advance to step2')
            model.eval();prompt=encodings[0][2];generation=model.generate(input_ids=torch.tensor([prompt],device='cuda'),attention_mask=torch.ones((1,len(prompt)),dtype=torch.long,device='cuda'),max_new_tokens=64,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            prediction=tokenizer.decode(generation[0,len(prompt):],skip_special_tokens=True)
            save_new(directory/'generation_probe.json',dict(source_id=rows[0]['id'],prediction=prediction,diagnostic_only=True,not_test_score=True))
            save_new(directory/'restore_receipt.json',dict(status='probe_restored_step2_and_generation_completed',losses=losses,generation_sha256=sha(directory/'generation_probe.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),not_uninterrupted_equivalence_test=True))
        if gate()!=frozen:raise ValueError('Probe inputs changed during GPU work')

def batch(keys,modes):
    with batch_lock():
        current=gate()
        subprocess.run([sys.executable,'scripts/prepare_8b_4090.py','check'],check=True)
        for key in keys:
            for mode in modes:
                name=key+'_'+mode;directory=BASE/name
                if directory.exists():raise FileExistsError('Existing diagnostic evidence retained; no automatic retry: '+name)
                directory.mkdir();save_new(directory/'inputs.json',current)
                for phase in ('train','restore'):
                    write_json(BASE/'execution_state.json',dict(status='running_gpu_probe',model=key,precision=mode,phase=phase))
                    try:
                        with (directory/(phase+'.stdout.log')).open('xb') as out,(directory/(phase+'.stderr.log')).open('xb') as err:
                            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--model',key,'--mode',mode,'--phase',phase],stdout=out,stderr=err,check=True)
                    except Exception:
                        write_json(BASE/'execution_state.json',dict(status='failed_preserved_no_auto_retry',model=key,precision=mode,phase=phase));raise
                save_new(directory/'complete.json',dict(status='diagnostic_pair_completed',inputs=current,train_receipt_sha256=sha(directory/'train_receipt.json'),restore_receipt_sha256=sha(directory/'restore_receipt.json')))
        write_json(BASE/'execution_state.json',dict(status='selected_gpu_probes_complete_not_formal_experiments',models=keys,modes=modes,all_conditions=(len(keys)==len(catalog()['models']) and len(modes)==len(MODES)),training_ready=False))
        print(json.dumps(load(BASE/'execution_state.json')))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--model',choices=list(catalog()['models']));p.add_argument('--mode',choices=MODES);p.add_argument('--phase',choices=['train','restore']);a=p.parse_args();os.chdir(ROOT)
    if not a.execute and not a.worker:
        print(json.dumps(dict(executable=False,conditions=[k+'_'+mode for k in catalog()['models'] for mode in MODES],source='two existing train records, diagnostic1step then fresh-process restore+step2',not_formal_training=True)));return
    if platform.system()!='Linux':raise RuntimeError('GPU probes only on owner Linux cloud')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
    if a.worker:
        if not all((a.model,a.mode,a.phase)):p.error('Internal worker arguments incomplete')
        worker(a.model,a.mode,a.phase)
    else:batch([a.model] if a.model else list(catalog()['models']),[a.mode] if a.mode else list(MODES))

if __name__=='__main__':main()
