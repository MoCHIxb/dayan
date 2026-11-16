"""CPU-only real-data/template/module preflight; never loads model weights or trains."""
import argparse
import json
import platform
import os
import shutil
from pathlib import Path
from personal_llm_data.personal_data import read_rows,sha
from deploy_research_models import ROOT,OUT,catalog,backend
from personal_llm_data.research_chat import message_view,POLICIES


def encode_record(tokenizer,row,max_length=1536,max_new=64,model_key=None):
    messages=message_view(row['messages'],model_key)
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    complete=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
    if complete[:len(prompt)]!=prompt or len(complete)<=len(prompt):raise ValueError('Native assistant prefix/target boundary mismatch')
    target=complete[len(prompt):]
    decoded=tokenizer.decode(target,skip_special_tokens=True).strip()
    if decoded!=row['messages'][-1]['content'].strip():raise ValueError('Rendered supervision differs from original label')
    if len(complete)>max_length or len(target)>max_new:raise ValueError('Native token budget exceeded; no truncation')
    return dict(prompt_tokens=len(prompt),target_tokens=len(target),sequence_tokens=len(complete),last_token_id=complete[-1])


def save_evidence(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

def failure_evidence(output,key,reports,error):
    path=output.with_name(output.stem+'.failure.json')
    save_evidence(path,dict(status='cpu_preflight_failed_gpu_blocked',active_model=key,completed_models=reports,error_type=type(error).__name__,error=str(error),training_ready=False,model_weights_loaded=False))
    return path

def tokenizer_dependencies(key):
    if key!='mistral7b':return {}
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for package in ('sentencepiece','protobuf'):
        try:versions[package]=version(package)
        except PackageNotFoundError as error:raise RuntimeError('Missing tokenizer dependency '+package+'; install requirements-tokenizers.lock in .venv-cloud8b without changing model files') from error
    return versions


def check(output):
    if platform.system()!='Linux':raise RuntimeError('Run real model/tokenizer preflight on designated Linux cloud only')
    if output.exists() or output.with_name(output.stem+'.failure.json').exists():raise FileExistsError('Keep previous preflight; use new --output after diagnosing')
    from run_qwen8b_banking500 import DATA,data_check
    data_check()
    records={split:read_rows(DATA/'n500'/f'{split}.jsonl') for split in ('train','dev','test')}
    records['fewshot_test']=read_rows(DATA/'fewshot_test.jsonl')
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    from accelerate import init_empty_weights
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight must not create a CUDA context')
    reports={};m=backend()
    for key,entry in catalog()['models'].items():
        try:
            print(json.dumps(dict(stage='cpu_preflight',model=key)),flush=True)
            m.assets(key)
            dependency_versions=tokenizer_dependencies(key)
            directory=ROOT/entry['directory'];tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
            if tokenizer.eos_token_id is None:raise ValueError('EOS token undefined')
            stats={};issues=[]
            for split,rows in records.items():
                sizes=[]
                for row in rows:
                    try:sizes.append(encode_record(tokenizer,row,model_key=key))
                    except Exception as error:issues.append(dict(split=split,id=row['id'],error=str(error)))
                stats[split]=dict(expected=len(rows),checked=len(sizes),max_sequence=max((v['sequence_tokens'] for v in sizes),default=0),max_target=max((v['target_tokens'] for v in sizes),default=0))
            cfg=AutoConfig.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
            if cfg.model_type!=entry['model_type']:raise ValueError('Architecture differs from roster')
            with init_empty_weights():model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False)
            names=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)]
            counts={leaf:sum(name.rsplit('.',1)[-1]==leaf for name in names) for leaf in entry['lora_projections']}
            meta_only=all(p.is_meta for p in model.parameters());parameters=sum(p.numel() for p in model.parameters());del model
            if not meta_only or not all(counts.values()):raise ValueError('Missing LoRA modules or non-meta model allocation')
            reports[key]=dict(status=('cpu_contract_passed_with_explicit_adapter' if key=='mistral7b' else 'native_cpu_contract_passed') if not issues else 'requires_explicit_adapter_revision',chat_policy=POLICIES[key],chat_adapter_sha256=sha(ROOT/'src/personal_llm_data/research_chat.py'),splits=stats,issues=issues,projection_module_counts=counts,total_parameters=parameters,meta_only=True,tokenizer_config_sha256=sha(directory/'tokenizer_config.json'),model_config_sha256=sha(directory/'config.json'),asset_lock_sha256=sha(OUT/'assets'/f'{key}.json'),asset_receipt_sha256=sha(OUT/'assets'/f'{key}.verified.json'),eos_token_id=tokenizer.eos_token_id,model_weights_loaded=False,tokenizer_dependency_versions=dependency_versions)
            save_evidence(output.with_name(output.stem+'.'+key+'.json'),reports[key])
        except Exception as error:
            failure_evidence(output,key,reports,error)
            raise
    passed=all(v['status'] in ('native_cpu_contract_passed','cpu_contract_passed_with_explicit_adapter') for v in reports.values())
    result=dict(status='cpu_contracts_passed_gpu_unverified' if passed else 'cpu_contract_issues_gpu_blocked',models=reports,registry_sha256=sha(ROOT/'configs/research_models.json'),preflight_code_sha256=sha(Path(__file__)),data_manifest_sha256=sha(DATA/'n500/manifest.json'),free_disk_bytes=shutil.disk_usage(ROOT).free,training_ready=False,cuda_context_created=torch.cuda.is_initialized(),source='all actual BANKING500/dev100/test400/fixed10shot400; no synthetic train samples')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],output=str(output),training_ready=False)))
    return 0 if passed else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT/'compatibility_preflight.json');a=p.parse_args()
    os.chdir(ROOT)
    return check(a.output)

if __name__=='__main__':raise SystemExit(main())
