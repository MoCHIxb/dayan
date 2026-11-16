"""Full400 native research generation and classification, explicit GPU opt-in."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from personal_llm_data.personal_data import sha,read_rows
from personal_llm_data.personal_runtime import gpu_guard
from personal_llm_data.research_sft import encode_row
from personal_llm_data.banking77 import score_intents
from train_research_grid import ROOT,validate_config,inputs_snapshot,verify_checkpoint

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)

def evaluate(config_path,condition,output):
    config=load(config_path);rows=validate_config(config);inputs=inputs_snapshot(config,config_path)
    prediction=output.with_suffix('.predictions.jsonl');partial=output.with_suffix('.partial.jsonl');receipt=output.with_suffix('.receipt.json')
    if any(p.exists() for p in (output,prediction,partial,receipt)):raise FileExistsError('Preserve previous generation evidence')
    source=ROOT/'data/processed/banking77_8b_v1/fewshot_test.jsonl' if condition=='fewshot' else ROOT/config['data_dir']/'test.jsonl'
    source_before=sha(source)
    records=read_rows(source)
    if len(records)!=400 or len({r['id'] for r in records})!=400:raise ValueError('Full400 required')
    if condition=='fewshot' and config['quantization']!='none':raise ValueError('Fixed baseline is BF16')
    adapter_hash=None
    run=ROOT/config['output_dir']
    if condition=='adapter':
        summary=load(run/'training_summary.json')
        if summary['status']!='completed_verified' or summary['inputs']!=inputs:raise ValueError('Incomplete/changed training')
        verify_checkpoint(Path(summary['best_checkpoint']),inputs);verify_checkpoint(run/'checkpoints/checkpoint-128',inputs)
        adapter_hash=sha(run/'adapter/adapter_model.safetensors')
        if adapter_hash!=summary['adapter_sha256']:raise ValueError('Export identity differs')
    output.parent.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from personal_llm_data.model_loading import load_causal_model
    with gpu_guard(ROOT):
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 4090':raise RuntimeError('Owner single4090 required')
        tokenizer=AutoTokenizer.from_pretrained(str(ROOT/config['model_name_or_path']),local_files_only=True,trust_remote_code=False)
        encoded=[encode_row(tokenizer,row,config['model_key'],1536,64) for row in records]
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        model=load_causal_model(str(ROOT/config['model_name_or_path']),config['quantization'])
        if config['quantization']=='none':model.to('cuda')
        if condition=='adapter':model=PeftModel.from_pretrained(model,str(run/'adapter'))
        model.eval();load_seconds=time.monotonic()-started;predictions=[];tokens=0;started=time.monotonic()
        eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else list(eos or [])
        if tokenizer.eos_token_id not in eos:eos.append(tokenizer.eos_token_id)
        with partial.open('x',encoding='utf-8') as f,torch.inference_mode():
            for i,(row,(_,prompt)) in enumerate(zip(records,encoded),1):
                ids=torch.tensor([prompt],device='cuda');mask=torch.ones_like(ids)
                generated=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=64,do_sample=False,eos_token_id=eos,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                result=generated[0,len(prompt):];tokens+=len(result)
                item=dict(id=row['id'],reference=row['messages'][-1]['content'],prediction=tokenizer.decode(result,skip_special_tokens=True).strip())
                predictions.append(item);f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
                if i%20==0:print(json.dumps(dict(condition=condition,completed=i,total=400)),flush=True)
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        score=score_intents(records,predictions)
        resources=dict(load_seconds=load_seconds,generation_seconds=elapsed,generated_tokens=tokens,tokens_per_second=tokens/elapsed,mean_seconds_per_record=elapsed/400,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if inputs_snapshot(config,config_path)!=inputs or sha(source)!=source_before:raise ValueError('Input identity changed during generation')
    partial.rename(prediction)
    save(output,dict(status='complete_full400_scored',condition=condition,quality=score,resources=resources))
    save(receipt,dict(condition=condition,config=str(config_path),inputs=inputs,source_sha256=sha(source),adapter_sha256=adapter_hash,predictions_sha256=sha(prediction),metrics_sha256=sha(output),evaluator_sha256=sha(Path(__file__)),command=[sys.executable,*sys.argv]))
    print(json.dumps(dict(condition=condition,samples=400,macro_f1=score['macro_f1'],invalid=score['invalid_outputs'])))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--condition',choices=['zero','fewshot','adapter'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();os.chdir(ROOT)
    if not a.execute:print(json.dumps(dict(executable=False,condition=a.condition,full_test=400)));return
    if platform.system()!='Linux':raise RuntimeError('Cloud only')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
    evaluate(a.config,a.condition,a.output)

if __name__=='__main__':main()
