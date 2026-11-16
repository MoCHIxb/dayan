"""Deterministic full-factorial BANKING design, config generation and disk estimates."""
import itertools
import json
from pathlib import Path
import random
import struct
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'configs/research_banking_v2/grid.json'

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def spec():return load(SPEC)
def identity(key,mode,rank,alpha,seed):return f'{key}_{mode}_r{rank}_a{alpha}_s{seed}'
def config(key,mode,rank,alpha,seed):
    design=spec()
    if key not in design['models'] or mode not in design['quantizations'] or rank not in design['ranks'] or alpha not in design['alphas'] or seed not in design['training_seeds']:raise ValueError('Condition outside registered full grid')
    registry=load(ROOT/'configs/research_models.json')['models'][key]
    return dict(protocol_id=design['protocol_id'],model_key=key,recipe_id=f'{mode}_r{rank}_a{alpha}',model_name_or_path=registry['directory'],data_dir=design['data_dir'],output_dir=f"{design['output_root']}/{identity(key,mode,rank,alpha,seed)}/run",seed=seed,quantization=mode,lora_r=rank,lora_alpha=alpha,target_modules=registry['lora_projections'],**design['fixed'])
def configs(model=None,mode=None,rank=None):
    design=spec()
    for key in ([model] if model else design['models']):
        for precision in ([mode] if mode else design['quantizations']):
            for rank_value in ([rank] if rank is not None else design['ranks']):
                block=[config(key,precision,rank_value,alpha,seed) for alpha,seed in itertools.product(design['alphas'],design['training_seeds'])]
                random.Random(f"{design['schedule_seed']}:{key}:{precision}:{rank_value}").shuffle(block)
                yield from block

def config_path(condition):return Path(spec()['output_root'])/'configs'/(identity(condition['model_key'],condition['quantization'],condition['lora_r'],condition['lora_alpha'],condition['seed'])+'.json')
def materialize():
    for condition in configs():
        path=ROOT/config_path(condition);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if load(path)!=condition:raise ValueError('Existing grid config changed: '+str(path))
        else:
            with path.open('x',encoding='utf-8') as stream:json.dump(condition,stream,indent=2)

def projection_coefficient(model_dir,leaves):
    """Read safetensors matrix headers only; no Torch/CUDA or tensor allocation."""
    coefficient=0;counts={leaf:0 for leaf in leaves};files=sorted(Path(model_dir).glob('*.safetensors'))
    if not files:raise ValueError('Official local weights required for disk forecast')
    for path in files:
        with path.open('rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0]
            if length>32*1024**2:raise ValueError('Unexpected safetensors header')
            header=json.loads(stream.read(length))
        for name,value in header.items():
            parts=name.split('.')
            if len(parts)<2 or parts[-1]!='weight' or parts[-2] not in counts:continue
            shape=value['shape']
            if len(shape)!=2:raise ValueError('Expected matrix projection')
            counts[parts[-2]]+=1;coefficient+=sum(shape)
    if not all(counts.values()):raise ValueError('Some registered projections absent')
    return coefficient,counts

def storage_forecast(conditions,coefficients):
    parameters=[coefficients[condition['model_key']]*condition['lora_r'] for condition in conditions]
    retained=sum(28*count+64*1024**2 for count in parameters)
    temporary=24*max(parameters,default=0)+10*1024**3
    return dict(pending_trainings=len(parameters),estimated_retained_bytes=retained,temporary_and_reserve_bytes=temporary,required_free_bytes=retained+temporary,estimate_not_guarantee=True)
