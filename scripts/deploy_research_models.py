"""Serial asset deployment for the reviewed research roster. Never starts a model."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
from personal_llm_data.personal_runtime import gpu_guard,write_json

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'configs/research_models.json'
OUT=ROOT/'experiments/cloud_8b_4090'

def catalog():return json.loads(REGISTRY.read_text(encoding='utf-8'))
def backend():
    spec=importlib.util.spec_from_file_location('research_asset_backend',ROOT/'scripts/prepare_8b_4090.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.SPEC=REGISTRY;m.MODELSCOPE_CANDIDATES={k:v['modelscope'] for k,v in catalog()['models'].items()}
    return m

def pin_once(key,m):
    lock=OUT/'assets'/f'{key}.json'
    if lock.exists():
        frozen=json.loads(lock.read_text(encoding='utf-8'))
        if frozen['repo_id']!=catalog()['models'][key]['repo_id']:raise ValueError('Prior model lock identity differs')
        print(json.dumps(dict(model=key,status='existing_official_revision_retained',revision=frozen['revision'])),flush=True)
    else:m.pin(key)

def selected(keys):
    names=list(catalog()['models']) if keys is None else keys
    if not names or len(set(names))!=len(names) or any(k not in catalog()['models'] for k in names):raise ValueError('Unique registered model keys required')
    return names

def deploy(keys,verify_only=False):
    m=backend()
    # One global transfer/GPU guard: do not mutate model assets during compute.
    with gpu_guard(ROOT):
        for key in keys:
            write_json(OUT/'deployment_state.json',dict(status='running_asset_verification' if verify_only else 'deploying_assets',active_model=key,training_started=False))
            try:
                if not verify_only:pin_once(key,m)
                m.assets(key,not verify_only,'modelscope')
            except Exception:
                write_json(OUT/'deployment_state.json',dict(status='failed_preserved_no_auto_retry',active_model=key,training_started=False))
                raise
        report=dict(status='selected_model_assets_verified_not_training_ready',models=keys,all_roster=keys==list(catalog()['models']),training_ready=False,registry_sha256=m.digest(REGISTRY),
            locks_sha256={key:m.digest(OUT/'assets'/f'{key}.json') for key in keys},
            verified_receipts_sha256={key:m.digest(OUT/'assets'/f'{key}.verified.json') for key in keys})
        write_json(OUT/'deployment_state.json',report)
        print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',default='plan',choices=['plan','pin-all','deploy','verify']);p.add_argument('--models',nargs='+');a=p.parse_args();keys=selected(a.models)
    if a.action=='plan':
        print(json.dumps(dict(executable=False,models={k:catalog()['models'][k] for k in keys},free_disk_bytes=shutil.disk_usage(ROOT).free,source='ModelScope weights verified against frozen official hashes; official small files',training_ready=False),indent=2));return
    if platform.system()!='Linux':raise RuntimeError('Asset deployment only on owner Linux cloud')
    if a.action=='pin-all':
        with gpu_guard(ROOT):
            m=backend()
            for key in keys:pin_once(key,m)
    else:deploy(keys,a.action=='verify')

if __name__=='__main__':main()
