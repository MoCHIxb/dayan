"""Download fixed model revisions; never starts training."""
import argparse
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--download',action='store_true'); args=p.parse_args()
    spec=json.loads((ROOT/'repro/local_assets.json').read_text(encoding='utf-8'))
    if not args.download:
        print('Use --download to fetch the fixed Qwen model and BERTScore cache.'); return
    from huggingface_hub import snapshot_download
    model=spec['model']
    snapshot_download(model['repo_id'],revision=model['revision'],local_dir=ROOT/'models/Qwen3-4B',
                      ignore_patterns=['*.gguf','*.onnx'])
    # Original BERTScore uses the public model identifier with its default ref.
    # Refuse if upstream main moved away from the verified local revision.
    bert=spec['bertscore_model']; path=Path(snapshot_download(bert['repo_id'],allow_patterns=list(bert['files_sha256'])))
    if path.name!=bert['revision']:
        raise ValueError('BERTScore main differs from local revision; stop and transfer the original cache instead of changing evaluation')
    print('Models downloaded; next run run_cloud_comparison.py --check-assets.')


if __name__=='__main__': main()
