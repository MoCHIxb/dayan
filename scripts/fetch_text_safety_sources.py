"""Opt-in fixed-version source download for the three-domain example."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from build_text_safety_sets import SOURCES,BT_REV,DIA_REV,ROOT

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/raw/text_safety_sources')
    p.add_argument('--download',action='store_true')
    a=p.parse_args(argv)
    if not a.download:
        print(json.dumps(dict(executable=False,sources=SOURCES,next='--download explicitly fetches public corpus files; no model/GPU'),indent=2));return 0
    for key,(relative,digest) in SOURCES.items():
        destination=a.output_dir/relative
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('Existing source differs; refusing overwrite')
            continue
        if key.startswith('beavertails'):
            from huggingface_hub import hf_hub_download
            hf_hub_download('PKU-Alignment/BeaverTails',relative.removeprefix('beavertails/'),repo_type='dataset',revision=BT_REV,local_dir=a.output_dir/'beavertails')
        else:
            url=f'https://raw.githubusercontent.com/thu-coai/DiaSafety/{DIA_REV}/{relative.removeprefix("diasafety/")}'
            req=urllib.request.Request(url,headers={'User-Agent':'dayan-public-research-source-fetch'})
            with urllib.request.urlopen(req,timeout=60) as response:payload=response.read()
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Source download checksum mismatch')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:stream.write(payload)
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('Source checksum mismatch')
    print('All five fixed corpus files verified; no model or GPU started')
    return 0

if __name__=='__main__':raise SystemExit(main())
