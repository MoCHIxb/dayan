"""Paired per-item retention intervals; never substitute unpaired score changes."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",required=True)
    parser.add_argument("--adapter",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    report={"bootstrap":2000,"seed":20251206,"contrasts":{},
            "primary_metric":"ifeval_prompt_level_strict_acc",
            "scope":"paired test-item bootstrap, conditional on one training seed; secondary CIs unadjusted"}
    rng=np.random.default_rng(20251206)
    for task,filter_name,metric,count in (("arc_challenge","none","acc_norm",1172),
        ("gsm8k_cot_zeroshot","flexible-extract","exact_match",1319),
        ("ifeval","none","prompt_level_strict_acc",541)):
        systems=[]
        for path in (args.base,args.adapter):
            path=Path(path)
            with path.with_name(path.stem+f".{task}.samples.jsonl").open(encoding="utf-8") as stream:
                rows=[json.loads(line) for line in stream if line.strip()]
            index={str(row['doc_id']):row for row in rows if row['filter']==filter_name}
            if len(index)!=count:
                raise ValueError(f"Incomplete paired samples: {path} {task}")
            systems.append(index)
        ids=sorted(systems[0])
        if set(ids)!=set(systems[1]) or any(systems[0][i]['doc_hash']!=systems[1][i]['doc_hash'] for i in ids):
            raise ValueError("Benchmark documents do not align")
        base=np.asarray([float(systems[0][i][metric]) for i in ids])
        adapted=np.asarray([float(systems[1][i][metric]) for i in ids])
        difference=adapted-base
        draws=np.asarray([difference[rng.integers(0,count,count)].mean() for _ in range(2000)])
        report['contrasts'][task]={"metric":metric,"samples":count,"base":float(base.mean()),
            "adapter":float(adapted.mean()),"delta":float(difference.mean()),
            "ci95_low":float(np.percentile(draws,2.5)),"ci95_high":float(np.percentile(draws,97.5)),
            "base_correct_adapter_wrong":int(((base==1)&(adapted==0)).sum()),
            "base_wrong_adapter_correct":int(((base==0)&(adapted==1)).sum())}
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
