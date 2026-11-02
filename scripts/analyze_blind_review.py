"""Validate two independent completed rating sheets before any blind conclusions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from personal_llm_data.pipeline import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("content_preservation", "style_similarity", "fluency", "instruction_following")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater1", required=True)
    parser.add_argument("--rater2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pack = ROOT / "experiments/blind_review_200"
    mapping = read_jsonl(pack/"private_do_not_share/system_mapping.jsonl")
    expected = {(r["item"],r["label"]) for r in mapping}
    sheets = []
    for path in (args.rater1,args.rater2):
        with Path(path).open(encoding="utf-8-sig",newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows)!=800 or {(r['item'],r['label']) for r in rows}!=expected:
            raise ValueError("Rating sheet must contain exactly the frozen 800 output IDs")
        sheet = {}
        for row in rows:
            values = []
            for dimension in DIMENSIONS:
                value = row[dimension].strip()
                if value not in {"1","2","3","4","5"}:
                    raise ValueError(f"Missing/invalid independent rating: {path}, {row['item']}, {row['label']}, {dimension}")
                values.append(int(value))
            sheet[(row['item'],row['label'])] = np.asarray(values,dtype=float)
        sheets.append(sheet)
    records = read_jsonl(ROOT/"data/processed/lamp7_bm25_k12/test.jsonl")
    by_id = {str(r['id']):r for r in records}
    lengths=sorted(len(r['messages'][-1]['content'].split()) for r in records)
    edges=[lengths[round((len(lengths)-1)*q)] for q in (.25,.5,.75)]
    def stratum(record):
        n=len(record['messages'][-1]['content'].split())
        return (sum(n>edge for edge in edges),sum(record['profile_size_original']>edge for edge in (10,20,50)))
    population={}
    for row in records:
        key=stratum(row)
        population[key]=population.get(key,0)+1
    systems=sorted({r['system'] for r in mapping})
    items=sorted({r['item'] for r in mapping})
    scores={system:np.zeros((200,4)) for system in systems}
    item_index={item:i for i,item in enumerate(items)}
    item_strata={}
    for row in mapping:
        key=(row['item'],row['label'])
        scores[row['system']][item_index[row['item']]]=(sheets[0][key]+sheets[1][key])/2
        item_strata[row['item']]=stratum(by_id[str(row['source_id'])])
    groups={key:np.asarray([item_index[item] for item in items if item_strata[item]==key]) for key in population}
    def average(values, sampled=None):
        total=np.zeros(4)
        for key,indices in groups.items():
            chosen=indices if sampled is None else sampled[key]
            total+=population[key]/1496*values[chosen].mean(axis=0)
        return total
    contrasts={}
    rng=np.random.default_rng(20251206)
    for candidate,reference in (("bf16_profile","base_profile"),("int8_profile","bf16_profile"),("nf4_profile","bf16_profile")):
        difference=scores[candidate]-scores[reference]
        draws=np.asarray([average(difference,{key:rng.choice(indices,len(indices),replace=True) for key,indices in groups.items()}) for _ in range(2000)])
        contrasts[f"{candidate}-{reference}"]={dimension:{"difference":float(average(difference)[i]),
            "ci95_low":float(np.percentile(draws[:,i],2.5)),"ci95_high":float(np.percentile(draws[:,i],97.5))} for i,dimension in enumerate(DIMENSIONS)}
    ordered=sorted(expected)
    result={"items":200,"raters":2,"ratings_per_rater":800,"bootstrap":2000,
            "primary_contrast":"bf16_profile-base_profile","primary_dimension":"style_similarity",
            "interval_scope":"stratified item bootstrap conditional on these two raters; unadjusted secondary comparisons",
            "stratum_weighted_mean_scores":{system:dict(zip(DIMENSIONS,map(float,average(values)))) for system,values in scores.items()},
            "quadratic_weighted_kappa":{dimension:float(cohen_kappa_score([sheets[0][key][i] for key in ordered],[sheets[1][key][i] for key in ordered],weights="quadratic",labels=[1,2,3,4,5])) for i,dimension in enumerate(DIMENSIONS)},
            "contrasts":contrasts}
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Two completed independent sheets validated and analyzed",flush=True)


if __name__=="__main__":
    main()
