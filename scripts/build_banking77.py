"""Explicit pinned-source BANKING77 subsets; CPU/tokenizer only, never trains."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

from personal_llm_data.banking77 import (BASE_URL, SOURCES, LABELS, SIZES, SEED, SYSTEM,
    MAX_LENGTH, MAX_NEW_TOKENS, source_record, catalog)
from personal_llm_data.personal_data import sha, validate_pairs, verify_prepared
from personal_llm_data.text_safety import text_hash, grams, similarity, verify_automated_audit

ROOT = Path(__file__).resolve().parents[1]
AUDIT = dict(method='source_prompt_character5_jaccard_v1', threshold=0.9)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def fetch(directory):
    for name, digest in SOURCES.items():
        destination = directory/name
        if destination.exists():
            if sha(destination) != digest:
                raise ValueError('Changed source; refusing overwrite: '+name)
            continue
        with urllib.request.urlopen(BASE_URL+'/'+name, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('Source download checksum mismatch: '+name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(payload)


def rows_from_source(directory, split):
    with (directory/f'banking_data/{split}.csv').open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def content(row):
    return json.loads(row['messages'][-2]['content'])['user_request']


def near(features, forbidden):
    return any(similarity(features, other) >= 0.9 for other in forbidden)


def build(source, output, tokenizer_path):
    if output.exists():
        raise FileExistsError('Keep existing dataset/evidence; choose a new output directory')
    for name, digest in SOURCES.items():
        if sha(source/name) != digest:
            raise ValueError('Frozen source differs: '+name)
    official = {split: rows_from_source(source, split) for split in ('train', 'test')}
    if len(official['train']) != 10003 or len(official['test']) != 3080:
        raise ValueError('Official source coverage differs')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    def check_budget(row):
        prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        answer = tokenizer(row['messages'][-1]['content'], add_special_tokens=False)['input_ids']
        return len(prompt)+len(answer)+1 <= MAX_LENGTH and len(answer)+1 <= MAX_NEW_TOKENS

    test = [source_record(row, 'test', i) for i, row in enumerate(official['test'], 1) if row['category'] in LABELS]
    if len(test) != 400 or Counter(r['intent'] for r in test) != Counter({key:40 for key in LABELS}):
        raise ValueError('All 40 official test records per intent must be retained')
    if any(not check_budget(row) for row in test):
        raise ValueError('Official test does not fit the frozen token budget; do not truncate or drop rows')
    test_groups = {text_hash(row['text']) for row in official['test']}
    test_features = [grams(content(row)) for row in test]
    grouped = defaultdict(list)
    exclusions = Counter()
    for i, raw in enumerate(official['train'], 1):
        if raw['category'] in LABELS:
            grouped[text_hash(raw['text'])].append(source_record(raw, 'train', i))
    eligible = defaultdict(list)
    for group, records in grouped.items():
        if group in test_groups:
            exclusions['official_test_exact_prompt_overlap'] += len(records)
            continue
        if len({row['intent'] for row in records}) != 1:
            exclusions['conflicting_training_prompt_labels'] += len(records)
            continue
        row = records[0]
        exclusions['duplicate_training_prompts'] += len(records)-1
        if near(grams(content(row)), test_features):
            exclusions['near_official_selected_test'] += 1
            continue
        if not check_budget(row):
            exclusions['token_budget'] += 1
            continue
        eligible[row['intent']].append(row)
    dev = []
    training = {}
    dev_features = []
    order = lambda row: hashlib.sha256(f'{SEED}/{row["id"]}'.encode()).hexdigest()
    for label in LABELS:
        ordered = sorted(eligible[label], key=order)
        if len(ordered) < 60:
            raise ValueError('Need at least 60 available train records for intent '+label)
        selected = ordered[:10]
        dev.extend(selected)
        dev_features.extend(grams(content(row)) for row in selected)
        training[label] = ordered[10:]
    for label in LABELS:
        filtered = [row for row in training[label] if not near(grams(content(row)), dev_features)]
        exclusions['near_development'] += len(training[label])-len(filtered)
        if len(filtered) < 50:
            raise ValueError('Insufficient disjoint records for intent '+label)
        training[label] = filtered[:50]
    # Interleaving gives exact nested 5/20/50-per-class prefix subsets.
    train = [training[label][i] for i in range(50) for label in LABELS]
    datasets = {size: dict(train=train[:size], dev=dev, test=test) for size in SIZES}
    for splits in datasets.values():
        validate_pairs([dict(row, split=split) for split, rows in splits.items() for row in rows])
        verify_automated_audit(splits, AUDIT)
    # Fixed ten-example prompt baseline, examples taken only from n50 training.
    demonstrations = '\n'.join(json.dumps(dict(customer_message=content(train[i]), label=train[i]['intent']),
                                          ensure_ascii=False) for i in range(10))
    fewshot = []
    for row in test:
        clone = json.loads(json.dumps(row))
        clone['messages'][0]['content'] = SYSTEM+'\nExamples from the training set:\n'+demonstrations
        if not check_budget(clone):
            raise ValueError('Few-shot baseline exceeds frozen token budget; no truncation allowed')
        fewshot.append(clone)
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for size, splits in datasets.items():
        directory = output/f'n{size}'
        directory.mkdir()
        files = {}
        for split, rows in splits.items():
            path = directory/f'{split}.jsonl'
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            files[path.name] = sha(path)
        manifest = dict(schema_version=1, purpose='personal_adapter_example_v1', task='banking77_card10_v1',
            counts={key:len(rows) for key, rows in splits.items()}, files_sha256=files,
            real_authorization_declared=True, manual_near_duplicate_review_declared=False,
            automated_near_duplicate_review=verify_automated_audit(splits, AUDIT),
            label_counts={split:dict(Counter(r['intent'] for r in rows)) for split, rows in splits.items()},
            independent_prompt_groups={split:len({r['source_group'] for r in rows}) for split, rows in splits.items()},
            source_sha256=SOURCES, tokenizer_config_sha256=sha(tokenizer_path/'tokenizer_config.json'),
            max_length=MAX_LENGTH, max_new_tokens=MAX_NEW_TOKENS, split_seed=SEED,
            limitations='Closed-set project-defined ten-intent subset. Original labels, no relabeling. Test is all selected official records; dev is taken only from official train. No OOS or personal-user guarantee.')
        write_json(directory/'manifest.json', manifest)
        verify_prepared(directory)
        manifests[str(size)] = manifest
    path = output/'fewshot_test.jsonl'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in fewshot:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    report = dict(status='cpu_data_prepared_training_unexecuted', catalog=catalog(), official_counts=dict(train=10003,test=3080),
        selected_unique_records=1000, counts={size:m['counts'] for size,m in manifests.items()},
        exclusions=dict(exclusions), eligible_train_by_intent={key:len(value) for key,value in eligible.items()},
        views_nested=True, dev_test_identical_across_sizes=True,
        fewshot_example_ids=[r['id'] for r in train[:10]], fewshot_sha256=sha(path),
        manifests=manifests, all_records_fit_token_budget=True, gpu_started=False,
        publication='Raw customer texts stay private; reproduce using fixed upstream source files')
    write_json(output/'construction_report.json', report)
    print(json.dumps(dict(status=report['status'],counts=report['counts'],exclusions=report['exclusions'])))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, default=ROOT/'data/raw/banking77_v1')
    p.add_argument('--output-dir', type=Path, default=ROOT/'data/processed/banking77_v1')
    p.add_argument('--tokenizer', type=Path, default=ROOT/'models/Qwen3-4B')
    p.add_argument('--download', action='store_true')
    p.add_argument('--build', action='store_true')
    args=p.parse_args(argv)
    if args.download: fetch(args.source_dir)
    if args.build: build(args.source_dir,args.output_dir,args.tokenizer)
    if not args.download and not args.build:
        print(json.dumps(dict(executable=False,catalog=catalog(),sources=SOURCES,next='--download and/or --build; CPU only')))
    return 0


if __name__ == '__main__': raise SystemExit(main())
