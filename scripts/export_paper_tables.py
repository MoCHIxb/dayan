"""Render aggregate-only paper tables without loading predictions or models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    names = ('small_sample_v1_aggregate.json', 'peft_quality_cost_analysis.json',
             'local_text_safety_validation.json', 'sms_spam_results.json')
    sources = {name: ROOT/'results'/name for name in names}
    small, peft, safety, sms = (json.loads(sources[name].read_text(encoding='utf-8')) for name in names)
    if peft['status'] != 'complete_original_statistics_audited' or peft['completed_original_bootstrap_tasks'] != 18:
        raise ValueError('Original PEFT statistics are not complete/audited')
    if safety['status'] != 'passed_local_gpu_functional_validation' or sms['status'] != 'complete_second_domain_framework_validation':
        raise ValueError('Example evidence is not complete')
    lines = ['# Aggregate tables for paper draft', '',
             'Descriptive local results only. SD is training-seed sample SD, not a confidence interval.',
             'No statistics are recomputed. Linux/RTX4090 remains unexecuted.', '',
             '| Train records | Profile BLEU mean | Seed SD |', '|---|---:|---:|']
    for size in ('50', '100', '200', '500'):
        score = small['aggregate'][size]['bleu']
        lines.append(f"| {size} | {score['mean']:.4f} | {score['sample_sd']:.4f} |")
    lines += ['', '| PEFT recipe | BLEU mean | SD | Parameters/M | Allocated/GiB | Training/min |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('bf16_r16_a32', 'bf16_r8_a16', 'bf16_r16_a16', 'nf4_r16_a32'):
        row = peft['aggregate'][name]
        lines.append(f"| {name} | {row['bleu']['mean']:.4f} | {row['bleu']['sample_sd']:.4f} | "
                     f"{row['trainable_parameters']['mean']/1e6:.3f} | {row['allocated_gib']['mean']:.3f} | "
                     f"{row['training_minutes']['mean']:.2f} |")
    lines += ['', '| Task | Condition | Test N | Accuracy | Macro-F1 | Risk-class recall | Other-class false alarm | Invalid |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for task, report in (('response safety (32 steps)', safety), ('SMS (128 steps)', sms)):
        for condition in ('base', 'adapter'):
            score = report['quality'][condition]
            recall = score['unsafe_recall'] if report is safety else score['per_class']['spam']['recall']
            false_alarm = score['safe_response_false_alarm_rate'] if report is safety else sms['ham_false_alarm_rate'][condition]
            lines.append(f"| {task} | {condition} | {score['samples']} | {score['accuracy']:.4f} | "
                         f"{score['macro_f1']:.4f} | {recall:.4f} | {false_alarm:.4f} | {score['invalid_rate']:.4f} |")
    lines += ['', 'Risk class = unsafe for response safety, spam for SMS. False alarm = safe/ham flagged.',
              'Training memory is allocated peak, not whole-device memory; see source reports for timing scope.',
              'Small-sample training uses another 1500 labeled dev and 1496 test records.', '', '## Source SHA256', '']
    for name, path in sources.items():
        lines.append(f"- results/{name}: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New Markdown path; existing output is never overwritten')
    args = parser.parse_args(argv)
    document = render()
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(document)
    else:
        print(document, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
