# Aggregate tables for paper draft

Descriptive local results only. SD is training-seed sample SD, not a confidence interval.
No statistics are recomputed. Linux/RTX4090 remains unexecuted.

| Train records | Profile BLEU mean | Seed SD |
|---|---:|---:|
| 50 | 20.7858 | 0.1369 |
| 100 | 20.6277 | 0.0319 |
| 200 | 21.8018 | 0.2474 |
| 500 | 22.0421 | 0.6995 |

| PEFT recipe | BLEU mean | SD | Parameters/M | Allocated/GiB | Training/min |
|---|---:|---:|---:|---:|---:|---:|
| bf16_r16_a32 | 22.0421 | 0.6995 | 33.030 | 9.169 | 31.66 |
| bf16_r8_a16 | 22.6734 | 0.3356 | 16.515 | 8.923 | 30.34 |
| bf16_r16_a16 | 22.6267 | 0.1523 | 33.030 | 9.169 | 30.39 |
| nf4_r16_a32 | 22.0167 | 0.5416 | 33.030 | 5.657 | 36.27 |

| Task | Condition | Test N | Accuracy | Macro-F1 | Risk-class recall | Other-class false alarm | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| response safety (32 steps) | base | 120 | 0.7000 | 0.6770 | 0.4333 | 0.0333 | 0.0000 |
| response safety (32 steps) | adapter | 120 | 0.8167 | 0.8162 | 0.7667 | 0.1333 | 0.0000 |
| SMS (128 steps) | base | 1000 | 0.8780 | 0.7986 | 0.9328 | 0.1305 | 0.0000 |
| SMS (128 steps) | adapter | 1000 | 0.9800 | 0.9566 | 0.9179 | 0.0104 | 0.0000 |

Risk class = unsafe for response safety, spam for SMS. False alarm = safe/ham flagged.
Training memory is allocated peak, not whole-device memory; see source reports for timing scope.
Small-sample training uses another 1500 labeled dev and 1496 test records.

## Source SHA256

- results/small_sample_v1_aggregate.json: daf1a22549171b7af187bf4a80b1f2f5da986782fc76a2b5a7a2ba406a7612e4
- results/peft_quality_cost_analysis.json: 2081cb59641cb4a4d5ad2bf7ae0ce4306de93cb814f0a735ce6cc49c6b441fc7
- results/local_text_safety_validation.json: 0ed9f8aae9bd3158093e8fe76b4fc4d547589ac041ce7a9b293439c8ed497073
- results/sms_spam_results.json: 3e980126c2ab858cad8e5c4516cb114e1ccb0ea7ff986e4b68b8dc04a5d55a24
