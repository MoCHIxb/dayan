# Three-domain text safety review example

See [the full source/data/cloud guide](../../docs/35_three_domain_text_safety.md)
and [source attribution and terms](NOTICE.md).

| Set | train | dev | test |
|---|---:|---:|---:|
| medical (DiaSafety Unauthorized Expertise) | 50 | 20 | 40 |
| financial (BeaverTails topic subset) | 50 | 20 | 40 |
| public_safety (BeaverTails topic subset) | 50 | 20 | 40 |
| joint (same records combined) | 150 | 60 | 120 |

Each split/domain is balanced Safe/Unsafe. There are 330 unique QA records;
joint is a duplicate view, not extra data. Inputs are one quoted request and
candidate response; the target is only the existing response-safety rating in
JSON. Source policies differ. Medical labels concern unauthorized expertise,
not clinical factual correctness. No images, constructed multi-turn attacks,
VLM-project data, newly synthesized cases, new model answers or generated labels.

Sources, grouping, counts, file hashes and an automatically recomputed
near-duplicate audit are recorded in manifests and construction_report.json.
The automated checks are not a claim of manual review or certified safety.
Dataset built and CPU-verified; no new GPU training or performance claim yet.
