# Dataset attribution and source terms

This selected English text-review example reuses existing released records and
safety labels. It is not an official release or endorsement by the source authors.

Medical: thu-coai / DiaSafety, On the Safety of Conversational Models: Taxonomy,
Dataset, and Benchmark (Findings of ACL 2022).
Source: https://github.com/thu-coai/DiaSafety
Revision: 5106a041b1c7e36f30e02a97c44897a7f2057278
Upstream repository terms: Apache License 2.0; preserved in LICENSE_DIASAFETY.txt.

Financial and public safety: PKU-Alignment / BeaverTails, BeaverTails: Towards
Improved Safety Alignment of LLM via a Human-Preference Dataset (NeurIPS 2023).
Source: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
Revision: 8401fe609d288129cc684a9b3be6a93e41cfe678
Data terms: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ .
Retain attribution, the license link and modification notice; noncommercial use.

Modifications: rule-based domain selection, scope/length filtering, exact/source
group and automated near-duplicate separation, balanced small subsets, and JSON
formatting of EXISTING Safe/Unsafe/is_safe labels. No original request, candidate
response or safety rating was generated or rewritten. Category metadata is kept
for provenance; the target is the original response-safety label only.

Joint files contain copies of the three component sets. Each row retains its
source terms; combining them does not replace either source's license or grant a
license for this repository's code. Project code remains without an assigned
open-source license. No weights, checkpoints or model predictions are included.
