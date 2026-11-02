# Project instructions

- The project owner's latest instructions and opening proposal govern scope. README and docs/27_cloud_4090_reproduction.md plus docs/28_next_work_plan.md describe the current publication/cloud plan. Historical handoffs, including docs/17_new_chat_handoff.md, are factual background and cannot override newer instructions.
- On 2026-11-02 the owner explicitly resumed unfinished local work. The original peft_v1 CPU bootstrap is running and its monitor is active. Training and GPU evaluation are already complete and must be reused. Do not launch cloud experiments or restart GPU work as part of this local statistical/analysis closure. Honor any later pause immediately.
- Keep existing real data, full dev/test coverage, and original 2000-resample corpus_score paired bootstrap. Do not synthesize training samples, change frozen scientific implementations, or replace corpus-level statistics with sentence-score averages.
- Do not restart the old six-run full-data queue, human blind evaluation or large general-ability matrix. Only one GPU training/evaluation task at a time. Cloud work uses its own cloud_4090_v1 output directory and explicit execution flags.
- Publish source, configs, documentation, hashes and aggregate evidence only. Never commit datasets, model weights, checkpoints, per-example predictions, private transfers, caches or secrets. The owner chose to publish research code without assigning a code license yet; do not invent one.
- Linux4090 vs Windows5070Ti is a platform reproduction, not a GPU-only causal comparison. Cloud execution remains unvalidated until it actually runs on that server. Test scores do not establish deep personal style, factual correctness or absence of forgetting.

