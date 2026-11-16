# 脚本用途与当前运行边界

当前研究方案为docs/55_research_realignment.md，不以开发软件框架为目标。这里的分类是导航，不改变冻结源码或授权GPU执行。

- 科学核心保留：train_qwen_lora.py、model加载/数据实现、evaluate_generation.py、evaluate_full_dev.py、bertscore_eval.py、paired_bootstrap.py、paired_adapter_comparison.py及风格/内容诊断。旧20/24等收据按其原版本核对。
- 真实数据构建/核验保留：prepare_lamp7.py、build_banking77.py、verify_banking77.py、SMS及安全构建/核验；不扩充安全/VLM来源，不合成。
- Qwen8单条件入口保留但先不运行：run_qwen8b_banking500.py及cloud/run_qwen8b_banking500.sh。它只支持Qwen BF16当前条件，不支持Mistral或全部量化矩阵。
- 云端资产工具：prepare_8b_4090.py提供Qwen及历史Llama锁定/候选下载；Mistral资产/模板/目标mask尚待实现，不将目录名替换当兼容。
- 已完成历史调度/旧方案，不自行重启：run_small_sample_sequence.py、run_peft_sequence.py、run_quantization_sequence.py、run_confirmation_sequence.py、run_cloud_comparison.py、run_intenttune_3090.py、resume_small_sample_evaluation.py。保留原路径以免破坏冻结证据及调用依赖。
- 历史SMS4090准备/导出：prepare_cloud_sms.py、check_cloud_sms_assets.py、download_cloud_models.py、export_cloud_inputs.py；不是当前研究自动队列。
- 归档：旧prepare_blind_review.py、analyze_blind_review.py已作为完整原字节.py.txt保存于docs/history/replanning_20261116，不是可直接运行的新主观评价协议。未来若采用人评，先按55固定新的真实抽样与评分安排，不伪造分数。
- 发布/维护：audit_publication.py、restore_workspace_archive.py及聚合导出保留，不改原数据许可与公开范围。

其余脚本保留为历史分析实现。未删除模型/数据/逐条预测/统计，也未为了清爽而移动训练/评价核心。一次一个云端GPU过程，失败留证、无自动重试；本地仅CPU检查。
