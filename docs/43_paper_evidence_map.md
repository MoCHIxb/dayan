# 论文主张与证据映射

2026-11-04。对应42号v0.1；原始实验不重算、不改协议/收据。主表数值直接提取见[44号](44_paper_tables.md)，通过 `python scripts/export_paper_tables.py` 可在干净源码克隆中从公开聚合JSON只读重现；--output只接受新文件路径。下列路径均相对仓库根，experiments为私有本地证据，不存在于干净公开克隆属于预期。

| 初稿部分 | 聚合证据/固定协议 | 私有原始证据 | 可支持与不可支持的结论 |
|---|---|---|---|
| 4.1、5.1 小样本 | docs/20_small_sample_protocol.md、21/23；experiments/refocused/small_sample_results.json | experiments/refocused及对应run目录：16训练、12所选适配器、36统计输出 | 固定子集/三种子观测；非用户泛化、非单因素策略创新 |
| 5.2 PEFT | docs/24_small_sample_peft_protocol.md、25/31；results/peft_quality_cost_analysis.json | experiments/peft_v1/results.json及各run；18原始统计与收据 | 参数/显存/时间取舍；不支持无损量化、等效、全局最优 |
| 5.3 风格/内容 | docs/29_objective_supplement_protocol.md、30；results/style_content_supplement.json | 28条件预测输入及29号manifest | 表层代理与43/1453分层；非事实正确/深层风格证明 |
| 5.4 三领域 | docs/35_three_domain_text_safety.md、36；results/local_text_safety_validation.json | experiments/text_safety_v1/local_verify，包括pause_evidence/checkpoint-9 | 32步功能验收及误报取舍；非128步平台对照或专业认证 |
| 5.4 SMS | docs/37_sms_spam_protocol.md、38；results/sms_spam_results.json、results/sms_spam_run_audit.json | experiments/sms_spam_v1：两组1000预测、最佳/最终checkpoint、原失败和修订 | 单种子任务适配；非SOTA、显著性、当前短信分布或用户风格 |
| 3.3 框架 | docs/33_developer_framework_guide.md、41；src/personal_llm_data、scripts、tests | 两次真实任务的输入快照与运行收据 | 已测接口可用与恢复可审计；非用户易用性研究 |
| 8 平台复现 | docs/39_cloud_4090_sms_guide.md、configs/cloud_sms_spam_v1/qwen3_4b.json | 尚无云端训练结果 | 仅准备完成，不能填实测分数 |

## 核稿规则

1. 结果表从聚合JSON/既定报告提取，保留四位小数及单位；均值±SD与bootstrap区间分开。不得将三种子预测池化后另称三种子区间。
2. 所有“节省显存”同时说明allocated峰值、平台、训练预算与比较条件。原reserved和nvidia-smi不是同一口径。
3. 新分类结果的macro-F1提高须同时报告召回/误报变化。没有预设检验时不追加“显著”措辞。
4. 50–500仅为train；完整dev/test、公开历史语料污染和个人身份缺失在摘要/方法/局限保留。
5. 不发布逐条短信、LaMP原文、预测或模型。三领域公开例外只覆盖已审查的examples/text_safety_v1文件。
6. 代码后续变化只影响新提交；原运行源码按原Git版本核对，不为新入口修改旧收据。doctor新增不改变六个冻结SMS核心源码。

## 定稿前清单

- 实际完成一轮独立云端SMS，填平台对照表；未完成时保持“待执行”。
- 作者逐项核对表格和文字，并按学校模板补作者/导师、页码、图表编号、参考文献样式及必要声明。
- 与导师确认以实证/工程为贡献的论文定位；如要求新算法，另立研究协议，不能将已有LoRA改称新算法。
- 对通用性、个人定制和易用性保持已验证边界；真实个人/用户研究未完成。
- 所有者决定代码许可与正式发布范围；数据上游条款单独保留。
