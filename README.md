# 低资源大模型适配：多基座PEFT实验研究

最新范围（2026-11-16）：以开题书和所有者最新指示为准，研究主体为多个基座模型下LoRA/QLoRA、rank/alpha和4/8-bit的效果/资源比较，以及主客观多维评价。通用/专业微调框架、IntentTune产品或GUI不再是主要交付或创新目标。仓库用于研究复现和维护证据。

## 当前入口

- [55：研究目标、模型选择、有限矩阵及评价重规划](docs/55_research_realignment.md)。
- [17：最新交接](docs/17_new_chat_handoff.md)、[文档索引](docs/INDEX.md)、[脚本分类](scripts/README.md)。
- 主比较Qwen3-8B与Mistral-7B-Instruct-v0.3，Phi-4-mini作为规模补充；Llama访问被拒、LLaVA取消。三模型资产、CPUv3及九项GPU诊断已由所有者回报通过，原始云端文件未本地导入。
- 当前设计：[65号全因子协议](docs/65_full_factorial_banking_protocol.md)，1125次训练不变；新批次推荐[66号分析保留入口](docs/66_analysis_evidence_retention.md)，逐条件完整评价/审计后释放大状态，保留预测/曲线/指标/收据和9套代表adapter，4513控制阶段。原全量保留入口保持原源码，不对已启动旧任务自动迁移或删文件。新r64探针及正式训练仍待云端回报。
- 目标服务器为所有者RTX4090 24GB／PyTorch2.12.1／CUDA13.0；[53号](docs/53_cloud_4090_dual_8b_setup.md)保留历史Qwen/Llama准备操作，只复用已核验的Qwen资产，不再要求取得Llama权限。

## 已有研究证据

| 实验 | 状态 | 正确解释 |
|---|---|---|
| LaMP4B小样本与PEFT | 完整训练、评价和原始bootstrap完成 | 小样本适配与质量/资源取舍；非深层风格或无遗忘保证 |
| SMS及联合安全功能验证 | 已完成，历史证据保留 | 分类任务结果，不证明通用安全或专业事实正确 |
| BANKING77 card10／Qwen4B／3090 | 云端聚合已导入复算 | macro-F1：基座0.8360、10示例0.8916、500条LoRA0.9401；原始云端文件未本地重审 |
| 三模型／4090 | CPUv3及九项两步GPU诊断通过；正式15次训练待执行 | 诊断峰值不是完整训练资源或质量成绩 |

完整历史证据见[23](docs/23_small_sample_interpretation.md)、[25](docs/25_small_sample_peft_results.md)、[31](docs/31_peft_quality_cost_analysis.md)、[38](docs/38_sms_spam_results.md)、[49](docs/49_banking77_results.md)。旧论文[42号](docs/42_paper_draft.md)是历史工作稿，需按新研究主线修订，不能直接作为定稿。

## 研究边界与文件整理

数据、模型、checkpoint、完整预测、原始统计和失败现场保留；20/24/46冻结协议与核心科学代码不改。旧框架与全历史重跑方案已无损归档，原路径保留指针，见[整理清单](results/research_cleanup_20261116.json)。旧调度脚本因证据/依赖保留，不能从存在脚本推导自动运行授权。此前至少82次不是当前完成门槛。

只公开代码、配置、聚合证据及已授权的12份text_safety示例JSONL；不公开LaMP/SMS/BANKING原文、权重、逐条预测或密钥，不合成训练样本。代码许可证未由所有者指定，不把公开维护称为正式开源发行。主观评价另定协议与执行安排，不伪造人评或产生未授权付费judge费用。

本地只做CPU准备与测试，新增模型实验均在所有者云端显式串行执行，失败留证，不自动重试。下一步先确认55号模型/任务/评价定义，再准备新实验配置；当前未启动GPU任务。

当前纯文本模型清单与一次性ModelScope资产部署见[56号](docs/56_text_model_roster_and_deployment.md)；LLaVA已按最新指示取消，部署不等于训练完成。

所有者已要求扩大参数分辨率并接受长时间训练；当前[65号](docs/65_full_factorial_banking_protocol.md)覆盖64号72次建议，五档Rank/Alpha完整交叉和五种子入口已实现并经CPU测试。云端实际进度尚未回报，不宣称已启动或完成。
