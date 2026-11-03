# 基于低资源微调的大模型个性化适配

Exploration of Personalized Adaptation Methods for LLMs via Low-Resource Fine-Tuning.

最终目标：帮助个人开发者用消费级显卡和少量自己的真实数据调整大模型，交付有实证依据、可扩展、易上手的开发框架仓库与方法指南。按所有者最新范围，不要求完整MVP或商业软件；通过启动命令、配置模板和模块接口即可使用，前端为可选扩展。

已有真实数据设计、小样本策略与PEFT消融为LoRA选型及候选配方提供证据。已验证的是LaMP-7英文共享适配器结合历史画像，不是任意个人任务；当前示例采用医学、金融、公共安全三类已有纯文本安全审查数据，沿用Qwen3-4B，取消VLM/MMDS路径。实际数据、来源与云端命令见[35号指南](docs/35_three_domain_text_safety.md)和[选定数据](examples/text_safety_v1/README.md)。每类50训练/20开发/40测试，联合150/60/120；本地32步功能验收已通过，完整结果和取舍见[36号报告](docs/36_local_framework_validation.md)。云端128步尚未运行。

开发者从[33号启动与接口指南](docs/33_developer_framework_guide.md)开始：prepare-personal导入真实配对，train-personal默认检查且显式执行，evaluate-personal覆盖完整个人test，infer加载适配器。新增包装器已在本地完成真实训练、暂停恢复、完整120条基座/适配器测试和独立推理；总体macro-F1为0.677 → 0.816，安全回答误报率3.3% → 13.3%，不能只报告改善。

总方案见[32号](docs/32_framework_master_plan.md)，接手见[当前交接](docs/17_new_chat_handoff.md)。

**本仓库是研究代码公开快照；按项目所有者要求暂不指定代码许可证，正式开源发布尚未完成。** 见[许可证状态](LICENSE_STATUS.md)和[第三方资产说明](THIRD_PARTY_ASSETS.md)。

## 当前可核验成果

- 小样本v1完成16次训练、12个所选适配器的完整评价及36项原始语料级配对bootstrap；三种子有画像BLEU均值为50条20.7858、100条20.6277、200条21.8018、500条22.0421。不是严格单调改善，也不能将多个训练因素变化归因为单一机制。
- 本地PEFT轮新增9次训练，复用3次BF16基线；20组完整test生成及9套辅助评价完成。2026-11-02已完成18/18项原始2000次配对bootstrap，完整收据、原始命令、样本/哈希及最终汇总均通过独立审计。最终[结果汇总](docs/25_small_sample_peft_results.md)和[质量/资源解释](docs/31_peft_quality_cost_analysis.md)已形成；没有候选在三个种子上均有名义BLEU优势，不据跨0判等效。
- 数据准备、训练、完整评价、独立推理、哈希核验和统一入口已经在本机验证；云端Linux/RTX4090尚未实机验证。

新增[客观风格/内容辅助证据](docs/30_objective_supplement_results.md)和[PEFT质量/资源分析](docs/31_peft_quality_cost_analysis.md)。后者现为18/18原始统计核验后的最终本轮报告，保留多重比较及适用范围限制。

结果和边界见[小样本解释](docs/23_small_sample_interpretation.md)、[本地500条参考汇总](results/local_500_reference.json)。50–500仅指训练条数，另用了1500条有标签dev；不能声称总标签预算仅有50–500条。

## 快速入口

```bash
python -m pip install -e . --no-deps
python -m personal_llm_data.workflow --help
python -m personal_llm_data.workflow prepare-personal --help
python -m personal_llm_data.workflow train-personal --help
python -m personal_llm_data.workflow evaluate-personal --help
python -m personal_llm_data.workflow plan --spec configs/framework_task_pending.json
python -m personal_llm_data.workflow infer --help
python scripts/run_cloud_comparison.py
```

plan仅检查声明式任务元数据：当前真实材料未到位，模板有意返回待补输入和退出码2，不加载模型或启动训练；即使声明完整也须继续实际数据/硬件审计。最后一条只展示云端计划，不训练。训练依赖和真实数据/模型须另行准备；正式入口参数见[本地指南](docs/22_local_reproduction_guide.md)。历史自动执行器依赖本机证据和Windows接口，不能直接用于干净云端。

## RTX4090复现

按[云端完整指南](docs/27_cloud_4090_reproduction.md)执行。第一轮固定500条、四种PEFT配置各三个种子，共12次训练；完整dev1500/test1496，固定128步和开发集选择，不在4090上追加调参。

- `configs/cloud_4090_v1/`：独立云端配置，不覆盖本地输出。
- `repro/local_assets.json`：数据、模型、核心代码与BERTScore模型哈希；没有原文或权重。
- `scripts/export_cloud_inputs.py`：在本机将已有真实数据打包为私有传输包，不上传GitHub。
- `scripts/download_cloud_models.py`：取得固定模型版本并核验。
- `scripts/run_cloud_comparison.py`：显式冻结、GPU阶段及CPU统计阶段；失败即停，不自动重试。

Linux4090对Windows5070Ti属于平台复现与资源对照，操作系统、驱动和后端均可能影响耗时，不能解释为显卡单因素实验。训练权重不要求跨硬件逐字节相同；模型/数据/科学参数必须匹配。

## 目录与公开范围

```text
configs/       历史及独立云端配置
src/           数据、加载和统一入口
scripts/       训练、评价、审计与执行工具
tests/         CPU单元测试
docs/          历史研究记录、协议、复现指南
results/       不含逐条文本的公开汇总
repro/         只含校验值的复现资产清单
```

`data/`、`models/`、`experiments/`、缓存、日志、checkpoint和逐条预测均不提交。LaMP/GYAFC等第三方资产按各自获取条件使用；GYAFC仅完成选型，未开展正式实验。不自行合成训练样本。

历史文档可能反映当时计划；当前状态以本README、32号总方案及results/local_progress.json为准，local_pause_status.json仅为历史快照，见[文档索引](docs/INDEX.md)。人工盲评、付费强模型打分、大规模通用矩阵及旧六次全量训练队列继续暂缓。

后续本地工作见[重新规划清单](docs/28_next_work_plan.md)。

## 第二领域验证（2026-11-03）

新增通信文本SMS ham/spam分类：原UCI语料及既有标签，私有500/200/1000划分，单次128步真实训练和两组完整1000test均完成。macro-F1 0.7986→0.9566，spam召回0.9328→0.9179，ham误报0.1305→0.0104，不得仅报告accuracy。新增通用score-classification与只读CPU审计，协议见[37号](docs/37_sms_spam_protocol.md)，完整结果和是否值得微调见[38号](docs/38_sms_spam_results.md)。83项测试通过，原始短信/模型/预测不公开。新领域任务验证不代表任意个人、任务或语言都有效。

## 当前云端入口与工作区整理

RTX4090首轮准备见[39号单轮SMS指南](docs/39_cloud_4090_sms_guide.md)：复现本地500/200/1000、128步实例，独立cloud_sms_spam_v1输出。新增固定资产准备、严格CPU/真实Linux4090预检与可核验的manifest换行兼容；当前仅本地CPU验证，服务器未连接、云端未训练。旧27号LaMP十二次矩阵不是当前启动入口。

清理范围、保留证据和恢复方法见[40号维护记录](docs/40_workspace_maintenance.md)。仅活动4B基座与正式数据保留在工作区常用路径；停用模型元数据、旧数据及大日志私有归档，关键checkpoint/原始统计保留。
