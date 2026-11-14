# IntentTune：客服意图识别的小样本适配框架

本项目原课题为“基于低资源微调的大模型个性化适配”。2026-11-13按所有者最新选择，当前专注**客服意图识别**，不以通用框架或通用安全审查能力作为主要交付目标。IntentTune是仓库工作名，交付命令、配置、标签目录、实验与方法指南，无需GUI或商业MVP。当前只要求规范维护GitHub研究仓库，不以立即正式开源或许可证决定作为阻塞项。

论文主线：**既有LaMP小样本/PEFT实验说明适配可行性 → 提出面向客服任务的IntentTune → 用提示、轻量模型和LoRA的完整对照验证实践价值**。旧LaMP/SMS/安全结果保留，不将其写成BANKING77成绩。安全数据扩充已暂停。

## 当前任务与运行入口

- BANKING77银行卡客服10意图，原文/原标签；不是全77类benchmark。三档嵌套训练50/200/500，每类5/20/50条，共享100dev和400官方test，共1000条不同源记录。
- 本轮数据源回放和14份构建文件字节重建已通过。RTX3090云端22阶段已报告完成；导入聚合结果的10组混淆矩阵在本地独立复算通过，云端逐条产物尚未导入。
- 后续全部新增实验在所有者准备的AutoDL RTX3090上执行。有限22阶段覆盖固定示例提示、轻量对照、三次128步LoRA、完整基座/适配器评价及审计；默认只展示计划，不运行GPU。
- [45：领域方案](docs/45_intenttune_plan.md)、[46：固定协议](docs/46_banking77_protocol.md)、[47：RTX3090完整运行指南](docs/47_autodl3090_intenttune.md)、[48：数据卡](docs/48_banking77_data_card.md)。[当前交接](docs/17_new_chat_handoff.md)、[论文工作稿](docs/42_paper_draft.md)。

```bash
# 已激活按锁文件准备的环境后，从仓库根目录运行：
export PYTHONPATH=src
python -m personal_llm_data.workflow run-intents
# 在真实Linux/RTX3090服务器上显式执行：
bash scripts/cloud/run_banking77_3090.sh prepare
bash scripts/cloud/run_banking77_3090.sh check
bash scripts/cloud/run_banking77_3090.sh run
# report只有所有阶段的输入/输出/收据均核对后才汇总完成：
bash scripts/cloud/run_banking77_3090.sh report
```

新环境的bootstrap、失败处理及完整暂停/恢复命令见47号。首轮结果见[49号分析](docs/49_banking77_results.md)及[公开聚合](results/banking77_card10_results.json)：500条LoRA macro-F1为0.9401，高于零示例0.8360、10示例0.8916和同档轻量0.8947。50条优于200条，不能宣称单调收益或普适最优。云端原始预测/checkpoint/阶段收据仍需私有保全，本地未重新逐文件读取。原始BANKING数据、模型、checkpoint和逐条预测不进Git；固定来源和构建代码可供云端重建。代码许可暂未指定，见[许可状态](LICENSE_STATUS.md)；各数据上游条款独立保留。

## 已完成的历史研究证据（不重跑）

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
python -m personal_llm_data.workflow doctor
python scripts/prepare_cloud_sms.py
```

plan仅检查声明式任务元数据：当前真实材料未到位，模板有意返回待补输入和退出码2，不加载模型或启动训练；即使声明完整也须继续实际数据/硬件审计。最后一条只展示当前单轮SMS资产准备计划，不训练。doctor只做CPU诊断，不认证GPU或训练就绪。训练依赖和真实数据/模型须另行准备；正式入口参数见[本地指南](docs/22_local_reproduction_guide.md)。历史自动执行器依赖本机证据和Windows接口，不能直接用于干净云端。

## 历史LaMP平台复现（独立保留，当前不执行）

当前云端入口是[47号RTX3090/BANKING指南](docs/47_autodl3090_intenttune.md)；历史[39号单轮SMS指南](docs/39_cloud_4090_sms_guide.md)，仅一次500/200/1000、128步复现。下列[27号LaMP矩阵](docs/27_cloud_4090_reproduction.md)属于独立保留的未执行方案，需另行决定；不是本轮启动队列。其设计为四种PEFT配置各三个种子、完整dev1500/test1496。

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

历史文档可能反映当时计划；当前状态以本README、45/46/47号及results/local_progress.json为准，local_pause_status.json仅为历史快照，见[文档索引](docs/INDEX.md)。人工盲评、付费强模型打分、大规模通用矩阵及旧六次全量训练队列继续暂缓。

后续本地工作见[重新规划清单](docs/28_next_work_plan.md)。

## 第二领域验证（2026-11-03）

新增通信文本SMS ham/spam分类：原UCI语料及既有标签，私有500/200/1000划分，单次128步真实训练和两组完整1000test均完成。macro-F1 0.7986→0.9566，spam召回0.9328→0.9179，ham误报0.1305→0.0104，不得仅报告accuracy。新增通用score-classification与只读CPU审计，协议见[37号](docs/37_sms_spam_protocol.md)，完整结果和是否值得微调见[38号](docs/38_sms_spam_results.md)。本轮历史验收83项测试通过，原始短信/模型/预测不公开。新领域任务验证不代表任意个人、任务或语言都有效。

## 历史4090准备与工作区整理（当前主入口为47号）

已被当前RTX3090/BANKING任务替代的历史准备见[39号单轮SMS指南](docs/39_cloud_4090_sms_guide.md)：复现本地500/200/1000、128步实例，独立cloud_sms_spam_v1输出。新增固定资产准备、严格CPU/真实Linux4090预检与可核验的manifest换行兼容；当前仅本地CPU验证，服务器未连接、云端未训练。旧27号LaMP十二次矩阵不是当前启动入口。

清理范围、保留证据和恢复方法见[40号维护记录](docs/40_workspace_maintenance.md)。仅活动4B基座与正式数据保留在工作区常用路径；停用模型元数据、旧数据及大日志私有归档，关键checkpoint/原始统计保留。

下载故障恢复：HF原站不可达或镜像大文件断连时，见[47号第5节](docs/47_autodl3090_intenttune.md)。可显式指定 `prepare --model-source modelscope`，仅接受原固定SHA一致的文件；已完成文件复用，失败保留partial且不自动重试。
