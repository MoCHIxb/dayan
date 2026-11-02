# 基于低资源微调的大模型个性化适配

Exploration of Personalized Adaptation Methods for LLMs via Low-Resource Fine-Tuning.

研究目标：在消费级硬件上，用50–500条真实训练记录研究LoRA/QLoRA的个性化适配策略、质量与资源权衡，并形成可复现的本地框架和使用指南。当前任务是LaMP-7英文短文本改写，共享适配器结合用户历史；不宣称已验证每人50条作品的独立适配器、深层风格学习或无遗忘。

**本仓库是研究代码公开快照；按项目所有者要求暂不指定代码许可证，正式开源发布尚未完成。** 见[许可证状态](LICENSE_STATUS.md)和[第三方资产说明](THIRD_PARTY_ASSETS.md)。

## 当前可核验成果

- 小样本v1完成16次训练、12个所选适配器的完整评价及36项原始语料级配对bootstrap；三种子有画像BLEU均值为50条20.7858、100条20.6277、200条21.8018、500条22.0421。不是严格单调改善，也不能将多个训练因素变化归因为单一机制。
- 本地PEFT轮新增9次训练，复用3次BF16基线；20组完整test生成及9套辅助评价完成。18项原始bootstrap尚未完成：用户关机暂停，第一项仅日志到800/2000，中间数组未保存，恢复时重算。**不提供尚未形成的PEFT最终置信区间。**
- 数据准备、训练、完整评价、独立推理、哈希核验和统一入口已经在本机验证；云端Linux/RTX4090尚未实机验证。

结果和边界见[小样本解释](docs/23_small_sample_interpretation.md)、[本地500条参考汇总](results/local_500_reference.json)。50–500仅指训练条数，另用了1500条有标签dev；不能声称总标签预算仅有50–500条。

## 快速入口

```bash
python -m pip install -e . --no-deps
python -m personal_llm_data.workflow --help
python -m personal_llm_data.workflow infer --help
python scripts/run_cloud_comparison.py
```

最后一条只展示计划，不训练。训练依赖和真实数据/模型须另行准备；正式入口参数见[本地指南](docs/22_local_reproduction_guide.md)。历史自动执行器依赖本机证据和Windows接口，不能直接用于干净云端。

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

历史文档可能反映当时计划；当前状态以本README、23/24/27号文档及results中的暂停记录为准，见[文档索引](docs/INDEX.md)。人工盲评、付费强模型打分、大规模通用矩阵及旧六次全量训练队列继续暂缓。

后续本地工作见[重新规划清单](docs/28_next_work_plan.md)。

