# 本地复现与运行指南

更新日期：2026-11-01。本项目尚未正式公开发布；本指南面向已有本地资产的复现。小样本v1已完成16次训练和完整评价，结果见21、23号文档；后续关键PEFT对照单独按24号协议执行。

## 环境与数据

当前实测训练硬件为RTX5070Ti 16GB，Python3.12，依赖版本见requirements-train.lock与requirements-eval.lock。在已有虚拟环境中运行下列命令，不在实验执行中升级依赖。单卡一次仅一个训练或GPU评价进程；CPU只读分析不占用GPU。

数据使用LaMP-7官方train/dev/test划分及既存真实嵌套50/100/200/500子集。原始文件置于data/raw/lamp7/{train,dev,test}/，每个目录包含对应questions.json与outputs.json。不得把问题ID当真实用户ID；官方配对输入包含模型改写，本项目未新增合成训练样本。保留数据访问与第三方使用限制，不把原始或处理后语料提交到公开仓库。GYAFC尚未完成获取与实验。

重新prepare会写出数据文件，因此当前运行阶段只使用审计入口；数据准备可在独立新输出目录运行，绝不覆盖冻结资产：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts/prepare_lamp7.py --raw-dir data/raw/lamp7 --output-dir data/processed/lamp7_reproduction --profile-k 12 --seed 20251206
```

prepare入口参数与种子对应现有BM25数据流程；是否逐字节一致仍需检查哈希与记录内容，不能仅凭参数相同认定复现成功。

## 核查和冻结

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/freeze_small_sample_protocol.py
```

第二条命令只核验8份配置、模型权重、数据与代码，不加载GPU模型或启动训练。首次冻结用--freeze，已冻结后禁止重复覆盖。发现哈希不符，应调查变更并另立版本，而非重写manifest掩盖差异。

## 单次训练与恢复

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -u scripts/train_qwen_lora.py --config configs/small_sample_v1/n50_lr1e4_seed20251206.json
```

以下为训练接口示例；v1已完成，勿对其非空目录重跑。新的GPU任务必须先检查现有进程。新协议在非空输出目录拒绝重跑。正常训练输出包含run_config.json、training_progress.jsonl、checkpoints/、最佳adapter/以及结束后的training_summary.json。progress中的一条step不等于训练完成；结束摘要、8次dev、128步和权重哈希需要一起核验。

显式恢复示例（须先确认旧进程已退出、故障已定位，且配置完全一致）：

```powershell
.\.venv\Scripts\python.exe -u scripts/train_qwen_lora.py --config configs/small_sample_v1/n50_lr1e4_seed20251206.json --resume-from-checkpoint experiments/runs/small-sample-v1-n50_lr1e4_seed20251206/checkpoints/checkpoint-16
```

只从本run的完整checkpoint恢复优化器、调度器和RNG。恢复点之后的失效日志不能混入有效训练轨迹；需要另行核验记录与分段资源成本。一次性程序不自动恢复失败任务。

## 完整评价和本地推理

训练完成后，用既有evaluate_generation.py加载基座及adapter执行批量本地生成与评分。它需要包含messages及真实参考答案的JSONL，不是自由对话接口。完整test示例：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_generation.py --model models/Qwen3-4B --adapter experiments/runs/small-sample-v1-n50_lr1e4_seed20251206/adapter --data data/processed/lamp7_bm25_k12/test.jsonl --output experiments/local_eval_reproduction.json --max-samples 1496 --batch-size 4 --max-input-tokens 768 --max-new-tokens 80 --quantization none
```

此示例只展示入口，不能在一次性程序执行期间另开GPU任务。单独评价入口本身没有新协议的所有输出保护，应先核对输出不存在。完整评价还需BERTScore、错误分析、完整dev NLL/PPL及配对bootstrap，不能以生成文件存在认定全部评价结束。

adapter/保存适配器和tokenizer，不等于独立完整基座。加载时必须使用相同Qwen3-4B基座。新增scripts/infer_qwen.py支持自由输入和无需参考答案的messages JSONL；GPU冒烟验证记录见experiments/framework_smoke/。权重合并与端侧部署仍未验证。

## 自动执行范围与状态

scripts/run_small_sample_sequence.py是有明确终点的一次性程序，不是周期任务。这是v1历史执行入口，已完成首个训练接管、全部候选、selection和补充种子；当前不应重新启动。任何失败停止，无自动重试。

- 当前阶段和错误：experiments/refocused/execution_state.json。
- 一次性程序日志：experiments/refocused/logs/small_sample_supervisor.stdout.log与stderr.log。
- 各任务日志：相应输出目录的*.log及training_progress.jsonl。
- 第一阶段选择：experiments/refocused/small_sample_selection.json（已产生）。
- 第二阶段配置哈希：experiments/refocused/small_sample_confirmation_manifest.json。
- 最终机器汇总：experiments/refocused/small_sample_results.json（已产生）。
- 最终可读汇总：docs/21_small_sample_results.md（已完成）。

一旦失败，保留现场，先读取state中的错误和对应日志。不要删除lock或重复启动来绕过仍在运行的任务。输出缺少完成收据时程序拒绝复用，不覆盖半成品。进程存在仅说明程序尚存，完成状态须以实际文件核验。

## 结论边界与待交付

目前任务为共享适配器结合用户历史的英文短文本改写；没有证明独立个人作品微调、中文长篇风格或无遗忘。目标在历史中重复、整句复制等风险见19号审计报告。完整1496条评价不删样本，分组仅作补充。

关键PEFT对照已按24号协议启动，独立推理入口已完成真实数据冒烟验证。后续仍需完成PEFT结果解读、代码许可证确定、新环境复现与正式发布；本机入口和教程不等于已完成公开发布。

## 独立推理入口与新PEFT阶段

推理入口不要求参考答案，最后一条messages必须为user。已有SFT文件需明确删除末尾assistant参考答案后才能作为推理输入，不能直接喂给推理器。已有真实dev冒烟输入已这样处理；不新增训练样本。

```powershell
$env:PYTHONPATH='src'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe scripts/infer_qwen.py --model models/Qwen3-4B --adapter experiments/runs/small-sample-v1-n500_lr1e4_seed20251206/adapter --input experiments/framework_smoke/real_dev_prompt.jsonl --output experiments/my_inference.jsonl --quantization none
```

自由输入可用--prompt替代--input；默认system为通用助手，应用任务应明确给出所需system/用户历史。此适配器只在英文短文本改写中验证，通用聊天效果未验证。默认greedy、关闭thinking；输入超768 token直接报错，需主动缩短；不自动删历史。该保护与正式评价固定的截断行为不同，不能用本入口输出替代正式指标。输出JSONL及receipt含输入/适配器/输出哈希；异常保留.partial，拒绝覆盖，不自动重试。

新阶段入口默认只读核验：

```powershell
.\.venv\Scripts\python.exe scripts/run_peft_sequence.py
```

首次冻结使用--freeze，仅一次；确认GPU空闲后--run执行9次串行训练和完整评价。--resume仅供故障诊断后显式恢复已完成步骤，训练中断需单独检查checkpoint，不自动覆盖或重训。新的状态为experiments/peft_v1/execution_state.json，不以v1的complete判断新阶段完成。结束核验results.json、25号可读汇总及全部逐run证据。

统一入口为python -m personal_llm_data.workflow；可编辑安装python -m pip install -e .之后亦提供personal-llm命令。子命令prepare/train/evaluate/infer/audit-peft转发到已验证脚本，需要保留源码仓库，不支持脱离scripts目录的独立wheel部署。本地研究实验已全部完成；不因本指南中的历史命令而重启。正式发布前需确认代码授权、依赖/数据/模型第三方说明并在干净环境复现。不能把本机验证写成已开源发布。


统一入口示例（下面仅查看帮助，不加载GPU）：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m personal_llm_data.workflow --help
.\.venv\Scripts\python.exe -m personal_llm_data.workflow infer --help
```

工程交付清单和机器环境记录见26号文档；第三方资产范围见根目录THIRD_PARTY_ASSETS.md。许可证尚未确定，因此不能将本地工程交付称为正式开源发布。


2026-11-02框架阶段更新：当前目标与前端计划见32号总方案。统一入口新增plan子命令，仅校验声明式任务元数据，不训练、不加载模型；模板configs/framework_task_pending.json反映真实个人材料尚未确定，正常返回待补输入/退出码2。按最新范围，完整前端与任务服务为可选扩展，非本轮必做；开发者命令、配置与模块接口见33号指南。旧研究冻结实现不改。
