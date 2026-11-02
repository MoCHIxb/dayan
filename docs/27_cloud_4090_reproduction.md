# RTX4090云端复现与本地对照指南

2026-11-02范围更新：本文件仅描述既有LaMP平台复现。最新优先准备的AI安全审查示例见34号方案，需独立数据/基座/输出配置；不据此启动本文件旧12次训练队列。本文件的科学对照范围保持原样。

## 目标与边界

首轮固定Qwen3-4B、现有真实500条训练子集、完整1500 dev与1496 test。比较BF16 r16/a32、BF16 r8/a16、BF16 r16/a16、NF4 r16/a32，各三个种子20251206/7/8，共12次训练。BF16 r16/a32也在云端重训，不能把本地适配器搬过去冒充云端训练结果。

所有科学参数和本地一致：128更新步、每16步完整dev评价和保存、最低dev loss选择checkpoint、输入768、microbatch1、梯度累积16、lr1e-4、固定目标模块。生成greedy、无thinking、batch4、新词80；完整BERTScore、错误分层、表层风格MAE、复制诊断和原始统计保留。不会重新选学习率、合成样本、增加batch或切换FlashAttention来追求速度。

云端Linux/4090与本地Windows/5070Ti是平台复现，不是显卡单因素实验。保持依赖主版本和精确包版本，记录操作系统、驱动、CUDA、GPU、Python补丁版本和完整pip freeze；不要求不同硬件训练出的适配器逐字节相同。模型/数据/科学配置/核心实现的哈希必须相同。

**工具已作CPU级验证；尚未在Linux/RTX4090上实机验收。没有租用、连接或付费开通云实例。** 第一次执行先完成下述验收，不能把文档当作已跑通。

## 1. 准备服务器

选择单张RTX4090 24GB、Python3.12环境、可用NVIDIA驱动的Linux实例。建议至少32GB系统内存、100GB可用持久磁盘；容量含基座、BERTScore、环境和checkpoint，是工程预留，不是最低需求实测。数据盘须在重启后保留；确认平台停止GPU计费的方法，不要只关闭SSH。

以下用`/workspace`举例，按实际持久目录修改。先检查资源：

```bash
nvidia-smi
df -h /workspace
free -h
python3.12 --version
```

不要同时运行其他GPU任务。不要把SSH私钥或云服务token发到聊天或提交仓库。

## 2. 克隆并固定代码版本

```bash
cd /workspace
git clone https://github.com/MoCHIxb/dayan.git
cd dayan
git rev-parse HEAD
# 把输出的完整commit写入本轮实验记录；整轮不要git pull或改代码。
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.lock
python -m pip install -e . --no-deps
python -m pip check
python -m unittest discover -s tests
```

训练锁文件含原始torch2.11.0+cu128、transformers4.57.6、peft0.21.2、bitsandbytes0.50.2等精确版本。此轮不需要requirements-eval.lock中的大规模通用矩阵依赖。若包不可安装、驱动不兼容或pip check失败，保留错误并处理，不擅自升级/降级后仍称同环境复现。Python3.12补丁号尽量匹配本地3.12.14，实际不同必须记录。

安装与CUDA检查依据[PyTorch官方入口](https://docs.pytorch.org/get-started/locally/)；原锁文件选择cu128以匹配本地，而非跟随安装器默认最新版。

## 3. 本机私有数据传输

GitHub不含真实数据。为保证字节级哈希一致，首轮优先传输本机已冻结的三个处理后文件，而不是在Linux上重建后忽略换行差异。仅用于已获准的研究用途，不公开分发。

在Windows本机的仓库根目录运行（本地原始实验路径仍为D:\dayan）：

```powershell
python scripts/export_cloud_inputs.py --local-root D:\dayan --output D:\dayan\experiments\cloud_transfer\inputs.private.tar.gz
```

如果使用原虚拟环境，可将`python`换成`D:\dayan\.venv\Scripts\python.exe`。只打包真实train_500/dev/test JSONL及私有哈希清单，不包含权重、密钥或合成训练数据。把SSH端口/主机按租用平台提供信息替换：

```powershell
scp -P <SSH端口> D:\dayan\experiments\cloud_transfer\inputs.private.tar.gz <用户>@<主机>:/workspace/
```

在服务器仓库根目录解包：

```bash
tar -xzf /workspace/inputs.private.tar.gz -C .
```

后续若第三方复现，应按LaMP官方入口合法取得相同版本，按既定流程准备并解释哈希差异；不能绕过数据条件，从公开代码仓库期待取得语料。

## 4. 固定模型与资产核验

```bash
python scripts/download_cloud_models.py --download
python scripts/run_cloud_comparison.py --check-assets
```

Qwen revision固定为`1cfa9a7208912126459214e8b04321603b3df60c`，下载到models/Qwen3-4B。BERTScore使用与本地相同的roberta-large缓存版本；若默认ref已变化，工具拒绝继续，应转移本机相同缓存而非改变评价。所有对应哈希见repro/local_assets.json。

模型下载使用Hugging Face的[snapshot_download与revision机制](https://huggingface.co/docs/huggingface_hub/en/guides/download)。如果网络不可达，按租用平台允许的方式私有转移原模型目录/缓存，随后仍需通过哈希核验，不凭模型名字相同认定版本一致。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/run_cloud_comparison.py --freeze
```

--freeze核对数据、模型、核心代码、BERTScore、12份配置与精确依赖后创建一次性manifest，不训练。已有manifest禁止覆盖。缺少数据、版本不符、已有输出目录均停止。

## 5. GPU阶段：12训练＋完整评价

使用tmux保持SSH断开后的进程运行；tmux若未安装，按平台方式安装后使用：

```bash
tmux new -s dayan4090
cd /workspace/dayan
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p experiments/cloud_4090_v1/logs
set -o pipefail
python -u scripts/run_cloud_comparison.py --phase gpu --execute 2>&1 | tee experiments/cloud_4090_v1/logs/supervisor_gpu.log
```

在tmux中按Ctrl+B再按D退出会话；`tmux attach -t dayan4090`查看。`tee`日志仅用于首轮启动；恢复时使用新文件名，避免覆盖原日志。

执行器顺序进行12次训练、4组基座生成（BF16/NF4×有/无画像）、24组adapter生成及12套辅助评价。单GPU串行、失败即停、不自动重试。结束证据为gpu_completion.json，需核验12/28/12及哈希，不仅看进程退出或终端文字。

```bash
cat experiments/cloud_4090_v1/execution_state.json
nvidia-smi
tail -n 10 experiments/cloud_4090_v1/logs/supervisor_gpu.log
```

CPU bootstrap在下一阶段，不会自动延长GPU阶段。GPU阶段全部完成且核验备份后，可以停止GPU实例计费，统计可在本机或CPU实例继续。

## 6. 下载结果与CPU统计

在服务器仓库根目录，打包时排除checkpoint和adapter，保留manifest、gpu_completion、训练摘要、全部预测/指标/收据/逐条dev及环境记录：

```bash
tar --exclude='*/checkpoints' --exclude='*/adapter' -czf /workspace/cloud_results.private.tar.gz experiments/cloud_4090_v1
```

在本机下载：

```powershell
scp -P <SSH端口> <用户>@<主机>:/workspace/cloud_results.private.tar.gz D:\dayan\experiments\cloud_transfer\
```

在**同一代码commit**的本机仓库根目录解包；若目标experiments/cloud_4090_v1已存在，先检查，不覆盖。归档含逐条预测，保持私有，不上传GitHub。CPU统计读取gpu_completion中的输出哈希，不加载模型，不需要将云端权重下载到CPU机器；原本地peft_v1不被覆盖。

```bash
tar -xzf /path/to/cloud_results.private.tar.gz -C .
python -u scripts/run_cloud_comparison.py --phase statistics --execute --resume
```

Windows亦可使用`tar -xzf`和既有`.venv\Scripts\python.exe`，设置`PYTHONPATH=src`。共21项统计：每个adapter的2×2原始bootstrap12项，9个候选对同种子BF16 r16/a32直接比较9项；均1496条、2000次、原始corpus_score。沿用既定评价，不以句分数均值或近似统计替代。最终生成experiments/cloud_4090_v1/results.json，状态cloud_round_complete。

若愿意继续在云端做CPU统计，也可以在GPU阶段完成后原目录执行上述命令；需要自行核对平台是否仍按GPU计费。

## 7. 和本地怎样比较

公开results/local_500_reference.json提供12个本地对应条件的汇总、dev最佳步、训练allocated/reserved显存、参数量、耗时、生成吞吐、BERTScore与复制率。云端最终results.json按相同variant/seed给出描述性差值。

报告三种子均值和样本SD；不池化成4488条独立测试样本。耗时含128步及8次dev，不含全部下载/加载/正式评价，租赁总时长另计。比较时同时检查科学参数、输入哈希和环境差异，不依据SHA不同断言训练损坏。

21项云端bootstrap检验云端方案之间的质量差异，**不等于云端对本地的差异检验**。若要对跨平台质量差异给置信区间，需要私有传输对应本地有画像预测，再对每个variant/seed用paired_adapter_comparison.py进行12项完整2000次原始配对统计，另立输出目录；不能从两个均值的区间相减冒充配对检验。当前默认只作跨平台描述性比较。

## 故障与暂停

首次失败保留state、日志、收据及半成品，不重复点启动。已有GPU计算进程或sequence.lock时拒绝新任务。中断训练须定位完整checkpoint再按train_qwen_lora.py的显式恢复入口审查处理；中断bootstrap没有中间数组/RNG保存，只能从0用原种子重算该项。已完成训练和收据继续复用。

诊断后如恢复，在原目录、配置和版本不变且旧进程确已退出的前提下，使用相应--phase、--execute、--resume；失效锁和半成品必须先单独审计归档，不能自动删锁绕过活进程。禁止修改运行中的冻结脚本、改batch或降低dev/test/bootstrap规模来追赶进度。

