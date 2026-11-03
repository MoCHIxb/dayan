# 当前RTX4090准备与单轮SMS复现指南

2026-11-03。当前首轮云端目标是复现最新SMS分类实例：500train/200dev/1000test，Qwen3-4B，128步、8次完整dev及两组完整test。选择它是因为已有本地完整同预算结果，便于平台对照；不是自动重启27号旧LaMP十二次训练矩阵，也不自动安排多个任务或网格。

**当前只完成本地CPU资产/配置/换行兼容检查，服务器尚未连接，Linux/RTX4090未实机验收，云端训练未执行。** 不以准备好的指南或配置宣称云端已成功。

## 1. 服务器就绪后克隆并固定版本

在真实Linux/RTX4090实例的持久目录执行。以下以/workspace举例，先确认路径确为持久磁盘、GPU名称、空闲资源和Python3.12；不从本地复制Windows .venv或整个工作区。本文不租机、不配置账号、不自动开通计费。

```bash
set -euo pipefail
nvidia-smi
df -h /workspace
free -h
python3.12 --version
cd /workspace
git clone https://github.com/MoCHIxb/dayan.git
cd dayan
git rev-parse HEAD
# 将完整commit记入本轮记录；开始后整轮不要git pull或改代码。
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.lock
python -m pip install -e . --no-deps
python -m pip check
python -m pip freeze > /workspace/dayan_cloud_environment.txt
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
```

锁文件是已验证本地版本；实际Linux轮子、驱动兼容性仍须服务器验证，不能为了安装通过而默默更换版本。CPU/显存/下载或权限失败时保留现场并停止，不自动换配方、重试或升级科学依赖。长任务可在平台持久终端或已安装的tmux会话运行；SSH断开是否影响进程与停止实例是否停止计费由平台实际确认，不能只关闭窗口。

## 2. 只准备本轮所需模型和真实数据

本轮需要固定Qwen3-4B和UCI SMS，不需要MiniLM、Qwen3-1.7B、通用能力数据或LaMP大矩阵。SMS原文继续不进Git；从固定上游重建，不合成。

```bash
python scripts/prepare_cloud_sms.py
# 无操作参数只显示计划；以下显式下载固定模型与固定原档、构建数据：
python scripts/prepare_cloud_sms.py --download-model --download-data --build-data
```

基座固定Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`，仅下载本轮冻结的13个模型/tokenizer/许可文件。UCI原档SHA和分组/选取规则沿用37号；输出500/200/1000，约13.4%spam。

Windows既有SMS manifest使用CRLF，Linux原构建器会输出LF。本准备器保留原构建器不改：只有JSON元数据完全等于本地冻结内容，且仅处理换行后的字节SHA256精确匹配本地记录时才接受。原短信、原标签、三个JSONL、分组和抽样均不修改；不能修复其他差异。它在新数据构建后执行；训练开始后禁止变更输入。

若已有从本机**私下**传输的原样四文件，可跳过重建，直接预检。若先手动跑了原构建器，只能在训练前显式执行`python scripts/prepare_cloud_sms.py --canonicalize-manifest`。已有目录/文件保护不允许反复覆盖。下载被阻断时停止，另行安排私人传输，不使用任意第三方镜像或换模型版本。

## 3. 真实服务器预检

```bash
python -m unittest discover -s tests -q
python scripts/check_cloud_sms_assets.py --strict-environment --check-gpu --output experiments/cloud_sms_spam_v1/assets_check.json
```

预检校验：本地已冻结的4个数据文件、13个模型文件、6个核心实现SHA；云端配置除output_dir外与本地逐字段相同；500/200/1000完整覆盖；依赖锁版本一致。--check-gpu还要求实际Linux、单张RTX4090、CUDA/BF16可用和无其他计算任务，记录驱动、CUDA、Python和设备。默认不加--check-gpu只作CPU检查，不能当服务器GPU验收。不要传输本机Windows可信桌面进程注册表。

科学配置`configs/cloud_sms_spam_v1/qwen3_4b.json`；独立输出`experiments/cloud_sms_spam_v1/run`。若run已存在，先检查进度/收据；不要重新启动或使用本地旧run覆盖。

## 4. 显式执行一次训练及完整评价

仅在上一步真实服务器预检通过后执行。保持set -euo pipefail，所有GPU阶段串行；日志用追加方式保留，任务完成后不要重复执行这些命令。

```bash
mkdir -p experiments/cloud_sms_spam_v1/logs
python -m personal_llm_data.workflow train-personal --config configs/cloud_sms_spam_v1/qwen3_4b.json --execute 2>&1 | tee -a experiments/cloud_sms_spam_v1/logs/train.log
python -m personal_llm_data.workflow evaluate-personal --config configs/cloud_sms_spam_v1/qwen3_4b.json --condition base --execute 2>&1 | tee -a experiments/cloud_sms_spam_v1/logs/base.log
python -m personal_llm_data.workflow evaluate-personal --config configs/cloud_sms_spam_v1/qwen3_4b.json --condition adapter --execute 2>&1 | tee -a experiments/cloud_sms_spam_v1/logs/adapter.log
python -m personal_llm_data.workflow score-classification --data-dir data/processed/sms_spam_v1 --labels ham spam --predictions experiments/cloud_sms_spam_v1/run/evaluation/base_metrics.predictions.jsonl --output experiments/cloud_sms_spam_v1/base_scores.json
python -m personal_llm_data.workflow score-classification --data-dir data/processed/sms_spam_v1 --labels ham spam --predictions experiments/cloud_sms_spam_v1/run/evaluation/adapter_metrics.predictions.jsonl --output experiments/cloud_sms_spam_v1/adapter_scores.json
python scripts/audit_classification_run.py --config configs/cloud_sms_spam_v1/qwen3_4b.json --labels ham spam --score-directory experiments/cloud_sms_spam_v1 --output experiments/cloud_sms_spam_v1/run_audit.json
```

该脚本复核真实128步、8次完整200dev、最早最低loss选模、最佳/最终全状态和导出哈希，以及各1000条test的ID/参考/命令预算/收据/分数。验收没有通过不能只依据进程退出、训练loss或run_status宣布完成。

暂停从另一个终端发`train-personal --config configs/cloud_sms_spam_v1/qwen3_4b.json --request-pause`；等待paused_checkpoint_verified与进程退出后才关机。恢复须读取实际最新完整checkpoint，用相同配置加--execute与--resume-from-checkpoint指定路径；不能从存在更晚证据的旧checkpoint恢复，不能恢复已经完成128步的任务。首次失败保留现场，不自动重新开始。

## 5. 与本地结果比较与交付

本地参考：results/sms_spam_run_audit.json、results/sms_spam_results.json和38号。基座/适配器accuracy为87.8%/98.0%，macro-F1为0.7986/0.9566，spam召回93.28%/91.79%，ham误报13.05%/1.04%。本地最佳step32；云端仍按自身完整dev选模，不要求最佳step相同，不因test差异追加调参。

先核对数据、模型、代码、科学参数及生成预算相同，再比较全部质量指标、混淆矩阵、训练时长、CUDA allocated/reserved峰值，附实际系统/驱动/CUDA/依赖记录。4090不保证分数更高；Linux4090与Windows5070Ti是**平台复现**，不是显卡单因素因果实验。适配器权重不要求跨平台逐字节相同，但本轮内导出/所选checkpoint哈希必须一致。

将实际服务器审计、环境和必要聚合证据私下带回审查后再更新仓库；短信原文、权重、checkpoint、逐条预测、日志和凭据不直接push。服务器上的证据须在停机或销毁前确认已保存到持久存储并完成校验。

三领域安全审查可作为未来另一个单独任务，沿用35号原128步配置；本地只有32步功能验收，不能把不同预算成绩当成同配置平台对照。旧LaMP平台矩阵仍另行保留，只有明确决定后才执行。没有自动训练队列或新监控任务。
