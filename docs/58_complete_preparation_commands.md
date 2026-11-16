# 数据、环境与CPU准备完整命令清单

2026-11-16。当前主线55/56：Qwen8B、Mistral7B及Phi4mini纯文本研究；云端GPU暂被占用，以下当前步骤不进行训练/推理。三模型已由所有者报告验收，无需重复bootstrap或deploy。数据、模型、原始统计和失败现场不能清空。

## 1. 当前服务器：进入环境并隐藏GPU

```bash
cd /root/autodl-tmp/dayan
df -h .
# 确认当前没有本项目脚本在运行，更新代码：
git status --short
git -c http.version=HTTP/1.1 pull --ff-only
source .venv-cloud8b/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
python -m pip check
python -c 'import sys,torch; print(sys.version); print("torch:",torch.__version__,"runtime:",torch.version.cuda,"CUDA context:",torch.cuda.is_initialized())'
python -m unittest discover -s tests -q
```

不要调用会设置CUDA_VISIBLE_DEVICES=0的旧训练Bash入口；下面用python直接执行CPU动作。隐藏GPU不改变其他终端的任务，不会终止其他进程。nvidia-smi显示的驱动CUDA13.2与torch运行时13.0不冲突。当前环境已搭建成功，不要重复bootstrap或升级pip/torch。

## 2. BANKING固定真实数据：构建与复核

```bash
source /etc/network_turbo
python scripts/run_qwen8b_banking500.py prepare
python scripts/verify_banking77.py   --source-dir data/raw/banking77_v1   --data-dir data/processed/banking77_8b_v1
python - <<'PY'
from personal_llm_data.personal_data import verify_prepared
for size in (50,200,500):
    rows=verify_prepared(f'data/processed/banking77_8b_v1/n{size}')
    print(size,{split:len(records) for split,records in rows.items()})
PY
```

prepare仅校验模型文件、下载固定真实CSV和CPU构建/审核，虽文件名含run仍不执行train。存在已构建集合时只验证，不覆盖。原九份JSONL和fewshot与原4B哈希相同，8B tokenizer manifest另存。100dev/400test共享，50/200/500严格嵌套；不因模型结果改标签/删难例。

## 3. 三模型CPU兼容性预检

```bash
python scripts/preflight_research_models.py
cat experiments/cloud_8b_4090/compatibility_preflight.json
```

核对native模板/目标监督边界、全数据/示例长度、meta构型投影，不加载模型权重，不用GPU。可能返回cpu_contract_issues_gpu_blocked：保留报告并诊断，不能自行忽略。已存在报告不覆盖；明确修订后用--output独立文件名。training_ready=false正常，GPU计算/保存恢复尚待空闲后验证。

## 4. LaMP：从本地冻结数据私有传输

### 4.1 在本地Windows PowerShell执行（不是云端）

```powershell
Set-Location D:\dayan
.venv\Scripts\python.exe scripts/export_cloud_inputs.py `
  --local-root D:\dayan `
  --output D:\dayan\experiments\private_transfer\lamp7_frozen_500.tar.gz
Get-FileHash -LiteralPath D:\dayan\experiments\private_transfer\lamp7_frozen_500.tar.gz -Algorithm SHA256
```

包已存在时脚本拒绝覆盖：先核对已有包，不删除或重复打包。包只含冻结train_500/dev/test与私有manifest，覆盖当前LaMP500主矩阵，不包含50/100/200曲线、权重或checkpoint，不冒充全部历史数据导出。脚本逐文件核对repro/local_assets.json；若缺失/哈希不同就停止，应先查原归档，不重建近似数据。

通过AutoDL文件上传界面或所有者已有私有传输方式，将包上传到云端/root/autodl-tmp/private_inputs/。不提交Git、不使用公开网盘链接，不需要提供SSH/密码给助手。

### 4.2 在云端执行

```bash
cd /root/autodl-tmp/dayan
source .venv-cloud8b/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
sha256sum /root/autodl-tmp/private_inputs/lamp7_frozen_500.tar.gz
# 与Windows输出的整个包SHA对比一致后导入：
python scripts/import_cloud_data.py   --archive /root/autodl-tmp/private_inputs/lamp7_frozen_500.tar.gz
python - <<'PY'
import json
from pathlib import Path
for split in ('train_500','dev','test'):
    p=Path('data/processed/lamp7_bm25_k12')/(split+'.jsonl')
    rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    print(split,len(rows),len({str(row['id']) for row in rows}))
PY
```

预期500/1500/1496且各集合ID唯一。import只接受冻结精确文件清单、普通文件及原SHA，拒绝额外成员/重复路径/目录链接/越界；不同的已存在数据不覆盖，同SHA数据可复用。不要对不明tar直接解压到仓库。当前不运行prepare_lamp7.py默认k8或重新生成检索变体，否则会改掉既有BM25-k12输入。LaMP的三tokenizer长度和画像/无画像编码仍需在后续正式统一接口单独核验，BANKING预检不代表LaMP也通过。

## 5. 可选：提前准备BERTScore固定资产（CPU下载，不评分）

此步骤可等网络/磁盘方便再做。先取消只对本终端设置的HF离线变量并启用HF访问，再固定revision下载与原SHA核验；不会取Qwen4B权重。

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
source /etc/network_turbo
python - <<'PY'
import json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
s=json.loads(Path('repro/local_assets.json').read_text(encoding='utf-8'))['bertscore_model']
p=Path(snapshot_download(repo_id=s['repo_id'],revision=s['revision'],allow_patterns=list(s['files_sha256'])))
assert p.name==s['revision']
for name,expected in s['files_sha256'].items():
    h=hashlib.sha256()
    with (p/name).open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
    assert h.hexdigest()==expected,name
print('Fixed BERTScore files verified:',p)
PY
```

这里只准备roberta-large的固定快照；后续评分入口还须绑定到该revision及原参考IDF/baseline rescale，不能默认使用漂移的main，不能据下载完成就称BERTScore已跑通。不要运行download_cloud_models.py旧入口，它还会获取旧Qwen4B。当前无预测输出，不执行BERTScore或bootstrap；后续LaMP仍是完整1496条和2000次原始corpus_score配对重采样。

## 6. 记录环境与空间，私有保留准备结果

```bash
mkdir -p experiments/cloud_8b_4090/preparation_notes
python -m pip freeze --all > experiments/cloud_8b_4090/preparation_notes/packages.txt
python -m pip check
# pip freeze可能包含私有包索引或安装路径，留云端私有，不直接上传Git/聊天。
df -h .
du -sh models .venv-cloud8b data experiments
```

这些记录不替代正式实验输入冻结。后续还需留出最佳/最终checkpoint、优化器和临时保存空间；扩容容量以实际df为准，不自动删除任何文件。

## 7. 仅新服务器需要的环境/模型步骤（当前不要重复）

如果是全新、没有.venv-cloud8b的指定镜像，才执行：

```bash
cd /root/autodl-tmp
git -c http.version=HTTP/1.1 clone https://github.com/MoCHIxb/dayan.git
cd dayan
bash scripts/cloud/prepare_8b_4090.sh bootstrap
# 模型未准备且GPU无其他计算占用时，以下是资产动作，不是训练：
source /etc/network_turbo
bash scripts/cloud/deploy_research_models.sh plan
bash scripts/cloud/deploy_research_models.sh deploy
bash scripts/cloud/deploy_research_models.sh verify
```

deploy/verify受全局GPU/传输互斥保护，GPU正被占用时不要运行；当前服务器三模型资产已经通过，无需重做。bootstrap复用镜像torch2.12.1+cu130，已有环境拒绝覆盖，不安装旧requirements-train.lock替换torch。

## 8. GPU空闲后才进行的步骤

在新终端进入同一环境，解除本终端的GPU隐藏：

```bash
cd /root/autodl-tmp/dayan
source .venv-cloud8b/bin/activate
export PYTHONPATH="$PWD/src"
unset CUDA_VISIBLE_DEVICES
nvidia-smi
bash scripts/cloud/prepare_8b_4090.sh check
```

check只核对硬件/运行时，仍不是正式训练。三模型BF16/NF4/int8前后向、LoRA保存/恢复、正式统一训练/完整评价入口仍待实现验证；目前没有可可靠提供的“一键训练全部模型”命令，不把旧Qwen单条件run当替代。SMS/三领域安全及CLINC已退出55号主研究，保留历史证据，不自动下载/重跑。
