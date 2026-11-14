# AutoDL RTX3090：IntentTune全部实验步骤

2026-11-13，当前云端入口。替代旧39号作为当前主任务入口；旧4090/SMS及LaMP矩阵不在本脚本队列。首轮已由所有者在云端完成22阶段，2026-11-14导入聚合结果，见[49号分析](49_banking77_results.md)。本地复算聚合指标与核对固定资产通过，未重新读取云端原始预测/checkpoint/阶段收据。以下命令保留为独立复现指南，已完成运行无需重跑。

## 1. 建立独立环境

在已确认保留策略的持久磁盘操作。`/root/autodl-tmp`只作路径示例，务必实际检查挂载、容量、关机与实例删除后的保留规则，并准备独立证据备份。不要复制Windows .venv、私有桌面进程注册表或整个本地工作区。

```bash
set -euo pipefail
nvidia-smi
df -h
python3.12 --version
# 将下面路径替换为你确认可用的工作目录：
cd /root/autodl-tmp
git clone https://github.com/MoCHIxb/dayan.git
cd dayan
git rev-parse HEAD
# 记录完整提交；整轮开始后不要git pull或修改源码/配置。
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.lock
python -m pip install -e . --no-deps
python -m pip check
export PYTHONPATH=src
python -m unittest discover -s tests -q
```

已有工作环境不必覆盖重建。新克隆也可用 `bash scripts/cloud/run_banking77_3090.sh bootstrap` 代替上面的venv/安装/测试命令；bootstrap仅在显式选择时安装，若.venv已存在就拒绝覆盖。运行结束后仍须在当前终端 `source .venv/bin/activate`。锁文件是本地已验证版本，Linux wheel、驱动和3090兼容仍须真实服务器验收；安装失败保留日志并停下，不静默升级依赖。单卡、单训练/生成任务，无自动OOM降配或重试。其他Bash子命令不会安装依赖。

## 2. 统一入口

```bash
# 默认plan只展示22阶段，不下载、不训练：
bash scripts/cloud/run_banking77_3090.sh plan
# 显式取得固定13文件4B基座、五份BANKING源文件，构建三档及固定示例输入：
bash scripts/cloud/run_banking77_3090.sh prepare
# 只CPU核对14份数据产物、三配置、模型/核心源码SHA及锁定依赖：
bash scripts/cloud/run_banking77_3090.sh check
# 显式串行运行本轮全部步骤：
bash scripts/cloud/run_banking77_3090.sh run
```

prepare使用独立prepare_model_assets核验并补齐固定13份模型文件；默认Hugging Face源，可显式选择ModelScope权重候选源，接受条件始终为原SHA256。不构建SMS。BANKING数据来自作者固定revision，所有JSONL及manifest用UTF-8/LF，云端应字节一致，不需要修改冻结SMS换行规则。已有数据会核验，不覆盖。首次运行先冻结新manifest，核对真实Linux/RTX3090、BF16和其他GPU进程；检查CUDA在短生命周期子进程完成，避免协调进程的CUDA上下文拦截后续训练。

run的有限队列：一次固定10示例基座完整400test；然后按500/200/50各执行轻量对照、128步训练、零示例基座400test、适配器400test、两次严格评分和运行审计，共22阶段。所有GPU生成/训练串行。固定100dev与400test、提示目录、学习率和预算不按测试成绩改变。

长任务可在服务器已有tmux/平台持久终端中运行；脚本不会自动创建任务服务或后台队列。SSH窗口关闭不等于安全保存，也不等于停止计费。

## 3. 查看进度与暂停

```bash
cat experiments/intenttune_banking77_v1/execution_state.json
ls experiments/intenttune_banking77_v1/attempts
# stdout/stderr分阶段、分attempt保存，不覆盖原失败日志：
tail -n 30 experiments/intenttune_banking77_v1/attempts/train_500/1.stdout.log
tail -n 30 experiments/intenttune_banking77_v1/attempts/train_500/1.stderr.log
# 另一个终端中显式请求暂停：
bash scripts/cloud/run_banking77_3090.sh pause
```

若当前是训练，调用既有request-pause，在下一安全保存点做完整dev与全状态checkpoint并核验，协调器随后停止；若当前是评价/CPU阶段，当前阶段完成后停止。模型加载/评价期间不会立刻退出。必须核对checkpoint receipt、实际进程退出及GPU释放，不仅看execution_state。暂停请求失败也要检查已写入的pipeline请求，不能据返回码推断所有任务已停。

正常阶段间暂停，所有者明确恢复后：

```bash
bash scripts/cloud/run_banking77_3090.sh resume
# 训练内暂停时，填实际核验过的checkpoint，不能照抄STEP：
bash scripts/cloud/run_banking77_3090.sh resume --checkpoint experiments/intenttune_banking77_v1/n500/run/checkpoints/checkpoint-STEP
```

resume只跳过已核验完整收据的阶段，显式恢复时归档消费暂停请求。发生失败或异常中断不会自动重试；有未完成attempt时先诊断，再用 `--acknowledge-failure 精确阶段名` 显式确认。有部分评价输出时仍拒绝覆盖；须先核对并私有归档现场后再决定。训练中断仅从本次目录、全状态/输入一致且没有更晚状态冲突的checkpoint显式恢复。最终128步已保存但汇总中断时不盲目续训，需要单独核验收尾。

## 4. 完成证据与结果

```bash
bash scripts/cloud/run_banking77_3090.sh report
```

主要私有输出：manifest.json、execution_state.json、attempts、receipts、fewshot、n50/n200/n500各自run、评分、run_audit.json和最终results.json。report要求22项收据、冻结代码/配置输入和全部输出SHA吻合，不仅凭进程结束宣布完成。最终含轻量模型、零示例、10示例、LoRA及每类混淆/资源；效果差或未达门槛也照实报告。

保留所有原始文本、预测、权重和失败证据在服务器/私有备份。后续只将审查后的代码、配置和聚合结论同步GitHub，不将run目录提交。当前没有任何本轮模型结果；运行脚本和CPU测试通过不能替代3090实测。

## 5. 下载中断或镜像速度过慢（2026-11-14）

用户报告：原站不可达，HF镜像可下载配置/tokenizer和第三个分片，但两个约4GB分片在约8%处断连。这是模型准备失败，尚无本轮训练结果；缺少construction_report是准备中断的连带现象，不据此删数据或改配置。

Qwen官方也提供ModelScope下载方式：[官方文档](https://qwen.readthedocs.io/en/latest/inference/transformers.html)。恢复工具不安装ModelScope SDK、不升级依赖；仅从官方ModelScope候选URL取得缺失safetensors，其余缺失元数据仍从固定HF revision取得。ModelScope分支名不等于HF提交，只有与冻结完整SHA完全一致才接受，不能用“同名模型”替代校验。

只在下载准备失败、训练尚未启动时更新仓库。用户自行在云端命令行运行，无需本地SSH：

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only origin main
source .venv/bin/activate
# 可先测1MiB，只是连通性/速度测试，不代替完整SHA核验：
curl --noproxy '*' -L --fail --range 0-1048575 --connect-timeout 10 --max-time 30 -o /dev/null -w 'HTTP %{http_code}, speed %{speed_download} bytes/s\n' https://modelscope.cn/models/Qwen/Qwen3-4B/resolve/master/model-00001-of-00003.safetensors
```

若测试可用，显式选择该来源。下面的代理设置仅作用于括号内进程；元数据HF镜像保留，以备确实缺少某个小文件：

```bash
(
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  bash scripts/cloud/run_banking77_3090.sh prepare --model-source modelscope &&
  bash scripts/cloud/run_banking77_3090.sh check
)
```

新工具会对已完成的每个文件核对原SHA并跳过；不会重新请求Hub，也不覆盖不匹配的现有成品。新来源自己的partial保存在models/Qwen3-4B/.cache/intenttune_download，每次显式调用可按Range续传；服务器不支持正确续传则停止并保留原partial。旧HF .incomplete不删除、不自动拼接为其他来源的partial。出错不自动重试，不关闭TLS证书校验；最终SHA/长度不符不提升为正式权重。

模型齐全后若BANKING的GitHub原档下载再失败，保留新的错误，单独处理数据网络问题，不启动run。训练开始后不得pull或运行模型恢复工具；现有run和旧六个科学核心实现不修改。本地只有小字节网络探测、临时小文件测试和已有完整模型的只读SHA验证，不能称云端下载已恢复。

本地另已核对ModelScope官方文件API声明：三个权重分片的SHA均与冻结清单一致；最终仍对实际完整下载字节校验。108项CPU测试和已有13文件只读校验通过，不代表云端下载已完成。机器记录见results/intenttune_download_recovery.json。
