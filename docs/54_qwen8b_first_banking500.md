# Qwen3-8B首轮：BANKING77 n500完整对照

2026-11-16。所有者提供Qwen8B14文件校验完成摘要，明确先运行Qwen实验，不等Llama。本入口只运行一个Qwen8B A配方/seed20261113，不启动至少82次的大队列或其他历史任务。新主机4090，环境已由所有者报告通过，首轮GPU路径尚待云端实测。

## 实验定义与独立路径

配置configs/cloud_8b_4090/qwen3_8b_banking_n500_A.json：Qwen3-8B官方锁定版本、BF16、r8/alpha16、lr1e-4、七投影、累积16、128步、每16步完整100dev、最低dev loss且同值最早。原500train/100dev/400test与固定10示例完整保留；没有按已看test改变配方。输出experiments/cloud_8b_4090/banking_n500_A_seed20261113，数据data/processed/banking77_8b_v1。不是4B结果改名。

现有train_personal_adapter.py明确支持Qwen3且实际全split校验token预算，故本轮可直接以独立配置复用冻结实现，无需改核心源码。它仍不支持Llama与int8，本入口不声称全部双8B执行器完成。

prepare使用固定官方BANKING源、既有构建器及8B tokenizer独立重建。全部三档九份train/dev/test JSONL及fewshot JSONL必须与原4B数据清单SHA完全一致，任何因tokenizer筛选导致记录变化都停止，不删除测试或替换样本。manifest中的8B tokenizer_config SHA允许反映新模型，原4B数据/锁不改。再次源回放1000条、嵌套和完整test并检查真实8B全数据/固定示例长度，输入1536/目标64，禁止静默截断。

## 云端命令

先确认下载进程已经退出、无训练进程。仅在开始前更新代码，运行冻结后不再pull或改脚本。

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
# 无新增依赖，沿用已准备的独立环境
bash scripts/cloud/run_qwen8b_banking500.sh plan
bash scripts/cloud/run_qwen8b_banking500.sh prepare &&
bash scripts/cloud/run_qwen8b_banking500.sh check
# 前面均成功后启动：
bash scripts/cloud/run_qwen8b_banking500.sh run
```

prepare下载固定BANKING源需访问raw.githubusercontent.com，可按需source /etc/network_turbo；此步模型已下载，仅本地重核验，无需重下载或重pin。check不加载模型，不能证明BF16显存能容纳。正式run在子进程中再次4090环境预检以避免协调进程留下CUDA context。OOM保留attempt日志即停，无自动重试，不自动换NF4或缩短测试。HF数据/模型需已落盘，训练/评价不自动获取其他模型。

run是有限8阶段：TF-IDF500 → 固定10示例400test → LoRA训练 → 零示例400test及评分 → 适配器400test及评分 → 完整审计。fewshot在train前，先测量未适配参照但不据test修改后续配方。各GPU子阶段串行持有全局锁。进度在attempts各阶段日志，协调终端安静不代表停滞；400题逐条生成可能先运行较久。

```bash
cat experiments/cloud_8b_4090/banking_n500_A_seed20261113/execution_state.json
tail -n 10 experiments/cloud_8b_4090/banking_n500_A_seed20261113/run/progress.jsonl
# 首先运行fewshot阶段时，训练progress还不存在属于正常；查看：
tail -n 20 experiments/cloud_8b_4090/banking_n500_A_seed20261113/attempts/fewshot/1.stderr.log
nvidia-smi
```

## 暂停、恢复与故障

在另一终端：

```bash
cd /root/autodl-tmp/dayan
bash scripts/cloud/run_qwen8b_banking500.sh pause
```

训练阶段请求完整checkpoint暂停；生成阶段允许当前阶段结束再停止，不能承诺立即释放GPU。必须等状态paused_checkpoint_verified或paused_between_stages且实际进程退出，才能关机。恢复只复用输入/命令/输出哈希完全匹配的完成阶段。

```bash
# 阶段之间暂停的恢复：
bash scripts/cloud/run_qwen8b_banking500.sh run --resume
# 训练内暂停时，改用状态中记录并核验的本轮checkpoint实际路径：
# bash scripts/cloud/run_qwen8b_banking500.sh run --resume --checkpoint /实际路径/checkpoint-N
```

失败/异常退出与用户正常暂停不同。未完成attempt自动拒绝重试，不删除或覆盖部分输出；先检查日志和checkpoint，由明确修复流程处理。不得仅凭路径存在手动补写完成收据，不把普通run --resume当作故障无限重试。若GPU预检失败也保留manifest，先定位再决定是否恢复。

## 完成门槛

```bash
bash scripts/cloud/run_qwen8b_banking500.sh report
```

独立结果位于experiments/cloud_8b_4090/banking_n500_A_seed20261113/results.json。report重新核对冻结命令、配置/源码/数据/模型哈希、核心依赖版本及全部8阶段输出，调用已有分类审计核验500/100/400、128步、8次完整dev、最佳/最终状态、适配器SHA和完整预测ID/参考，再复算固定示例及TF-IDF。不能以GPU空闲或execution_state完成就代替这些核验。

只在CPU测试下验证新增计划、收据拒绝篡改与配置，不声称云端端到端已完成。训练损失很低不能代替正式test。首轮单种子描述性结果，非完整三种子或所有8B任务的完成。新模型、原始客户消息、checkpoint/日志和逐条预测私有保全；后续可导入聚合结果解读。
