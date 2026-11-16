# 三模型 BANKING500 首种子正式实验协议与执行指南

本协议 research_banking_v1 独立于冻结的4B实验、46号数据协议和九项GPU诊断。当前是正式执行入口已实现、CPU测试通过后的待云端运行批次；不把九项两步诊断等同完整实验成功。目标是跨基座PEFT质量与资源比较，不提出新LoRA算法或商业微调软件。

## 1. 固定范围

基座为官方冻结版本的Qwen3-8B、Mistral-7B-Instruct-v0.3和Phi-4-mini-instruct。Qwen/Mistral是主要比较，Phi约3.8B是规模补充。沿用ModelScope候选权重与官方SHA核验资产，不重新下载、升级环境或使用诊断适配器。

每个基座运行以下五种配方，种子20261113：

|配方|基座存储/计算|Rank|Alpha|比较用途|
|---|---|---:|---:|---|
|A|BF16|8|16|基础LoRA|
|B|BF16|16|16|固定Alpha的Rank对照|
|C|BF16|16|32|固定Rank的Alpha对照|
|D|4-bit NF4双重量化，BF16计算|16|32|与C比较量化|
|E|LLM.int8，BF16混合计算|16|32|与C比较量化|

A→B同时改变Alpha/Rank比例，不能称为固定有效缩放的纯Rank效应；A→C固定比例2、改变Rank，作为补充解释。C/D/E固定Rank与Alpha，但量化训练的prepare_model_for_kbit_training也改变冻结层精度，属于常规PEFT管线整体比较。Qwen/Mistral覆盖注意力和MLP的七类投影；Phi使用实际融合的qkv_proj/o_proj/gate_up_proj/down_proj。报告真实可训练参数量，不能声称模型等规模、等可训练参数或纯架构因果效应。

数据保持原始BANKING77 card10真实客户文本和类别，不合成或改写。三模型均使用同一500 train / 100 dev / 400 official test。冻结的50/200嵌套视图保留，首批不运行其训练。此前4B测试结果已经看过，本批不能声称从未接触该测试集；不得根据这400条反馈修改配方。

## 2. 训练、选择与完整评价

每次128个优化器步，micro-batch=1、梯度累积16，AdamW lr=1e-4、weight_decay=.01、warmup_ratio=.05、cosine，BF16、TF32、sdpa，dropout=.05。训练集循环、无提前停止。每16步对完整100条dev计算assistant目标token损失，八次评价中选最小loss，平局取更早步。目标边界使用原生完整chat渲染，保留终止符；prompt标签为-100，无截断。Mistral沿用60号显式system合入user的视图；原始数据和官方模板不变。max_length1536、max_new_tokens64、贪心batch1生成、使用原生EOS集合、不修复非法JSON。

导出最佳适配器并核对SHA；保留最佳和最终128步全状态（适配器、optimizer、scheduler、trainer、RNG和收据），独立进程在完整400条test生成并评分。严格JSON、原始有序ID/reference、无效输出计错。报告accuracy、macro-F1、逐类precision/recall/F1、最差类recall、混淆矩阵、invalid rate、选定dev loss及其token-level PPL。PPL在不同tokenizer之间不可直接解释为同一单位的质量排名。

每个模型另评估BF16零样本、BF16固定10条few-shot（只取冻结n50训练集，每类一条）、NF4零样本、int8零样本。一个共同的训练集拟合字符TF-IDF质心基线。共15次训练、27次LLM完整test生成、15次独立审计、1次CPU基线，合计58阶段。对量化适配器优先报告同精度基座差值，同时给出BF16基座参考。零样本结果不能参与dev选模。

资源报告实际可训练参数、训练本次会话时间与峰值allocated/reserved、独立推理加载时间/生成时间/token吞吐/平均样本延迟。单个4090、同样工程配置；不能将GPU allocated等同板卡总显存。暂停恢复时，最终会话时间/峰值不能代表整次运行，须查全部attempt日志。首批单种子只作描述性比较，不作显著性/稳健提升结论；补充种子、50/200规模对照、LaMP完整生成和原始corpus_score配对bootstrap属于后续独立批次，不自动加入此队列。主观评价尚未实施；不启动付费judge、人工盲评或大规模通用能力矩阵。

## 3. 运行

只在云端执行。现有九项诊断和CPUv3文件必须仍在原路径，不要删除。正式入口check会重新验证三套官方模型字节、诊断全状态/收据、数据来源回放和所有4200条模型输入编码与诊断编码的一致性，不使用GPU。已有数据不再构建；缺文件则停止定位，不自动覆盖。

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
bash scripts/cloud/run_research_banking.sh check
bash scripts/cloud/run_research_banking.sh plan
bash scripts/cloud/run_research_banking.sh run
```

run在创建冻结manifest后调用短进程GPU环境检查，再串行运行58阶段。协调器不持有CUDA模型，每个训练/生成独立进程；共享GPU锁拒绝其他计算进程。不要再启动旧Qwen单模型入口或重复开第二个run。不要在manifest冻结后git pull、改配置、升级环境；输入/模型/代码变化会阻断收据核验。哈希核验期间可能无GPU占用，不能仅据此判断停滞。

可在tmux/screen会话内运行run，避免网页终端断开（工具需已安装）。另一个终端只读监控：

```bash
cat experiments/research_banking_v1/execution_state.json
find experiments/research_banking_v1/attempts -name '*.stderr.log' -printf '%T@ %p\n' | sort -nr | head
nvidia-smi
```

按状态active_stage找到attempts/<active_stage>/<attempt>.stdout.log及.stderr.log。训练逐步记录在各条件run/progress.jsonl。日志/进程只是过程证据；完成必须由58项收据及检查点、独立评分重算证明。

```bash
bash scripts/cloud/run_research_banking.sh report
cat experiments/research_banking_v1/results.md
```

最终机读experiments/research_banking_v1/results.json与可读results.md。完成状态complete_research_banking15_first_seed_audited只代表此批，不代表整个课题完成。原始文本、逐条预测、模型、检查点和全部experiments目录保持私有，不能直接git add。

## 4. 暂停与故障

```bash
bash scripts/cloud/run_research_banking.sh pause
cat experiments/research_banking_v1/execution_state.json
```

训练在下一个既定16步边界保存，不增加dev选模机会；生成完成当前整组400条后暂停。等待paused_between_stages或paused_checkpoint_verified且实际进程退出后再关闭电脑/实例。仅发出pause不等于已经安全落盘。

阶段之间暂停后：

```bash
bash scripts/cloud/run_research_banking.sh run --resume
```

训练中暂停后，将状态文件checkpoint字段的完整路径填入：

```bash
bash scripts/cloud/run_research_banking.sh run --resume --checkpoint /root/autodl-tmp/dayan/experiments/research_banking_v1/具体条件/run/checkpoints/checkpoint-实际步数
```

只能恢复原条件、输入一致、收据核验通过的显式暂停检查点。失败/非正常退出一律保留attempt、partial预测和检查点，不自动重试或删除证据；须先诊断。初始r8两步探针不能保证r16完整训练无OOM，出现OOM同样保留现场，不私自降长度、改精度或预算。暂停训练可能在恢复时重载模型，累计时间须结合全部会话日志解释。
