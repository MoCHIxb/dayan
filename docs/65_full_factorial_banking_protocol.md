# 三模型五种子 Rank×Alpha×精度全因子实验（v2）

日期：2026-11-16。所有者明确可以长时间训练，要求趋势分辨率和结果可靠性优先，忽略两档Rank起步方案并直接扩大训练脚本。本协议research_banking_v2替代63号15次和64号72次方案作为当前执行主线；不修改它们已绑定的源码、配置和证据，不先跑旧15次。当前只验证新入口的CPU逻辑，新增r64探针与正式训练必须在云端实际执行。

## 1. 完整交叉设计

|变量|取值|
|---|---|
|模型|Qwen3-8B、Mistral-7B-Instruct-v0.3、Phi-4-mini-instruct|
|Rank|4、8、16、32、64（五档，倍增间隔）|
|Alpha|8、16、32、64、128（五档，倍增间隔）|
|基座精度|BF16、NF4双重量化/BF16计算、LLM.int8混合计算|
|训练种子|20261113、20261117、20261123、20261129、20261205|

每个模型/精度覆盖全部5×5=25个Rank/Alpha点，每点5次独立重训；三模型×三精度×25点×5种子=1125次训练。不是只看alpha=2r的对角线。可输出固定alpha的Rank曲线、固定Rank的Alpha曲线、二维响应热图、不同量化下的交互、质量与资源曲线、逐种子及均值/样本标准差。5个离散点用于观察跨度内趋势，不证明连续参数全局最优。

标准LoRA为DeltaW=(alpha/r)BA，固定alpha改变r会改变显式缩放，完整二维网格用于将固定r、固定alpha与固定比率切片分开解释。本批不启用rsLoRA、DoRA或新增初始化算法。alpha/r覆盖0.125至32，包括可能过强/过弱的配方；不预设Rank/Alpha越大越好。不同模型大小、预训练、tokenizer和融合投影不同，不能称纯架构因果。相同r也不是相同可训练参数。Qwen/Mistral七投影，Phi实际四类融合投影，按原始权重头计算预期参数并与实际训练参数核对。

所有条件预先列出，不能根据test删除差配方、换种子或只公布赢家。学习率/预算固定，结论是该统一配方下的响应面，不是各模型分别充分调优后的能力上限。1125次扩大参数和优化随机性覆盖，但不增加测试集规模、不能弥补单任务或数据源的泛化边界。

## 2. 数据、训练和选择保持一致

固定真实BANKING77 card10，500train/100dev/400official test，保持原始客户文本和类别，无合成、改写、重新随机拆分或追加安全/VLM来源。数据选择seed仍20261113；不是全77类或OOS任务。历史已经看过该test，明确不是从未接触的新测试集。

每次从官方冻结基座加载，128优化器步，micro-batch1、累积16，lr1e-4，AdamW/weight_decay .01、warmup .05/cosine、BF16/TF32/sdpa、LoRA dropout .05。每16步完整100dev，八次最低loss选checkpoint，同值取更早步。开发集loss是Trainer按样本汇总的assistant token平均NLL，exp(loss)为该汇总口径的派生PPL，不冒充LaMP的corpus token加权PPL，也不跨tokenizer直接解释为统一质量尺度。

原生完整chat目标和终止符，prompt mask=-100，无截断，max_length1536/max_new_tokens64。Mistral沿用已验证system前置user视图，不改原数据/官方模板；Qwen关闭thinking。每个选定适配器独立进程贪心生成完整400test，用原始有序ID/reference、严格JSON、原类别评分，无效计错；保留最终128步与最佳全状态、adapter SHA及收据，独立重算指标。

## 3. 种子的含义与统计

种子是伪随机数生成器的初始整数，数字大小无优劣含义。种子相同、软硬件和算法相同时便于重复随机过程，但CUDA实现仍不保证跨环境逐位一致。预训练基座不会被重新随机初始化；变化的是LoRA新增矩阵初始化、dropout和训练采样顺序等。每个种子从原始基座独立训练，不从其他种子的adapter继续，也不能拿同一次训练的5个checkpoint当5个种子。

本批训练seed和Trainer data_seed同步改变，五个种子全部使用同一500/100/400材料和固定few-shot示例，不测数据抽样或用户变化的稳健性。同数字seed用于各模型/配方便于配对，但形状不同不等于完全相同的初始化张量和随机路径。调度顺序另用固定schedule_seed20261116，在每个模型/精度/Rank块内打乱25个alpha×训练seed运行次序，不依赖分数。

三模型×三精度×25个Rank/Alpha点构成225个模型×精度×r×alpha单元；只有收齐5个预定种子才计算该单元均值、样本SD(ddof=1)、min/max。保留每种子Macro-F1/accuracy/invalid/最低召回、混淆、dev loss、资源和相对同精度基座增益；还输出同种子相对r16/a32的差值均值与SD。部分单元标记incomplete_no_mean，不混成“五种子结果”。不挑最好种子；5次重复也不自动构成统计显著、等效或无遗忘证明，225个单元的多重比较限制必须披露。

五种子研究的是优化随机性；LaMP原始corpus_score的2000次配对bootstrap研究给定预测的测试抽样不确定性，两者不能替代。LaMP仍是后续独立完整生成协议，不在本1125次分类矩阵里，不重算或覆盖4B原统计。

## 4. 基线、资源与审计

每模型BF16 zero/fixed10-shot、NF4 zero、int8 zero，共12组完整400test模型基线；共同CPU字符TF-IDF质心一次。固定贪心基线无需按5个训练seed重复。1125组adapter test加12基线=1137组生成；1125训练+1125生成+1125审计+12基线+1CPU=3388阶段，r64资源探针另计，不充作正式成绩。

保留allocated/reserved、可训练参数、会话训练窗口、独立推理加载/生成时间与吞吐；训练worker的CLI耗时和所有正常暂停会话资源有单独汇总。worker耗时含加载、哈希和保存，不称GPU纯计算时间。暂停最多在既定16步dev边界，不增加选模机会；累计worker时间求和、峰值取所有会话最大值，allocated不等于板卡总显存。

当前旧r8两步探针不能保证r64，因此新建gpu_probe_grid.py，在每模型/精度用最长两条真实训练编码以r64/a128执行step1保存、新进程恢复到step2与生成，共9个条件。探针仍非128步/所有5种子通过，也非不中断等价验证。原始r8探针源码与结果不动；新探针绑定旧CPUv3/九项收据、新grid/newprobe代码与NumPy/tokenizers版本。run要求所选模型/精度的r64原始文件/收据核验通过。

所有GPU在所有者Linux4090显式、串行、共享GPU锁，协调器无持久CUDA模型。失败/非有限loss/OOM保留attempt/partial/checkpoint并停止，不自动重试、删除、改长度/预算/精度；现有恢复只支持明确暂停，失败需诊断。不能静默跳过失败条件后宣布完整1125次完成。

## 5. 磁盘与分块

权重头读取矩阵shape，adapter参数P=r×sum(in+out)，按两个保留checkpoint各adapter+Adam moments和导出adapter估计28P字节，再加每次64MiB tokenizer/证据余量、24×最大P的临时保存余量和10GiB运营保留。是FP32 LoRA/Adam常规存储预测，不是空间保证；训练前重复检查最低余量，磁盘不足不清理证据。

1125次保留全状态预计需要TB级持久磁盘，具体以prepare输出和真实文件大小为准，不把150G现盘宣称足够。支持模型/精度/Rank显式分块：一个块5alpha×5seed=25次训练；分块只改变执行顺序，不删全矩阵条件。可以先运行较小空间块测实际时间/空间，再扩容继续；全局汇总仍要求所有1125条件，不把块完成当全部完成。跨机器迁移/归档后仍须保留可核验的原始路径或另行设计来源感知导入，本入口不自动转移/删除检查点。

## 6. 云端命令

本地仅CPU测试，不启动训练。若原v1正在运行，先用旧入口pause并等待已核验暂停及进程退出，再更新代码；原run/source/manifest不改，不把旧15次当本批的额外种子或自动复用记录。若v1尚未运行，直接使用新入口，无须先跑它。

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
bash scripts/cloud/run_research_grid.sh plan
bash scripts/cloud/run_research_grid.sh prepare
```

plan不执行，prepare仅生成私有1125配置、重核官方资产/数据来源/4200条编码与旧probe一致性、读取头估算空间，不启动GPU；不会下载或覆盖数据。当前机器须保留原CPUv3和九项r8探针文件。

先以一个模型/精度开始（仍属于完整预定矩阵）：

```bash
bash scripts/cloud/run_research_grid.sh probe --model qwen3_8b --mode nf4
bash scripts/cloud/run_research_grid.sh run --model qwen3_8b --mode nf4 --rank 4
```

后续块显式继续同一全局manifest，例如：

```bash
bash scripts/cloud/run_research_grid.sh run --model qwen3_8b --mode nf4 --rank 8 --resume
```

对尚未做r64探针的模型/精度，先同入口probe对应组合；已有完整探针复核后复用，残缺失败证据禁止自动重试。若持久空间足够且希望一次串行全跑：

```bash
bash scripts/cloud/run_research_grid.sh probe
bash scripts/cloud/run_research_grid.sh run
# 若全局manifest已创建，最后一行改为 run --resume
```

全局manifest绑定所有条件、命令、数据/代码/依赖与原资产锁。模型实际权重由check及独立worker起止全SHA核验；频繁阶段检查用轻量代码/数据/锁/收据哈希，不用mtime冒充完整模型重哈希。manifest冻结后不要pull、改代码/配置/环境。无需SSH、无需再安装新依赖；先前环境和模型仍沿用。

```bash
cat experiments/research_banking_v2/execution_state.json
nvidia-smi
bash scripts/cloud/run_research_grid.sh pause
```

训练下一个16步保存，生成完成整组400再暂停。等待paused_between_stages或paused_checkpoint_verified以及实际进程退出后才能关机。阶段间暂停/已完成块继续用run同选择--resume；训练中暂停还须加--checkpoint并传状态checkpoint完整路径。失败不是暂停，不能直接重试。

```bash
bash scripts/cloud/run_research_grid.sh report
cat experiments/research_banking_v2/progress.md
```

progress_summary.json保留逐次证据与分组，trends.tsv为聚合趋势表。手工report只读重审不把失败状态改为成功；块结束状态selected_block_complete_global_incomplete，只有3388项收据与1125次审计全部通过才生成results.json/complete_grid1125_audited。全部原始数据/逐条预测/模型/检查点/experiments保持私有。

## 7. 与课题目标的关系与当前状态

本批主要回答开题书第一方向跨基座LoRA/量化、Rank/Alpha质量资源取舍；第三方向覆盖客观指标、五种子稳定性与工程成本。第二方向的50–500数据量关系、个性化生成/风格、主观评价及论文整合仍需独立实验；这1125次不能代替它们，也不是整个课题完成条件。

新源码、配置、协议和CPU测试已准备；旧GPU诊断是所有者回报，新r64诊断/1125正式训练均没有执行成功证据。长训练是所有者允许的资源取向，不推算“已开始”或“必然效果好”。代码许可证仍未指定，公开仓库维护不等于正式开源发行。

一手依据：[PEFT LoRA缩放与参数](https://huggingface.co/docs/peft/package_reference/lora)、[Dodge等：初始化和数据顺序的随机性](https://arxiv.org/abs/2002.06305)。五档参数与五种子是本课题的预先设计选择，不是文献保证的最优区间或显著性门槛。
