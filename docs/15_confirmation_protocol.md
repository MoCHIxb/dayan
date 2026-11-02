# 完整 dev、多种子、盲评、能力保持、模型规模与检索补充协议

**最新覆盖指令（2026-10-28 22:54）：用户要求立即暂停当前评价并清理，不再等待其跑完。评价和等待程序已停止，当前base_4B正式结果未生成，最后生成364/1860；未完成日志与等待脚本已清理，基准数据和可复用代码保留。当前研究优先级重新对齐小样本与PEFT/框架交付，见 `docs/17_new_chat_handoff.md`。本文件以下旧扩展矩阵暂缓，不能自动续跑。**

## 2026-10-28 22:45 用户调整优先级

用户要求当前 Qwen3-4B 基座通用能力评价完成后暂停，并重新对齐计划书三个核心方向。已停止派发后续任务的队列父进程，保留当前 `evaluate_general_abilities.py`（实际PID=23716）正常完成整组三项基准。一次性 `finish_current_and_pause.py --pid 23716` 仅等待此进程退出、核对结果完整性并记录暂停，不启动新训练或评价。实时状态为 `finishing_current_then_pause`，其后转为 `paused_after_current_completed` 或未完成故障状态。六次新训练尚未开始，暂不执行；人工盲评继续暂缓，不调用付费裁判。

以下冻结矩阵是已经准备但暂缓的扩展方案，不是当前必须全部执行的清单。下一步重新优先考虑少样本50–500条的训练预算与dev选择、注意力层范围对照、风格/语义证据及开发者框架交付。不要因旧队列文件仍存在而擅自恢复全量六次训练。

冻结日期：2026-10-28。该方案是已查看过 LaMP-7 测试结果后的后续研究，不能声称既有 test 是从未查看的独立确认集。不得根据后续 test 结果反复挑选设置。未收到独立人工评分前，不写出人工风格改进结论；用户可安排两名评价者，暂不使用付费裁判接口。

## 1. 完整 dev 与历史产物边界

保留官方 1,500 条 dev，全量计算 assistant-only teacher-forced loss，沿用 768 token 训练序列处理。报告逐样本平均 loss（与原 batch=1 的验证指标口径一致）以及按监督 token 加权的 NLL/PPL，二者不可混称。逐条保存 ID、监督 token 数和 loss，记录模型、适配器和数据哈希。

历史 BF16 r16/a32 第一轮最佳权重已完成全 dev：平均逐样本 loss=1.74100246，监督 token 加权 NLL=1.77764651，PPL=5.91591695。前 500 条均值 1.74734632，与原训练验证 1.74749 接近但非逐位一致；复核使用推理加载和 autocast，保留该数值差异，不修改旧报告。

历史较差轮次曾按授权清理，r16/a32 的第二轮完整权重已不存在。第一次尝试加载该目录时失败，日志保留；已修正为仅重评实际存在的历史权重。此复核不能冒充完整 dev 重新选择所有轮次。为消除该缺口，正式两模型三种子全部重新训练并在每轮使用完整 1,500 条 dev 选最佳 checkpoint；不复用原 500-dev 选择结果作为新正式训练。

## 2. 严格模型规模与种子矩阵

- 模型：Qwen3-1.7B、Qwen3-4B，均为本地已保存权重。
- 种子：20251206、20251207、20251208，两模型逐种子配对，共六次新训练。
- 固定：同一 10,437 条 train、BM25-k12、rank=16、alpha=32、dropout=0.05、七类 projection、学习率 1e-4、有效 batch=16、两轮、BF16、max_length=768，完整 1,500 条 dev、每轮验证选最优。
- 不根据 test 或历史 alpha64 观察值改变本矩阵的 alpha。历史 alpha/rank/量化最佳适配器另行做全 dev 复核，属于探索性补充。
- 模型大小变化会改变 LoRA 参数数目、计算量和原始能力，报告这些资源差异；该矩阵是相同训练配方对照，不是严格等可训练参数或等 FLOPs 对照。
- 每个适配器在全部 1,496 条 test 上评价有画像/无画像，基座两组预测按模型分别固定；BERTScore 与逐种子模型规模对照的 2,000 次原始语料级 bootstrap 均保留。
- 汇总三个独立种子的均值和样本标准差。逐种子的测试 bootstrap 不代替训练随机性；三个种子不足以支持很强的总体统计结论。

## 3. 检索改进

固定现存 4B/r16/a32 适配器，先做推理阶段检索对照，避免将重训练收益混入检索效应。候选为 BM25 k=4/8/12/20、原始历史顺序 k=12、语义向量 k=12、BM25+向量 reciprocal rank fusion k=12（常数60）。数据没有被证实的时间戳顺序，原始顺序不得称为“最近历史”。

向量模型为 all-MiniLM-L6-v2，固定 revision=1110a243fdf4706b3f48f1d95db1a4f5529b4d41；CPU 编码、attention-mask mean pooling、L2归一化、256 wordpiece 上限。向量只来自当前样本的任务和其用户历史，不用目标答案进行排序，不跨用户召回。

发现历史评价入口在超过768 token时可能右截断整个提示，影响尾部Task。因此七个新候选统一按实际 Qwen chat template 计数，超预算时依次移除最低优先的历史，始终保留完整 system 和Task；Task自身超限则失败，不静默截断。预算修正后的 BM25-k12 单列新对照，不把其分数直接当作历史提示完全相同的结果。当前 BM25-k20 在 dev/test 分别有13/14条需裁减，其他候选也完整记录。

七组均使用完整1,500条dev，以dev chrF为预定主要选择指标，分数相同按候选名确定性排序。选择后只对胜出策略和预算对齐的BM25-k12在全test比较，进行2,000次原始corpus_score bootstrap。不保证改进一定成立；若BM25-k12仍最佳，保留该负结果。

## 4. 通用能力保持

使用 lm_eval==0.4.13 官方任务与评分实现，固定数据revision、样本数量及数据文件SHA256，见 `experiments/confirmation/manifest.json`。

| 能力 | 任务 | 完整样本数 | 指标 |
|---|---|---:|---|
| 知识/科学常识推理 | ARC-Challenge test | 1172 | acc_norm（主要）、acc |
| 数学推理 | GSM8K test，gsm8k_cot_zeroshot | 1319 | flexible-extract exact_match（主要）、strict-match |
| 可验证指令遵循 | IFEval官方train（该基准评价划分） | 541 | prompt-level strict accuracy（主要），instruction-level和loose辅助 |

不设置limit，不抽小子集，不替换官方检查器。两种基座与六种适配器共八个系统，全部跑三个基准；同一chat template、enable_thinking=False、0-shot、batch=1、context=4096。GSM8K生成上限1024；IFEval保留官方1280，确定性解码。生成上限仍可能约束答案，应在结果中记录，不能把本协议等同于无限推理预算。

保存逐样本问题哈希、预测、官方评分，按doc ID/hash配对比较微调前后。主要遗忘指标为IFEval strict；ARC和GSM8K补充知识与推理。配对二元分数差值做2,000次样本重采样区间，另列基座正确→适配器错误及反向数目。区间跨0不是“无遗忘”证明；不设置事后有利的容忍阈值。次要比较未作多重校正，明确标注。

## 5. 真实人工盲评

已生成 `experiments/blind_review_200/for_raters/`：200道真实test题，四个现有系统（基座有画像、BF16、int8、NF4）的输出按题随机匿名为A/B/C/D；每名评价者800条输出、每条四维1–5分。按参考长度四分位×原始画像规模分层，罕见非空层至少2条，再按比例分配。显示Task、用户历史及匿名输出，不显示参考答案、系统身份或自动分数。

两名评价者独立填写自己的CSV，不讨论、不用LLM代填。维度为内容保持、风格相似、流畅自然、指令遵循；原始评分表与映射分离。`private_do_not_share/` 仅供分析，不交给评价者。自动分析会拒绝空白、非法分数、缺失或重复ID；不会补造评分。

`analyze_blind_review.py` 汇总分层人口权重均值、2,000次分层题目bootstrap区间、二次加权Cohen kappa。主要人工对比为BF16−基座的风格相似，内容保持为必要伴随指标。int8/NF4相对BF16为次要对比。区间条件于这两名评价者，不冒充对所有人类评价者的泛化推断。该包评价既有系统，不代表未来全部新种子/检索系统已经获得人评。

## 6. 执行与恢复

单次串行入口 `scripts/run_confirmation_sequence.py`，状态 `experiments/confirmation/state.json`，锁 `gpu_queue.lock`，每步骤独立日志。已完成结果先核验再复用，任何一步失败即退出，保留产物，不自动无限重启。六次新训练及后续完整基准会持续较长时间；真实完成后才写 `machine_summary.json` 和 `docs/16_confirmation_results.md`。

旧量化监控已随上一批完成而删除，本次未擅自创建新的周期自动化。机器实验完成与人工评分完成分别报告；两份独立评分未返回时项目保持人工证据待补状态。

## 官方实现依据

- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml
- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/ifeval.yaml
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
