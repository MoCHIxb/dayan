# 文献收集与阶段性综述

检索更新时间：2026-10-17。32 篇分组论文目录和精读顺序见 [`04_paper_catalog.md`](04_paper_catalog.md)，机器可读清单见 [`papers.csv`](papers.csv)。建议用 Zotero 建立 `PEFT`、`low-resource-data`、`personalization-style`、`evaluation` 四个集合，并在每篇论文笔记中固定记录：问题、方法、数据、指标、资源、局限、可复现实验。

## 一、主线结论

本课题不应只被表述成“用 LoRA 模仿某人的措辞”。更严谨的研究问题是：在固定的消费级计算预算和极少样本下，怎样选择参数化方式、数据构造方式和正则化策略，使模型获得可辨识的个性化特征，同时保持内容忠实、指令遵循和通用能力。

LoRA 提供低成本参数更新，QLoRA 进一步降低基座权重的显存占用；LIMA 等工作说明高质量小数据可能胜过低质量大数据。风格迁移文献则强调，风格准确率、内容保持与流畅度是相互制约的三个目标。因而本项目的真正研究对象应是“方法 × 数据质量 × 资源预算”的联合权衡，而不是单独追求训练损失下降。

## 二、必读论文（第一层）

### 1. LoRA: Low-Rank Adaptation of Large Language Models

- Hu et al., 2021/ICLR 2022，<https://arxiv.org/abs/2106.09685>
- 核心：冻结预训练权重，在权重更新上学习低秩分解 `ΔW = BA`。训练参数和适配器存储显著减少，合并后通常不引入额外推理延迟。
- 对本项目的启示：将 `r`、`alpha`、目标层和 dropout 作为主要自变量；不仅报告总参数，还报告可训练参数比例和适配器大小。
- 注意：低秩是否足够与任务、层、数据量有关，不能把 `r=8` 当作普遍最优。

### 2. QLoRA: Efficient Finetuning of Quantized LLMs

- Dettmers et al., NeurIPS 2023，<https://arxiv.org/abs/2305.14314>
- 核心：冻结的 4-bit 基座模型上训练 LoRA，引入 NF4、double quantization 和 paged optimizers。
- 对本项目的启示：消费级显卡优先采用 QLoRA；实验中区分“存储量化”和“计算 dtype”，记录峰值显存与吞吐。
- 注意：论文中 65B/48GB 的结论不是任意环境的硬件承诺；序列长度、优化器、实现版本都会改变显存。

### 3. LIMA: Less Is More for Alignment

- Zhou et al., NeurIPS 2023，<https://arxiv.org/abs/2305.11206>
- 核心：用 1,000 条精心筛选的提示—回答进行 SFT，强调数据质量和多样性。
- 对本项目的启示：50–500 条数据必须进行质量分层、去重和覆盖度审计；样本数消融应从同一高质量母集嵌套抽样。
- 注意：该结论依赖强基座模型，不能推导出任何 50 条数据都足够。

### 4. Deep Learning for Text Style Transfer: A Survey

- Jin et al., Computational Linguistics 2022，<https://aclanthology.org/2022.cl-1.6/>
- 核心：系统整理平行/非平行文本风格迁移、数据集与评估方法。
- 对本项目的启示：把“风格”定义成可标注属性，而不是某位作者的模糊整体印象；评估至少覆盖风格、内容和流畅度。

### 5. Are Large Language Models Actually Good at Text Style Transfer?

- Mukherjee et al., INLG 2024，<https://aclanthology.org/2024.inlg-main.42/>
- 核心：比较零样本、少样本提示与参数高效微调；专门微调能显著改善部分风格迁移任务。
- 对本项目的启示：必须保留 zero-shot、few-shot prompting 作为非训练基线，否则无法说明微调是否必要。

### 6. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

- Zheng et al., 2023，<https://arxiv.org/abs/2306.05685>
- 核心：强模型裁判可扩展开放式生成评估，但存在位置、冗长度和自我偏好等偏差。
- 对本项目的启示：裁判采用盲测、交换 A/B 顺序、固定 rubric；抽取一部分做人工复核并报告一致性。

## 三、拓展阅读（第二层）

| 主题 | 文献 | 与课题的关系 |
|---|---|---|
| Adapter | Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, 2019 | LoRA 之外的结构基线 |
| Prefix Tuning | Li & Liang, *Prefix-Tuning*, 2021 | 连续前缀参数与低秩更新对比 |
| P-Tuning v2 | Liu et al., 2021/2022 | 适合补全项目计划中的 P-Tuning 阅读 |
| AdaLoRA | Zhang et al., *Adaptive Budget Allocation for PEFT*, 2023 | 在层间动态分配秩预算 |
| DoRA | Liu et al., *Weight-Decomposed Low-Rank Adaptation*, 2024 | 将权重幅值与方向分解，可作为后期扩展 |
| LoftQ | Li et al., *LoftQ*, 2023 | 为量化基座联合设计 LoRA 初始化 |
| PiSSA | Meng et al., *Principal Singular Values and Singular Vectors Adaptation*, 2024 | 改进 LoRA 初始化和收敛的候选方法 |
| 风格评估 | Ostheimer et al., LREC-COLING 2024 | LLM 裁判用于风格、内容、流畅度三维评价 |
| 个性化生成综述 | Xu et al., ACL 2025，<https://aclanthology.org/2025.acl-long.1201/> | 将个性化任务、上下文和评估统一分类 |

拓展论文的版本和最终发表信息应在正式写作前从论文主页、ACL Anthology、OpenReview 或会议论文集再次核验。

## 四、可以直接写入开题/中期报告的研究空白

1. 许多 PEFT 工作以通用基准为主，极少样本的中文个人风格适配证据仍不充分。
2. 工程项目常只展示训练成功或主观样例，缺少“资源—风格—通用能力”共同度量。
3. 文本风格评估缺乏统一标准，单一 BLEU 与目标并不一致；LLM 裁判又有系统偏差。
4. 小样本研究容易发生来源泄漏、近重复泄漏以及提示模板泄漏，使效果被高估。
5. “像某人”可能是主题、词汇或句法的混合，需要显式解耦，验证模型学到深层风格而非关键词。

## 五、建议的论文实验结构

- RQ1：数据量从 50 增至 500 时，风格收益何时饱和？
- RQ2：相同显存预算下，LoRA 与 QLoRA 的质量和效率差异如何？
- RQ3：`r`、`alpha` 与目标模块对风格和通用能力的影响是否一致？
- RQ4：质量筛选能否优于等量随机采样？
- RQ5：加入 10%–30% 通用指令回放，能否减轻通用能力退化？

每个结论至少使用 3 个随机种子，报告均值和标准差。训练前冻结测试集，先注册主要指标，避免挑选有利样例。

