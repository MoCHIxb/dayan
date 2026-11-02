# 相关论文批量收集目录（32 篇）

检索与核验日期：2026-10-17。优先链接会议论文集、ACL Anthology、PMLR、NeurIPS、OpenReview 或 arXiv 原始页面。机器可读版本见 [`papers.csv`](papers.csv)，核心论文 BibTeX 条目见 [`references.bib`](references.bib)。

## 如何使用这份目录

- A 级：第一轮精读，必须做一页结构化笔记；共 16 篇。
- B 级：用于补齐方法谱系、对照实验或评价依据；共 16 篇。
- 每篇笔记固定回答：研究问题、核心方法、训练数据、基座模型、评价指标、硬件成本、主要结论、局限、可复现实验。
- 论文中的“优于”只能在原论文设定内理解；不要跨数据集、跨基座模型直接比较表格数字。

## A. 参数高效微调与量化

| ID | 论文 | 年份/出处 | 优先级 | 对本项目的直接作用 |
|---|---|---:|---|---|
| P01 | [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html) | ICML 2019 | B | Adapter 起点；理解“冻结主干、插入小模块”的基本范式 |
| P02 | [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/) | ACL 2021 | B | 生成任务中的连续前缀基线，适合与 LoRA 做轻量对照 |
| P03 | [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/) | EMNLP 2021 | B | 说明软提示的效果与模型规模相关 |
| P04 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | ICLR 2022 | A | 本课题的算法主基线；定义 rank、alpha、目标层消融 |
| P05 | [BitFit: Simple Parameter-efficient Fine-tuning](https://aclanthology.org/2022.acl-short.1/) | ACL 2022 | B | 极简稀疏更新基线，帮助判断 LoRA 收益是否来自参数量本身 |
| P06 | [P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks](https://aclanthology.org/2022.acl-short.8/) | ACL 2022 | B | 对应计划书中的 P-Tuning；补齐软提示方法谱系 |
| P07 | [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://papers.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html) | NeurIPS 2022 | A | 引入 IA³/T-Few，并直接讨论 few-shot PEFT 与 ICL 的成本效果 |
| P08 | [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://openreview.net/forum?id=lq62uWRJjiY) | ICLR 2023 | B | AdaLoRA；研究不同层是否需要相同 rank |
| P09 | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | NeurIPS 2023 | A | 消费级显卡主方案；NF4、double quantization、paged optimizer |
| P10 | [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659) | AAAI 2024 | B | 研究量化误差与 LoRA 初始化的耦合，可作为 QLoRA 扩展 |
| P11 | [DoRA: Weight-Decomposed Low-Rank Adaptation](https://proceedings.mlr.press/v235/liu24bn.html) | ICML 2024 | A | 把权重幅值和方向分解，适合作为 LoRA 变体对照 |
| P12 | [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354) | ICML 2024 | B | A/B 矩阵使用不同学习率；适合低成本优化实验 |
| P13 | [PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html) | NeurIPS 2024 | B | SVD 初始化与量化结合；可研究小样本收敛速度 |

## B. 小样本指令数据与数据选择

| ID | 论文 | 年份/出处 | 优先级 | 对本项目的直接作用 |
|---|---|---:|---|---|
| P14 | [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) | ACL 2023 | A | 指令生成、无效样本过滤和相似指令去重的经典流程 |
| P15 | [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | NeurIPS 2023 | A | 支持“高质量小数据”假设，但要求强基座和严格筛选 |
| P16 | [AlpaGasus: Training a Better Alpaca with Fewer Data](https://openreview.net/forum?id=FdVXgSJhvz) | ICLR 2024 | A | 用强模型质量评分过滤指令数据；对应质量分层实验 |
| P17 | [What Makes Good Data for Alignment? / DEITA](https://arxiv.org/abs/2312.15685) | ICLR 2024 | A | 把好数据拆为复杂度、质量、多样性三个维度 |
| P18 | [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://proceedings.mlr.press/v235/xia24c.html) | ICML 2024 | B | 用低秩梯度相似度选择对目标能力有影响的数据 |

## C. 文本风格迁移与个性化

| ID | 论文 | 年份/出处 | 优先级 | 对本项目的直接作用 |
|---|---|---:|---|---|
| P19 | [Deep Learning for Text Style Transfer: A Survey](https://aclanthology.org/2022.cl-1.6/) | CL 2022 | A | 定义任务、平行/非平行数据和风格—内容—流畅度三目标 |
| P20 | [A Recipe for Arbitrary Text Style Transfer with Large Language Models](https://aclanthology.org/2022.acl-short.94/) | ACL 2022 | A | 零样本风格改写强基线；判断微调是否真的必要 |
| P21 | [A Call for Standardization and Validation of Text Style Transfer Evaluation](https://aclanthology.org/2023.findings-acl.687/) | Findings ACL 2023 | B | 说明 TST 评价的标准化与验证缺口 |
| P22 | [LaMP: When Large Language Models Meet Personalization](https://aclanthology.org/2024.acl-long.399/) | ACL 2024 | A | 个性化文本任务、用户级划分和检索增强的重要基准 |
| P23 | [Are Large Language Models Actually Good at Text Style Transfer?](https://aclanthology.org/2024.inlg-main.42/) | INLG 2024 | A | 比较零/少样本提示和 PEFT；直接贴合研究问题 |
| P24 | [Text Style Transfer Evaluation Using Large Language Models](https://aclanthology.org/2024.lrec-main.1373/) | LREC-COLING 2024 | A | 用 LLM 分别评价风格、内容保持和流畅度 |
| P25 | [Personalized Soups: Personalized LLM Alignment via Post-hoc Parameter Merging](https://openreview.net/forum?id=EMrnoPRvxe) | NeurIPS 2024 | B | 多种个性维度的独立适配与推理时组合 |
| P26 | [Personalized Generation In Large Model Era: A Survey](https://aclanthology.org/2025.acl-long.1201/) | ACL 2025 | A | 个性化生成的统一分类、数据集和评价指标总览 |

## D. 生成评价与通用能力保持

| ID | 论文 | 年份/出处 | 优先级 | 对本项目的直接作用 |
|---|---|---:|---|---|
| P27 | [MAUVE: Measuring the Gap Between Neural Text and Human Text](https://proceedings.neurips.cc/paper_files/paper/2021/hash/260c2432a0eecc28ce03c10dadc078a4-Abstract.html) | NeurIPS 2021 | B | 分布层面比较生成文本和人类文本，不依赖逐句参考答案 |
| P28 | [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | 2023 | A | LLM 裁判的可行性、位置偏差、冗长偏差及缓解方式 |
| P29 | [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://aclanthology.org/2023.emnlp-main.153/) | EMNLP 2023 | B | rubric + 分步骤评分范式，可改造成风格评分表 |
| P30 | [Prometheus: Inducing Fine-Grained Evaluation Capability in Language Models](https://openreview.net/forum?id=8euJaTveKw) | ICLR 2024 | B | 开源、细粒度 rubric 评价模型的候选方案 |
| P32 | [BERTScore: Evaluating Text Generation with BERT](https://openreview.net/forum?id=SkeHuCVFDr) | ICLR 2020 | B | 用上下文化表示评价生成语义相似度；用于补充 BLEU/ROUGE 的词面指标 |

## E. 本项目基座模型

| ID | 论文 | 年份/出处 | 优先级 | 对本项目的直接作用 |
|---|---|---:|---|---|
| P31 | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | 2025 | A | 确定 Qwen3-1.7B 架构、上下文能力和非思考模式；作为首轮 LoRA 基座 |

## 第一轮精读顺序

1. P04 LoRA → P09 QLoRA：确定低资源训练基线。
2. P15 LIMA → P16 AlpaGasus → P17 DEITA：确定“质量、复杂度、多样性”的数据选择框架。
3. P19 TST Survey → P20 Recipe → P23 LLM for TST：定义任务并建立 prompting 基线。
4. P22 LaMP → P26 Personalized Generation Survey：明确个性化的用户级划分和泛化问题。
5. P24 TST Evaluation → P28 MT-Bench：设计多维评估和裁判偏差控制。
6. P07 IA³/T-Few、P11 DoRA：确定第二种 PEFT 方法，不让实验只剩 LoRA 自身参数搜索。

## 从文献直接导出的研究假设

- H1：在 50–500 条范围内，质量与覆盖度对结果的影响大于单纯增加相似样本。
- H2：QLoRA 显著降低显存，但量化误差可能在极小样本与低 rank 下放大，需要与 BF16 LoRA 对照。
- H3：模型可能只学习主题词而非风格，因此测试集必须包含“已见风格 + 未见主题”。
- H4：加入少量通用指令回放可以减轻指令遵循能力下降，但过多回放会稀释风格信号。
- H5：单一自动指标不足以支持结论；需要自动指标、LLM 裁判与人工评价三角验证。

