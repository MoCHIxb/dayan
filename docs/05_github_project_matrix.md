# GitHub 典型项目收集与分层选型（28 个）

检索更新时间：2026-10-12。这里不是按 star 排名，而是按本课题从数据到训练再到评估的完整链路分组。正式实验时记录仓库 commit SHA，避免接口更新导致无法复现。

## 1. 模型、PEFT 与训练基础设施

| 项目 | 作用 | 本课题如何使用 | 建议 |
|---|---|---|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | 模型、tokenizer、chat template、Trainer | 加载 Qwen/Llama，固定模型 revision 和模板 | 主线依赖 |
| [huggingface/peft](https://github.com/huggingface/peft) | LoRA、AdaLoRA、IA³、DoRA 等 | 精确控制 target modules、rank、alpha 和适配器保存 | 主线依赖 |
| [huggingface/trl](https://github.com/huggingface/trl) | SFT、DPO 等后训练器 | `SFTTrainer`，优先使用 conversational/prompt-completion 格式 | 主线依赖 |
| [huggingface/accelerate](https://github.com/huggingface/accelerate) | 单/多 GPU、混合精度与分布式启动 | 统一 BF16、梯度累积和设备配置 | 主线依赖 |
| [huggingface/datasets](https://github.com/huggingface/datasets) | Arrow 数据加载、映射、缓存 | 读取 JSONL、划分数据、保存版本化数据集 | 推荐 |
| [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 4/8-bit 量化与低精度优化器 | 实现 NF4 QLoRA；版本和 CUDA 兼容性必须记录 | QLoRA 必需 |

## 2. 一体化微调框架

| 项目 | 特点 | 适合场景 | 风险/注意点 |
|---|---|---|---|
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 多模型、CLI/WebUI、数据模板丰富 | 快速跑通 Qwen/Llama 和批量消融 | 配置默认值多，必须导出完整配置 |
| [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | 中文生态、训练推理部署一体 | Qwen 系列、国内模型与 ModelScope | 与 HF 主线的模板语义要逐项对齐 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 单卡显存和速度优化 | 8–16GB 显卡上的快速原型 | 优化内核可能使跨框架速度比较不公平 |
| [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | YAML 配置、数据 packing、多卡支持 | 大批量实验和后期扩展 | 初期学习成本较高 |

主实验建议仍用 Transformers + PEFT + TRL，因为变量最透明；再选 LLaMA-Factory 或 ms-swift 复核一组关键结果。不要让四个框架同时成为实验自变量。

## 3. 论文官方或代表性实现

| 项目 | 对应论文/方法 | 用途 |
|---|---|---|
| [microsoft/LoRA](https://github.com/microsoft/LoRA) | LoRA 原始实现 | 阅读算法源代码，不作为当前主训练框架 |
| [artidoro/qlora](https://github.com/artidoro/qlora) | QLoRA | 对照 NF4、paged optimizer 和原始实验配置 |
| [QingruZhang/AdaLoRA](https://github.com/QingruZhang/AdaLoRA) | AdaLoRA | 研究层间动态 rank 分配 |
| [NVlabs/DoRA](https://github.com/NVlabs/DoRA) | DoRA | LoRA 变体对照和幅值/方向分解 |
| [yxli2123/LoftQ](https://github.com/yxli2123/LoftQ) | LoftQ | 量化感知 LoRA 初始化 |
| [princeton-nlp/LESS](https://github.com/princeton-nlp/LESS) | LESS | 面向目标能力的数据影响/梯度选择参考 |
| [hkust-nlp/deita](https://github.com/hkust-nlp/deita) | DEITA | 质量、复杂度和多样性数据选择 |
| [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct) | Self-Instruct | 合成指令生成、过滤与去相似流程 |

## 4. 数据清洗、去重、隐私与标注

| 项目 | 作用 | 建议用法 |
|---|---|---|
| [datajuicer/data-juicer](https://github.com/datajuicer/data-juicer) | 可组合的数据清洗、过滤、去重和分析算子 | 数据达到万条以上或需要复杂规则时引入；当前 50–500 条先保持自写流程透明 |
| [huggingface/datatrove](https://github.com/huggingface/datatrove) | 大规模文本过滤、MinHash/句子去重和统计 | 借鉴 MinHash 与污染检测；规模扩大后替换 O(n²) 近重复检查 |
| [microsoft/presidio](https://github.com/microsoft/presidio) | PII 检测与匿名化 | 邮箱、电话、人名、地址等隐私字段的二次扫描 |
| [argilla-io/argilla](https://github.com/argilla-io/argilla) | 面向 NLP/LLM 的协作标注与反馈 | 多人标注、质量评分、分歧处理 |
| [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) | 通用数据标注平台 | 自定义风格标签界面或多人协作 |

本项目第一版保留 CSV 人工标注，因为 50–500 条规模下更容易审计。Argilla/Label Studio 是多人协作升级路径，不是当前必需依赖。

## 5. 评估、裁判和实验管理

| 项目 | 作用 | 本课题如何使用 |
|---|---|---|
| [lm-sys/FastChat](https://github.com/lm-sys/FastChat) | MT-Bench 和 LLM judge | 借鉴成对比较、位置交换和裁判提示 |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 标准化语言模型基准 | 测量微调前后通用能力保持 |
| [open-compass/opencompass](https://github.com/open-compass/opencompass) | 多模型、多数据集综合评测 | 中文通用能力和模型横向评估的备选框架 |
| [prometheus-eval/prometheus-eval](https://github.com/prometheus-eval/prometheus-eval) | rubric 驱动的开源 LLM 裁判 | 闭源强模型裁判之外的复核方案 |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | 实验参数、指标和模型记录 | 记录 rank、seed、显存、训练时长和指标 |
| [iterative/dvc](https://github.com/iterative/dvc) | 数据与模型版本控制 | 数据超过 Git 适用范围后保存数据版本与依赖关系 |

## 6. 推荐的最小技术栈

第一阶段只安装必要组件：

```text
personal_llm_data 数据管线
        ↓ JSONL
datasets + transformers
        ↓
PEFT + TRL + accelerate
        ↓
bitsandbytes（仅 QLoRA）
        ↓
自定义风格评估 + lm-evaluation-harness
```

第二阶段再按痛点引入：

- 多人标注困难：Argilla 或 Label Studio；
- 近重复/数据量扩大：DataTrove 或 Data-Juicer；
- 消融实验太多：MLflow；
- 需要 GUI/快速复核：LLaMA-Factory；
- 显存不足：Unsloth；
- 中文综合评测：OpenCompass。

## 7. 框架对比实验必须统一的项目

1. 基座模型 ID、revision、tokenizer 与 chat template；
2. assistant-only/completion-only loss mask；
3. 截断方向、最大 token 长度和 packing；
4. LoRA target modules、rank、alpha、dropout、bias；
5. batch size、gradient accumulation 和每步实际 token 数；
6. optimizer、learning rate、scheduler、warmup、epoch/step；
7. 量化格式、compute dtype、double quantization；
8. seed、数据顺序、checkpoint 选择和生成参数。

只要上述任一项不同，就不能把结果差异直接归因于框架。

## 8. 当前项目选型结论

- 数据：继续使用本仓库的透明小数据管线；增加近重复、PII、授权和来源审计。
- 训练：Transformers + PEFT + TRL 为论文主实现。
- 快速复核：LLaMA-Factory 与 ms-swift 二选一，取决于最终基座模型。
- 基线：zero-shot、few-shot prompting、LoRA、QLoRA，第二种 PEFT 选 DoRA 或 IA³。
- 评估：自定义风格/内容/流畅度 rubric + 人工盲评 + 通用能力基准。

## 9. 仓库审计字段

每个实际使用的仓库记录：URL、commit SHA、许可证、访问日期、依赖版本、运行命令、配置文件、GPU、峰值显存、运行时长和本地修改。GitHub star 只能反映热度，不能作为方法选择依据。

