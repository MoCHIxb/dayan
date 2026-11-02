# GitHub 典型项目收集与选型

检索更新时间：2026-10-12。本页保留核心项目速查；包含数据、标注、评估和实验管理在内的 28 个项目完整矩阵见 [`05_github_project_matrix.md`](05_github_project_matrix.md)。以下均链接到项目官方仓库。星标数和接口会变化，因此本表关注定位与可复现用途，不以热度代替技术判断。

## 推荐项目

| 项目 | 定位 | 适合本课题的用途 | 学习成本 |
|---|---|---|---|
| [Hugging Face PEFT](https://github.com/huggingface/peft) | PEFT 算法库 | 理解并直接控制 LoRA/QLoRA、AdaLoRA、DoRA 等适配器 | 中 |
| [Hugging Face TRL](https://github.com/huggingface/trl) | SFT/偏好优化训练器 | 用 `SFTTrainer` 建立透明、可编程的研究基线 | 中 |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) | 多模型训练 CLI/WebUI | 快速跑通 Qwen/Llama 的 LoRA/QLoRA 与批量实验 | 低 |
| [ms-swift](https://github.com/modelscope/ms-swift) | ModelScope 训练部署框架 | 中文生态、Qwen 系列、训练到部署一体化 | 低—中 |
| [Unsloth](https://github.com/unslothai/unsloth) | 单卡训练优化 | 显存非常紧张时做快速 LoRA/QLoRA 原型 | 低—中 |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | YAML 驱动训练框架 | 大量消融、分布式扩展、配置复现 | 中—高 |
| [QLoRA 官方实现](https://github.com/artidoro/qlora) | 论文参考代码 | 对照论文细节、理解 NF4 与原始实验 | 中 |
| [Microsoft LoRA](https://github.com/microsoft/LoRA) | LoRA 原始实现 | 理解原始算法和早期实现；不建议作为当前主训练框架 | 中 |
| [FastChat LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) | MT-Bench/模型裁判 | 借鉴双盲、位置交换和裁判提示设计 | 中 |

## 选型结论

建议采用“两层实现”：

1. 主实验使用 Transformers + PEFT + TRL。代码路径短、变量透明，便于论文解释和精确消融。
2. 工程复核使用 LLaMA-Factory 或 ms-swift。在同一数据上复现关键结论，验证结果不是自写脚本偶然造成。

若只有单张 8–16GB 显卡，可把 Unsloth 作为训练加速后端；若计划多机、多卡或复杂批量实验，再考虑 Axolotl。不要同时以多个框架作为主线，否则配置语义、chat template 和 loss mask 差异会让结论难以比较。

## 项目阅读顺序

1. 在 PEFT 查看 `LoraConfig`、目标模块选择、适配器保存/合并。
2. 在 TRL 查看 `SFTTrainer` 接受的 conversational 与 prompt-completion 数据格式，以及 completion-only loss。
3. 用 LLaMA-Factory WebUI 跑一次极小模型 Demo，认识训练配置和日志。
4. 对照 QLoRA 官方实现理解 4-bit 加载、NF4、double quantization 和计算 dtype。
5. 最后再研究 Unsloth/Axolotl 的性能优化，避免把框架特性误当作算法贡献。

## 复现实验审计表

收集每个仓库时记录以下字段：

- 仓库 URL、commit SHA、许可证、访问日期；
- Python、PyTorch、CUDA、Transformers、PEFT、TRL 版本；
- 基座模型精确 ID 与 revision；
- tokenizer 和 chat template；
- 数据格式、loss mask、截断/packing 策略；
- LoRA target modules、`r`、`alpha`、dropout；
- 量化格式、计算 dtype、优化器；
- GPU 型号、峰值显存、wall-clock 时间；
- seed、验证集选择、checkpoint 选择规则。

## 不建议直接照搬的做法

- 只因 README 声称“更快/更省显存”就跨框架比较，未统一模型、序列长度和 batch token 数。
- 用模型名里的 `4bit` 代替完整量化配置说明。
- 把训练 loss 最低的 checkpoint 当作风格质量最好的 checkpoint。
- 合并 LoRA 后才做所有评估，且不保留未合并适配器与配置。
- 直接复制公开人物文本并发布训练数据，而未核验许可证与人格/隐私风险。

