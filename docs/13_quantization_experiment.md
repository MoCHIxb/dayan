# Qwen3-4B 全量量化对照

更新日期：2026-10-28 18:57。int8 与 NF4 全量两轮训练、各四组 1,496 条测试、BERTScore、错误分析及各自 2,000 次原始语料级配对 bootstrap 全部完成。最终检查通过：控制配置、训练/dev 记录数、两轮 dev、1,306 步、252 个量化 projection、最佳适配器哈希、八组测试 ID/参考答案与解码条件、BERTScore 协议、重采样数均符合约定。状态为 complete，正式报告已追加两种模式的实际结果。本轮持续检查完成后停止。以下先前运行叙述为历史记录。

## 固定设计

比较 BF16 LoRA、LLM.int8 LoRA 和 NF4 LoRA。固定 Qwen3-4B、10,437 条 BM25-k12 训练记录、500 条固定 dev、种子 20251206、rank 16、alpha 32、学习率 1e-4、有效 batch 16、两轮、max_length 768、全部注意力与 MLP projection、AdamW 优化器和余弦调度。BF16 复用既有正式基线。量化新增配置仅改变 quantization 与 output_dir。

8-bit 使用 `load_in_8bit=True`、outlier threshold 6.0。NF4 使用 4-bit NormalFloat、double quantization 和 BF16 compute。训练前采用 PEFT 的 `prepare_model_for_kbit_training`，冻结主干并启用非重入梯度检查点。该官方准备函数还会把未量化的浮点参数转换为 FP32，因此本实验比较的是实际可复现的量化训练配方，不能把差异全部归为位宽本身；不额外更换为 paged/8-bit 优化器。

实现参考 [Transformers 4.57.1 bitsandbytes 文档](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes) 与 [PEFT 量化指南](https://huggingface.co/docs/peft/developer_guides/quantization)。项目安装版本为 Transformers 4.57.6、PEFT 0.21.2、PyTorch 2.11.0+cu128、bitsandbytes 0.50.2。后者已加入 `requirements-train.lock`，依赖检查通过。

## 训练与评价入口

共享加载实现在 `src/personal_llm_data/model_loading.py`，支持 none/int8/nf4，量化模式强制使用同一张 CUDA GPU，并检查实际量化标志以拒绝静默回退。训练入口沿用 `scripts/train_qwen_lora.py`；两份配置为 `configs/qwen3_4b_lamp7_full_lora_r16_int8.json` 与 `configs/qwen3_4b_lamp7_full_lora_r16_nf4.json`。输出目录分别为 `experiments/runs/qwen3-4b-lamp7-full-lora-r16-a32-int8/` 与 `experiments/runs/qwen3-4b-lamp7-full-lora-r16-a32-nf4/`。

评价入口 `scripts/evaluate_generation.py --quantization int8` 或 `--quantization nf4`，记录量化方式、模型、输入/生成长度限制、确定性解码、峰值显存及单独同步计时的生成耗时。原有 elapsed 包含自动指标计算，不当作纯推理时间。

每个量化配置采用相同的 1,496 条完整 test。为区分量化基座和适配器作用，分别生成相同精度的“基座无画像/有画像”和“适配器无画像/有画像”四组预测；量化 2×2 分析不能把既有 BF16 基座预测直接当作同精度基座。另将量化适配器的绝对分数与 BF16 正式适配器比较，并报告 2,000 次原始语料级配对 bootstrap。BERTScore 沿用 roberta-large、IDF、baseline rescaling。

正式训练完成后核验两轮 dev、最佳 checkpoint、最终适配器哈希、量化线性层数量、资源统计及日志。任何时刻只运行一个 GPU 任务。量化模型的参数与冻结层表示发生变化，可训练参数量、分配显存、实际运行时间分别报告，不把权重位宽直接当作端到端显存缩减倍数。

## 已执行检查

12 项现有与新增单元测试通过；新增检查覆盖原 BF16 加载参数保持、int8 单卡映射、NF4/double-quant/BF16 compute 配置、无效模式拒绝和量化标志检查。`pip check` 通过。CUDA 量化训练兼容性需由随后正式训练的实际前向/反向验证；目前没有量化质量或资源结果可报告。

## 当前运行

### 2026-10-27 恢复

宿主时间 18:22 重新从头启动原定 int8 配置。启动前只读进程检查确认无 Python worker，GPU 空闲；磁盘约 77 GiB 可用。沙箱内后台启动未存活，确认没有遗留进程后在正常宿主环境启动，未形成重复训练。训练实际 worker PID=272352，启动包装进程 PID=273412。未改变模型、数据、rank、alpha、学习率、batch、epoch 或评价标准。

增加启动阶段 flush 输出和 Trainer `on_log` 回调，日志事件立即追加至 `training_progress.jsonl`；标准输出和错误分别写入 `train.stdout.log`、`train.stderr.log`。第 10 个优化步已落盘：loss=6.7118，grad_norm=15.14311695，epoch≈0.02。这验证目前训练已执行前向/反向，不能据此判断最终收敛或量化效果。bitsandbytes 报告 int8 matmul 输入转换为 FP16，保留原始日志；不把 BF16 配置误记为所有内核运算均为 BF16。

单次批处理入口 `scripts/run_quantization_sequence.py --training-pid 272352` 已启动，包装进程 PID=276884；实际进程 PID 写入 `experiments/quantization_sequence.lock`。它等待既有 worker 退出，先核验两轮 dev、1,306 个优化步、实际量化 projection 数、最佳 checkpoint 与最终适配器哈希，再顺序运行四组各 1,496 条预测、roberta-large/IDF/baseline-rescaling BERTScore、分层错误分析以及相对 BF16 有画像适配器的 2,000 次原始语料级配对 bootstrap。保持原 `paired_adapter_comparison.py` 的逐次 `corpus_score`，不做充分统计量向量化。

int8 全套完成后检查 NF4 目录和可用磁盘，再执行原定全量 NF4 训练与同样评价。命令串行执行；任一步非零退出、缺失产物、协议/哈希不符或目标结果已存在均停止，不覆盖结果、不自动重启。每种模式完成才写入运行目录 `training_audit.json`、`quantization_result.json` 并向正式报告追加实际结果。阶段状态写入 `experiments/quantization_sequence_state.json`，批处理日志为 `experiments/quantization_sequence.stdout.log`、`.stderr.log`。这是一批有限命令，没有恢复半小时周期任务。

恢复后 15 项测试通过，新增三项验证 int8 核验失败时不得启动 NF4、现有锁阻止重复批处理、已有日志阻止覆盖执行。两份量化配置与 BF16 正式基线除量化方式和输出目录外逐项一致，冻结预测的 1,496 条 ID 与参考答案对齐预检通过。测试使用工作区内临时目录，不接触正式运行状态。

### 2026-10-26 暂停历史

int8 使用原定全量 10,437 条、两轮共 1,306 步配置，启动命令已核验。本轮按用户暂停要求发送中断，训练会话退出码为 1；随后只读进程核验确认无 Python worker 遗留，GPU 使用已降回桌面占用水平。此退出为人为暂停，不是自然完成或已确认的 CUDA 故障。

运行目录保留。`checkpoints/` 当前为空，无最终适配器和 `training_summary.json`；`train.stdout.log`、`train.stderr.log` 存在但为 0 bytes，无法从中确认末步数，也不能据此认定前向/反向兼容性已完成验证。本次内存中的未保存更新无法恢复。下次获得继续指令后，先检查进程和目录变化，改进日志实时落盘后重新启动该配置；不得直接对现有空目录调用断点续训，不得删除已有正式结果。训练完成后核验两轮 dev 和最佳适配器，再生成相同 int8 精度的四组预测，NF4 随后执行。


## int8 全量对照完成（2026-10-28）

训练与评价均已完成核验。训练 10437 条，两轮 dev loss 为 [1.7498958110809326, 1.7546252012252808]，最佳 epoch=1.0，峰值分配显存 9711863808 bytes，训练耗时 15231.91 秒。量化 projection 数 252；最终适配器 SHA256=f1c16c240dd6730f1b95c0c9ac752a86b23347b13ae0884ba6793196fb2987a7。

同精度基座/适配器×无画像/有画像四组均使用完整 1,496 条冻结测试集。

| 条件 | BLEU | chrF | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| base_no_profile | 11.979229 | 37.543578 | 0.411877 | 0.394263 |
| base_profile | 14.879620 | 39.970736 | 0.433972 | 0.418308 |
| adapter_no_profile | 17.294621 | 41.162252 | 0.482709 | 0.450435 |
| adapter_profile | 24.544717 | 45.515969 | 0.518166 | 0.484864 |

有画像适配器相对 BF16 的原始语料级配对 bootstrap（2,000 次；候选−BF16）：

- bleu: -0.161852，95% CI [-0.401296, 0.072023]。
- chrf: -0.045985，95% CI [-0.342315, 0.230716]。
- rougeL: -0.001523，95% CI [-0.004411, 0.001214]。

机器结果：`experiments\runs\qwen3-4b-lamp7-full-lora-r16-a32-int8/quantization_result.json`。以上为单训练种子的名义逐项区间，未进行多重比较校正。量化配方包含 k-bit 准备函数的非量化参数转换，不能把差异完全归为位宽。


## nf4 全量对照完成（2026-10-28）

训练与评价均已完成核验。训练 10437 条，两轮 dev loss 为 [1.763455867767334, 1.7680071592330933]，最佳 epoch=1.0，峰值分配显存 6462965760 bytes，训练耗时 8719.39 秒。量化 projection 数 252；最终适配器 SHA256=3c085d53ee31f6430b0e8d12db640f6feaf517b349416ae92f64ac81c1c0e982。

同精度基座/适配器×无画像/有画像四组均使用完整 1,496 条冻结测试集。

| 条件 | BLEU | chrF | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| base_no_profile | 11.409053 | 36.886502 | 0.404012 | 0.384401 |
| base_profile | 14.510065 | 39.512524 | 0.433347 | 0.414129 |
| adapter_no_profile | 17.065337 | 41.066489 | 0.482655 | 0.450498 |
| adapter_profile | 24.350505 | 45.385832 | 0.518201 | 0.484787 |

有画像适配器相对 BF16 的原始语料级配对 bootstrap（2,000 次；候选−BF16）：

- bleu: -0.356064，95% CI [-0.691419, -0.049884]。
- chrf: -0.176121，95% CI [-0.479904, 0.112396]。
- rougeL: -0.001488，95% CI [-0.005031, 0.002042]。

机器结果：`experiments\runs\qwen3-4b-lamp7-full-lora-r16-a32-nf4/quantization_result.json`。以上为单训练种子的名义逐项区间，未进行多重比较校正。量化配方包含 k-bit 准备函数的非量化参数转换，不能把差异完全归为位宽。
