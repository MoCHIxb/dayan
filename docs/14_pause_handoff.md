# 项目暂停与恢复记录

## 当前最高优先级：立即暂停并交接（2026-10-28 22:54）

用户已覆盖先前“完成当前后暂停”的安排，要求立即停止、清理未完成实验，并转到新对话。评价与等待程序均已停止；未完成Qwen3-4B基座通用能力评价最后为364/1860条生成，没有正式分数，当前日志及失效等待程序/PID已清理。已有正式实验、完整dev、检索结果、基准数据及可复用代码保留。实时状态 `experiments/confirmation/state.json=paused_by_user`，清理明细 `cleanup_20261028.json`。没有本项目周期自动检查。

**新对话首先阅读 `docs/17_new_chat_handoff.md`，该文档包含完整目标、结果、限制、用户偏好和建议下一步。不要运行下面历史记录中的等待程序、旧PID命令或大扩展队列。**

## 最新指令：完成当前评价后暂停（2026-10-28 22:45）

当前 Qwen3-4B 基座通用能力评价仍在运行，实际worker PID=23716、包装PID=13980；队列派发进程25228/9328已停止，后续六次新训练不会自动开始。单次等待程序 `finish_current_and_pause.py` 包装PID=30444，只负责当前评价结束后的完整性核验和暂停状态落盘。原队列锁已归档为 `gpu_queue.lock.archived_finish_current_20261028`；这不表示当前GPU空闲，必须检查当前评价worker和实时状态后再行动。

请以 `experiments/confirmation/state.json` 为准。当前为 `finishing_current_then_pause`，结束后应为 `paused_after_current_completed`；若结果不完整则记录 `paused_current_evaluation_incomplete`，不自动重启。等待程序日志为 `finish_current_then_pause.stdout.log`、`.stderr.log`。用户要求重新围绕计划书三个核心方向讨论后续优先级，人工评分继续暂缓；不要自动恢复旧扩展队列或创建周期任务。

## 当前活动阶段：2026-10-28 完整研究补充

用户已授权补齐六项局限；单次队列 `scripts/run_confirmation_sequence.py` 正在运行，包装进程 PID=9328，实际进程见 `experiments/confirmation/gpu_queue.lock`。当前任务状态和日志路径见 `experiments/confirmation/state.json`；主日志 `queue.resume_after_checkpoint_audit.stdout.log`/`.stderr.log`。完整协议见 `docs/15_confirmation_protocol.md`。只有一个 GPU 子任务，异常时停止。若用户要求再次暂停，先停止该队列父进程，再核对并停止实际训练/评价子进程，保留 checkpoint、输出和日志；不要只停父进程后让子进程继续。

历史第二轮权重缺失引起的首次尝试已保留日志，现已修正为六次完整 dev 新训练；不能把该历史失败当成当前队列仍失败。旧量化批处理已完成，不要重复启动；本补充阶段尚未创建新周期任务。人工评分包位于 `experiments/blind_review_200/for_raters/`，只有两份真实独立评分返回后才运行人工分析。下面量化完成、暂停和恢复内容为历史记录。

## 最新里程碑：2026-10-28 量化批处理已全部完成

int8、NF4 训练与八组完整测试、BERTScore、错误分析、各 2,000 次原始配对 bootstrap 已完成并核验，实时状态为 complete，批处理锁已正常移除。实际结果已写入各自运行目录 `quantization_result.json`，以及 `docs/09_full_scale_qwen3_4b_experiment.md` 和 `docs/13_quantization_experiment.md`。本轮自动检查的目标已达成，停止此监测。以下暂停和恢复指引为历史记录；以后不要重新运行本批处理，也不要重复训练量化配置。下一研究阶段为既定模型规模、检索与多种子/综合评价。

## 最新状态：2026-10-28 已恢复实验

收到用户“继续未完成的工作”后，重新核验 int8 两轮训练、1,306 步、252 个量化 projection、两轮 dev 和最终适配器/最佳 checkpoint 哈希，均通过。base_no_profile 的 1,496 条完整预测已核验 ID、参考答案、量化方式、模型、解码参数及无适配器条件，直接复用。

恢复入口为 `scripts/run_quantization_sequence.py --resume-evaluation`。该入口只复用完整且协议一致的已完成评价，缺少预测/指标任一文件时停止并要求检查，不覆盖产物。中断的 base_profile 使用新的 `base_profile.eval.resume-*.log` 重新生成全部 1,496 条，旧中断日志保留。接下来串行完成两组适配器评价、原定 BERTScore、错误分析和 2,000 次原始语料级 bootstrap，再训练与评价 NF4。没有重训 int8，没有降低数据规模或评价标准。

17 项单元测试通过，新增恢复检查覆盖禁止重算已完成条件、拒绝量化协议不符和拒绝不完整成对产物。启动前宿主进程检查无 Python worker，磁盘约 75 GiB 可用。一次性批处理于 13:37 启动，包装进程 PID=5904，实际 worker 见 `experiments/quantization_sequence.lock`。新批处理日志为 `experiments/quantization_sequence.resume_20261028.stdout.log` 和 `.stderr.log`，实时阶段仍为 `experiments/quantization_sequence_state.json`。

2026-10-28 随后收到用户明确授权，已将既有半小时自动检查 `qwen` 恢复为 ACTIVE；仅阶段完成、异常或需要用户处理时通知。检查使用本次 resume 日志，不重复启动任务。下面的暂停记录为历史信息。

## 最新暂停：2026-10-27 关机前

用户明确要求暂停自动检查、暂停并保存实验，然后关闭电脑。半小时 heartbeat `qwen` 已通过应用工具设置为 PAUSED。已停止单次批处理 PID 265640/276884 和 int8 base_profile 评价 PID 279416/270272；随后系统只读进程检查没有 Python worker 遗留，GPU 显存降至约 1.2 GiB 的桌面占用。没有删除实验产物。

### 此次已完成并保存

- int8 全量 10,437 条、两轮、1,306 步训练完成。dev loss 为 1.7498958111 / 1.7546252012，最佳为第 1 轮 checkpoint-653。耗时 15,231.91 秒，峰值分配显存 9,711,863,808 bytes；实际量化 projection 为 252 个。
- 最终适配器与 checkpoint-653 的 SHA256 一致：`f1c16c240dd6730f1b95c0c9ac752a86b23347b13ae0884ba6793196fb2987a7`。checkpoint-653、checkpoint-1306、最终 adapter、训练摘要与日志全部保留，关机前又单独保存了哈希记录。
- int8 base_no_profile 已完成全部 1,496 条，指标与预测均保存：BLEU 11.979229，chrF 37.543578，ROUGE-L 0.411877，无空输出。文件在 `experiments/runs/qwen3-4b-lamp7-full-lora-r16-a32-int8/base_no_profile_test_metrics.json` 与同名 `.predictions.jsonl`。
- 此前全部 BF16 容量、alpha、六档数据规模实验及正式结果保持完整。

### 中断位置与恢复顺序

int8 base_profile 最后日志为 `generated 1200/1496`。当前评价脚本在整组生成结束后才写预测，因此这组没有落盘预测，不能从第 1,201 条继续。日志已保留，恢复时从头重跑且只重跑这一组，随后运行 adapter_no_profile、adapter_profile，各完整 1,496 条；再进行原定 BERTScore、错误分析、2,000 次原始 corpus_score 配对 bootstrap，核验并写入正式报告，最后执行 NF4 全量训练与评价。NF4 尚未启动。

**不要重训 int8，也不要直接重新运行旧 `run_quantization_sequence.py --training-pid ...`：旧 PID 已退出，而且该单次脚本会因已有完整评价或日志而拒绝覆盖。** 继续前先核对 GPU/进程与现有产物，采用识别已完成条件的恢复入口或逐条件执行；保留此次中断日志，在新日志文件中记录重跑。不能把缺失评价标成完成。

状态写入 `experiments/quantization_sequence_state.json`，stage=`paused_by_user`；旧运行状态另存 `quantization_sequence_state.before_pause_20261027.json`。批处理锁已归档为 `quantization_sequence.lock.paused_20261027`，未删除，旧 PID 文件只作为历史记录。关机前进程快照为 `experiments/pause_20261027_process_snapshot.json`，适配器哈希为 `experiments/pause_20261027_adapter_hashes.json`。

等待用户明确继续，不自动重启实验，也不自动恢复半小时检查。

## 2026-10-26 暂停及 2026-10-27 恢复历史

记录日期：2026-10-26。状态：用户明确要求暂停，等待下一次继续指令。没有后台自动续跑任务。

状态更新（2026-10-27）：已收到“请继续工作”，现已恢复原配置全量 int8 训练；本文件下述暂停状态为历史记录。当前运行与日志入口见 `docs/13_quantization_experiment.md`，实时阶段为 `experiments/quantization_sequence_state.json`。另有单次串行批处理等待训练完成后执行评价及 NF4，不是周期任务。若再次暂停，须同时停止该批处理与当时实际训练/评价子进程，避免只停父进程后仍续跑；保留 checkpoint、状态和所有产物。

## 已完成并保存在磁盘的工作

- 论文与 GitHub 项目调研，以及 LaMP-7 数据清洗、检索画像、指令格式转换和数据检查。
- Qwen3-4B 全量 rank 8/16/32 容量消融，固定 rank 16 的 alpha 16/32/64 缩放对照，均已完成训练及既定全测试评价。
- 50/100/200/500/2,000/10,437 条数据规模曲线：六档训练、两种完整 1,496 条 test 条件、BERTScore、分层错误分析和各小子集相对全量的 2,000 次原始语料级配对 bootstrap 均已完成。
- 六档统一检查通过：配置、子集嵌套、记录数、两轮 dev、最佳适配器哈希、测试 ID 与参考答案对齐均通过。机器汇总为 `experiments/data_scale_summary.json`；曲线图为 `experiments/figures/data_scale_test_metrics.png`、`experiments/figures/data_scale_dev_loss.png`。
- 正式报告与计划见 `docs/09_full_scale_qwen3_4b_experiment.md`、`docs/10_controlled_ablation_plan.md`；数据规模详情见 `docs/12_data_scale_experiment.md`。

## 暂停时的量化阶段

已安装 bitsandbytes 0.50.2 并锁定版本，保留原 PyTorch/Transformers/PEFT 版本。共享加载器 `src/personal_llm_data/model_loading.py` 支持 none/int8/nf4，训练与评价入口已接入。12 项单元测试与 pip check 均通过；CUDA 训练兼容性仍待正式训练记录验证。

int8 配置为 `configs/qwen3_4b_lamp7_full_lora_r16_int8.json`，NF4 配置为 `configs/qwen3_4b_lamp7_full_lora_r16_nf4.json`。int8 于当日 23:13 启动；收到暂停指令后向训练会话发送中断，会话退出码为 1。随后通过系统进程只读检查确认无 Python worker，GPU 已释放。

int8 目录 `experiments/runs/qwen3-4b-lamp7-full-lora-r16-a32-int8/` 已保留：`checkpoints/` 为空、无 `training_summary.json`、无最终适配器；两份日志文件为 0 bytes。没有可恢复 checkpoint，末步数未知，未落盘更新不能恢复；不能把该轮记作完成。NF4 尚未启动。已完成的 BF16 实验结果、预测、适配器和最佳 checkpoint 未删除。

## 收到继续指令后

1. 先检查 Python/GPU 进程和 int8 目录，避免重复任务，并确认是否有新的摘要或 checkpoint。
2. 当前无可恢复点，需从头重新运行同一 int8 配置；先保证标准输出和错误日志实时落盘。不要对空 checkpoint 目录执行 resume，不修改既有正式结果。
3. int8 完成后核验训练记录、两轮 dev、最佳 checkpoint 与最终适配器哈希、量化层数及资源统计；评价同精度的基座/适配器×无画像/有画像四组，补充 BERTScore 与原始配对 bootstrap，再运行 NF4。
4. 之后继续既定模型规模、检索、完整 dev、多训练种子、人工/LLM 评价和通用能力保持实验。只允许一个 GPU 任务；未经用户重新授权，不创建周期自动化。
