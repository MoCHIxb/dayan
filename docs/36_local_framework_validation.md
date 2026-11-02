# 本地框架真实运行验证

2026-11-02，Windows / RTX 5070 Ti 16GB / Qwen3-4B。结论：本轮数据核验、GPU互斥、LoRA训练、完整checkpoint暂停恢复、开发集选模、适配器导出、完整基座/适配器test、严格标签评分和独立推理均实际通过。聚合机器证据见[local_text_safety_validation.json](../results/local_text_safety_validation.json)。不是仅根据进程或状态文件判断。

## 配置与验收

使用[独立本地配置](../configs/text_safety_v1/qwen3_4b_local_verify.json)，150条train、60条dev、120条test；完整英文现成数据不改。BF16 LoRA，r8/alpha16，学习率1e-4，梯度累积16，长度1024、生成64，32个优化器步，每8步完整评价60条dev。默认云端128步/每16步配置没有修改；本轮是功能验收预算，不能与128步服务器运行当作同一训练条件比较。

实际核验：

- 真实第二进程在GPU锁占用时被拒绝，训练输出目录未创建；独立推理在锁占用时同样被拒绝，未加载模型或创建partial。
- 第8步常规保存后请求暂停，第9步完成额外完整dev评价并保存退出；适配器、优化器、scheduler、RNG、trainer_state及输入收据全部核验，私有复制保留暂停证据。随后从原第9步checkpoint恢复至32步。
- 优化器训练日志覆盖1–32步；完整dev评价在8、9、16、24、32步发生。最低dev loss为0.0524473898，选择第32步；最终32步完整状态及导出适配器与所选checkpoint的SHA256一致。
- 基座与适配器各生成全部120条test，逐条ID、顺序、参考标签、预测数量及输入/输出哈希均核对；从预测重新计算总体及三领域分数，与CPU评分结果一致。
- 从每个领域各取一条既有dev提示、移除参考assistant标签，验证独立infer接口；3条输出ID、JSON结构和收据哈希通过。这3条不是新增训练样本，也不计入正式test分数。
- 训练环境76项CPU测试通过；独立精简CPU环境71项通过、5项因可选训练依赖缺失跳过。训练/评价/推理进程均已退出，GPU保护再次通过，结束时显存约1133MiB（桌面占用），空闲约14863MiB。

暂停引入第9步额外dev评价，本轮没有证明与“不暂停的32步运行”数值一致。云端不要直接套用本轮模型分数来声称跨平台一致。

## 完整test结果

各领域Safe/Unsafe各20条，总体各60条；以下均为基座 → 适配器。无效JSON两者均为0。

| 集合 | 条数 | accuracy | macro-F1 | 不安全回答检出率 | 安全回答误报率 |
|---|---:|---:|---:|---:|---:|
| 总体 | 120 | 70.0% → 81.7% | 0.677 → 0.816 | 43.3% → 76.7% | 3.3% → 13.3% |
| 医学 | 40 | 50.0% → 70.0% | 0.333 → 0.697 | 0.0% → 60.0% | 0.0% → 20.0% |
| 金融 | 40 | 85.0% → 95.0% | 0.847 → 0.950 | 70.0% → 95.0% | 0.0% → 5.0% |
| 公共安全 | 40 | 75.0% → 80.0% | 0.744 → 0.799 | 60.0% → 75.0% | 10.0% → 15.0% |

总体正确数84/120 → 98/120；不安全回答检出26/60 → 46/60，安全回答误报2/60 → 8/60。适配器更符合这批源数据的安全判定，但更容易将安全回答判为不安全，不能只展示检出率改善。医学源任务是Unauthorized Expertise：基座在该子集把所有回答判Safe，适配后仍漏检8/20个Unsafe并误报4/20个Safe。医学不是临床事实正确性测试。

这是一种子、小规模平衡英文子集上的描述性改善，无显著性或最优超参数结论；120条问答来自113个问题组，不能假设全部独立。各源安全政策不同、每领域仅40条，不证明普遍强安全、金融合规、临床可靠性、中文能力或无遗忘。既有LaMP实验与原始corpus_score统计没有改动，本示例也没有以JSON分类分数替换它们。

## 资源与修复

可训练参数16,515,072，总参数4,038,983,168，约0.409%。恢复训练进程记录CUDA峰值allocated 8.80GiB、reserved 10.81GiB，session训练/保存/末尾核验约111.4秒。首次训练和恢复两个子进程合计墙钟193.9秒，包含各次加载/核验；不是从准备数据到所有评价结束的总耗时。CUDA allocator峰值不是全机显存占用。本轮证明此长度/批量/模型配置在这张16GB显卡上可运行，不承诺任意模型、文本长度或其他GPU都可运行。

真实预检发现Windows WDDM驱动将桌面图形进程也列入compute查询、全部标C+G。已增加WDDM XML与实际OS进程交叉检查：仅排除明确列出的桌面进程和已核查的受保护厂商安装路径；Python、模型服务、C-only或未知进程继续拒绝。实际名称须与XML一致，不能仅凭C+G放行。跨进程OS锁继续生效。它是操作保护而非安全隔离，仍须检查其他软件占用；Linux原有compute检查不放宽。训练依赖显式声明psutil；独立infer也使用相同GPU锁。

## 命令和证据位置

以下为本轮实际使用的接口顺序。运行目录现已完成，不要重复执行、覆盖或再从第9步恢复；新验证需先复制配置并选择新的output_dir。训练中请求暂停须从另一个终端发出；必须等待paused_checkpoint_verified和进程退出才能关机。继续前核对实际checkpoint，不从存在更晚证据的旧步续跑。

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m personal_llm_data.workflow train-personal --config configs/text_safety_v1/qwen3_4b_local_verify.json --execute
# 本轮曾在第8步checkpoint完整保存后，从另一个终端请求：
.venv\Scripts\python.exe -m personal_llm_data.workflow train-personal --config configs/text_safety_v1/qwen3_4b_local_verify.json --request-pause
# 第9步暂停完成并核验后，本轮曾执行：
.venv\Scripts\python.exe -m personal_llm_data.workflow train-personal --config configs/text_safety_v1/qwen3_4b_local_verify.json --execute --resume-from-checkpoint experiments/text_safety_v1/local_verify/run/checkpoints/checkpoint-9
.venv\Scripts\python.exe -m personal_llm_data.workflow evaluate-personal --config configs/text_safety_v1/qwen3_4b_local_verify.json --condition base --execute
.venv\Scripts\python.exe -m personal_llm_data.workflow evaluate-personal --config configs/text_safety_v1/qwen3_4b_local_verify.json --condition adapter --execute
.venv\Scripts\python.exe scripts/score_text_safety.py --data-dir examples/text_safety_v1/joint --predictions experiments/text_safety_v1/local_verify/run/evaluation/base_metrics.predictions.jsonl --output experiments/text_safety_v1/local_verify/base_scores.json
.venv\Scripts\python.exe scripts/score_text_safety.py --data-dir examples/text_safety_v1/joint --predictions experiments/text_safety_v1/local_verify/run/evaluation/adapter_metrics.predictions.jsonl --output experiments/text_safety_v1/local_verify/adapter_scores.json
```

私有证据根目录`experiments/text_safety_v1/local_verify/`：run内training_summary/progress/run_inputs、checkpoints、adapter、evaluation；logs与commands.json记录执行；pause_evidence/checkpoint-9保留暂停全状态；pause_verification.json、lock_negative.json、inference_predictions.jsonl及收据、final_audit.json记录末尾核验。常规Trainer轮换可能删除原checkpoint-9，所以保留的暂停副本是证据，不应拿它重跑已完成任务。

公开仅代码、配置、聚合分数和哈希，原始日志、模型、checkpoint及逐条预测不提交。下一步是在实际Linux/4090上准备固定环境和独立云端输出，按35号128步示例计划显式执行；服务器目前未执行。项目代码许可证仍待所有者决定。
