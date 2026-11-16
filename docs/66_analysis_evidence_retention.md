# 分析证据优先保存：无需永久保留1125套完整检查点

2026-11-16，所有者明确本轮目标是得到足够分析的数据，不要求全量永久保留checkpoint。本补充改变65号的保存/复审方式，不改五Rank×五Alpha×三精度×三模型×五种子1125训练、数据、128步/8次dev选择或400test科学口径。新控制器scripts/cloud/run_research_analysis.sh是当前新批次推荐入口。原全量保留入口与所有旧结果/协议/源码保持不变。

## 1. 保留与释放的内容

|内容|策略|
|---|---|
|训练中、明确暂停、失败的run|保留原状态供定位/恢复，不清理|
|已完成run的最佳/最终checkpoint|先实际核验完整适配器/optimizer/scheduler/RNG/trainer状态和SHA，再清理指定大二进制|
|每条件全部400条test预测、原ID/reference|保留，用于指标复算、错误分析和后续预先设计的统计|
|八次完整dev loss、128步训练曲线、最佳步数、checkpoint metadata|保留，用于选模核对、拟合与稳定性分析|
|全部配置、输入/模型/源码哈希、命令、阶段收据、会话与资源记录|保留，记录原二进制曾在清理前通过验证|
|导出adapter权重|固定保留r16/a32/seed20261113的3模型×3精度共9套，其余评价/审计成功后释放|
|原始官方基座、数据、旧4B与历史实验、r8/r64诊断|不清理|

精确删除范围仅新控制器自建run内checkpoints/checkpoint-N/{adapter_model.safetensors,optimizer.pt,scheduler.pt,rng_state.pth}，以及不属于9套预定代表的adapter/adapter_model.safetensors。不是递归删除整个目录；JSON metadata、tokenizer、预测与日志留下。保存的plan.json列出每个删除文件的原SHA/字节和所有保留文件SHA；complete.json绑定该清单与工具源码。

保留9套是预先确定的复现实例，不是看test选赢家。之后如果要对任何其他条件追加新测试/生成或继续训练，可能需要重新训练；不能用保留的哈希恢复被释放权重，也不能声称清理后仍可逐个进行原二进制文件的重新审计。原全状态审计是清理前真实执行的历史事实，清理后复核的是保留下来的元数据和分析证据。

## 2. 执行和完成判定

每条件顺序固定：训练→完整test生成→独立full-state/score审计→保留快照与精确清理，随后开始下一个条件。清理有额外CPU阶段，故模型实验仍1125次、生成1137组；控制阶段总数由3388增为4513。清理不参与训练优化或test选模，不影响科学参数矩阵。

原生训练/评价/审计仍使用scripts/train_research_grid.py、evaluate_research_grid.py、audit_research_grid.py和相同1125配置，源码未改。科学run/config/probe仍在experiments/research_banking_v2；新控制器的manifest/receipts/attempts/origins/汇总单独在experiments/research_banking_analysis_v1。共享原v2操作系统流程锁和GPU锁，不允许两个入口同时运行。

只有新控制器在训练前记录过ownership、全部train/test/audit收据匹配、run_status=completed_verified且原始完整审计重算一致，才能写清理意图并释放登记文件。任何阶段失败或未评完都不清。删除前所有目标先校验，拒绝路径越界/符号链接/变更文件；只逐个删除白名单文件，不触碰models/data、失败证据或已有历史run。

清理后控制器只允许“已完成清理收据中原SHA吻合的明确删除文件”缺失；其他缺失/篡改仍报错。之后每次report重核保留文件、session收据、128步/八次dev选择、完整400预测与严格JSON指标，报告evidence_level明确不能重放已删除二进制。全部4513阶段通过时状态complete_grid1125_analysis_evidence_verified，不使用旧全量保留完成标签。

## 3. 磁盘需求

不再累加1125份adapter+optimizer训练状态。空间主要为：官方基座/数据、原诊断、保留分析文件与tokenizer、9份代表adapter，以及一个正在运行的条件。估算每个待完成条件64MiB分析/JSON/tokenizer余量，加代表adapter、52×所选最大LoRA参数的活动/保存余量和10GiB固定保留。

这是保守预测而非保证；实际空间以prepare/训练前disk guard输出为准。通常远小于此前TB级全状态累加，但不保证任意150G机器都够；已有缓存、其他实验和失败现场仍占空间。空间不足时停止，不删除未完成/旧实验，不静默减数据或条件。

## 4. 新批次启动

尚未执行旧run时，可在GPU空闲、原CPUv3/九项r8诊断保留的云端直接使用：

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
bash scripts/cloud/run_research_analysis.sh plan
bash scripts/cloud/run_research_analysis.sh prepare
bash scripts/cloud/run_research_analysis.sh probe
bash scripts/cloud/run_research_analysis.sh run
```

新probe复用原r64探针目录；已完整通过且SHA一致的探针会复核复用，未通过的残缺证据不会自动重试。prepare仅CPU，probe/run才GPU。可沿用--model/--mode/--rank分块，每块25训练；全局manifest创建后的继续须--resume。新run会自动逐条件完成清理，CLI选用本分析入口即采用analysis_retention.json所列策略，不另产生人工逐run确认。

如果已用旧run_research_grid.sh启动，先不要更新运行中源码或手动删文件。旧完整保留manifest存在时，新控制器明确拒绝自动迁移/清理；先用原入口pause并保存状态，再按实际进度另做来源感知迁移，不重跑已完成条件。本次没有该云端状态，所以不假设旧任务未启动，也不删除或覆盖它。

```bash
cat experiments/research_banking_analysis_v1/execution_state.json
bash scripts/cloud/run_research_analysis.sh pause
bash scripts/cloud/run_research_analysis.sh report
cat experiments/research_banking_analysis_v1/progress.md
```

pause在训练已初始化后写入原生per-run请求，于下一个16步保存；如恰处于训练启动竞态，可能完成当前训练阶段后暂停。生成完成400条再停，清理完成其登记事务再停。必须等已核验暂停和进程退出，不能发出pause就关机。训练中恢复仍用同一分析入口run --resume --checkpoint 状态给出的完整路径。

## 5. 清理中断

清理意图先写plan.json，原始分析证据始终保留。中断后不能装作清理完成，也不能自动再运行GPU。排查后可显式：

```bash
# 将路径换为失败条件的实际config路径
.venv-cloud8b/bin/python scripts/compact_research_grid.py   --config experiments/research_banking_v2/configs/实际条件.json   --finish-cleanup --execute
# 随后在相同选择下确认CPU清理阶段收据并继续，训练不会自动重试
bash scripts/cloud/run_research_analysis.sh run --resume --finish-cleanup
```

只允许完成原登记的同一CPU清理事务，不能用该参数重试失败训练/生成。旧失败attempt和操作清单保留。未完成清理前，部分检查点二进制可能已按计划释放，不能再宣称可从该条件checkpoint恢复训练；重新分析则仍以保留预测/曲线为准。

## 6. 当前状态

本地仅CPU临时夹具测试清理、篡改拒绝、代表适配器保留、越界保护和中断恢复，没有删除实际项目文件，没有云端GPU或真实清理成功回报。新入口不降低原始训练/评价审计：清理前验原文件，清理后如实限定复审能力。20/24/46及LaMP原始corpus_score实现与历史证据不变，代码许可证未指定。
