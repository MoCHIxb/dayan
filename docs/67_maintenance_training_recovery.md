# 服务器检修紧急中断：一次性训练恢复

2026-11-19，所有者明确中断原因是服务器检修紧急停机，并要求重启实验。终端记录停在qwen3_8b_none_r64_a16_s20261205_train、attempt1，最后progress step88、01:32:32；GPU空闲且未见相关进程输出，状态仍running为旧记录。最近预期保存点80需实际核验，不把日志88当成可恢复状态，也不重跑1125次。

新增独立scripts/recover_research_maintenance.py；不修改冻结的训练/评价/审计、65/66协议、grid或既有收据。此次更新只增加恢复工具/测试/文档及状态，原manifest输入和源码字节保持一致；工具会再次比较当前inputs与原manifest，不一致则停止。

## 操作

确认当前没有旧任务继续运行，在原目录/原venv中执行：

```bash
cd /root/autodl-tmp/dayan &&
git -c http.version=HTTP/1.1 pull --ff-only &&
export PYTHONPATH="$PWD/src:$PWD/scripts" &&
export PYTHONDONTWRITEBYTECODE=1 &&
.venv-cloud8b/bin/python scripts/recover_research_maintenance.py
```

默认只读、无GPU启动；核验模型/环境/数据/源码、origin、保存点的adapter/optimizer/scheduler/trainer/RNG及收据SHA。打印restore_step/restore_checkpoint。若没有完整状态、已存在训练摘要或完整128保存点，则拒绝盲目续训；后两种需做专门finalization审计。

默认核验通过后，以下命令显式授权一次检修恢复，并在成功后继续原队列：

```bash
.venv-cloud8b/bin/python scripts/recover_research_maintenance.py --execute &&
bash scripts/cloud/run_research_analysis.sh run --resume
```

执行前持有原流程锁、检查活进程和GPU占用、重新核验冻结输入及所选r64探针、核对临时保存空间。恢复仅当前中断训练，继续原128步预算，仍完整8次dev选模与400test，源训练实现不变。每个已完成条件按原收据跳过，不重训。

## 保留和衔接

恢复资料在experiments/research_banking_analysis_v1/recoveries/<active_stage>/，包含plan.json、完整原progress副本及SHA、被更早有效保存点排除的后续残缺checkpoint目录、实际恢复命令与日志、完成事件和cost_caveat.json。任何移动前确认原checkpoint目录和目标恢复证据目录均在指定实验范围内；不递归删除文件，不清理任何失败证据。

原progress先无损备份，再将活动曲线恢复到保存点的真实dev历史，避免后续重复评价或未保存尾部被当作恢复后的新进度。尾部如81–88保存在原始副本，随后在恢复计算中重算。原state不是手改成paused；用maintenance_recovery_running/maintenance_training_recovered记录真实操作。存在恢复attempt目录即禁止自动再次执行；出错保留现场，需诊断，不无限重试。

原生Trainer从已核验的完整保存点恢复optimizer/scheduler/RNG/数据位置等，不能承诺跨维护硬件/驱动环境与完全不中断逐位相等。恢复到128后核验最终与最佳全状态、八次dev与导出SHA，再新增此前不存在的train阶段收据，保存原命令和实际resume命令，不覆盖旧收据。

training_sessions保留已结束的正常暂停事件和本次已测恢复会话，另明确记录未测中断会话。run/maintenance_recovery.json和recoveries/cost_caveat.json标记full_training_cost_incomplete。原始汇总的会话累计不包含未知中断成本；该条件不得直接用于完整训练耗时/峰值横向比较，需排除资源计时或带明确限定报告。质量指标仍待完整原预算和dev/test审计后才能使用。

## 监控

恢复单条件期间：

```bash
tail -n 20 experiments/research_banking_analysis_v1/recoveries/qwen3_8b_none_r64_a16_s20261205_train/stderr.log
cat experiments/research_banking_analysis_v1/execution_state.json
```

恢复成功后，原队列仍将执行该条件的完整test、独立审计和分析保留，不把train恢复到128当成该条件全部完成。不中途改代码/环境或重复执行run。若恢复本身失败，先提供recovery日志，不删除恢复目录。

本地仅6项CPU单元测试与全套回归通过；未连接服务器、未验证云端80步实际文件、未本地启动GPU。用户在云端运行命令后才更新恢复成功状态。
