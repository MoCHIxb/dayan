# 三模型CPU通过后的独立GPU功能验证

2026-11-16。用户v3总状态cpu_contracts_passed_gpu_unverified，Qwen/Phi原生与Mistral显式视图已通过CPU。资产与模板不是实际GPU训练验证。此入口只用已有BANKING前两条真实train记录，每条件一步诊断更新、独立新进程恢复后再一步及一条训练prompt生成；没有test评分，不进入正式实验统计。

## 执行顺序

确认当前预检/下载进程已退出，GPU空闲。保留v2/v3报告，不再改模型或数据。入口只认既有v3路径，绑定registry/数据manifest/各资产锁与消息视图/源码；版本不同应先重新诊断，不手动改report。

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
source .venv-cloud8b/bin/activate
unset CUDA_VISIBLE_DEVICES
nvidia-smi
bash scripts/cloud/gpu_probe_research.sh
# 上一条仅显示九条件计划；明确执行：
bash scripts/cloud/gpu_probe_research.sh --execute
```

默认Qwen、Mistral、Phi各依次BF16、NF4、int8，共9条件；每条件两个独立GPU子进程（train/restore），前者退出释放显存后才加载后者。协调进程不创建CUDA上下文，不并发占卡。也可先只执行一个明确条件：

```bash
# 单独选条件时不要同时启动默认全矩阵；已经完成的证据不会自动跳过。
# bash scripts/cloud/gpu_probe_research.sh --execute --model qwen3_8b --mode nf4
```

单条件成功不等于全部9条件通过。目录已存在即拒绝覆盖或重复运行；失败保留完整日志/状态，不自动重试或换精度。若中断/失败，先诊断，不删除证据再全跑。当前入口无自动resume/重试，后续恢复须明确方案；未开始的其他条件可在诊断确认后用--model/--mode显式单选，不把失败条件偷偷排除。

## 验证范围

- 使用冻结加载函数的BF16/NF4/int8路径，七投影或Phi融合投影，r8/alpha16、AdamW、实际原生模板和相同message_view、assistant-only监督，无合成训练记录。
- 两条真实train分别前后向、累积一次优化器更新；检查有限loss、有限且非零LoRA梯度，保存适配器、optimizer、scheduler、torch CPU/CUDA RNG、Python/NumPy RNG及步数1。checkpoint文件全部哈希收据保留。
- 新进程重新加载同一精度基座与适配器，先逐文件校验再恢复优化器/scheduler/RNG，验证step1并推进至step2，再用原真实train prompt做greedy生成、最多64token。生成仅用于接口检查，不解析成正式分类成绩。
- 每条件记录allocated/reserved峰值，保留18个阶段日志及各自输入/完成收据。loss低、格式偶然正确都不构成效果结论。
- 当前只检查实际两条输入长度，不证明1536极限长度/所有正式训练条件显存均可容纳；不比较恢复与不中断训练的数值等价。恢复后的第二步不另导出正式模型，不用于后续test。

结果在experiments/cloud_8b_4090/gpu_probe_v1。协调状态：

```bash
cat experiments/cloud_8b_4090/gpu_probe_v1/execution_state.json
# Qwen BF16首阶段示例日志：
tail -n 30 experiments/cloud_8b_4090/gpu_probe_v1/qwen3_8b_none/train.stderr.log
```

失效、OOM、量化kernel或保存恢复失败会停并写failed_preserved_no_auto_retry。CLI进程退出不等于模型任务完成，需核对对应complete.json及train/restore收据、源码/模型/数据哈希、进程释放。成功状态selected_gpu_probes_complete_not_formal_experiments表示所选诊断完成，all_conditions才说明是否覆盖9条件，training_ready仍为false。

三精度诊断通过后，才可固定正式跨模型实验协议、128步/完整dev/test/选择规则与统一评价执行器。当前不是42次或旧82次队列；不要拿诊断输出代替正式训练或补进结果表。用户云端执行，助手本地仅测试监督mask/边界/不覆盖与有限矩阵逻辑，未本地加载模型或执行GPU。
