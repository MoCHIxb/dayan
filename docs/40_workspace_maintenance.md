# 工作区清理、证据保全与恢复记录

2026-11-03。所有者要求为RTX4090准备整理工作区，并明确确认约6.3GiB具体清理范围。现已完成，机器证据见[workspace_cleanup_20261103.json](../results/workspace_cleanup_20261103.json)。服务器和云端训练均未启动。

## 清理结果

- 清理2498个目录/文件目标，主要是再生bytecode缓存；按已移除目标减去新增归档计算，逻辑净回收6.30GiB。统计时D盘空闲25.14GiB。此为逻辑文件大小口径，不把整个磁盘其他活动造成的变化归因为清理。
- models活动区仅保留Qwen3-4B。停用Qwen3-1.7B与MiniLM目录移除前，配置、tokenizer、许可证、HF元数据全部ZIP归档并校验；权重先核对固定上游版本与原SHA256，后续可按版本重新下载。没有把权重内容打包进Git。
- 早期data/processed/lamp7与停止的lamp7_retrieval_confirmation已无损ZIP归档；保留当前lamp7_bm25_k12、SMS处理集和全部原始来源。正式研究数据未改样本、未改划分。
- 六份历史int8大日志采用gzip无损压缩，解压内容SHA与原字节一致，再移除原件；包含历史错误输出，未丢弃失败信息。
- 仅删除字节等同于规范目录的text_safety reproduction_check、SMS data_rebuilt副本，保存其逐文件SHA及对应规范路径。旧独立CPU测试环境和未使用的ARC/IFEval/GSM8K缓存、bytecode清理；活动Windows训练环境和NLTK资源保留。
- 一个pip wheel因Windows权限保护而保留，不重复强删、不修改ACL或获取所有权；这不影响已安装环境与云端准备。

在具体确认前，自动审批曾拒绝批量删除；明确范围获所有者确认后才执行。所有目标均解析为D:/dayan内的绝对路径，检查Git追踪与链接路径，操作逐项留痕。没有清理D:/dayan以外的系统/用户缓存。

## 清理后独立核验

| 项目 | 核验结果 |
|---|---|
| 冻结LaMP/PEFT资产 | 164项SHA全部匹配 |
| 小样本训练 | 16次，128步/8次完整dev/最佳适配器/最终状态保留 |
| PEFT训练及复用 | 12组，最佳权重及最终128步状态核验通过 |
| 原正式结果收据 | 159项输出SHA匹配 |
| 原始配对bootstrap | 54项输出保留，均1496条/2000次；未重新统计 |
| SMS与三领域例子 | 最佳/最终checkpoint、预测和收据SHA核验通过 |
| 三领域暂停 | 第9步完整暂停副本及收据保留、SHA匹配 |
| 私有无损归档 | 10份归档SHA匹配；生成时逐文件/解压后原字节SHA校验 |
| 真实恢复/跨平台准备 | 在临时目录验证真实MiniLM元数据归档恢复、真实SMS manifest的LF→冻结CRLF处理 |
| 测试 | 主环境87通过；临时无第三方包CPU环境82通过/5可选依赖跳过，临时环境已清理 |

所有checkpoint、适配器、optimizer、scheduler、RNG、原始统计、逐条预测和正式JSON收据均保留；没有重跑GPU、bootstrap或覆盖旧结果。保留这些证据约34GiB是研究复核需要，不作为“无用文件”删除。

## 归档与恢复

私有记录位于experiments/maintenance_20261103：inventory_before.json、cleanup_manifest.json、deletion_actions.jsonl、deletion_receipt.json及archives目录。清单包含原路径、归档SHA、ZIP逐文件SHA、原日志SHA、停用模型repo/revision/权重SHA和再生副本映射。只在本地或私人证据备份中保存，不将归档上传GitHub。

只查看清单、不恢复：

```powershell
$env:PYTHONPATH='src'
python scripts/restore_workspace_archive.py
```

明确需要复查某项旧证据时，一次恢复一个精确路径；现有目标不覆盖，归档及原内容SHA不符即停止：

```powershell
python scripts/restore_workspace_archive.py --restore data/processed/lamp7_retrieval_confirmation
python scripts/restore_workspace_archive.py --restore experiments/runs/qwen3-4b-lamp7-full-lora-r16-a32-int8/train.stderr.log
# 停用模型只恢复配置/tokenizer/许可/元数据，不自动下载权重或训练：
python scripts/restore_workspace_archive.py --restore models/Qwen3-1.7B
```

权重确需重取时，根据清单的repo_id、revision和filename显式调用huggingface_hub.hf_hub_download(local_dir=对应模型目录)，并核对清单SHA。Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。恢复元数据不表示模型权重已经就绪。重建副本无需恢复，可以在新目录从现有规范数据或固定来源重建；原核验结果是历史证据，副本删除不代表规范数据丢失。

重启旧统计、旧训练队列或人工盲评不属于恢复操作。旧运行源码哈希按其原Git版本核验，不修改旧收据来迎合新代码。当前云端单轮准备统一见[39号](39_cloud_4090_sms_guide.md)。
