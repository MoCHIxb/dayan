# 三模型GPU诊断：云端回报与核验边界

2026-11-16。所有者回报九条件诊断完成，随后只读复核inputs、checkpoint文件、train/restore收据及生成SHA，输出All 9 diagnostic receipts verified。本地未收到云端原始文件，不冒充本地重新逐文件审计。

| 模型 | BF16 allocated GiB | NF4 | int8 |
|---|---:|---:|---:|
| qwen3_8b | 16.05 | 9.14 | 13.39 |
| mistral7b | 14.01 | 6.63 | 10.56 |
| phi4mini | 7.85 | 4.98 | 6.78 |

来源为两条既有真实train、r8/alpha16、一次更新保存、独立新进程恢复后继续一步及一条train prompt生成；峰值取train/restore两子进程较大值，已按用户输出四舍五入。不是模型权重磁盘大小、整机显存、1536最大长度压力测试或r16完整训练保证。

这解决了当前三模型/三精度路径的功能可行性疑问；不提供正式效果分数，不能宣称NF4无损、8B优于4B或三模型适配收益已验证。后续不重做这九条件诊断，不把诊断适配器用于正式test。全部GPU探测目录及收据应私有备份，不删除。

正式研究按55号预先固定模型/任务/配方与主观客观评价；统一训练、完整dev选模、模型原生渲染/同一message_view、完整test生成及审计入口仍需实现和冻结。Qwen单模型旧run不是通用入口。前置功能成功不构成自动启动42/54/82次队列的授权。

机器可读摘要：[gpu_probe_owner_report.json](../results/gpu_probe_owner_report.json)。
