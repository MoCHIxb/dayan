# 当前文档索引

最高优先级为所有者开题原文与最新指示；[55号研究重规划](55_research_realignment.md)覆盖旧框架、IntentTune和全历史大队列的后续目标。[17号交接](17_new_chat_handoff.md)记录事实与最新覆盖。

- 研究规划：[55](55_research_realignment.md)。模型为Qwen8B＋Mistral7B候选，主客观多维评价，不以软件框架为贡献。
- 当前8B状态：[53准备记录](53_cloud_4090_dual_8b_setup.md)、[54单条件Qwen入口](54_qwen8b_first_banking500.md)。云端尚未运行，Llama访问被拒后不再要求继续准备它；不自动执行旧队列。
- 历史LaMP科学协议：[20](20_small_sample_protocol.md)、[24](24_small_sample_peft_protocol.md)，结果23/25/31，原始完整统计保留。
- 历史辅助诊断：29/30。表层风格/复制不是主观深层风格证明。
- 历史分类证据：36本地安全功能、37/38 SMS、46/48 BANKING协议数据卡、49结果。数据覆盖/风险边界按原报告保留。
- 论文：42历史工作稿、43证据映射、44历史主表；需按55重新整合，不是新主线已完成的论文。
- 维护：40归档恢复；本次[无损整理清单](../results/research_cleanup_20261116.json)。
- 32/33/41/45/50/51/52已移出活跃规划，其原路径保留指向docs/history/replanning_20261116的完整原文。归档内相对引用沿用原文件上下文，核对original路径及SHA后阅读；不是新执行依据。
- [脚本分类索引](../scripts/README.md)区分科学核心、已完成历史调度、当前待启动入口与归档工具。

没有删除实验数据、模型、checkpoint、预测或失败证据。只公开窄范围授权示例、代码和聚合；模型访问/许可各自遵守，来源与结果不可混用。

- [56：纯文本主流模型选择与ModelScope统一部署](56_text_model_roster_and_deployment.md)：最新取消LLaVA；Qwen/Mistral/Phi资产入口，统一训练器仍待验证。

- [57：扩容后的三模型CPU兼容预检](57_cpu_model_preflight.md)：真实数据、原生模板/监督边界和meta投影检查；不加载权重、不启动训练。

- [58：数据集、环境、CPU预检完整命令](58_complete_preparation_commands.md)：当前GPU忙时可做的步骤及LaMP私有传输，新机/GPU空闲步骤分列，不启动训练。
