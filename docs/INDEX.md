# 文档索引与优先级

开题计划与所有者最新指示优先；历史记录不是自动重启任务的授权。

- [39：当前4090单轮SMS指南](39_cloud_4090_sms_guide.md)、[40：清理与恢复记录](40_workspace_maintenance.md)：云端尚未执行；原始证据保留和归档位置。
- [37：SMS新领域协议](37_sms_spam_protocol.md)、[38：完整结果](38_sms_spam_results.md)：私有500/200/1000真实数据、128步、严格分类评分与实际效果边界。
- [36：本地框架功能验证](36_local_framework_validation.md)：32步真实训练、暂停恢复、完整test与资源/误报取舍；云端仍未执行。
- [35：三类纯文本审查数据](35_three_domain_text_safety.md)：当前医学/金融/公共安全来源、实际数据规模、标签边界和独立云端命令。
- 34及MMDS来源报告：已取消的历史方案，不能恢复。

- [33：开发者启动与接口指南](33_developer_framework_guide.md)：本轮命令集、真实配对格式、配置、暂停/恢复与实现边界。
- [32：框架总方案](32_framework_master_plan.md)：当前目标、已有证据、开发者仓库范围、架构和分阶段验收。
- [17：当前交接](17_new_chat_handoff.md)、[28：执行清单](28_next_work_plan.md)：接手入口与已完成/待实现边界。
- [31：最终质量/资源解释](31_peft_quality_cost_analysis.md)：18/18原始统计后的最终本轮解释；`results/peft_quality_cost_analysis.json`为聚合机器证据。
- 29/30：探索性客观风格/内容协议与28条件结果；表层代理不证明深层风格或事实正确。
- 20/21/23：冻结小样本协议、结果与解释；24/25：冻结PEFT协议与结果。两轮本地训练和统计均完成。
- [27：Linux/RTX4090指南](27_cloud_4090_reproduction.md)：平台复现尚未实机验证。
- 22/26：本地工程与历史验证记录；其中“运行中”等历史措辞不是当前状态。
- 19：去重、复制与风格证据审计；01–16/18和`history/`：文献与历史研究背景。

当前机器进度见`results/local_progress.json`；`results/local_pause_status.json`保留为历史暂停快照。本地已完成监控已删除，不自动重建。

公开仓库仅包含所有者授权的examples/text_safety_v1选定数据；不含全量原始语料、旧LaMP数据、权重、checkpoint和逐条模型预测。文档内`experiments/`路径指向本地私有证据，干净克隆中不存在属于预期。代码尚未指定许可证，不把公开研究快照称为正式开源发布。
