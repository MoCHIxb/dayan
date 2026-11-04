# 开发者框架迭代与论文工作计划

2026-11-04。范围：补齐使用流程、CPU诊断和论文初稿；不新增GPU训练，不修改已冻结实验。

## 本轮交付

新增 `personal-llm doctor`：只读CPU诊断，报告Python、源码目录、可选依赖安装版本；指定个人配置时检查prepared数据及其manifest、Qwen3模型类型、权重文件存在性和输出目录冲突。不导入Torch，不查询GPU，不联网、不下载、不写任务文件，不输出训练原文。退出0仅表示这些CPU检查通过，退出2表示需要处理。它不是完整配置验证器、依赖兼容性认证、权重完整性认证或显存适配器。

```powershell
$env:PYTHONPATH='src'
python -m personal_llm_data.workflow doctor
# 已完成的本地SMS输出会报告fresh_output=false，这是防止误重跑的预期结果：
python -m personal_llm_data.workflow doctor --config configs/sms_spam_v1/qwen3_4b.json
# 为尚未执行的独立云端配置做CPU诊断，不启动GPU：
python -m personal_llm_data.workflow doctor --config configs/cloud_sms_spam_v1/qwen3_4b.json
```

模型权重存在不代表所有分片完整。正式执行前继续用 `train-personal --config PATH` 做原有严格验证；云端固定版本/完整SHA/依赖检查使用39号专用工具。不要把doctor的通过当作已有实验完成或GPU可用的证明。

README、33号指南和32号总方案统一将39号单次SMS复现作为当前云端入口；27号旧LaMP矩阵只保留历史独立方案。新增论文初稿与证据映射，主张与已完成范围逐项对应。

## 开发者最短路径与验收点

| 阶段 | 入口 | 应核查的产物 | 失败时处理 |
|---|---|---|---|
| 定义任务 | plan、33号数据契约 | 原始授权、任务指标、总标签预算、组级划分 | 材料不足先补真实配对，不合成 |
| 环境定位 | doctor | CPU问题清单；版本仅供查看 | 缺依赖按锁文件配置独立环境 |
| 准备数据 | prepare-personal | train/dev/test、manifest、分组/重复审计 | 保留原文，另建修订数据目录 |
| 核验配方 | train-personal不带--execute | 实际配置与prepared数据有效 | 不自动降配或修改冻结协议 |
| 执行与恢复 | train-personal显式执行/暂停/恢复 | 完整状态、receipt、最佳与最终checkpoint | 留痕，人工确认诊断结果后决定 |
| 完整比较 | evaluate-personal、对应任务评分器 | 两组完整ID/参考答案、同预算指标、误报漏报 | 不修复预测追分、不看test调参 |
| 审计与使用 | audit_classification_run、infer | SHA、dev选模规则、完整评价、独立推理 | 不根据状态字段单独宣布完成 |

## 论文安排

初稿：[42号](42_paper_draft.md)。表格证据与可支持主张见[43号](43_paper_evidence_map.md)。论文定位为实证研究和可复现工程，算法基础采用已有LoRA/NF4方法。没有新增算法、未做非专业用户可用性研究，不能将“命令少”表述为已验证易用性。

下一阶段先由作者/导师确认题目、学校格式和论文章节要求，再调整行文与参考文献格式。云端租用后按39号实际运行一次独立SMS复现，补平台表和偏差分析；在执行前不填写推测分数。真实个人材料案例、轻量分类器对照、主观评价与通用能力检查仅列局限/未来工作，本轮不扩展队列。

代码许可证由所有者决定；尚未完成正式开源发布。源码可公开与数据可再分发是不同判断，SMS原文及历史LaMP原文仍私有。

论文主表的可核对提取见[44号](44_paper_tables.md)。运行 `python scripts/export_paper_tables.py` 只读显示；`--output NEW.md` 只允许新文件，不覆盖既有表格。输入仅四份公开聚合JSON，不需要原文/模型/私有预测。
