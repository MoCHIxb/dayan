# BANKING77 card10：IntentTune数据卡

2026-11-13。任务为单条银行卡客服消息的10意图识别，英文原文，closed-set。源为作者BANKING77，13,083条、77意图；本项目选定子集不称新独立数据集或全77类官方benchmark成绩。

## 来源与许可

- 作者论文：[Casanueva et al., 2020](https://arxiv.org/abs/2003.04807)。
- 仓库：[PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets)。固定revision `57ec275d8078af65b7731c2a98be812d844a6d6b`。
- 作者LICENSE原件为CC BY 4.0；本项目格式化目标、创建固定子集/提示目录，未改客户原文或原category。源文件原件私有保留，SHA见banking77.py及repro/banking77_assets.json。
- 未授权扩展原先12份安全示例JSONL公开清单。本轮仅发布构建代码、配置、聚合报告与哈希；云端可从固定来源重建，无需把CSV/JSONL提交GitHub。

## 数据视图

| 视图 | train | dev | test | 独立性 |
|---|---:|---:|---:|---|
| n50 | 50 | 100 | 400 | train是n200/n500前缀 |
| n200 | 200 | 100 | 400 | train是n500前缀 |
| n500 | 500 | 100 | 400 | 最大集合共1000不同源记录 |

每个集合各类数量相等，三档共享同一dev/test。n50每类5条、n200每类20条、n500每类50条。少样本条数指train，不隐去dev/test标签预算。固定示例提示只取n50训练的每类1条，共10示例，不产生新客户内容。

所选10类：激活卡、卡片未到、配送时间预期、卡片失效、无法识别的卡支付、卡片遗失/被盗、PIN锁定、修改PIN、待处理卡支付、被拒卡支付。实际输出原始英文标签，目录与项目定义见personal_llm_data.banking77；中文只作说明。

## 处理与验证

读取官方CSV并保存源行号、原标签、原文SHA、固定revision。问题按归一化哈希分组，原始客户文本不改。官方test的精确重叠从train排除；所选集合跨split字符5gram Jaccard<0.9。先固定全部所选官方400test，再从原train选每类10dev和50train，按固定SHA排序。不存在模型test分数参与选择。

本轮实际排除：3条重复训练问题、2条官方test精确重叠训练问题、1条与选定test近重复训练问题；没有超长排除和开发集近重复排除。原官方400test全部保留。1000不同原记录已逐条回放原文、原标签及provenance，14份构建文件独立重建字节一致；机器证据见results/banking77_preparation.json及banking77_data_verification.json。所有记录和10示例提示通过固定1536/64 token预算。

## 局限

这是银行领域中的银行卡客服子集，不覆盖贷款、投资、保险等所有客服业务。没有OOS样本或真实个人用户数据，不能宣称未知意图检测、所有客服领域泛化或个人风格迁移。公开历史数据可能已被基座预训练见过。原标签可能含噪声，相近意图边界亦可能模糊；本轮不根据预测自动纠错。近重复审计不是语义独立性或人工业务审校。训练与提示/轻量基线的效果均待RTX3090实测。
