# BANKING77银行卡客服10意图协议 v1

2026-11-13。在任何本轮模型预测之前固定。数据已CPU构建/核验，模型实验未执行。新输出为experiments/intenttune_banking77_v1，不覆盖历史结果。

## 来源与任务

来源为Casanueva等的[Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)及[作者BANKING77仓库](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)。固定revision `57ec275d8078af65b7731c2a98be812d844a6d6b`，官方10003条train、3080条test、77类；LICENSE原件为CC BY 4.0，SHA与五份源文件在代码常量和repro清单中。公开维护代码、配置、归属与聚合证据，BANKING原文不扩展此前12份安全JSONL的公开例外。

首轮是银行卡客服closed-set子集，不是完整77类成绩。10个标签按业务范围及训练可用性选定，而非模型test表现：activate_my_card、card_arrival、card_delivery_estimate、card_not_working、card_payment_not_recognised、lost_or_stolen_card、pin_blocked、change_pin、pending_card_payment、declined_card_payment。初选contactless_not_working只有35条官方train，不足以独立留dev后取50条真实训练，故在预测前替换为card_payment_not_recognised。保留到货/时效等相近意图，不能为了分数人为只留容易区分的类。

输入为原客户消息，系统提示包含同一10标签目录与项目编写的简短定义；目标仅把原category包装为 `{"label":"原标签"}`。不合成问句、客服答案、解释、翻译或伪标签。目录定义是任务元数据，不声称原作者逐字定义，也不重新标注原文。

## 数据协议

| 视图 | train | 每类train | dev | test |
|---|---:|---:|---:|---:|
| n50 | 50 | 5 | 100（每类10） | 400（每类40） |
| n200 | 200 | 20 | 同一100 | 同一400 |
| n500 | 500 | 50 | 同一100 | 同一400 |

测试为所选10类全部官方test记录，保持原顺序、原文与原标签，不删除测试难例。开发和训练只从官方train选择。先排除官方完整test归一化问题的精确重叠，再排除与选定400test的char5 Jaccard≥0.9训练候选。训练候选按NFKC/大小写/空白归一问题分组，每组保留首条，冲突标签组整体排除；记录排除计数。

固定种子20261113，以seed/源ID的SHA排序；每类先取10dev，再检查训练对全部dev的近重复约束，取50条。类别交错排列，使前50/200/500条精确嵌套。完整选中集合穷举近重复复核。分组只用于划分，存储文本不变。dev/test/训练共1000不同源记录；有标签总预算不能只写训练50–500。

Qwen3-4B tokenizer固定；输入＋目标＋EOS≤1536、目标＋EOS≤64，过长训练候选明确排除。任何选定官方test或固定示例提示超预算则停止，不截断或删除test。最终所有记录及固定示例输入均通过预算检查。

公开文献指出BANKING77可能存在标注噪声，不能将原标签等同业务真理，也不按本轮预测自行改标：[Ying & Thomas, 2022](https://aclanthology.org/2022.insights-1.19/)。本轮保留源标签并报告混淆；数据公开历史和预训练污染可能性均是局限。

## 比较条件与训练

1. 字符3-gram TF-IDF cosine质心分类器，每档训练集分别拟合词表、IDF和质心；不使用dev/test构造特征，不调参，固定目录顺序处理并列。
2. Qwen3-4B基座，统一意图目录，零示例；为配合现有逐run审计，每档各完整生成一次同一400test，不把它们算作三个独立测试集。
3. Qwen3-4B基座，目录＋固定10示例（每类1条，全部取自n50前10条）。三档共用该提示参照，不从dev/test选示例，也不按结果修改示例。
4. 各档LoRA，Qwen3-4B固定revision、BF16、r8/alpha16、dropout0.05、7投影、lr1e-4、microbatch1、累积16；128优化器步，每16步完整100dev并保存，共8次。三个配置只改数据/输出路径；学习率不按本轮test选择。

同预算greedy、无thinking、batch1、1536输入/64新token。选择完整dev loss最小的checkpoint，同值最早；正常完成全部128步后导出最佳权重，最终状态单独保留。固定更新数意味着不同数据重复曝光，不能称纯数据量因果实验。先执行n500，再n200、n50；没有隐含追加网格或自动重试。

## 评价与结束条件

主指标macro-F1，辅以accuracy、各类precision/recall/F1、最差类别召回、混淆矩阵、主要混淆对、invalid比例及训练参数/allocated/reserved/会话耗时。严格单键label JSON、目录白名单；重复键、额外文字、未知标签均无效且计错。BLEU不是主效果指标。

绝对可用性参照：macro-F1≥0.85、最差类别召回≥0.60、invalid=0；这是项目预设参考而非线上服务保证。还必须分别报告相对零示例、10示例和轻量模型的差值，达标不等于微调有价值；不设未经测量的提升保证。未达标照实报告，不据test调配方。单种子描述性研究，无本轮显著性/非劣效结论；不新增bootstrap，旧54项LaMP统计保持原样。

完整1000源记录回放、14份构建产物字节重建、三份dev/test一致且train嵌套先通过。GPU阶段每run核查128步、8次完整dev、最佳和最终checkpoint、适配器SHA；完整400预测ID/参考/收据核对后评分，run审计完成才进入总汇总。全部22阶段有完成收据并核验命令、输入、输出SHA才写complete。暂停、失败和未执行均不能称完成。
