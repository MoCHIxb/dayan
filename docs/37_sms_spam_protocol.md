# 第二领域框架验证协议：SMS垃圾短信分类

2026-11-03，sms_spam_v1。所有者授权更换领域并实际训练，独立输出，不重跑既有实验。先固定本协议、来源、数据及配置，再开始任何模型test；不根据test改配方或反复试跑。

## 任务、数据与来源

通信文本单条分类，与此前“请求＋候选回答”的安全审查任务不同。来源：Almeida & Hidalgo (2011), UCI SMS Spam Collection, DOI 10.24432/C5CC84；https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection 。UCI标示CC BY 4.0。2011年历史英文语料，不代表当前垃圾短信分布，不是个性化写作数据。无官方train/dev/test，本项目按预先固定规则创建本地留出集，不称官方划分。

原文与ham/spam标签完全保留。只添加固定任务指令并将原标签格式化为JSON：{"label":"ham"}或{"label":"spam"}；不生成新短信、解释或伪标签。user_request是兼容现有数据审计接口的JSON字段，内容只有原短信。短信作为引用数据，不执行其中的指令。原始数据和处理集均保留data私有目录，不扩大此前公开选定数据的授权例外。

固定原档：ZIP SHA256 1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3；SMSSpamCollection SHA256 7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d。逐行核对标签及原文，记录源行号、文本SHA和原标签。

规范化仅用于分组：NFKC、大小写/空白归一、将URL/email/连续数字替换为占位符；存储的训练/评价短信原文不修改。每个模板组保留首个原记录，标签冲突组剔除，全部集合在tokenizer下预先检查完整prompt＋目标＋EOS≤512、目标＋EOS≤16，超长先排除并计数。种子20261103按原ID哈希排序，先选test再dev再train；同组不跨集合，已选跨集合原文char5 Jaccard≥0.9的候选跳过。完整穷举重算审计，不冒充人工或语义去重。模板约束不能保证发送者独立或同集合内没有近似文本。

| 集合 | ham | spam | 合计 |
|---|---:|---:|---:|
| train | 433 | 67 | 500 |
| dev | 173 | 27 | 200 |
| test | 866 | 134 | 1000 |

比例接近原始类不平衡，未刻意平衡成50/50；总标签预算1700，500仅指训练条数。若按规则无法凑齐则停止，不补造或放松门槛。

## 单次训练与评价

Qwen3-4B固定本地模型；BF16 LoRA r8/alpha16/dropout0.05，七类projection；学习率1e-4、累积16、单卡microbatch1、128步，每16步完整200dev评价保存。其余复用现有独立开发者训练入口。配置configs/sms_spam_v1/qwen3_4b.json，输出experiments/sms_spam_v1/run。正常完成全部预算后按最低dev loss、同值取最早步选权重；不按test选模型。一个种子，一次配方，不自动追加搜索或重试。OOM/失败留痕后定位，不覆盖或降配重跑。

基座与适配器各完整1000test，相同greedy/无thinking、512输入/16新token。严格JSON、唯一键、标签白名单；无效输出计错误，逐条ID/顺序/参考标签必须完整相同。统一score-classification入口输出accuracy、macro-F1、各类precision/recall/F1、混淆矩阵和invalid比例；单列spam召回与ham→spam误报。多数类ham基线只作描述性参照；不以高accuracy掩盖少数类失败。词面BLEU等不是本任务主要指标。

预设可用性参考：macro-F1≥0.90、spam召回≥0.90、ham误报≤0.05、无效输出为0；另外完整报告相对基座变化，达标不等于微调有增益。若基座更强或收益有限，也照实交付，不调整test或改配方追分。一种子描述性结果，不进行显著性/非劣效/跨用户泛化宣称，不增加本例bootstrap；既有LaMP原2000次corpus_score统计不变。

验收必须核查500/200/1000记录、128步、8次完整dev、最早最低loss选模、适配器/最佳checkpoint同SHA、最终128步完整状态、两组完整test及收据、严格分类指标重算。全部GPU阶段串行；如用户要求暂停，立即发起现有保存请求并等待核验退出，不自动恢复。

## 框架迭代及复现命令

新增通用字符串标签分类评分模块和统一score-classification CLI，支持其他预先声明标签词表，不再把分类任务硬编码成is_safe。新增固定版本数据下载/构建器、配置、保护测试与本任务操作指南；保持旧科学实现和已完成证据不变。

```powershell
$env:PYTHONPATH='src'
python scripts/build_sms_spam.py
python scripts/build_sms_spam.py --download --build
python -m personal_llm_data.workflow train-personal --config configs/sms_spam_v1/qwen3_4b.json
python -m personal_llm_data.workflow train-personal --config configs/sms_spam_v1/qwen3_4b.json --execute
python -m personal_llm_data.workflow evaluate-personal --config configs/sms_spam_v1/qwen3_4b.json --condition base --execute
python -m personal_llm_data.workflow evaluate-personal --config configs/sms_spam_v1/qwen3_4b.json --condition adapter --execute
python -m personal_llm_data.workflow score-classification --data-dir data/processed/sms_spam_v1 --labels ham spam --predictions experiments/sms_spam_v1/run/evaluation/base_metrics.predictions.jsonl --output experiments/sms_spam_v1/base_scores.json
python -m personal_llm_data.workflow score-classification --data-dir data/processed/sms_spam_v1 --labels ham spam --predictions experiments/sms_spam_v1/run/evaluation/adapter_metrics.predictions.jsonl --output experiments/sms_spam_v1/adapter_scores.json
```

默认下载/构建命令仅展示计划；必须显式--download/--build，拒绝覆盖处理集。完成后不重复执行已完成训练，复现选择新输出配置。只公开代码、协议、配置、哈希及聚合证据，短信原文、日志、权重和逐条预测不提交；源码许可证仍未指定。Linux/4090未执行，本轮不租机。
