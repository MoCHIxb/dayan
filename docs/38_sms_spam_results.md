# 第二领域框架验证结果：通信文本分类

2026-11-03。本轮真实训练、完整test、独立收据与checkpoint审计完成。机器证据：[sms_spam_results.json](../results/sms_spam_results.json)、[运行审计](../results/sms_spam_run_audit.json)。固定方案见[37号协议](37_sms_spam_protocol.md)。这是一项新的单条短信任务，不是重跑LaMP或三领域安全审查。

## 来源与实际数据

采用UCI SMS Spam Collection，Almeida & Hidalgo (2011)，DOI 10.24432/C5CC84；[UCI来源及CC BY 4.0说明](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection)。原始5574条，ham4827/spam747。按数字、URL、email等模板规范化分组去除439条重复模板记录，剩5135个候选组；未触发长度排除。此规范化仅用于分组，选中短信原文不修改。没有合成短信、翻译扩样、新标签或生成解释。

| 集合 | ham | spam | 总数 |
|---|---:|---:|---:|
| train | 433 | 67 | 500 |
| dev | 173 | 27 | 200 |
| test | 866 | 134 | 1000 |

1700条均回放源文件核验原文和标签；每个模板只保留一个源记录，组不跨集合，跨集合char5 Jaccard检查800000对全部通过。独立重建train/dev/test/manifest四文件字节一致。无官方数据划分，本轮为固定种子本地留出，不能宣称官方benchmark可比成绩或发送者级独立。约13.4%的类别比例接近原始语料，不能只用accuracy评价。

## 实际训练与完整test

Qwen3-4B，RTX5070Ti 16GB，BF16 LoRA r8/alpha16，500条训练、128步、每16步完整200dev，共8次评价。best step=32，dev loss=0.00655748；由开发集选取，未按test调参。最终128步全状态、最佳checkpoint和导出适配器SHA全部通过。完整测试基座和适配器各1000条，相同512输入/16新token、greedy和无thinking；ID、参考答案、覆盖、收据与哈希均匹配。

| 指标 | 基座 | LoRA适配后 |
|---|---:|---:|
| 准确率 | 0.8780 | 0.9800 |
| macro-F1 | 0.7986 | 0.9566 |
| 无效JSON比例 | 0.0000 | 0.0000 |
| spam召回率 | 0.9328 | 0.9179 |
| ham误报率 | 0.1305 | 0.0104 |

固定全部预测为ham的参照：accuracy=0.8660，macro-F1=0.4641。成对正确性计数：共同正确874，仅适配器正确106，仅基座正确4，共同错误16。无效JSON按错误计入，不修复或默认当ham。

## 结论与选择建议

适配器相对基座macro-F1提高，差值+0.1581；协议预设的macro-F1≥0.90、spam召回≥0.90、ham误报≤0.05、零无效输出目标全部达到。各门槛状态见机器报告，不能把“绝对成绩达标”和“微调相对基座有增益”混为一谈。这是一种子描述性结果，不宣称显著、非劣效或最优，不因分数不理想而追加训练或调整test。

框架贡献是换用原始数据和标签后可复用训练、开发集选模、完整评价与适配器使用流程，并增加通用字符串标签评价、混淆矩阵、严格JSON和CPU审计接口。该例验证了另一种任务的数据接入与评价，不能据两个分类示例宣称任意领域、任意模型都通用。它没有证明某个真实个人的表达定制，也不是当前短信场景的线上部署验收。公开历史语料可能与模型预训练数据重叠，本轮未排除此可能。若基座已经满足目标，应结合训练成本与误报风险决定是否使用适配器，不能把微调当必选项；独立短信过滤应用还应比较轻量分类器，本轮没有开展该比较。

## 运行保护修复、资源与证据

首次尝试在GPU预检阶段因本机EdgeGameAssist和UU远控服务被拒，模型未加载，run目录未创建。保留logs/train.log及preflight_failure_01.json。经所有者明确授权、核实UU有效NetEase签名及微软游戏辅助Appx身份后，新增私有本机桌面程序注册表：只允许限定的桌面程序名、完整实际路径和当前文件SHA完全一致，且仅限WDDM G/C+G。Python、模型服务、未知程序和C-only仍拒绝，跨进程GPU锁不变。

Edge辅助exe无独立Authenticode，Appx SignatureKind为Developer，因此没有以发布者文本冒充密码学认证；本轮信任依据包括所有者授权及固定本机文件哈希。注册表是操作保护，不是安全隔离；桌面程序更新后会重新拦截。该私有注册表不提交仓库、不按进程ID永久放行、不自动信任未来版本。

只授权并执行了一次修复后重试，original manifest保留；operational_revision.json只记录保护源码改变，数据、模型、科学配置和37号协议未变。旧已完成实验收据不改；旧源码哈希应按其原Git版本核验，不能为适配新代码重写旧证据。

训练可训练参数16,515,072，CUDA峰值allocated 8.17GiB、reserved 10.11GiB，会话训练/保存/末尾核验1094.6秒；不是整机显存或含全部数据准备/测试的总耗时。83项CPU测试通过；精简CPU环境78通过/5项因可选训练依赖缺失跳过。完成后GPU保护复核通过，库存：NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 1103 MiB, 14893 MiB, 1 %。

私有根目录experiments/sms_spam_v1包含原始冻结manifest、一次授权修订、逐步日志、完整checkpoint/adapter、两组预测/收据、CPU分数和最终审计。data/raw/sms_spam_v1与data/processed/sms_spam_v1保留短信原文；不扩大此前仅text_safety选定数据的公开例外。

## 新的可复用接口

```powershell
$env:PYTHONPATH='src'
python -m personal_llm_data.workflow score-classification --data-dir data/processed/sms_spam_v1 --labels ham spam --predictions experiments/sms_spam_v1/run/evaluation/adapter_metrics.predictions.jsonl --output experiments/sms_spam_v1/adapter_scores.json
python scripts/audit_classification_run.py --config configs/sms_spam_v1/qwen3_4b.json --labels ham spam --score-directory experiments/sms_spam_v1 --output results/sms_spam_run_audit.json
```

以上为已执行接口记录，已有结果禁止覆盖。复现时使用独立新目录和配置；具体下载、构建、训练和基座命令见37号。通用分类接口要求显式标签词表和严格{"label":"类别"}格式，不把其他任务硬编码为is_safe。CPU审计不加载模型或启动训练。服务器Linux/4090仍未执行，代码许可证仍未指定。
