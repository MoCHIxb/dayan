# 第三方资产与发布范围

本文件记录项目的资产边界，不替代各上游许可证。项目所有者选择先公开研究代码，暂不指定代码许可证；见LICENSE_STATUS.md。

| 资产 | 本地来源 | 处理方式 |
|---|---|---|
| Qwen3-4B/1.7B权重与tokenizer | models/下缓存，4B含LICENSE和README.md | 不随代码包分发，使用者单独取得并核对上游条件 |
| LaMP-7原始及处理后数据 | data/raw与data/processed | 不随代码包分发；保留官方划分，不新增合成训练样本 |
| roberta-large与MiniLM缓存 | 本机Hugging Face缓存 | 不随代码包分发；按各模型说明取得 |
| Python训练/评价依赖 | requirements-train.lock、requirements-eval.lock | 本机详细元数据保存在私有experiments/framework_smoke/environment.json；版本锁不代表完整依赖授权审计 |
| 适配器、checkpoint、逐条预测 | experiments/runs及其他实验目录 | 只作为本地证据；正式发布前单独检查，不自动纳入代码发布包 |

最新公开范围新增所有者明确授权的examples/text_safety_v1选定数据：DiaSafety医学行按上游Apache-2.0条款，BeaverTails金融/公共安全行按CC BY-NC 4.0保留来源及改动说明，见该目录NOTICE。其他全量原始数据、模型、缓存和逐条模型预测不公开。该数据例外不为仓库代码新增许可证。没有执行权重合并或模型发布。第三方资产不因代码公开而自动获准重新分发。
