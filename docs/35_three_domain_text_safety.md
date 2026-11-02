# 三领域纯文本安全审查示例

2026-11-02最新范围：医学、金融、公共安全，纯文本、单轮，沿用Qwen3-4B。取消使用vlm-long-context-safety-eval及MMDS，不开发Qwen-VL、不恢复旧实验队列。这里只选取已有论文/公开成果中的数据及标注，不合成问题、候选回答或风险标签。

## 数据来源与任务

| 领域 | 实际采用的已发表来源 | 选取内容 | 监督信号 |
|---|---|---|---|
| 医学 | [DiaSafety](https://github.com/thu-coai/DiaSafety)，[Findings of ACL 2022论文](https://aclanthology.org/2022.findings-acl.308/) | Unauthorized Expertise医学场景的上下文与回复 | 原Safe/Unsafe标签，格式转换为is_safe布尔 |
| 金融 | [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)，[NeurIPS 2023论文](https://arxiv.org/abs/2307.04657) | 金融咨询、金融违法/欺诈等主题问答 | 原is_safe人工标签，原风险类别保留在provenance |
| 公共安全 | 同一BeaverTails公开语料 | 暴力、威胁、违法与公共风险相关主题问答 | 同上 |

本轮使用英文原文，不翻译扩样，也不声称已验证中文审查。原始语料包含模型生成的候选回复；沿用已有成果不等于全部由真人撰写，本项目没有自行生成新案例或标签。

这三个是基于已有语料抽取的领域子集，不能宣传成三个原作者发布的独立新数据集。金融/公共安全的领域归属由可复现关键词规则选取，非原始官方领域标注。医学优先使用现成医学审查标签，不把MedSafetyBench的安全答复生成任务强行当作分类任务。

统一任务：给定一条用户请求和一个候选回答，审查**候选回答在该上下文中是否安全**。输出仅为`{"is_safe": true}`或`{"is_safe": false}`。请求中存在攻击意图不意味着安全拒绝回复应标Unsafe；这些不是“请求是否为攻击”的标签。候选回答只在外层user字段作为待审查的引用证据，监督assistant目标只含原始安全标签，不训练模型复现危险回答。没有原始审查理由时不编写或生成理由。

医学源标签主要评估越权医疗建议，不能当作诊断、剂量、治疗方案的临床正确性金标准。不同语料采用不同安全政策，系统提示保留政策区别，结果须按领域单独报告；统一布尔格式不代表跨领域标注标准完全相同。

## 实际规模与划分

| 数据 | train | dev | test |
|---|---:|---:|---:|
| medical | 50 | 20 | 40 |
| financial | 50 | 20 | 40 |
| public_safety | 50 | 20 | 40 |
| joint（以上三类并集） | 150 | 60 | 120 |

每个领域、每个集合均为Safe/Unsafe各半。共330条不同问答，joint只是同一批数据的联合视图，不能重复计成660个独立样本。标注预算为150训练＋60开发＋120测试，不能声称只有150条总标签。

实际不同问题组：医学42/20/39、金融43/19/40、公共安全31/18/34；联合116/57/113（依次train/dev/test）。同一问题可有多个候选回答，每组每集合最多3条，问答条数不等于独立问题数。

医学从原train/val/test各取对应集合；金融和公共安全的test仅来自原官方test，dev从原train中按问题组留出。所有官方test的问题哈希均从train/dev排除；医学官方val问题也从train排除。开发和训练进一步按问题组隔离，选中集合间穷举检查归一化原问题的字符5-gram Jaccard，阈值0.9；这是自动近重复检查，不冒充人工审校或语义独立性的证明。

所有源文件使用固定版本与SHA256；330条选中记录逐条回放上游，核对原问题、原回复和原标签完全一致。出处、源文件行号/数组序号、原类别和原标签映射均在provenance中。部分高风险细节、识别信息模式、超长内容及标签冲突被排除；具体规则/计数保留在构建脚本和construction_report中，不能称为原数据集完整覆盖。

## 构建与使用

公开数据在`examples/text_safety_v1/`，每个领域和joint均含train/dev/test.jsonl与manifest.json。原始全量语料只在Git忽略的data/raw目录；本次公开例外仅覆盖已选定的示例文件，模型、checkpoint、逐条模型预测继续私有。

固定版本：DiaSafety `5106a041b1c7e36f30e02a97c44897a7f2057278`；BeaverTails `8401fe609d288129cc684a9b3be6a93e41cfe678`。原始数据按各上游条款使用；医学行保留上游Apache-2.0条款，BeaverTails行遵循CC BY-NC 4.0，joint保留各来源条款，不改变项目代码尚未指定许可证的状态。详细归属见示例目录NOTICE与许可证副本。

在仓库根目录、已准备好的环境中：

```bash
export PYTHONPATH=src
# 干净克隆已含选定数据。下面只检查，不训练：
python -m personal_llm_data.workflow train-personal --config configs/text_safety_v1/qwen3_4b_joint.json

# 需要从上游重建时，先显式下载，再用新目录避免覆盖冻结数据：
python scripts/fetch_text_safety_sources.py --download
python scripts/build_text_safety_sets.py --output-dir experiments/text_safety_v1/rebuilt
python scripts/verify_text_safety_data.py
```

PowerShell用`$env:PYTHONPATH='src'`代替export。下载脚本无--download只展示文件计划，不下载模型。构建需本地Qwen3-4B tokenizer以核对所有样本token，不加载模型权重或使用GPU。默认长度1024、输出64；超长样本明确排除，不静默截断。预处理及近重复检查为CPU过程。

## 云端训练与评价

使用独立`experiments/text_safety_v1/`，不启动cloud_4090_v1旧12次平台复现队列。默认只训练一个150条联合适配器，不自动安排三套单领域训练或大网格。LoRA r8/alpha16、BF16、128步、每16步完整60条dev并保存，按最低dev loss最早步选取；这些是新示例预设起点，不保证最优。

Linux/RTX4090先按27号的环境检查与锁文件准备依赖（只复用环境说明），下载固定Qwen3-4B版本，固定当前Git commit。新示例尚未在服务器上执行，不能以本地CPU检查代替Linux/GPU验收。

```bash
export PYTHONPATH=src
# 新服务器首次环境（先确认Python 3.12与驱动；已有合格环境可跳过）：
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.lock
python -m pip install -e . --no-deps
python -m pip check
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c', local_dir='models/Qwen3-4B')"
nvidia-smi
python -m personal_llm_data.workflow train-personal --config configs/text_safety_v1/qwen3_4b_joint.json
# 确认实际GPU进程与资源后，显式执行：
python -m personal_llm_data.workflow train-personal --config configs/text_safety_v1/qwen3_4b_joint.json --execute
python -m personal_llm_data.workflow evaluate-personal --config configs/text_safety_v1/qwen3_4b_joint.json --condition base --execute
python -m personal_llm_data.workflow evaluate-personal --config configs/text_safety_v1/qwen3_4b_joint.json --condition adapter --execute
python scripts/score_text_safety.py --data-dir examples/text_safety_v1/joint --predictions experiments/text_safety_v1/run/evaluation/base_metrics.predictions.jsonl --output experiments/text_safety_v1/run/base_review_scores.json
python scripts/score_text_safety.py --data-dir examples/text_safety_v1/joint --predictions experiments/text_safety_v1/run/evaluation/adapter_metrics.predictions.jsonl --output experiments/text_safety_v1/run/adapter_review_scores.json
```

基座和适配器使用相同完整120条test、相同提示/解码预算，逐条ID和参考标签必须一致。主要看每领域及总体macro-F1、不安全回答检出、安全回答误报、无效JSON比例；无效输出计为错误，不默认当Safe。这里的“安全回答误报”不能称为“正常请求误报”。生成包装器附带的BLEU等词面指标不用于证明安全审查能力。

暂停/恢复沿用33号显式请求与完整checkpoint核验；不自动重试，不覆盖运行证据。该新示例不替换旧small_sample_v1/peft_v1的完整dev/test及2000次原始corpus_score统计。新test为研究性平衡子集，不代表自然部署风险比例；每领域仅40条，且部分共享问题来源，结果只能作为小样本框架示例，不足以认证医疗/金融合规或宣称普遍强安全能力。

## 当前状态

三类数据及联合视图已在本地构建，源文本/标签回放核对通过；当前没有模型预测或新训练结果。只交付数据、代码、配置、归属说明与CPU核验，云端训练需实际服务器就绪后显式启动。
