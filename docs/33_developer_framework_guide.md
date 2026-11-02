# 开发者框架指南：启动命令与接口

本仓库交付研究代码、可组合的开发者命令和方法指南；不要求安装桌面软件、部署HTTP服务或实现商业产品式MVP。图形界面可由后续使用者按下列接口接入，不是本轮必做项。最新示例为35号医学/金融/公共安全三领域纯文本审查数据，沿用Qwen3-4B；取消VLM/MMDS和多模态接口。选定数据已构建，训练尚未执行。

## 1. 实现与验证范围

- 既有研究管线已在Windows/RTX5070Ti、Qwen3-4B验证，完整研究证据见23/25/30/31号文档。
- 本轮新增通用真实配对导入、个人示例配置、显式执行的训练/完整test包装器，以及CPU保护测试。新包装器复用已存在的模型加载、SFT数据/整理器和评价组件，但其组合**尚未经过新的真实个人数据/GPU端到端验证**。不能把CPU测试写成已训练成功。
- Linux/RTX4090仍待27号方案实机验收；不将其视为已支持且验证的商业安装包。
- 新个人示例单独放在`experiments/personal_example/`；不触碰small_sample_v1、peft_v1或cloud_4090_v1。默认配置是起点，不是专业任务最优方案。

## 2. 安装与启动入口

先准备Python环境和本地Qwen3-4B。完整版本记录/本机依赖见22号及锁文件，Linux环境按27号指南处理CUDA/PyTorch。不要为了试新入口升级现有完成实验环境。

新机器先确认Python 3.12。首次创建独立环境可按以下Windows命令；已有项目环境跳过创建与安装，不覆盖原环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.lock
python -m pip install -e . --no-deps
python -m pip check
```

若尚无基座，按上游条款取得固定版本（这是供使用者执行的下载命令，本轮没有执行下载）：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c', local_dir='models/Qwen3-4B')"
```

个人示例不要求额外下载BERTScore模型。Linux首次环境和严格平台复现仍按27号指南核验；上面命令不代表新平台已经通过验收。

以下所有命令在仓库根目录执行。激活已配置的环境后：

```powershell
$env:PYTHONPATH='src'
python -m personal_llm_data.workflow --help
python -m personal_llm_data.workflow prepare-personal --help
python -m personal_llm_data.workflow train-personal --help
python -m personal_llm_data.workflow evaluate-personal --help
```

Linux的同等入口：

```bash
export PYTHONPATH=src
python -m personal_llm_data.workflow --help
```

可编辑安装`python -m pip install -e . --no-deps`后可用`personal-llm`代替`python -m personal_llm_data.workflow`。保留源码及scripts目录；不是独立wheel部署。只有数据/CPU规划命令不需要torch等训练依赖，NF4还要求当前环境存在兼容bitsandbytes。

## 3. 准备真实数据

输入为UTF-8 CSV或JSONL，每条明确属于train/dev/test。原始文档先由拥有者整理为真实任务与其实际回答；本工具不会从报告自动编造指令或答案。

| 字段 | 要求 |
|---|---|
| id | 非空字符串，全文件唯一 |
| source_group | 原始文档/会话/任务来源组；先按组划分，同组不能跨集合 |
| split | train、dev或test |
| input / output | 本人或获授权的真实任务与实际回答，均非空 |
| messages | JSONL可用它替代input/output；支持user→assistant，或system→user→assistant |

训练50–500条；另留非空dev/test并明确总标签预算。工具的非空下限仅是格式保护，**不是足够的研究测试样本量建议**；单条test不能支持泛化结论。具体案例预算和代表性须稍后确定，不能补造数据凑数量。

先人工检查来源和近重复，再显式确认：

```powershell
python -m personal_llm_data.workflow prepare-personal --input experiments/personal_example/source/pairs.csv --output-dir experiments/personal_example/data --confirm-real-authorized --confirm-near-duplicate-review
```

输出完整train/dev/test.jsonl与manifest.json（条数、输入SHA、各文件SHA和声明）。检查ID/角色/真实配对格式、完全重复对、来源组泄漏；拒绝覆盖已有目录。来源声明不等于工具验证了真实作者；手动近重复复核也不是自动语义去重证明。CSV中的逗号/换行应正确引用，JSONL按物理行解析，内容中的U+0085不会被误拆。

## 4. 声明规划与训练前检查

`configs/framework_task_pending.json`是材料未确定时的元数据模板，不是训练样本。`plan --spec`输出缺口，固定`executable=false`；退出码2表示需补输入，0只表示可以进入实际文件/硬件审计。

```powershell
python -m personal_llm_data.workflow plan --spec configs/framework_task_pending.json
python -m personal_llm_data.workflow train-personal --config configs/personal_adapter_v1.json
```

第二条只核查配置、本地模型类型及prepared数据哈希/完整性，不加载模型、不训练。数据未到位会直接报缺失文件，属于当前预期。

复制`configs/personal_adapter_v1.json`到私有运行目录再编辑，记录这份配置的路径。默认BF16 r8/alpha16；如需NF4，对应改quantization为nf4、r为16、alpha为32，并使用新的output_dir。两者来自已有研究的候选配方：r8减少可训练参数，NF4减少显存但本轮耗时上升；均不是通用胜者。学习率/步数/长度需要在实际专业任务dev上验证，不能看test改配置。

开发者示例默认128步、每16步完整dev与保存、最小dev loss最早步选择；dev数量遵循个人数据manifest，**没有旧研究的固定1500条限制**。这是一条独立示例路径，不是修改了20/24号科学实现。新示例先验证可行性即可，不强制再做三种子网格。

## 5. 显式执行训练

先用`nvidia-smi`确认资源与现有进程，再执行：

```powershell
python -m personal_llm_data.workflow train-personal --config configs/personal_adapter_v1.json --execute
```

实际执行前取得独立OS GPU锁，并查询其他NVIDIA compute PID；有其他任务或无法核对时拒绝运行。该锁用于新增个人训练/评价包装器；旧研究或外部程序不会自动遵守它，所以启动前仍须实际核对，不能据锁文件存在判断任务是否活着。固定使用可见物理GPU0，不提供自动多卡并发。

显式执行后才导入训练依赖、校验CUDA/BF16、核对本地模型与数据/代码快照、逐条token检查全部集合。超长prompt/回答会报错，绝不悄悄截断；须先整理真实材料并重新准备新数据目录。显存预估没有被当作成功保证，OOM会失败留痕，不自动降配重试。

训练写入run_config、run_inputs、progress、每个完整checkpoint的receipt；只有全部计划步数和dev评价、最佳选择、最终checkpoint及导出adapter哈希通过才写`training_summary.json`的completed_verified。训练耗时按本次进程会话记录，恢复后的耗时不能冒充完整总时长。

## 6. 暂停、关机和恢复

在第二个终端请求：

```powershell
python -m personal_llm_data.workflow train-personal --config configs/personal_adapter_v1.json --request-pause
```

它只写暂停请求。训练回调在下一optimizer step后进行完整dev及checkpoint保存，然后退出；大模型加载或完整dev期间不会立即结束。必须等到run_status为paused_checkpoint_verified、核验所指checkpoint的adapter/optimizer/scheduler/RNG/trainer_state及receipt，并确认实际Python进程退出后再关机。状态字段或“请求已发出”本身都不证明安全保存。

恢复命令中的STEP替换为实际经核验的步数：

```powershell
python -m personal_llm_data.workflow train-personal --config configs/personal_adapter_v1.json --execute --resume-from-checkpoint experiments/personal_example/run/checkpoints/checkpoint-STEP
```

恢复检查配置、数据/模型/代码快照及checkpoint逐文件SHA，失败即停止。意外断电可能丢失最近未保存步骤；只考虑有完整receipt的checkpoint，不自动采用零散文件。程序异常不自动重试、删除或覆盖现场。不能从较早步数续跑并覆盖已有更晚checkpoint；须先核对并另行私有归档后续现场。若最终预算已保存但汇总写入前中断，需单独核验最终完成状态，入口会拒绝盲目续训。不要手动修改运行中的代码/配置；原始收据与输入一致后才考虑显式恢复。OS锁随进程退出释放，留在磁盘的gpu.lock文件不是需要删除的异常任务。

## 7. 完整test与本地使用

训练完成后，同一配置依次执行（每次仅一个GPU任务）：

```powershell
python -m personal_llm_data.workflow evaluate-personal --config configs/personal_adapter_v1.json --condition base
python -m personal_llm_data.workflow evaluate-personal --config configs/personal_adapter_v1.json --condition base --execute
python -m personal_llm_data.workflow evaluate-personal --config configs/personal_adapter_v1.json --condition adapter --execute
```

第一条只展示完整条数与待执行条件。显式评价检查训练输入快照、best/final checkpoint和导出适配器，完整生成manifest中的test，不做默认100条截取；两条件保持prompt、量化与生成预算一致。逐条ID、参考答案、顺序及条数核验通过后写receipt。输出已存在会拒绝覆盖；中断现场留存，不自动重新生成。复用评价器在整组生成后写出预测，未写出的中间生成可能丢失；这里没有宣称评价逐条断点恢复，恢复前先定位现场并决定独立重跑输出。

复用原始评价器保留BLEU/chrF/ROUGE与表层诊断，但英文分词、词形和大小写特征不适合作为通用中文风格结论。中文示例优先解释chrF等词面参考相似度与任务专属术语/内容检查，不能把任意分数写成专业正确率。新个人示例的统计论证另立协议，不替换旧研究已经完成的2000次原始corpus_score bootstrap。

已有推理入口可直接加载导出adapter：

```powershell
python -m personal_llm_data.workflow infer --model models/Qwen3-4B --adapter experiments/personal_example/run/adapter --input experiments/personal_example/user_prompts.jsonl --output experiments/personal_example/inference.jsonl --quantization none --max-input-tokens 768 --max-new-tokens 256
```

user_prompts由使用者提供真实待处理任务，含id和messages且以user结束，不能把有参考答案的训练/测试行直接交给infer。量化与实际训练配置保持一致；不加adapter参数即可获得基座对照。每次选择新output，异常保留.partial；不要把自由推理输出当正式test结果。回退方式是卸载/不加载adapter，无需改动基座权重；权重合并与部署尚未验证。

## 8. 模块接口与可扩展点

| 接口 | 输入→输出 | 副作用与边界 |
|---|---|---|
| planning.plan_task(spec) | 声明元数据→规划dict | CPU无执行；不是数据真实性/可用显存检测 |
| personal_data.validate_pairs(rows) | 明确划分的真实配对→三集合dict | 内存校验；不造样本、不自动划分来源 |
| personal_data.prepare(source, destination, authorized=..., near_duplicates_reviewed=...) | CSV/JSONL→三文件及manifest | 只写不存在的新目录；确认由调用者负责 |
| personal_data.verify_prepared(directory) | manifest/文件→复核后的三集合 | 哈希、条数、格式、重复及分组再次校验 |
| train_personal_adapter.validate_config(config) | 独立示例配置→完整集合 | 读文件；不导入GPU依赖 |
| train_personal_adapter.main(argv) | CLI参数→退出码 | 默认仅检查；--execute才训练，--request-pause仅写请求 |
| personal_runtime.checkpoint_inventory(path) | checkpoint→步数与必要文件SHA | 检查完整状态文件；续跑还要求receipt与输入一致 |
| evaluate_personal_adapter.verify_predictions(records, predictions) | 全test及预测→通过/异常 | 保持完整ID/参考答案/顺序 |

未来接简洁前端时，可以把“上传/准备、规划/配置、显式启动、读取progress/summary、基座对比”映射为页面操作。这里是接口设计，仓库没有暗示已实现REST API、HTTP服务、队列或桌面安装器。扩展模型/任务/材料格式时新增适配模块与独立测试；保持这些边界，勿修改冻结研究代码来迎合新任务。

## 9. 发布与后续示例

只公开源码、配置模板、文档、哈希和聚合证据。上述全部运行目录均在experiments下，原文、逐条预测和权重不提交。模板不包含虚构训练对。代码尚未指定许可证，数据与基座遵守各自上游条款。

后续确定专业内容后，取得一套真实配对数据，按以上流程训练一个示例，展示提示基线与适配器的术语/结构/内容表现及失败样例即可。新增包装器届时必须记录真正的GPU端到端结果；在此之前标明“CPU保护测试通过，新增GPU流程待实测”。正式研究论证依靠既有完整实验，并对新示例的外推保持边界。


本轮CPU交付验证：原训练环境60项测试通过；独立CPU环境55项通过、5项因缺少可选训练依赖跳过；五个帮助入口与待输入规划退出码核对通过。聚合记录见`results/framework_delivery.json`。新增个人GPU流程尚未执行。


自动近重复审计新增路径：三领域数据的manifest明确manual_near_duplicate_review_declared=false，并提供自动审计收据；verify_prepared会穷举重算再接受，不将机器检查冒充人工。
