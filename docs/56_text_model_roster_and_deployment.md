# 纯文本研究模型选择与ModelScope统一部署

2026-11-16最新：所有者先提出将LLaVA加入候选，随后明确“不使用LLaVA，最好主流LLM”。LLaVA已从待部署清单移除，没有部署或恢复任何图文任务。当前统一资产清单configs/research_models.json包括Qwen3-8B、Mistral-7B-Instruct-v0.3、Phi-4-mini-instruct；后者作为较小规模补充，不直接充作7/8B的容量匹配对照。Llama3访问被拒后不再作为必须条件。

## 1. 模型建议

| 模型 | 研究用途 | 需说明的限制 |
|---|---|---|
| Qwen3-8B | 已有主模型资产，建立完整PEFT对照 | 当前无8B训练结果，关闭thinking须固定 |
| Mistral-7B-Instruct-v0.3 | 另一家族主对照，Apache-2.0 | 版本较早；原生模板/目标监督/结束token须验证 |
| Phi-4-mini-instruct | Microsoft较小模型补充，MIT | 3.8B规模、融合投影结构与7/8B不同，不能只复用七projection名称 |
| DeepSeek-R1-Distill-Qwen-7B | 可选知名推理模型扩展，不默认下载 | 基于Qwen2.5，不是独立DeepSeek7B从头预训练家族；推理/格式预算须单独固定 |
| IBM Granite-3.3-8B-Instruct | 可选8B量级独立补充，Apache-2.0 | 生态关注度不同；ModelScope来源/模板/兼容待核验，不默认下载 |

一手来源：[Qwen](https://huggingface.co/Qwen/Qwen3-8B)、[Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)、[Phi](https://huggingface.co/microsoft/Phi-4-mini-instruct)、[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)、[Granite](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)。具体版本而非品牌名是固定实验单位。

建议先完成Qwen/Mistral主对照，Phi补资源规模取舍。如果要求三个模型都近8B，可用Granite替换较小Phi，但须先锁定可获取来源；不要为了品牌数量加入多个同源蒸馏模型并称独立架构证据。这里只更新选择与资产部署，不把可选模型自动展开全部配方/数据量网格。

## 2. ModelScope候选仓库

三项仓库存在性和HF格式权重文件名已通过只读ModelScope API检查，未下载模型字节，不等于候选SHA已通过：

- Qwen/Qwen3-8B（已有官方锁及14文件验收继续复用）。
- mistralai/Mistral-7B-Instruct-v0.3（注意不是不存在的AI-ModelScope同名地址）。
- LLM-Research/Phi-4-mini-instruct。

使用官方HF一次性commit/checksum锁，权重大文件从ModelScope下载候选并逐文件SHA验证；小config/tokenizer/许可保持官方固定版本以避免候选格式化差异。所谓“ModelScope部署”不是免除官方身份/许可检查，也不是所有元数据都改信候选master。原Qwen锁存在就保留，不再pin主分支。Mistral仓库同时有原生consolidated与HF分片，准备工具只选择HF model/pytorch_model分片格式，避免重复下载两套权重。文件名相同不等于字节相同，SHA不一致保留partial并停，不能换SHA迁就候选。

## 3. 一次性串行准备所有当前模型

确认当前没有下载或模型进程后更新；在仓库无改动时：

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
bash scripts/cloud/deploy_research_models.sh plan
# pin新模型元数据仍可能需要AutoDL的HF加速；ModelScope域名由脚本NO_PROXY绕开代理。
source /etc/network_turbo
bash scripts/cloud/deploy_research_models.sh deploy
# 全部完成后独立只读复核：
bash scripts/cloud/deploy_research_models.sh verify
```

此入口沿用.venv-cloud8b，不增加ModelScope SDK或自动升级依赖。默认次序Qwen→Mistral→Phi；Qwen已有正确文件校验后跳过下载。任一权限/连接/SHA失败即停，已完成文件和partial保留；手动再次deploy仅继续当前固定资产，不自动循环重试、不覆盖既有错文件。不向ModelScope传HF令牌、不关闭TLS。若要只准备某些已列模型，可用--models qwen3_8b mistral7b；这时汇总all_roster=false，不能当三模型齐备。

全局GPU/传输互斥拒绝计算进程占用；不要在训练中改模型文件。私有状态experiments/cloud_8b_4090/deployment_state.json记录active_model、失败或selected_model_assets_verified_not_training_ready；每个模型官方锁和SHA收据仍在assets。不要为出错删除这些证据。尚未验证实际下载速度和所有候选字节匹配，不承诺ModelScope必定比官方快。

## 4. 部署与统一实验是两道门槛

该命令完成资产准备，不加载模型、不启动训练。接下来需统一两任务数据/标签与留出评价、原生chat template、结束token、assistant-only loss、模型模块映射、BF16/低比特计算/保存恢复和4090显存预检，并冻结55号正式矩阵；当前现成训练入口仍只有Qwen单条件，不能直接用于Mistral/Phi或int8。

Phi可能使用qkv_proj/gate_up_proj等融合投影；模型真实模块必须在云端读取核对，可训练参数逐次记录，不能用相同r/模块名字声称参数量对齐。Native Transformers支持优先，任何必要自定义代码先固定并审查，不默认trust_remote_code=True。新模型独立输出不覆盖Qwen或历史科学收据。

因此不提供伪装为已完成的“一键训练全部模型”命令。模型资产齐备后先提交统一兼容/训练执行方案再由所有者显式串行运行；分类与LaMP原完整评价和原始corpus_score统计保留，主观评价不伪造。当前云端尚未运行实验，此次仓库只准备资产与模型选择。
