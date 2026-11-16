# Mistral原生模板：显式system消息视图修订

2026-11-16。用户v2报告：Qwen与Phi各1400实际记录/示例全部通过；Mistral32层七投影存在，但1400条均Native assistant prefix/target boundary mismatch。不是长度问题、不是权重错误，也不是放宽计数就能通过的失败。

## 根因与固定规则

已只读核对官方Mistral-v0.3 tokenizer_config.chat_template：user分支仅在loop.last且存在system_message时将系统说明插入[INST]；构造推理prompt时末消息是user会保留system，完整训练序列末消息是assistant则不插入system。这导致两边prompt不同，甚至可能训练看不到完整任务目录。检查拒绝是必要保护，不应把这种差异简单当BPE边界误报。

新src/personal_llm_data/research_chat.py定义mistral_system_prefix_in_user_v1：仅对当前单轮system/user[/assistant]结构，以副本把完整system内容＋两个换行＋原user内容作为user编码视图，然后调用同一官方模板。训练和推理都应用同一规则；原JSONL、原客户文本、标签、system内容、示例与模型文件均不编辑。拒绝多轮/未知模型格式，不自动猜测。Qwen/Phi仍保留原生消息视图。

原生模板继续负责BOS/EOS/assistant格式；仍检查完整序列以prompt token精确开头、监督目标解码等于原JSON标签、完整长度1536/64不截断。报告显式记录chat_policy和chat_adapter_sha256；Mistral通过状态为cpu_contract_passed_with_explicit_adapter，不冒充native完全无改动。

## 云端只做一次显式v3预检

确认v2进程已退出，保留v2完整及分模型报告，不重新下载权重、不重新pin、不改数据。

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
source .venv-cloud8b/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="" bash scripts/cloud/preflight_research_models.sh   --output experiments/cloud_8b_4090/compatibility_preflight_chat_v3.json
cat experiments/cloud_8b_4090/compatibility_preflight_chat_v3.json
```

此修订不增加依赖，不必重装环境。新报告前缀独立保留先前故障和v2证据。若边界/长度仍失败则停止诊断，不手动改状态为通过。

## 训练与论文边界

该规则已接入CPU预检，后续统一训练dataset和推理prompt必须复用同一个message_view函数并冻结其SHA，不能预检用新视图、实际运行仍用旧Qwen专属helper。当前只完成预检修订，不宣称跨模型训练器或GPU验证已通过。冻结4B运行实现/数据/结果/原统计均未改变。

论文记录模型原生模板差异与固定规范化政策；比较的是相同任务内容在各基座的适配效果，不是完全相同token字符串或模板架构因果。此前没有8B正式模型运行，修订发生在正式预测之前。

本地148项CPU测试通过，包括原生末消息分支复现、修订前拒绝/修订后边界对齐、原消息不可变、训练/生成视图一致、原标签校验仍严格及多轮/未知策略拒绝。未在本地加载Mistral权重或伪造云端通过结果。

一手诊断来源：[官方tokenizer_config](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/tokenizer_config.json)。云端继续用既有锁定资产，不改为重新下载main。
