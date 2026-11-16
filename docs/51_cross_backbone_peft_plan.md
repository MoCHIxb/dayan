# Llama-3与Qwen：跨基座PEFT云端对照方案

2026-11-16。所有者明确要求将开题计划中的Llama-3纳入后续云端研究。本文件将其加入研究范围；尚不是已冻结运行协议或可直接执行的队列。未启动模型、下载权重、连接服务器或更改既有运行实现。其余多领域/稳定性扩展仍按50号及交接中的暂停边界管理。

## 1. 与开题计划、已有成果的对应

开题书原文明确要求研究“不同基座模型（Llama-3, Qwen）下的秩（Rank）选择、缩放系数（Alpha）配置及量化精度（4-bit/8-bit）对适配效果的影响”。目前LaMP、SMS、安全示例和BANKING正式结果均为Qwen系列；没有Llama-3训练或效果证据。历史Qwen实验不能代替跨基座研究。

新增问题是：相同数据与训练预算下，既有LoRA/低比特配方在两种基座中的效果、资源取舍及相对各自基座的增益是否一致。贡献仍是受控实证与方法指南，不是新LoRA算法或架构因果结论。

## 2. 模型选择与比较边界

- Llama：Meta官方[meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。明确是Llama-3 8B Instruct，不静默换成3.1/3.2、预量化再打包或第三方微调版本。须所有者自行满足官方访问与许可要求，不在聊天或Git中传令牌/权重。
- Qwen：官方[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)，明确关闭thinking。8B量级主对照可减少与Llama8B大小不同带来的混杂，但实际参数、结构、预训练和指令调优仍不同，不能称架构单因素实验。
- 原Qwen3-4B/BANKING结果完整保留，作为已完成的4B参考；不是8B对照的复用训练，也不能将4B到8B差值全归于模型家族。若后续改为节约资源的4B对8B，须在运行前修改方案并明确模型规模混杂，不静默降级。

在云端准备前锁定两个官方仓库的完整commit revision、文件列表、实际SHA256、tokenizer/chat template、config、generation config及许可文件。没有锁定资产前不得进入正式训练。现有Qwen4B专用prepare_model_assets.py及其13文件锁不用于下载/认证新模型；旧资产清单、原始收据不改。

## 3. 有限主矩阵：两基座、每基座五配方

| 配方 | 主干精度 | r | alpha | alpha/r | 主要对照用途 |
|---|---|---:|---:|---:|---|
| A | BF16 LoRA | 8 | 16 | 2 | 较小秩候选 |
| B | BF16 LoRA | 16 | 16 | 1 | A→B固定alpha改变rank；缩放比同时变化 |
| C | BF16 LoRA | 16 | 32 | 2 | B→C固定rank改变alpha；A→C固定alpha/r比较rank配方 |
| D | 4-bit NF4＋LoRA | 16 | 32 | 2 | C→D量化配方对照，BF16计算、double quant |
| E | 8-bit＋LoRA | 16 | 32 | 2 | C→E与D→E比较精度/资源配方 |

共10次新训练，每配方先一种子。D称NF4 QLoRA式适配；E称8-bit主干LoRA，不把全部低比特训练混称同一种QLoRA算法。默认标准LoRA缩放alpha/r，固定use_rslora=False、bias=none、dropout0.05和七个q/k/v/o/gate/up/down投影。两模型实际可训练参数不同，逐次记录，不用相同r声称参数预算完全相同。

A→B不是只改变低秩容量而完全固定有效缩放；A→C同时改变rank与alpha。报告这两种控制视角，不能把任何一对夸大为完全隔离机制。BF16与低比特训练的数据类型准备可能有额外差异，记录实际模块dtype、优化器与prepare_model_for_kbit_training行为；称配方对照，不宣称纯量化因果或无损。

这不是2模型×所有数据量×所有领域×所有种子的全网格。正式主实验先只用现成BANKING77 card10固定n500。跨领域扩展与本矩阵分开排期。若要补稳定性，可预先固定D为共同主配方，各模型再增加两个种子，共4次额外训练；尚不自动加入首轮队列，不依据test挑最好的配方才补种子。

## 4. 固定数据、训练与评价

- 使用现有500train/100dev/400官方test，原始问题/标签、顺序、ID与文件SHA保持不变；不合成、不翻译、不重采样、不裁test。新模型tokenizer校验失败时停止，在预测前另行修订预算，不截断官方test或自动替换样本。
- 128优化器步、每16步完整100dev并保存，共8次；最低dev loss且同值取最早，最终128步全状态与最佳导出另存。lr1e-4、microbatch1、累积16、固定seed20261113、七投影、梯度检查点和共同环境。固定步数不保证两模型FLOPs或耗时相同。
- 同一10标签目录/定义和user消息内容，各基座使用其官方原生chat template。Qwen无thinking；Llama不用Qwen专属提示参数。最大输入1536、生成64、greedy、batch1为拟定共同预算，正式前须通过完整数据与固定10示例长度检查。
- 每模型每精度分别评价其无适配器基座：BF16、NF4、int8各完整400test。每模型另评价BF16固定10示例提示；示例仍取既有n50前10，每类1条，不用dev/test选例。轻量TF-IDF使用同一n500真实训练，原始预测/输入哈希核验后可复用，与模型家族无关。
- 五个适配器各评价完整400test，主推理主干精度与该配方训练精度匹配。各模型共5组适配器＋3组零示例＋1组BF16固定示例，两模型共18组LLM预测，不计重复基线为独立实验。NF4与int8适配器分别报告相对同精度基座、BF16基座及BF16提示参照的差值，区别量化影响与适配增益。
- macro-F1主指标，accuracy、每类precision/recall/F1、最低召回、invalid和混淆完整保留；invalid计错。模型之间主要比较相对各自基座/提示的增益和质量-成本曲线，不用绝对分数直接宣称微调方法优越。不得跨模型直接比较dev loss大小来选择获胜基座。
- 记录训练allocated/reserved峰值、参数量、实际dtype、加载/训练/测试各阶段时间、推理吞吐/延迟、token数和适配器大小；统一预热/计时规则并区分生成token数差异。峰值不是整机占用，模型文件字节不是运行显存；单3090可行性要以实际预检和运行核验为准。

目前BANKING test已被观察，新增研究按固定历史任务的后续对照报告，不称全新未见确认集。首轮单种子描述性研究，不新增显著性或bootstrap结论；LaMP54项原始corpus_score统计及20/24协议不变。

## 5. 运行前必须补齐的兼容层与云端门槛

当前接口有AutoModelForCausalLM和标准投影注入，但这不等于Llama运行已验证。既有train_qwen_lora.ChatSFTDataset带enable_thinking=False并拼接tokenizer.eos_token_id；旧评价器也有模型相关模板/终止逻辑。新实验须使用独立版本化模型适配层，保留冻结实现与旧收据，不能直接套用旧BANKING执行器。

1. 在CPU测试中验证各模型prompt、assistant边界、训练只监督目标、padding mask和结束token；Llama官方示例使用EOS及<|eot_id|>终止。训练目标结束与官方模板一致，生成同时尊重官方终止标记。Qwen关闭thinking但不剥离其他模型文本；不能靠输出修复隐藏格式失败。
2. 正式运行前在云端检查两tokenizer的全量训练/dev/test和10示例长度、七投影是否存在、目标token是否被完整监督；不加载本地模型做实验。
3. 仅在所有者AutoDL Linux RTX3090进行显式兼容/显存预检；一次一个GPU过程。8B BF16 LoRA可能接近24GB上限，当前未证实能在给定长度和环境下完成。OOM保留日志/现场、停止，不自动重试、降低长度/精度或删矩阵格。资源修订须另存协议版本，保留不可行性结果。
4. 实现独立资产准备、输入冻结、有限串行执行、暂停/完整checkpoint恢复和独立汇总；每新训练核查500/100、128步与8dev、最佳/最终/导出SHA，每组400预测检查ID/参考、命令、输入/输出及收据哈希。进程或状态文件不足以宣布完成。
5. 先完成云端模板/终止和小型独立功能检查，成功也不计入正式性能结果；随后才允许显式启动固定主矩阵。失败不自动重试，既有Qwen4B管线、配置和结果均不覆盖。

本文件只有方案，没有提供尚未验证的run命令或伪装已兼容的配置。可执行新云端入口与资产锁是后续实现项；所有者直接操作云端命令行，不要求SSH。其他多领域/安全/旧矩阵不随此请求恢复。

## 一手参考

- [Meta Llama-3-8B-Instruct模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)：官方聊天和终止示例、访问与许可。
- [Qwen3-8B模型卡](https://huggingface.co/Qwen/Qwen3-8B)：模型版本和无thinking用法。
- [PEFT LoRA文档](https://huggingface.co/docs/peft/package_reference/lora)：rank、alpha及缩放；实际执行依赖沿用锁文件，不能直接按最新文档升级原环境。
- [Transformers bitsandbytes文档](https://huggingface.co/docs/transformers/quantization/bitsandbytes)：NF4、double quant和低比特适配。新实现需按本仓库锁定依赖验证，不追改历史运行。
