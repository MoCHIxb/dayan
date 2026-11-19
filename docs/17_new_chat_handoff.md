# 当前交接：IntentTune客服意图识别与RTX3090

2026-11-14更新。项目D:\dayan；[GitHub](https://github.com/MoCHIxb/dayan)。开题计划和所有者最新指示优先。完整MVP/商业软件不是必做项；交付可运行仓库、启动命令、配置、接口与方法指南。

## 当前入口

先读45号领域方案、46号BANKING77协议、47号AutoDL RTX3090脚本指南、48号数据卡。当前任务为IntentTune银行卡客服10意图，首轮云端聚合结果已导入复算，见49号；安全扩充暂停，39号已非主入口。详情与保留边界见文末2026-11-13最新覆盖。

## 先前接手入口（历史记录）

1. [35号三领域数据与云端指南](35_three_domain_text_safety.md)、[公开数据卡](../examples/text_safety_v1/README.md)。
2. [32号总方案](32_framework_master_plan.md)、[33号开发者接口](33_developer_framework_guide.md)、README和AGENTS.md。
3. `results/text_safety_data_verification.json`、`results/text_safety_delivery.json`与`results/local_progress.json`。
4. 既有研究解释23/25/30/31；27号仅是另行保留的平台复现，不是新示例执行队列。

## 最新范围

所有者明确取消使用vlm-long-context-safety-eval及其数据，不做图文或多轮攻击。34号、MMDS来源报告与旧模态问题均已取消，只保留历史记录，不能恢复。当前只使用已有论文/公开成果的纯文本数据，沿用Qwen3-4B。

三个领域：医学（DiaSafety Unauthorized Expertise），金融与公共安全（BeaverTails主题子集）。监督为现有Safe/Unsafe/is_safe标签转换为JSON；不新写请求、候选回答或审查理由。输入是单轮请求和待审查候选回答，回答作为引用证据，绝不作为有害回答的SFT目标。标签表示回答安全性，不能当请求攻击标签。医学标签不代表诊疗事实正确。

## 已构建数据

`examples/text_safety_v1/{medical,financial,public_safety}`各50训练/20开发/40测试，各集合Safe/Unsafe各半。joint为三者并集：150/60/120，共330条不同问答；不能把联合视图再算作额外330条。

问题组数依次train/dev/test：医学42/20/39，金融43/19/40，公共安全31/18/34；joint为116/57/113。组间无跨集合共享，官方test问题均从train/dev排除，医学官方val问题也从train排除。选中集合间穷举原问题字符5gram Jaccard检查，阈值0.9；是自动复核，不是人工审校或语义独立性认证。

330条文本和标签已回放固定源文件逐条核对；独立重建17份数据/manifest/构建报告，字节完全一致。当前76项CPU测试通过；独立CPU环境71通过/5因可选训练依赖缺失跳过。已完成独立32步本地GPU功能验收，详见下节及36号。

## 下一步云端

配置`configs/text_safety_v1/qwen3_4b_joint.json`，独立输出`experiments/text_safety_v1/run`。先训练一个联合150条适配器，不自动扩展三套模型或大网格。默认r8/alpha16、BF16、128步、每16步完整60条dev与保存；完整120条test，基座/适配器同预算；用`score_text_safety.py`计算各领域及总体macro-F1、不安全回答检出、安全回答误报和无效JSON。

训练入口无--execute只检查文件；显式执行时才使用GPU。服务器尚未租用/连接/执行，云端仍未执行；所有者随后授权的本地32步功能验收已完成。模型下载固定版本、环境准备、完整命令和暂停恢复见35/33号。不能把CPU验证当Linux4090/GPU已跑通，也不能先宣称审查能力强。

## 已完成且不重跑的历史研究

small_sample_v1：16训练、12所选适配器完整评价、36项原始bootstrap。peft_v1：9新训练＋3BF16对照，20组完整1496条test、9套辅助评价、18项各2000次原始corpus_score统计及独立最终解释。28条件客观辅助证据完成；原监控删除。

20/24号协议及冻结科学实现不改。新示例不缩减或替换历史dev1500/test1496/原始统计；旧六次全量队列、人工盲评、付费裁判和大规模通用矩阵继续暂缓。原平台复现cloud_4090_v1仍独立且未启动。

## 发布与执行边界

所有者授权选定三领域数据随仓库同步，窄范围例外仅为examples/text_safety_v1已审查文件。DiaSafety行保留上游Apache-2.0条款，BeaverTails行遵循CC BY-NC 4.0，归属和改动见NOTICE。源码仍未指定许可证，不能据数据许可宣称代码已正式开源。

原始全量数据、原VLM数据、模型、checkpoint、逐条模型预测与密钥不提交。`scripts/audit_publication.py`核对公开范围及快照；仅--update-manifest修改清单。一次只运行一个GPU任务，失败留痕不自动重试；已完成产物不覆盖。旧2026-10-28交接存于history/17_handoff_20261028.md，只作事实背景。

## 最新本地验收：已完成，不重跑

36号和results/local_text_safety_validation.json为当前证据。独立configs/text_safety_v1/qwen3_4b_local_verify.json，32步/每8步完整60dev，150train/120test；第9步完整暂停保存并核验后恢复，最佳32步适配器哈希与checkpoint一致。基座/适配器各120条完整test与3条移除gold的独立dev推理通过，进程退出、GPU已释放。结果accuracy70.0%→81.7%、macro-F1 0.677→0.816、不安全检出43.3%→76.7%，安全回答误报3.3%→13.3%。一轮小样本描述性结果，不声称强安全或无遗忘。

修复WDDM桌面进程误拦；未知/Python/模型服务仍拒绝，跨进程锁保留，infer入口也受锁保护。不要修改本轮捕获的训练输入源码后尝试沿用旧收据；代码/数据/模型哈希已记录。GPU原始证据位于experiments/text_safety_v1/local_verify，checkpoint-9暂停副本在pause_evidence。日志、逐条预测、权重不发布。

未来云端使用原128步配置与独立run，不把本地32步和云128步成绩解释为硬件对比；Linux/4090尚未执行。新环境复现和许可证决定仍待完成。

## 2026-11-03第二领域迭代：已完成，不重跑

所有者授权新增通信领域SMS分类，37号固定500train/200dev/1000test，Qwen3-4B单次128步。实际训练、8次完整dev、基座/适配器各1000test及独立CPU审计完成，见38号与results/sms_spam_results.json。新增score-classification与scripts/audit_classification_run.py，不改旧科学实现。首次GPU预检拒绝两个桌面进程，所有者明确授权核实并修正、重试一次；私有路径+SHA注册表保留远控，原失败/原manifest/operational_revision均保留。当前全部GPU阶段结束，不重启该任务。

结果macro-F1 0.7986→0.9566，spam召回0.9328→0.9179，ham误报0.1305→0.0104；门槛全部达到，完整取舍与局限见38号。短信原文、日志、预测和模型私有，仅公开代码、配置、聚合与哈希。旧3181101本地验收的源码哈希按原提交核验，不改旧收据。

## 2026-11-03清理与云端准备

当前云端主入口39号、configs/cloud_sms_spam_v1/qwen3_4b.json；首轮只复现SMS500/200/1000、128步，不启动旧LaMP12训练矩阵。prepare_cloud_sms负责固定资产并在匹配冻结SHA的前提下处理manifest换行差异；check_cloud_sms_assets默认只CPU，--check-gpu才核验真实Linux4090。尚未租机、连接或执行云端训练。

所有者明确确认约6.3GiB具体清理范围后完成整理。停用Qwen3-1.7B/MiniLM权重按固定上游版本与SHA可重取，元数据已归档；旧检索数据和大日志无损归档后删除原件，重复重建副本/旧CPU环境/缓存清理。正式SMS、LaMP BM25、4B、所有checkpoint/适配器/原始统计/预测/收据保留；权限保护的pip缓存不强制改ACL。恢复与独立保全核验见40号及results/workspace_cleanup_20261103.json。原始reproduction_check/data_rebuilt已删除，对应哈希/再生证据在维护manifest，不能把重复副本缺失误判为原始数据丢失。

## 2026-11-04框架与论文迭代

所有者要求继续完善框架与撰写论文。新增CPU只读doctor统一入口、41号操作与后续计划、42号论文v0.1及43号主张证据映射。论文定位为既有LoRA上的实证/工程研究，不称新算法；云端结果不填猜测数值。README/33/32统一39号单轮SMS为当前入口，旧27号LaMP矩阵不是队列。测试与检查见results/framework_iteration_20261104.json。没有启动GPU、改冻结协议或原运行核心源码；真实个人/用户研究、云端实测及代码许可证仍待完成。

## 2026-11-13最新覆盖：IntentTune与BANKING77

当前主线改为客服意图识别，不再追求通用框架或扩充安全语料。框架工作名IntentTune；45号方案、46号固定card10协议、47号AutoDL RTX3090全部阶段脚本、48号数据卡优先于前文旧云端入口。所有者已准备3090，尚未提供/核验本项目连接；本轮没有云端或本地模型实验。

选定10类银行卡客服原始问句/标签，嵌套50/200/500训练，共享100dev/400官方test，1000不同源记录。官方源固定版本，14文件独立重建完全一致，原文/标签回放通过；见results/banking77_preparation.json、banking77_data_verification.json和repro/banking77_assets.json。原文私有，只上传构建代码、配置、聚合与SHA，不扩大安全数据公开例外。

云端脚本默认plan，显式prepare/check/run/report。run含固定10示例基座、三档轻量对照、三训练、每run完整base/adapter、评分与审计，共22阶段，无旧训练队列。运行/暂停/恢复/失败留证详见47号。Linux/3090仍未实测，不能凭CPU检查称训练完成或效果好。

论文主线：LaMP研究提供小样本/配方可行性证据→客服领域框架→BANKING实践验证。42号已加入当前范围与待填实验位置；安全/SMS仅历史证据。GitHub正规维护为当前目标，立即正式开源和许可证决定不阻塞工作。既有结果与原始统计不改不重跑。此前Aegis仅探查源文件私有保留，未构建安全v2或加入任何队列。

## 2026-11-14云端下载故障与恢复

用户在AutoDL自行命令行操作，不需要本地SSH。已完成克隆/环境准备；HF原站不可达、镜像大分片约8%断连，训练未开始。新增prepare_model_assets与prepare --model-source modelscope，严格复用原13文件SHA，Range续传保留partial，失败不自动重试。详见47号第5节。用户须云端实测恢复，不能将本地网络探测/CPU测试当作云端恢复完成。原模型/数据/实验配置/六个核心源码不变。


## 2026-11-14最新结果交接（覆盖此前待运行状态）

首轮RTX3090已由所有者报告完成22阶段，聚合文件已导入。本地复算10组混淆矩阵、核对固定资产及24次dev选择和checkpoint哈希声明通过；未导入云端逐条预测、checkpoint和阶段收据，未作本地逐文件重审。结果见[49号](49_banking77_results.md)与results/banking77_card10_results.json。500条LoRA macro-F1为0.9401，零示例0.8360、10示例0.8916、同档轻量0.8947；50条优于200条，只有单种子10类证据。

原始results.json从experiments/banking77_v1移到experiments/intenttune_banking77_v1/imported_cloud，字节不变，SHA256为3a9b0a6d5d3a880548401b822afebc70e025629248a011e719a7fbda29d3f80f。目录仅为导入归档；import_receipt.json是本地迁移记录。公开只含聚合，不含客户文本、逐条预测或模型。CPU测试113项通过；未启动模型或修改冻结协议/历史统计。下一步优先私有保全云端原始产物、作者审阅论文。其他模型/种子须先固定独立方案，不据本轮test反复选配方。


## 2026-11-14暂停与开题计划复核（最新接手依据）

所有者已要求先暂停、回顾项目。本节只记录讨论结论与后续建议，不是恢复实验的指令；没有启动新模型任务、建立自动队列或修改冻结协议。恢复须等待所有者明确指示。当前已有结果、原始统计与失败证据继续保留。

最初开题计划书仍可从本聊天中所有者贴出的内容阅读。其第四阶段时间安排在“r=8 vs…”处截断，不能据此声称掌握完整原文；三条核心研究方向清楚：PEFT适配与资源优化、50–500条小样本数据与适配、多维评价及方法指南。原计划提到的所有模型、精度和评价方法不必全部铺成实验矩阵，按当前IntentTune范围补足关键证据。

### 原目标、已有成果与主要缺口

| 开题书目标 | 当前成果 | 仍缺的关键证据 |
|---|---|---|
| 消费级硬件低成本适配 | Windows5070Ti与Linux3090分别完成不同任务 | 完整流程成本：加载、训练、评价、存储及推理；不是同任务硬件单因素比较 |
| LoRA/QLoRA及Rank、Alpha研究 | LaMP关键配方对照、质量/资源解释与原始统计完成 | 历史配方取舍能否延伸至当前客服任务尚未验证 |
| 50–500条小样本适配 | LaMP及BANKING已有数据量实验 | 客服任务目前单种子，50条优于200条，不能主张单调收益或稳定最优 |
| 风格、内容与任务评价 | 生成评价、客观辅助诊断和完整分类评价已有 | 深层个人风格、事实正确与通用能力保持未获证明；不自动恢复暂缓评价 |
| 可复用框架与最佳实践 | 配置、训练、dev选模、暂停恢复、导出、评价和审计已建立 | 部分入口仍依赖BANKING；方法指南需要明确选择依据和适用边界 |

### 建议的后续优先级（尚未启动）

1. **多领域、同任务验证**：按[50号草案](50_multi_domain_validation_plan.md)，以银行、旅游、企业内部服务意图识别为候选，统一数据契约、训练预算及零示例/固定示例/轻量模型/LoRA四类对照。每领域独立适配器、完整测试；不是银行适配器无训练跨领域识别。CLINC两个领域共用一个来源，不称三个独立数据集。新来源revision、许可、类别、数据审计和配置尚须正式固定。
2. **稳定性与失败解释**：优先给预先固定的500条主比较补充种子，不展开所有档位、模型和参数的大网格。50条优于200条先检查类别错误、样本覆盖、完整dev曲线和选择过程；不能直接归因为过拟合。若研究机制，另立小型受控方案，不依据已看test反复追分。50号建议总计12次新训练仍只是预算草案。
3. **客服任务的资源取舍**：可在固定500条条件下补一个BF16与NF4配方比较，分别报告质量、allocated/reserved峰值与时间，把历史LaMP经验连接到当前任务。此为新增建议，尚无协议、配置或运行授权；不重复完整Rank/Alpha网格，不将配方比较夸大为纯量化单因素或无损证明。
4. **完成可决策的方法指南**：说明真实数据整理/授权、留出与泄漏检查；何时提示或轻量模型已足够；训练预算/显存配方/checkpoint选择；收益与类别退化判断；适配器导出、独立加载、备份和复现。补充推理延迟、吞吐、适配器大小、加载与完整评价耗时，区分已有训练会话时间和端到端成本。工程先将BANKING专用标签目录、输入抽取、基线与执行入口参数化，不改历史运行收据或冻结核心。

### 论文范围与交付定位

当前“私人定制”验证应准确描述为用户自定义标签体系下的服务意图任务适配；个人写作风格属于原课题背景和历史研究方向，不冒充已经完成真实个人风格验证。论文论证链为：历史实验说明小样本PEFT可行 → IntentTune规范适配流程 → 多领域对照验证复用价值 → 质量与成本实验形成方法指南。

现有框架贡献是可执行、可审计的领域适配流程，不是新LoRA算法。没有无框架训练的受控消融，不能将全部模型增益因果归于框架；没有用户研究，不能宣称易用性或节约操作时间已实证。无需为贴合原始措辞而重启风格训练、医疗安全扩充、人工盲评、付费裁判、大规模通用能力矩阵或旧队列。正式GitHub维护继续，GUI/商业MVP和立即正式开源仍不是必做项。

恢复后先确认上述优先级，再准备新协议和有限云端脚本；所有新增模型实验仍在所有者云端RTX3090显式执行，不在本地启动GPU任务。云端原始预测、checkpoint和阶段收据需私有保全，本地聚合复算不替代逐文件审计。


## 2026-11-16新增授权范围：Llama-3/Qwen跨基座对照

所有者确认希望补足开题书明确提出的Llama-3与Qwen、LoRA/低比特、Rank和Alpha研究，并在云服务器运行。现有正式实验均无Llama-3；历史Qwen不能替代该目标。新增[51号方案](51_cross_backbone_peft_plan.md)优先作为此方向的设计入口，不修改46号或既有统计。

建议采用Meta-Llama-3-8B-Instruct与Qwen3-8B的同量级主对照，Qwen3-4B作为历史参考，避免把4B/8B差值写成纯模型家族差异。固定BANKING77 n500/100dev/400test，每基座BF16 r8/a16、BF16 r16/a16、BF16 r16/a32、NF4 r16/a32、int8 r16/a32，共10次新训练；各模型/精度独立基座与固定示例参照，详见51号。首轮单种子，稳定性另作有限扩展，不混入全部数据量/领域大矩阵。

这次要求解除暂停只覆盖新增对照的方案/准备方向，不恢复其他旧队列；当前没有启动GPU、下载权重或连接服务器。新模型revision/SHA、访问、官方聊天模板/结束token、训练目标mask、长度和3090显存都尚需验证。不能直接替换路径后调用旧Qwen4B队列或声称Llama已兼容。独立适配层与有限云端可执行入口仍需实现；BF16 8B/24GB可行性待实测，OOM保留并停止不自动改条件重试。云端由所有者自行命令行操作，不请求SSH或令牌。


## 2026-11-16最新主基座调整：后续以Qwen3-8B完整重建

所有者要求后续所有实验以Qwen3-8B为主再次完整开展，当前先审阅方案、做好准备，未要求立即启动训练。[52号8B主线审阅稿](52_qwen3_8b_campaign_plan.md)覆盖50号后续4B主基座设计及51号预算，51号跨基座方法边界保留。已有4B结果/配置/原始统计不改，不称其为8B证据。

52号建议三领域×50/200/500×三训练种子共27次Qwen8B主实验；BANKING500的五配方三种子复用共同A三次后额外12次，共39次Qwen；Llama五配方×三种子15次，总计54次。首种子全覆盖18次，之后补36次。54次仅为待审阅完整预算，不是自动运行授权。历史LaMP/SMS/安全是否也全部以8B重跑已向所有者澄清，答案到来前不安排历史GPU队列。

机器可读草案results/qwen3_8b_campaign_plan.json标记revision未固定、runner_ready=false、新CLINC子集尚未构建。下一步先确认范围，再锁模型资产、数据、模板/监督mask/终止、长度与3090显存并实现独立云端入口；不得直接把4B旧执行器换路径后运行。此次没有启动GPU、下载权重或连接云端。正式实验均在所有者云端串行显式执行，失败留证不自动重试，其他暂停任务不自行恢复。


## 2026-11-16所有者已确认：全部历史实验也用8B重跑

所有者已回答范围问题：“全部历史实验也用8B重跑，包括LaMP、SMS和安全”。这覆盖上节“历史范围待确认”，不再把历史模块默认为保留4B而不复现。当前请求仍是做好准备、先审阅方案，未启动GPU任务。主线/预算以[52号第7节](52_qwen3_8b_campaign_plan.md)和results/qwen3_8b_campaign_plan.json最新字段为准。

已清点正式历史模块：LaMP小样本16次、关键PEFT至少额外9次（须核对新8B共同对照）、SMS128步1次、安全联合32步功能暂停恢复1次、安全独立128步正式预算1次，共至少28次，加IntentTune主线54次得到至少82次、Qwen至少67次/Llama15次。82不是最终总数：早期已完成LaMP两epoch各数据量、全数据rank/alpha/量化等仍须逐项完成证据清点，增加预算或完全相同条件去重复；不能把漏清点的研究称已覆盖。失败调试、取消任务和从未执行旧队列不当已完成科学条件，不调用旧队列恢复。

保留4B原始配置、协议、checkpoint/预测与统计，不改20/24等冻结文件。新8B重新依据dev选学习率/最佳步，不继承4B选择；同配置/种子/数据/资产匹配才复用新8B共同对照。历史任务保留自己的完整样本数/长度/步数和风险指标，不统一缩成客服预算。

LaMP新8B完整1500dev/1496test、画像/无画像、原始全指标/PPL/BERTScore、错误/风格/复制和43/1453分层；新8B预测单独运行2000次原始corpus_score配对bootstrap，不能替代或重写4B既有54项，不能用句分数均值近似。确切新比较清单须连同早期8B调度基线一起固定，不能混用4B历史预测冒充8B内部对照。分类不因此自动新增bootstrap。安全复用原330真实QA及联合视图，不合成、不扩充来源、不恢复VLM。

下一步为全部完成证据清点、新8B运行/评价映射、模型资产锁、兼容与3090资源预检设计；仍不自动启动训练、租机或请求SSH。各任务独立新输出，云端显式串行，失败留证、不自动重试。全部历史的最终训练/生成/统计和时间/磁盘预算待清点及云端测量后固定。


## 2026-11-16最新云端硬件：RTX4090双8B

所有者重新租用Ubuntu22.04/Python3.12/PyTorch2.12.1/CUDA13.0、单RTX4090 24GB、16vCPU/62GB RAM。后续Qwen3-8B与Meta-Llama-3-8B-Instruct均以此环境为目标，覆盖52/51中未来3090硬件；旧4B/3090结果不改。准备入口为53号、configs/cloud_8b_4090/environment.json及scripts/cloud/prepare_8b_4090.sh。独立.venv-cloud8b复用镜像torch，不直接安装含旧torch的requirements-train.lock。

已实现plan/bootstrap/check/pin/download/verify，无训练动作；模型锁固定官方commit/checksum、私有保存、已有错字节拒绝覆盖。当前仅本地CPU验证，不宣称云端安装、bitsandbytes CUDA13计算、模板或模型训练通过。双模型训练/评价执行器仍待实现，不直接调用旧Qwen专用执行器。全部历史8B范围保持，失败保留、无自动GPU重试；只由所有者云端命令行执行，不请求SSH或token。


### 2026-11-16双8B下载来源更新

所有者因官方速度过慢明确授权ModelScope候选下载。prepare_8b_4090.py新增download --source modelscope，Qwen/Qwen3-8B与LLM-Research/Meta-Llama-3-8B-Instruct的大权重仍按既有官方LFS SHA256逐文件验收，小config/tokenizer/许可保持官方commit；不能重新pin改掉官方锁。复用旧安全Range/SHA传输原语而非4B资产清单，保留HF缓存和候选独立partial，失败即停无自动重试；不向候选发送HF凭据。121项本地CPU测试通过，尚无候选云端验收结果，未启动模型。53号第4节为最新命令。


## 2026-11-16先运行Qwen8B首轮BANKING500

所有者报告Qwen8B14文件asset_bytes_verified_not_training_ready，授权先做Qwen实验。54号为新入口，run_qwen8b_banking500.sh只运行BF16 r8/a16/seed20261113的一个500/100/400、128步/8dev完整四参照实验，独立数据/输出，不启动全部大矩阵。复用Qwen3支持的冻结训练/评价核心，不改其源码；Llama/int8仍另待兼容。prepare以8B tokenizer重建后强制十份原JSONL SHA与4B一致，再完整源回放及长度检查；新tokenizer manifest独立。

新增入口包含8阶段收据、源码/数据/模型/环境版本冻结、完整测试评分与审计、暂停/完整checkpoint显式恢复，失败无自动重试。125项本地CPU测试通过；首轮尚未在云端实测，OOM须保留日志停止，不自动改精度/长度。下一步由所有者云端prepare/check/run执行，运行后冻结期间不再pull。完成后提供本轮results.json，不把状态文件或资产校验当训练完成。


## 2026-11-16最终覆盖：研究优先，云端未运行，撤回框架主线

所有者再次提供开题原文，明确以多个基座PEFT比较为实验主体、主客观多维评价为另一主线；通用/专业微调框架不是计划范围。随后明确云端尚未运行，先调整实验目标。当前以55号新研究方案为主，覆盖旧IntentTune/CLINC/全历史重跑与至少82次预算；不再自动要求全部历史8B重跑。旧成果与原冻结协议完整保留，绝不改名成8B结果。

Llama访问被拒，不绕权限；建议Qwen3-8B与官方Apache-2.0 Mistral-7B-Instruct-v0.3作为主对照，Phi-4-mini作为可选规模补充。主矩阵待审：BANKING两基座五配方三种子30次＋LaMP两基座A/D三种子12次=42次；BANKING50/200曲线可另加12次，第三模型/注入位置不全交叉。模型获取、模板/结束token、训练mask、量化与4090可行性须先验证，当前代码不宣称支持Mistral。Qwen8量化实验没有完成证据，目前仅有环境和14文件资产验收。

LaMP完整1496test/1500dev、原corpus_score2000次配对统计与辅助指标保留；分类不能证明风格。主观评价是拟新增协议，不自动恢复旧盲评或授权付费judge；可规划真实60题/两评者，未评分不能称已完成。历史七投影不等同计划书的仅注意力层，须解释方法差异或另立受控注入位置比较。

已无损归档7份过时规划，原路径保留指针，README与INDEX换成研究入口；旧2份未被当前代码/测试调用的盲评脚本转为.py.txt历史原文。记录results/research_cleanup_20261116.json含原路径/归档路径/SHA；冻结核心、当前Qwen8单条件入口、原始数据/模型/checkpoint/预测/统计/失败现场均不动。旧队列因依赖和哈希保留原路径，在scripts/README.md明确非当前启动入口。本次不是磁盘清空，没有删实验资产。

当前无GPU任务由本助手启动。下一步先审阅55号研究问题、模型/任务与评价定义，再冻结新跨模型/量化协议与实现，不直接恢复旧54/82次队列。


## 2026-11-16最新模型清单与统一部署

所有者要求更新远程并使用ModelScope一次性准备模型；提出LLaVA后又明确取消，不做多模态。56号为当前模型选择与部署入口，configs/research_models.json选Qwen3-8B/Mistral7Bv0.3/Phi4mini三个纯文本模型，Phi作为规模补充；DeepSeek蒸馏与Granite8B仅可选，未加入下载。主流品牌不代表独立架构，DeepSeek7B蒸馏基于Qwen，需披露。

新增cloud/deploy_research_models.sh默认plan，显式deploy/verify串行复用已有Qwen官方锁和已校验文件，再固定/准备Mistral与Phi。大权重ModelScope候选精确对官方SHA，小元数据保留官方commit；失败即停/保留partial，不绕过授权。模型仓库存在和文件名只读核查已完成，候选全部字节/云端下载尚未核验。全局GPU锁保护，原Qwen单条件科学源码不改。

资产准备不等于统一训练器完成，Mistral/Phi的模板、target mask、终止、融合projection/量化和显存仍需验证。不得直接套旧Qwen专用入口或宣称训练已跑通。LLaVA从未部署，取消VLM来源仍禁用。当前仍无GPU模型任务由本助手启动。


## 2026-11-16资产齐备、扩容与下一步CPU预检

所有者提供Qwen14/Mistral12/Phi13文件及all_roster=true的云端验收摘要；GPU4090空闲，原磁盘剩14GiB后又报告扩容50G，尚待df确认实际挂载容量。三个模型均仅资产通过，training_ready=false；当前无正式8B训练结果。57号新增真实数据/模板/监督/长度及meta模型LoRA投影CPU预检，先复用prepare构建并核对原BANKING划分，不恢复旧run。

预检不加载模型权重或使用GPU，全部本地逻辑测试通过不等于真实三tokenizer/meta兼容已通过。云端结果compatibility_preflight.json需核对；若原生Mistral模板或Phi融合投影失败，记录后固定适配规则，不能静默更改数据。CPU成功仍需独立GPU功能/保存恢复/资源预检与新统一训练入口，当前不启动大矩阵。


## 2026-11-16完整准备命令清单

58号集中当前服务器环境/隐藏GPU、BANKING固定源重建与三模型CPU预检、LaMP私有500/1500/1496导出上传导入、可选固定BERTScore快照、环境/容量记录、新机环境与GPU空闲后动作。新增import_cloud_data.py只接受repro/local_assets.json冻结的三数据文件与私有manifest，先全SHA核验再写，同SHA复用/不同文件不覆盖，拒绝越界/非普通/额外成员。没有创建或上传真实数据包。

当前GPU忙，命令使用CUDA_VISIBLE_DEVICES空串与python直接CPU动作，不调用强制设置0的旧训练wrapper。LaMP不使用默认k8重新生成，不把BANKING预检当LaMP也通过；BERTScore只下载固定资产，实际评分的revision绑定仍待新入口实现。SMS/安全历史不自动纳入。主训练统一接口未完成，不提供伪一键全部训练。


## 2026-11-16 Mistral tokenizer依赖故障与明确修复

云端CPU预检在Mistral实例化失败，尚未GPU/训练。已核对官方配置LlamaTokenizer/add_prefix_space=true/legacy=false与Transformers4.57.6源码：该配置触发from_slow SentencePiece转换，此前环境遗漏SentencePiece/protobuf。新增requirements-tokenizers.lock（sentencepiece0.2.1、protobuf4.25.8）并加入cloud依赖，不修改模型文件、分词参数或原始锁。59号为补依赖、pip check及独立v2报告一次显式重试命令；不是自动重试或训练恢复。

预检新增每模型进度与failure报告，保留前序结果并继续抛错，不吞掉失败或跳过Mistral。141项本地CPU测试通过，尚无云端修复成功证据，接下来仍须核验原生模板/标签边界、融合模块和真实GPU兼容。


## 2026-11-16 Mistral1400条模板边界问题修订

用户v2报告Qwen/Phi完整各1400通过、Mistral投影正常但全1400边界不匹配。已核对官方模板：system仅user为loop.last时插入，导致prompt有system而含assistant完整训练序列不含。新增research_chat.py版本化mistral_system_prefix_in_user_v1，单轮视图复制完整system前置原user，再用原生模板，确保训练/推理一致，不改JSONL/标签/官方文件，不放宽严格检查。预检记录policy和源码SHA，Mistral适配通过状态单列。

60号为云端独立v3一次显式重试命令，保留v2与此前failure。148项CPU测试通过，但无云端v3结果；统一训练/推理必须复用同一message_view，当前GPU训练接口仍待实现，不因为预检修订就宣称正式训练已可全模型运行。原4B冻结核心与统计未改，无GPU启动。


## 2026-11-16 v3 CPU通过，下一步独立GPU诊断

用户提供v3总状态cpu_contracts_passed_gpu_unverified，三模型资产与真实数据/原生或显式模板契约通过，但没有GPU/正式训练证据。61号新增gpu_probe_research.sh默认plan、--execute仅9条件诊断：3模型×BF16/NF4/int8，各取既有前2条真实train一步更新保存，然后独立新进程核验哈希并恢复optimizer/scheduler/torch/Python/NumPy RNG，推进第2步和一条真实训练prompt生成。绝不计test成绩/正式训练次数。

所有GPU在云端显式串行、全局锁保护；已有目录拒绝重复执行，失败停留日志不自动重试/降精度。CPU v3/registry/模型锁/数据/核心/消息视图及依赖版本绑定，原始资料不改。诊断成功也不证明全长度/完整训练显存或不中断等价；统一正式矩阵仍待后续冻结。用户云端尚未反馈GPU诊断结果，本地仅CPU逻辑测试，无GPU执行。


## 2026-11-16九条件GPU诊断及收据核验通过（云端回报）

用户报告九条件GPU诊断完成且all_conditions=true，随后只读校验所有inputs、train/restore收据、checkpoint文件和生成SHA，All 9 diagnostic receipts verified。诊断allocated GiB：Qwen BF16/NF4/int8=16.05/9.14/13.39，Mistral=14.01/6.63/10.56，Phi=7.85/4.98/6.78。62号及results/gpu_probe_owner_report.json只记录所有者回报，本地没有原云端文件，不宣称本地独立重审。

功能检查已完成，不再重复；诊断r8/a16、2条train、步骤1保存/新进程恢复步骤2，不算正式128步/完整dev/test，不提供质量结论，不证明完整1536长度/r16显存或与不中断数值等价。下一步实现并冻结正式统一训练/评价/审计入口，遵守55号研究主线与三模型不同模块/原生渲染；当前不自行运行旧Qwen或大队列。保留GPU诊断目录/收据私有备份。


## 2026-11-16 三模型正式BANKING500首种子入口已实现

所有者请求提供直接运行的正式实验命令。新增63号独立协议与configs/research_banking_v1共15配置：Qwen/Mistral/Phi各A–E，BF16 r8a16/r16a16/r16a32及同r16a32 NF4/int8。固定真实500/100/400，128步、每16步完整dev、八次最早最小loss选模；原生目标含EOS、prompt mask、Mistral既定视图、无截断。Qwen/Mistral七投影、Phi融合四投影，不声称等参数或架构因果。

cloud/run_research_banking.sh check核验已有CPUv3、九项GPU诊断原收据与保存文件、模型实际SHA、真实来源回放及三模型4200条编码与探针实现一致；run显式串行58阶段：1CPU质心+12基座/fewshot+15训练+15完整test+15审计。输入/原命令/输出冻结，保留最终128步与最佳全状态，导出adapter SHA一致，逐条ID/reference与严格JSON指标重算。失败不自动重试，显式暂停在既定16步保存，不增加dev选择次数；生成整组完成后暂停。仅阶段间或已核验的显式暂停可恢复。

新增166项CPU测试（5项环境条件跳过）通过，包括掩码/探针一致、长度拒绝、八次选模、防篡改、失败留证和禁止自动重试、阶段间暂停。正式云端训练未执行，未宣称Linux完整128步/r16资源/Trainer恢复已验证；此前r8两步诊断与新正式证据区分。原registry/chat/probe/20/24/46和4B科学文件未变，GPU未本地启动。LaMP全dev/test及原始bootstrap、多种子/50/200训练、主观评价仍待后续独立方案，不恢复旧82队列。

云端先git pull，再bash scripts/cloud/run_research_banking.sh check、plan、run。manifest冻结后不要更新代码或改环境；查看experiments/research_banking_v1/execution_state.json与attempt日志。完成后同入口report重核，机读results.json/可读results.md均私有。下一位不得将首批完成等同整个课题完成。


## 2026-11-16 Rank/Alpha与多种子扩充、进度更新

所有者询问更广Rank/Alpha与多种子，并要求维护进度。新增64号与results/research_extension_plan.json：推荐BF16固定alpha/r=2的r4/8/16/32；固定r16的alpha16/32/64；合并6个BF16+已有NF4/int8两条件=每模型8配方。三训练种子预先列20261113/20261117/20261123，数据选择/划分seed仍20261113、500/100/400与few-shot不变。3模型×8配方×3种子=72次规划，已有A–E首批15包含其中；补已有配方两种子30次，新增三配方三种子27次。首批若全部核验通过则新增57，不能在未核验情况下直接减15。报告逐种子、均值/样本SD与配方差值，不取最佳种子、不按test删配方；3种子和原始配对bootstrap回答不同不确定性，不能互替或自动称显著。

当前已完成三模型资产/CPUv3/九项GPU两步诊断（所有者回报），63号15次/58阶段入口已推送a24b57e，166项CPU测试含5项条件跳过。尚未收到正式run状态/日志/结果：实际是否启动、当前阶段未知，不能声称“云端肯定未启动”或“正式已完成”。本地无新增模型运行。扩充为设计记录，独立配置、执行器、r32资源/全状态预检、跨版本来源感知复用审计尚未实现，不提供虚构可执行扩充命令，不自动启动72队列。

63号被当前manifest绑定，不改其协议/配置/训练核心或运行中的脚本；原15次配置仍只接受20261113。后续须独立目录和manifest，旧收据按原Git版本核验，不覆盖SHA或重跑已核验实验。64号更新预算解释，55号早期42/54仅历史建议，现有58阶段含义不变。50/200、LaMP完整多维评价/原始corpus_score bootstrap、主观评价和论文更新仍待独立阶段。先审计首批，再补种子和扩充；不得将72当整个课题已完成门槛。


## 2026-11-16 最新：直接换用五档Rank/Alpha全因子、五种子大矩阵

所有者可接受长时间训练、优先详细可靠趋势，要求忽略两档Rank起步并直接修改训练脚本。65号及configs/research_banking_v2/grid.json为当前执行依据，覆盖63号15次和64号72次方案。五Rank=4/8/16/32/64，五Alpha=8/16/32/64/128，全部交叉，再×BF16/NF4/int8×Qwen/Mistral/Phi×5种子20261113/17/23/29与20261205，共1125训练、1137完整模型生成、3388阶段。数据选择仍20261113及原真实500/100/400，128步、8次完整dev/最早最低loss选模和完整test严格评分不变。225单元均需5种子才输出均值/样本SD和同seed相对r16/a32差值；不按test删配方/取最佳seed，不宣称全局最优或自动显著。

新独立scripts/research_grid.py、train_research_grid.py、evaluate_research_grid.py、audit_research_grid.py、gpu_probe_grid.py及run_research_grid.py，当前包装为scripts/cloud/run_research_grid.sh。1125配置仅prepare生成到私有experiments/research_banking_v2/configs，public只保留grid设计。plan不执行，prepare/check仅CPU/源回放/实际模型SHA/4200编码核验，probe新增r64/a128最长两条真实train保存step1/新进程恢复step2及生成；原r8诊断不冒充r64。run所选模型/精度须新probe通过；默认全量，支持--model/--mode/--rank的25训练块，后续块--resume，全局最终仍核验全部1125。

按官方safetensors矩阵头估adapter参数/存储、核对实际LoRA参数量。保留最佳/最终全状态/导出和收据，完整保留预计TB级空间，150G不能称足够全批；容量不足停止，不自动删证据或减少参数点。训练暂停在既定16步、生成完成400条；新增会话结束/累计时间峰值证据、partial report不改失败状态，失败留现场不自动重试。check与每独立worker起止全SHA核验模型；频繁阶段轻量哈希不以mtime冒充全字节重审。

180项CPU测试运行、5项环境条件跳过；新增14项验证1125完整交叉/3388数量、私有配置防覆盖、五seed/配对统计、磁盘预算、失败禁止重试、暂停会话累计和partial报告不冒充完成。没有本地GPU训练，新r64/正式云端结果未提供，当前实际进程/阶段未知。旧GPU探针、v1训练/评价/审计及20/24/46源码保持原SHA，未覆盖诊断数据或运行中脚本；第一次批量生成被自动审批误判会改旧探针而拒绝，未执行，后改成明确只新建独立副本并核验原SHA。

种子是伪随机起点：变化LoRA新增矩阵初始化、dropout和训练采样顺序，不随机重建基座、不换数据集，也不是同一训练5个checkpoint。保持相同记录测试5次独立优化随机性；不替代LaMP原始corpus_score配对bootstrap的测试样本不确定性。更多网格/种子不弥补单任务500/400限制，LaMP生成、多维主观评价、50/200规模与论文仍为后续独立工作，不把1125批次当整个课题完成。若原v1在云端运行，先原入口pause/核验退出再更新，不先重跑旧15；旧结果保留且不当v2额外seed。


## 2026-11-16 最新保存偏好：分析证据即可，不永久留1125套检查点

所有者询问是否必须全量保留并明确本轮只需足够分析的数据。新增66号/analysis_retention.json及scripts/cloud/run_research_analysis.sh（独立run_research_analysis.py、compact_research_grid.py），1125科学训练/五种子/原数据和原v2训练评价核心不变。每条件训练→完整400test→原始全状态/8次dev/选模/预测独立审计→写保留文件SHA和精确删除意图→释放checkpoint的adapter_model.safetensors/optimizer.pt/scheduler.pt/rng_state.pth及非代表导出权重；保留r16/a32/seed20261113的每模型/精度9套export。新控制总阶段4513，完整分析完成标签complete_grid1125_analysis_evidence_verified，不冒充原二进制仍可重审。

全部预测ID/reference/严格JSON评分、loss/dev曲线、checkpoint JSON metadata、配置、输入/模型/源码hash、命令、资源/会话与审计/删除收据保留。每次report重核保留元数据/完整预测和指标；被释放模型不再能继续训练/逐个重生成/重新核验原二进制。只保留SHA不能恢复权重。9套代表预先固定而非按test挑赢家。无需再累加TB级训练状态；新预算按分析文件余量+9代表export+一个活动run估算，不能保证任意150G盘必然够。

新控制器记录在experiments/research_banking_analysis_v1，科学run/config/probe仍在research_banking_v2并复用不变native脚本。共享v2流程锁/GPU锁。新控制器仅清理自己训练前记录ownership的新run，若旧v2 full-retention manifest存在，明确拒绝自动接管或删除；未知当前云端是否已启动旧入口，若已有旧任务先原入口暂停并按实际状态另做迁移，不能手删文件或重跑已完成结果。本次仅写新代码和文档，没有删除任何真实项目文件，没有动失败/暂停/旧4B/20/24/46/原始LaMP统计及原v2科学源码。

189项CPU测试运行、5项环境条件跳过；新增9项临时夹具验证无执行计划、审计后删除/预测复算、代表export保留、未归属/暂停/篡改拒绝、路径越界拒绝、清理中断必须显式finish和旧manifest保护。清理事务中断保留plan并报错，支持明确--finish-cleanup，不自动重试GPU。新r64/正式云端/真实清理均无成功回报。当前新批次推荐66号分析入口；65号仍为科学矩阵定义，full-retention旧脚本和冻结输入保持字节一致。


## 最新状态确认：旧全量保留实验尚未运行

所有者在说明当前实验流程/产物/空间占用之前明确：旧实验还未运行。记录为所有者确认旧full-retention正式run未启动，不是远程进程独立核验；新analysis入口亦尚无运行成功回报。因而当前无需先做旧manifest迁移，直接按66号新批次准备/探针/启动；不据此否定此前r8 GPU诊断已通过的回报，r64是否运行仍以真实证据为准。当前解释重点：1125次单卡串行；每次128步/八次完整100dev选模/完整400test与原文件审计后精确释放大二进制，分析产物在analysis_v1汇总、科学run/config在research_banking_v2。1125×64MiB=70.3125GiB是保守分析文件余量预算，不是预占/实测量；另加9代表adapter、一个活动run余量与10GiB保留，模型/诊断/缓存占用以云端du/df为准，不能保证150G必定够。


## 云端分析保留prepare已通过（所有者终端回报）

用户提供4f41383上的完整prepare输出：150G挂载39G已用/112G可用，4090空闲；plan1125训练/4513阶段，Qwen14/Mistral12/Phi13资产字节验收通过，grid_cpu_inputs_verified_formal_gpu_unexecuted_by_this_check。实际空闲119625474048 bytes（约111.4GiB），分析余量75497472000、9代表adapter预算1303904256、活动run与保留19815989248，合计所需96617365504 bytes（约90.0GiB），prepare空间检查通过。此为所有者CLI回报，不是本地重跑或导入云端原文件；不改任何运行源码/协议。输出未出现r64探针或正式训练启动，下一步分析入口probe再run，run会按探针后的实际空闲量重查。executable:false属于plan输出，asset_bytes_verified_not_training_ready是资产层状态，不是报错。


## 2026-11-19 检修紧急停机，所有者明确要求重启

当前云端进程输出为空/GPU闲，state旧running指向qwen3_8b_none_r64_a16_s20261205_train attempt1；进度88/128，最后01:32:32。所有者确认服务器检修紧急停机，授权恢复。67号和独立recover_research_maintenance.py默认只读核验原manifest/inputs/origin和最高完整checkpoint（预期80但尚未实际核验）；--execute一次性从有效全状态续训，恢复到128后补原train阶段此前缺失收据，再原analysis入口run --resume接着test/audit/retain及剩余条件，不重跑已完成条件。原训练/评价/清理/协议源码字节不变，Git更新只新增未被旧manifest绑定的辅助文件。

工具备份完整原progress和SHA，再恢复活动日志到保存点；81–88尾部保留并重算，后续残缺保存点移入受限恢复证据目录而非删除。保存真实检修恢复状态/原命令+实际resume命令，不伪造paused；存在恢复尝试则阻止自动重试。已完成128或已生成summary的情形拒绝续训，须finalization专门审计。中断会话完整时间/峰值不可知，run/maintenance_recovery.json及recoveries/.../cost_caveat.json明确标记，资源横向比较应排除或限定，不能把已测恢复段累计当全程成本；原质量评价仍需原128步/8dev/400test完整审计。195项CPU测试运行含5项条件跳过、新6项恢复测试通过；无本地GPU启动，无云端保存点核验或恢复成功回报。
