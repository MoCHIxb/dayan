# 当前交接：低资源开发者框架与三领域文本安全审查

2026-11-02更新。项目D:\dayan；[GitHub](https://github.com/MoCHIxb/dayan)。开题计划和所有者最新指示优先。完整MVP/商业软件不是必做项；交付可运行仓库、启动命令、配置、接口与方法指南。

## 接手入口

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

330条文本和标签已回放固定源文件逐条核对；独立重建17份数据/manifest/构建报告，字节完全一致。代码71项CPU测试通过；独立CPU环境66通过/5因可选训练依赖缺失跳过。没有新GPU训练或真实模型预测。

## 下一步云端

配置`configs/text_safety_v1/qwen3_4b_joint.json`，独立输出`experiments/text_safety_v1/run`。先训练一个联合150条适配器，不自动扩展三套模型或大网格。默认r8/alpha16、BF16、128步、每16步完整60条dev与保存；完整120条test，基座/适配器同预算；用`score_text_safety.py`计算各领域及总体macro-F1、不安全回答检出、安全回答误报和无效JSON。

训练入口无--execute只检查文件；显式执行时才使用GPU。服务器尚未租用/连接/执行，本轮只做本地CPU构建与公开同步。模型下载固定版本、环境准备、完整命令和暂停恢复见35/33号。不能把CPU验证当Linux4090/GPU已跑通，也不能先宣称审查能力强。

## 已完成且不重跑的历史研究

small_sample_v1：16训练、12所选适配器完整评价、36项原始bootstrap。peft_v1：9新训练＋3BF16对照，20组完整1496条test、9套辅助评价、18项各2000次原始corpus_score统计及独立最终解释。28条件客观辅助证据完成；原监控删除。

20/24号协议及冻结科学实现不改。新示例不缩减或替换历史dev1500/test1496/原始统计；旧六次全量队列、人工盲评、付费裁判和大规模通用矩阵继续暂缓。原平台复现cloud_4090_v1仍独立且未启动。

## 发布与执行边界

所有者授权选定三领域数据随仓库同步，窄范围例外仅为examples/text_safety_v1已审查文件。DiaSafety行保留上游Apache-2.0条款，BeaverTails行遵循CC BY-NC 4.0，归属和改动见NOTICE。源码仍未指定许可证，不能据数据许可宣称代码已正式开源。

原始全量数据、原VLM数据、模型、checkpoint、逐条模型预测与密钥不提交。`scripts/audit_publication.py`核对公开范围及快照；仅--update-manifest修改清单。一次只运行一个GPU任务，失败留痕不自动重试；已完成产物不覆盖。旧2026-10-28交接存于history/17_handoff_20261028.md，只作事实背景。
