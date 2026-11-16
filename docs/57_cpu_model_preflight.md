# 扩容后：三模型CPU兼容性预检

2026-11-16。用户报告三模型asset verify通过，GPU空闲、原剩余14GiB，随后已额外扩容50G。扩容是否落到当前挂载及剩余量尚需df实测；不据配置变化宣称具备完整矩阵的存储预算。当前仍先预检，不启动训练。

## 云端顺序

```bash
cd /root/autodl-tmp/dayan
df -h .
git -c http.version=HTTP/1.1 pull --ff-only
# 仅重建/核验真实BANKING数据，无训练；需要访问固定GitHub源
source /etc/network_turbo
bash scripts/cloud/run_qwen8b_banking500.sh prepare
# 模型已经本地齐备，CPU预检强制离线
bash scripts/cloud/preflight_research_models.sh
```

prepare入口仍以既有Qwen tokenizer独立重建数据，核对九份train/dev/test与fixed10shot JSONL对历史SHA、1000条源记录回放和长度。它不启动单模型run；此处复用数据审计，不代表恢复IntentTune框架路线。不需要重新下载模型或安装环境。

CPU预检对每个模型实际检查500train、100dev、400test、400固定示例输入。使用本地官方tokenizer的原生chat template及关闭thinking参数；要求完整assistant模板以前缀prompt开始、监督目标解码等于原JSON标签、序列/输出长度完整不过1536/64。问题写入报告，不修复标签、剥离模型输出或截断数据。使用AutoConfig＋init_empty_weights构造meta模型，检查各模型的LoRA投影是否存在及实际参数数量，不读取权重进模型，不创建CUDA上下文。Phi融合投影不能当成Qwen七投影。

结果：experiments/cloud_8b_4090/compatibility_preflight.json。通过状态为cpu_contracts_passed_gpu_unverified；模板不匹配为cpu_contract_issues_gpu_blocked并退出2。报告包含原记录ID/问题计数、数据manifest、源码、registry、tokenizer/config和资产锁SHA，不含客户原文。原生模板若因角色规则或边界不一致失败，需先阅读报告，写明新适配规则后重新核验，不能自动拼接或忽略系统说明。

```bash
cat experiments/cloud_8b_4090/compatibility_preflight.json
```

已存在报告拒绝覆盖；诊断并明确修订后用--output指定独立新报告名。程序不自动重试，也不因为CPU通过就写training_ready=true。后续仍需逐模型/精度的独立GPU加载、前后向、保存恢复与资源测量，然后冻结统一正式训练入口。不能现在执行旧Qwen单模型run代替跨模型统一实验。

本地仅验证预检逻辑/拒绝边界错误的CPU单元测试，不在本地加载三模型或做GPU实验。扩容不改变24GB GPU显存，BF168B实测仍不可省略。模型/数据/checkpoint/预测不清理，全部原统计和来源边界保持。
