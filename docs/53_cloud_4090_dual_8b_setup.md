# Linux RTX4090双8B云端准备指南

2026-11-16。当前所有者服务器：Ubuntu22.04、Python3.12、PyTorch2.12.1、CUDA13.0、RTX4090 24GB一张、16vCPU、62GB RAM。本指南替代新8B方案中的3090目标；旧4B/3090结果与入口不改。当前交付为可执行环境/模型资产准备，不是已完成的多模型训练执行器。

## 1. 取得代码与保留镜像PyTorch

以下路径按所有者确认的持久目录替换；autodl-tmp是否持久由实际实例保留策略决定。不要覆盖旧仓库中的模型、环境或实验。先检查磁盘，两个8B模型权重本身约数十GB，全部checkpoint/预测还需另做容量预算。

```bash
cd /root/autodl-tmp
# 若此处没有dayan目录：
git -c http.version=HTTP/1.1 clone https://github.com/MoCHIxb/dayan.git
cd dayan
# 若已有此仓库，先git status确认无需要保留的改动，再更新：
git -c http.version=HTTP/1.1 pull --ff-only
git log -1 --oneline
nvidia-smi
df -h .
bash scripts/cloud/prepare_8b_4090.sh plan
bash scripts/cloud/prepare_8b_4090.sh bootstrap
bash scripts/cloud/prepare_8b_4090.sh check
```

bootstrap新建.venv-cloud8b，采用--system-site-packages复用镜像torch；requirements-cloud8b.lock不含旧torch/cu128源，并将实际镜像torch版本写入精确constraints防止依赖解析替换。检查不通过即停，已有环境不覆盖、不自动重试、不静默降回旧torch2.11。镜像声称CUDA13.0但torch.version.cuda若不同，需要先查明；nvidia-smi的CUDA版本仅表示驱动支持，不能替代运行时。

check只做Linux/Python/torch/CUDA、单卡4090/显存、BF16及bitsandbytes导入和全局GPU进程保护，不加载模型。通过不等于低比特kernel、8B BF16显存、训练或暂停恢复已经验证。CPU单元测试不替代云端检查。依赖先沿用已知框架版本，CUDA13上的bitsandbytes真实计算兼容仍待后续GPU功能测试。

## 2. 固定两个官方模型并下载/核验

Llama须先在官方模型页满足访问和许可要求，并在服务器终端执行下面的隐藏输入认证；不要把token写入命令参数、聊天、仓库或日志。

```bash
source .venv-cloud8b/bin/activate
hf auth login
# 不会下载权重：解析官方仓库当前commit并固定文件尺寸与校验值
bash scripts/cloud/prepare_8b_4090.sh pin --model qwen3_8b
bash scripts/cloud/prepare_8b_4090.sh pin --model llama3_8b
# 固定后下载；网络失败保留缓存，检查原因后才手动再次执行
bash scripts/cloud/prepare_8b_4090.sh download --model qwen3_8b
bash scripts/cloud/prepare_8b_4090.sh download --model llama3_8b
# 完全只读复核模型文件；不训练
bash scripts/cloud/prepare_8b_4090.sh verify --model qwen3_8b
bash scripts/cloud/prepare_8b_4090.sh verify --model llama3_8b
```

pin仅从https://huggingface.co官方端点取得完整commit及文件checksum元数据，在experiments/cloud_8b_4090/assets创建一次模型锁；存在时拒绝覆盖。以后下载始终按固定commit，不继续跟踪main。权重检查LFS SHA256，普通文件检查Git blob SHA1，再为全部实际文件记录SHA256。已存在文件先核验，不匹配立即停并保留；下载库缓存保留，没有外围自动重试循环。准备期间全局GPU保护拒绝已有计算进程，不能在训练中修补模型。首次pin之后查看并私有备份两个模型锁；后续正式实验输入冻结还需把它们与配置/代码/数据一起绑定。

固定模型目录为models/Qwen3-8B、models/Meta-Llama-3-8B-Instruct。云端已有8B模型也只能在核验固定版本/字节一致后使用，不能仅靠目录名称复用。现已提供显式ModelScope权重候选，见第4节；网络、权限或SHA失败先保留输出定位，不绕过TLS。原Qwen4B专用ModelScope恢复和13文件清单不适用这里。

## 3. 新环境下的训练配置范围

configs/cloud_8b_4090/environment.json记录两个模型身份、目录、五PEFT配方及硬件契约，training_ready=false；这是准备配置，不可传给旧train-personal。现有train_personal_adapter.py显式限制Qwen3且只支持none/nf4，不能直接执行Llama或int8。新双模型模板/结束token/监督mask、低比特计算功能、每任务数据长度与独立训练/评价/审计执行器仍需实现与验证。旧3090/4B队列不作为替代run入口。

方案仍为52号主线54训练＋至少28历史任务；全部已完成历史科学条件的清点尚未结束，82是下界，不是最终自动队列。所有新实验以Qwen8B为主、Llama8B对照，在该4090串行执行，数据划分及完整LaMP原始统计不缩减。

8B BF16、NF4、int8均须先独立云端功能/保存恢复检查；4090与3090同为24GB，换卡不能保证原先不确定的显存条件必然可行。OOM保留证据即停，不自动换精度/减少长度或删测试。后续运行资产与输出独立于旧4B目录，失败不自动重试。

下一步请提供check的输出及两个verify的摘要（不含token）。这能确认云端准备到哪一步；尚不能据此宣布训练就绪。代码许可证未指定不阻塞维护仓库；模型各自的许可/访问要求独立适用。


## 4. 官方大权重下载过慢：显式ModelScope候选

所有者已授权使用候选来源。Qwen候选为Qwen/Qwen3-8B，Llama候选为LLM-Research/Meta-Llama-3-8B-Instruct；候选master不冒充官方commit，接受条件仍是既有官方锁中每个大权重的完整SHA256。候选仓库可能有不同tokenizer/config/README，因此只对.safetensors切换来源，小文件仍从已固定官方版本获取。不是重新pin到ModelScope或忽略差异。

先在旧下载终端按一次Ctrl+C并等待退出，保留HF缓存/incomplete/已完成文件，再更新代码。不要在另一下载或训练进程运行时更新运行脚本。

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
source .venv-cloud8b/bin/activate
# 两份pin已成功时不要重做pin；不存在时先按第2节固定官方版本。
bash scripts/cloud/prepare_8b_4090.sh download --model qwen3_8b --source modelscope &&
bash scripts/cloud/prepare_8b_4090.sh download --model llama3_8b --source modelscope
```

候选下载不增加依赖，不需安装ModelScope SDK。利用既有Range/SHA传输原语，独立partial位于各模型目录.cache/intenttune_download，不删除/拼接HF的不同格式断点。已完成文件逐项核验后跳过；候选SHA不匹配保留partial并停止，不能去掉校验继续。手动再次执行同命令只续传同一候选partial；无外围自动重试。HF令牌不发送给ModelScope候选请求；Llama上游访问与许可仍需满足。没有在本地下载权重，云端候选实际通过以verify摘要为准。

```bash
bash scripts/cloud/prepare_8b_4090.sh verify --model qwen3_8b &&
bash scripts/cloud/prepare_8b_4090.sh verify --model llama3_8b
```

这仍只认证资产，不表示模型加载、模板/低比特计算或训练已经通过。
