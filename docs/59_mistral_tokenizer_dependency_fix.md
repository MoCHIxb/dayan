# Mistral CPU预检：补齐SentencePiece转换依赖

2026-11-16。云端预检经过Qwen后在Mistral AutoTokenizer实例化报Cannot instantiate this tokenizer from a slow version。没有加载大模型/GPU或开始训练；不能据Qwen资产日志宣布三模型兼容通过。

官方Mistral-v0.3 tokenizer_config包含tokenizer_class=LlamaTokenizer、add_prefix_space=true、legacy=false。Transformers4.57.6的LlamaTokenizerFast对非空add_prefix_space设置from_slow=true，触发官方SentencePiece路径。此前requirements-cloud8b.lock缺少SentencePiece/protobuf是环境准备遗漏，不是权重损坏或Llama3授权错误。不要移除add_prefix_space/legacy设置绕过，也不要改名tokenizer.model.v3或修改官方tokenizer.json。

## 修复命令（仅CPU）

先确认之前预检进程已退出，保留其终端日志；无须删除模型/版本锁/缓存或重做prepare/模型部署。

```bash
cd /root/autodl-tmp/dayan
git -c http.version=HTTP/1.1 pull --ff-only
source .venv-cloud8b/bin/activate
# 仅补两个锁定依赖，不重装torch/transformers或覆盖环境：
python -m pip install --only-binary=:all: -r requirements-tokenizers.lock &&
python -m pip check
```

安装失败或pip check报告冲突则停止并提供输出，不手动升级整个栈、不自动改版本重试。新增锁定sentencepiece0.2.1、protobuf4.25.8；主cloud锁也引用此小依赖文件，新机不再漏装。若国内pip源因学术代理慢，可在独立子shell取消代理再执行安装，不影响HF代理设置，不关闭TLS。

依赖安装/检查均通过后，用新报告名明确重试一次，保留此前证据：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="" bash scripts/cloud/preflight_research_models.sh   --output experiments/cloud_8b_4090/compatibility_preflight_tokenizers_v2.json
cat experiments/cloud_8b_4090/compatibility_preflight_tokenizers_v2.json
```

新预检保存每模型独立进度（报告stem.model.json）；模型初始化失败保存stem.failure.json，含前序已完成模型报告、错误类型、未训练状态，然后抛出原异常。原输出/失败报告拒绝覆盖，禁止静默跳过Mistral。检查仍使用原生模板与真实数据；下一个可能暴露的角色/边界差异须另定位，不以安装成功就宣称兼容。新shell子进程重新导入Transformers以刷新可选依赖探测。

此修复仅在本地CPU逻辑测试通过，尚未取得云端Mistral tokenizer成功证据。完成预检再进行GPU功能验证和统一训练准备，原始模型/data/科学协议不改。

## 来源

- [官方Mistral tokenizer配置](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/tokenizer_config.json)（诊断读取公开配置，不改变云端固定revision）。
- [Transformers4.57.6 LlamaTokenizerFast源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/tokenization_llama_fast.py)。
- [SentencePiece0.2.1](https://pypi.org/project/sentencepiece/0.2.1/)与[protobuf4.25.8](https://pypi.org/project/protobuf/4.25.8/)。
