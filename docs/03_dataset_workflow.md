# 个性化小样本数据集构建实操手册

这份手册对应“分类—分段—标注—格式转换”的完整路径。建议第一次只处理 20 条样本，确认规范后再扩展到 500 条。

## 0. 先定义你要模型学什么

推荐先选一个可操作的目标，而不是宽泛的“像某作家”：

- 风格改写：输入一段中性文本，输出同义的目标风格文本；
- 个性化问答：输入问题，输出符合个人表达偏好的答案；
- 领域表达：输入事实或提纲，输出符合专业术语和结构习惯的成文。

第一轮建议选择“风格改写”，因为输入与输出语义可对照，比较容易测量内容保持。

## 1. 收集与登记

每个来源建立一条登记记录：`source_id`、作者/来源、URL 或文件、日期、许可证/授权、语言、体裁、能否再分发。原文放入 `data/raw/`，不要覆盖或人工改写原文件。

建议来源优先级：本人明确授权文本 > 公共领域或宽松许可文本 > 仅允许研究但不可再分发的文本。不要抓取私人聊天或含真实姓名、地址、账号、联系方式的内容。

## 2. 数据分类

分类分为两个互相独立的轴：

### 风格标签 `style_label`

先限制为 3–6 个互斥主标签，例如：

- `narrative_calm`：克制、叙事、低情绪强度；
- `argument_concise`：短句、结论先行、论证紧凑；
- `explanatory_warm`：解释性强、面向读者、语气温和；
- `dialogue_humorous`：对话化、反差或幽默。

标签必须有正例、反例和边界说明。无法判断时标 `uncertain`，不要强行归类。

### 主题标签 `topic_label`

例如 `technology`、`campus`、`daily_life`、`fiction`。主题用于检查风格是否只在某一主题上成立，不应与风格合并成一个标签。

本仓库脚本不自动“猜”风格，只生成候选段落和标注表。50–500 条数据时人工标注通常更可靠；自动分类可以作为候选建议，但必须复核。

## 3. 清洗与分段

运行：

```powershell
personal-data prepare --input data/raw --output data/interim/segments.jsonl --config configs/data.example.json
```

工具会进行 Unicode NFKC 归一化、换行统一、规则删除、段落/句子切分、长度约束和精确去重。中文默认按 `。！？；` 等句末标点切句，在 `max_chars` 内尽量保持完整句子。

分段原则：

- 每段表达一个相对完整意思；不要在引号、代码或列表中间切断。
- 过短段落缺乏风格信息；中文通常从 30–80 字起步。
- 超长段落会增加截断和显存压力；先用 300–800 字作为上限。
- 不要在训练/测试划分后再切分；应先切分，再按来源分组划分。
- 相邻窗口如有重叠，必须归入同一 split。

## 4. 人工标注

生成 CSV：

```powershell
personal-data export-annotation --input data/interim/segments.jsonl --output data/interim/annotation.csv
```

主要字段：

| 字段 | 填法 |
|---|---|
| `keep` | `1` 保留，`0` 丢弃 |
| `source_group` | 同源文档或衍生样本使用相同值，保证不跨 split |
| `task_type` | `rewrite`、`outline_to_text`、`qa`、`cross_topic` |
| `style_label` | 从预先定义的标签表选择 |
| `style_traits` | 具体风格属性，多项用逗号分隔 |
| `topic_label` | 主题标签 |
| `quality` | 1–5；建议训练集只保留 4–5 |
| `instruction` | 用户指令；同类任务准备多种自然表达 |
| `input` | 原始中性内容或提纲；纯续写任务可留空 |
| `text` | 目标回答，保持为作者原文或经授权的目标文本 |
| `consent` | 是否获得个人数据/文本使用授权 |
| `redistributable` | 是否允许公开再分发 |
| `pii_flags` | 脚本检出的敏感信息类型，必须人工处理 |
| `near_duplicate_of` | 近重复候选的已有样本 ID，人工决定保留哪条 |
| `notes` | 疑问、删改、隐私处理说明 |

质量量表：5=风格鲜明、完整、无噪声；4=可直接训练；3=需轻微修改或风格一般；2=明显噪声；1=错误、敏感或不可用。

至少抽取 10%–20% 做双人重复标注。分类标签可报告 Cohen's kappa；质量分可报告加权 kappa 或相关性。分歧样本经讨论后形成最终标签，并把规则补回标注手册。

## 5. 校验与去泄漏

导入和校验：

```powershell
personal-data import-annotation --input data/interim/annotation.csv --output data/interim/annotated.jsonl
personal-data validate --input data/interim/annotated.jsonl
personal-data stats --input data/interim/annotated.jsonl
```

除脚本的硬校验外，还需人工抽查：

- 回答是否真的执行 instruction；
- `input` 与 `text` 是否语义一致；
- 是否含网页导航、广告、版权行、乱码、模型生成免责声明；
- 是否含邮箱、电话、身份证、住址、账号等敏感信息；
- 是否出现同一事件或固定短语在训练集和测试集重复。

## 6. 划分

```powershell
personal-data split --input data/interim/annotated.jsonl --output-dir data/processed --train 0.8 --validation 0.1 --test 0.1 --seed 20251206
```

脚本按 `source` 分组做确定性划分，同一来源不会跨集合。小数据集的测试集建议至少 30–50 条；若总量太小，保留固定测试集，再对训练集做交叉验证或多随机种子重复。

做 50/100/200/500 条消融时使用嵌套子集：50 条集合必须完全包含在 100 条中，以此类推；各风格/主题尽量分层。

## 7. 格式转换

统一中间格式使用 JSONL，一行一条：

```json
{"id":"...","source":"sample.txt","instruction":"请用目标风格改写。","input":"中性内容","text":"目标风格回答","style_label":"narrative_calm","topic_label":"fiction","quality":5,"keep":true}
```

转换为 TRL/Hugging Face 推荐的对话格式：

```powershell
personal-data convert --input data/processed/train.jsonl --output data/processed/train_messages.jsonl --format messages
```

输出形如：

```json
{"id":"...","messages":[{"role":"user","content":"请用目标风格改写。\n\n中性内容"},{"role":"assistant","content":"目标风格回答"}]}
```

也支持 `alpaca` 和 `prompt-completion`：

```powershell
personal-data convert --input data/processed/train.jsonl --output data/processed/train_alpaca.jsonl --format alpaca
personal-data convert --input data/processed/train.jsonl --output data/processed/train_pc.jsonl --format prompt-completion
```

不要手工拼接 `<|im_start|>` 等特殊 token。由基座模型自己的 tokenizer/chat template 在训练时处理，否则很容易模板错配。

## 8. 训练前验收门槛

- 100% 有来源和授权状态；100% 通过结构校验。
- 训练目标中无 PII、网页噪声、空回答和明显乱码。
- 精确重复为 0；近重复完成抽查或专门检测。
- 所有 split 按来源隔离，测试集冻结。
- 每个主风格至少有可评估的测试样本。
- 随机抽 30 条，人工确认指令—输入—回答关系正确。
- 保存数据版本号、哈希、脚本 commit 和统计报告。

## 9. 常见错误

- 把长篇原文直接切成 completion，却没有有意义的 instruction/input；模型只学到续写而非指令适配。
- 用目标文本本身生成几乎等价的输入，导致任务过于简单和评估泄漏。
- 只按行随机划分，使同一文章的相邻段落跨集合。
- 用 BLEU 衡量创造性风格生成，并把低词面重合误认为质量差。
- 用同一个强模型生成训练对、测试参考和裁判分数，形成闭环偏差。

