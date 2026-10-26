# C03 课程交接（截至 Lesson03）

更新日期：2026-10-26（Asia/Shanghai）。

本文件保留 Lesson01 交接详情和 Lesson02 历史结果，并更新截至 Lesson03 的进度。当前接续文档为 [Lesson03→04 交接](C03-Lesson03-handoff.md)。早期交接中的未开始状态属于历史快照，不能覆盖当前进度。

## 1. 接续入口

接手时先阅读本文件，以及下方索引中的讲义、作业和验收记录。C02 Lesson01–05 按此前交接已完成，无需重复授课或重新登记验收。

**C03 Lesson01 已讲授，四道作业答案已提供，用户索要答案按约定接受为私下完成确认；用户原课堂脚本已检查，27 项本地检查通过。** 原文件、助手补充和运行证据分别记录，不把参考答案当成用户原有独立实现。

**最新接续位置：Lesson03 已讲授，四题答案已给出，用户索要答案按约定接受为私下完成确认；课堂与参考代码完成 31 项检查。下一次要求“下一节课”时进入 Lesson04。** Lesson04–08 尚未开始。

## 2. 教学约定

- 聊天是主要课堂：先讲完整知识点，再给教师代码和逐段解释，说明易错点，最后统一布置作业。
- 用户索要作业答案，按既有约定接受为私下完成确认，不另要求截图或证明材料。
- 区分“已经授课”“用户确认完成”“助手补充参考实现”和“实际运行通过”，据实更新进度。
- 保留用户文件及注释。原文件名 `lesosn01.py` 虽有拼写错误，仍原样保留，使用标准入口解决运行命名问题。
- 用户说“下一节课”时继续教学；只有明确要求同步时才提交 GitHub。不自动创建新聊天。

## 3. Lesson01 的目标与范围

课题：真实 Tokenizer 与模型输入编码。

已讲解以下内容：

1. Tokenizer 如何将文本转换为 Token 和词表编号；Token 数与汉字数、英文单词数并不等同。
2. WordPiece 子词与 `##` 前缀；当前分词器的大小写行为和未知标记 `[UNK]`。
3. `encode`、`decode`、`convert_ids_to_tokens`，以及加入、跳过特殊 Token 的影响。
4. BERT 的 `[CLS]`、`[SEP]`、`[PAD]`、`[UNK]`、`[MASK]`。
5. `input_ids`、`attention_mask` 与 BERT 的 `token_type_ids`；输入编号为整数张量。
6. 批量编码、`padding`、`truncation`、`max_length`、`return_tensors="pt"`。
7. `[B,T]` 的含义，以及 `attention_mask.sum(dim=1)` 得到的有效长度。
8. 有效位置 Mask、因果 Mask、不同注意力 API 的布尔约定。

Lesson01 仅运行分词与张量操作，没有加载模型权重、进行模型前向、训练或文本生成。后续 Lesson02 的模型运行结果单独见第 8 节，不能反写为 Lesson01 已完成的内容。

## 4. 固定配置与关键结果

Tokenizer：`google-bert/bert-base-chinese`。

固定 revision：`8f23c25b06e129b6c986331a13d8d025a92cf0ea`。

实现为 `BertTokenizer`，词表大小为 21128，`do_lower_case=False`；本课统一右侧补齐和右侧截断。

| 特殊 Token | ID | 说明 |
|---|---:|---|
| `[PAD]` | 0 | 批量补齐 |
| `[UNK]` | 100 | 未知 Token，仍为有效位置 |
| `[CLS]` | 101 | 本课单句编码的起始标记 |
| `[SEP]` | 102 | 本课单句编码的结束标记 |
| `[MASK]` | 103 | BERT 的掩码 Token，与 `attention_mask` 含义不同 |

以下是固定版本下的真实结果，不应推广为所有 Tokenizer 的规律：

| 输入 | Token（不加边界标记） | ID |
|---|---|---|
| 查询订单 | 查、询、订、单 | `[3389,6418,6370,1296]` |
| 订单金额 | 订、单、金、额 | `[6370,1296,7032,7583]` |
| playing | play、##ing | `[8942,8221]` |
| transformer | t、##ran、##s、##form、##er | `[162,10477,8118,12725,8196]` |
| Transformer | `[UNK]` | `[100]` |
| 🧠 | `[UNK]` | `[100]` |

课堂批量输入：`["查询订单", "我想查询最近一个月的订单金额"]`。

- 默认加入 `[CLS]`、`[SEP]` 后，有效长度为 `[6,16]`，动态补齐得到 `[2,16]`。
- 第一条 Mask 为 6 个 1、10 个 0；第二条为 16 个 1。
- 固定补齐并截断到 10，形状为 `[2,10]`，有效长度为 `[6,10]`。长句只保留 8 个正文 Token，解码为“我 想 查 询 最 近 一 个”。

作业批量输入：`["订单", "查询订单", "请查询本月订单金额"]`。

| 长度策略 | input_ids / attention_mask 形状 | 有效长度 |
|---|---|---|
| `padding=True` | `[3,11]` | `[4,6,11]` |
| `padding="max_length", max_length=12` | `[3,12]` | `[4,6,11]` |
| `padding="max_length", truncation=True, max_length=8` | `[3,8]` | `[4,6,8]` |

长度 8 包含两个边界标记。第三条截断后为 `[CLS] 请 查 询 本 月 订 [SEP]`，丢失“单 金 额”。

## 5. Mask 与解码易错点

| 接口或约定 | 允许读取 | 屏蔽 |
|---|---|---|
| 本课 Hugging Face `attention_mask` | `1` | `0`（PAD） |
| C02 自定义布尔 `allowed` | `True` | `False` |
| PyTorch SDPA 布尔 `attn_mask` | `True` | `False` |
| PyTorch MHA 布尔 `key_padding_mask` | `False` | `True` |

- Mask `[1,1,1,1,1,1,0,0]` 的有效长度是 6，其中可以包含边界标记；不等于正文长度。
- 转换到 MHA 的布尔补齐 Mask：`~attention_mask.bool()`。
- 仅将有效 Key 扩展为可广播形状，可用 `attention_mask[:,None,None,:].bool()`；课堂批量结果为 `[2,1,1,16]`，尚不包含因果限制。
- 有效位置 Mask 本身没有表达“不能看未来”。需要因果规则时，应另行组合或使用模型提供的因果机制。
- 屏蔽 PAD Key 不保证 PAD Query 的最终输出为零。
- `[UNK]` 的 Mask 是 1。`[MASK]` 是词表中的一个 Token，不能与注意力 Mask 混为一谈。
- `decode` 不保证逐字恢复原文；空格格式可能变化，未知字符信息会丢失，`skip_special_tokens=True` 也会移除 `[UNK]`。
- BERT 示例中的特殊 Token、`token_type_ids` 和输入规则不能直接套用到后续 Qwen 模型，应使用配套 Tokenizer。

## 6. 文件与运行入口

本地仓库根目录：`/Users/jace/agent学习/hugging-learning`。

公开 GitHub 仓库：[hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。

| 文件 | 用途与来源 |
|---|---|
| [lesosn01.py](../hugging_learn/lesosn01.py) | 用户原有课堂文件，内容与注释原样保留 |
| [lesson01.py](../hugging_learn/lesson01.py) | 助手补充的标准运行入口，执行上述原文件 |
| [common.py](../hugging_learn/common.py) | 助手补充的统一 Tokenizer、revision 和缓存配置 |
| [download_tokenizer.py](../hugging_learn/download_tokenizer.py) | 首次准备分词文件缓存 |
| [lesson01_answers.py](../hugging_learn/homework/lesson01_answers.py) | 助手补充的四题可运行参考答案 |
| [Lesson01 讲义](C03-Lesson01-notes.md) | 知识点与教师代码说明 |
| [Lesson01 作业及答案](C03-Lesson01-homework.md) | 四题题目、代码、实际结果与解释 |
| [Lesson01 验收说明](C03-Lesson01-acceptance.md) | 2026-10-24 的检查范围、来源与边界 |
| [Lesson01 运行记录](C03-Lesson01-verification.json) | 2026-10-24 真实运行产生的机器可读结果 |
| [verify.py](../verify.py) | 固定样例复核入口 |
| [C03 八节课表](C03-syllabus.md) | 模块安排 |
| [原 Lesson01→02 交接](C03-Lesson01-handoff.md) | 2026-10-24 历史快照，保留原文 |

沿用 `fde` 环境，Python 路径为 `/Users/jace/miniforge3/envs/fde/bin/python`。已记录版本：Python 3.11.16、PyTorch 2.14.0、Transformers 5.20.2、huggingface_hub 2.2.0、tokenizers 0.23.3。

从仓库根目录运行：

```sh
cd /Users/jace/agent学习/hugging-learning
conda activate fde
python -m hugging_learn.lesson01
python -m hugging_learn.homework.lesson01_answers
python verify.py --report /tmp/c03-lesson01-verification.json
```

日常入口默认只读本地缓存。新机器缺少分词文件时，先运行 `python -m hugging_learn.download_tokenizer`；Lesson01 不需要下载 BERT 模型权重。

## 7. Lesson01 验收与同步状态

2026-10-24 已完成 27 项检查：原文件内容与执行、标准入口输出一致性、参考答案独立运行、真实 Token/ID、UNK 有效性、批量形状、有效长度、补齐与截断、边界位置、编号范围、Mask 转换，以及不等长列表直接转张量的异常行为。

用户原文件 SHA256：

```text
07206df58b7dcd8415b11664025c51ded13c3d6c1fca4eaa33ef5d803db07c81
```

原课堂脚本检查、用户按约定确认完成、助手参考代码和实际运行证据分别成立。27 项检查支持固定样例下的执行结果，不额外声明完成了用户独立能力测验、模型推理质量评估或训练。

Lesson01 的代码、讲义与验收记录此前已同步到公开仓库，最近记录的提交为 `cbcac34f4fda9ed48115f94a08debda9b84517f0`。此前仅在本地的 `docs/handoff.md` 与本轮 Lesson02 文件，一并纳入用户 2026-10-26 明确要求的 GitHub 同步范围。具体提交以仓库历史为准。

## 8. Lesson02 历史结果与当时接续

本节单独记录 Lesson02，不计入 Lesson01 的历史验收。

Lesson02 已讲授配置、Tokenizer、权重职责，from_pretrained 与 from_config，CausalLM，设备与 dtype，eval 与 inference_mode，输入字典展开以及一次前向。

2026-10-26 本轮直接复核用户新增的 `lesson02.py`，原文件已经能运行。仅规范官方模型名大小写及两处打印提示，保留代码结构和注释。四题答案与验证器为助手补充，索要答案按约定接受为私下完成确认。

固定模型 `Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU、float32、参数量 494032768。课堂输入“我想查询订单金额。”为 `[1,5]`，logits 为 `[1,5,151936]`；三条作业文本 T 为 2、5、10；输出有限且不记录梯度。Linear 实验表明 eval 本身不关闭梯度，推理模式前向不记录梯度。26 项检查通过。

此前位于 `/tmp` 的教师代码与结果已不在原位置，本轮重新实际运行，并将可复现结果归档到课程仓库：

- [用户课堂文件](../hugging_learn/lesson02.py)
- [可运行四题答案](../hugging_learn/homework/lesson02_answers.py)
- [Lesson02 讲义](C03-Lesson02-notes.md)
- [作业题目与解释](C03-Lesson02-homework.md)
- [验收与来源](C03-Lesson02-acceptance.md)
- [真实运行结果](C03-Lesson02-verification.json)
- [Lesson02→03 交接](C03-Lesson02-handoff.md)

运行：`python -m hugging_learn.lesson02`、`python -m hugging_learn.homework.lesson02_answers`、`python verify_lesson02.py --report /tmp/c03-lesson02-verification.json`。本轮未训练、未生成文本，也未评估生成质量。

Lesson02 当时交接至 Lesson03：从隐藏表示、词表输出头与每个位置的 logits，继续到最后位置的下一 Token 分数、概率与候选。当前这部分已在 Lesson03 授课并复核，最新状态见第 9 节。


## 9. Lesson03 最新结果与 Lesson04 接续

2026-10-26 已提供 Lesson03 四题参考答案，本轮按用户指令补齐文件并同步。用户新增 `lesson03.py` 已实际执行，其全部原始字节、注释和无句号输入保留；同时保留并同步用户对 Lesson02 两行参数的缩进整理，不改动既有历史报告。

固定使用同一 Qwen 模型与 revision，CPU float32；隐藏维度 896，词表输出维度 151936。用户文本“请统计最近一个月各部门的订单总金额”T=9，hidden `[1,9,896]`、logits `[1,9,151936]`；手动输出头差异为 0，最高分编号 8997（`'。\n'`）。作业三条带句号文本 T=[2,5,10]；完整词表 float64 概率和约 1，top5 仍是原分布中的概率。

31 项检查通过，包括课堂运行、输出头、概率、候选、作业数值与因果对照。不同长度前缀最大差异约 2.18e-4，按记录的 float32 容差处理；同长度改变后续 Token，前两个位置差异为 0。参考答案模块单独运行成功。来源和边界见验收文件；没有训练或执行多步生成。

- [用户课堂](../hugging_learn/lesson03.py)、[可运行答案](../hugging_learn/homework/lesson03_answers.py)
- [讲义](C03-Lesson03-notes.md)、[作业与答案](C03-Lesson03-homework.md)
- [验收与来源](C03-Lesson03-acceptance.md)、[实际运行 JSON](C03-Lesson03-verification.json)
- [Lesson03→04 交接](C03-Lesson03-handoff.md)、[复核入口](../verify_lesson03.py)

运行：`python -m hugging_learn.lesson03`、`python -m hugging_learn.homework.lesson03_answers`、`python verify_lesson03.py --report /tmp/c03-lesson03-verification.json`。

下一课 Lesson04 手写自回归循环：预测一个编号、追加 input_ids 与 attention_mask、再次前向、EOS 与最大新增长度、完整续写解码。先用单条无补齐输入和 `use_cache=False`，理解循环后再进入后续 generate、批量与缓存课程。Lesson04–08 尚未开始。
