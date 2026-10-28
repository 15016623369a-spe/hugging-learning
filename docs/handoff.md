# C03 课程交接（截至 Lesson06）

更新日期：2026-10-28（Asia/Shanghai）。

本文件保留 Lesson01–05 历史结果，并更新截至 Lesson06 的进度。当前接续文档为 [Lesson06→07 交接](C03-Lesson06-handoff.md)。早期交接中的未开始状态属于历史快照，不能覆盖当前进度。

## 1. 接续入口

接手时先阅读本文件，以及下方索引中的讲义、作业和验收记录。C02 Lesson01–05 按此前交接已完成，无需重复授课或重新登记验收。

**C03 Lesson01 已讲授，四道作业答案已提供，用户索要答案按约定接受为私下完成确认；用户原课堂脚本已检查，27 项本地检查通过。** 原文件、助手补充和运行证据分别记录，不把参考答案当成用户原有独立实现。

**最新接续位置：Lesson06已讲授，四题详细答案已给出；用户索要答案按既有约定登记私下完成确认。课堂与参考代码完成40项检查，实际执行批量生成、模板与手动历史。下一次要求“下一节课”时进入Lesson07。** Lesson07–08尚未开始。

## 2. 教学约定

- 聊天是主要课堂：先讲完整知识点，再给教师代码和逐段解释，说明易错点，最后统一布置作业。
- 用户明确要求讲解更详细：先回顾前课与动机，术语逐项说明，数学用具体数字推导，张量标出每步形状；先解释每段目的，再逐行解释关键代码、参数和输入输出，结合实测结果与参数变化说明，最后统一给四题。
- 用户索要作业答案，按既有约定接受为私下完成确认，不另要求截图或证明材料。
- 区分“已经授课”“用户确认完成”“助手补充参考实现”和“实际运行通过”，据实更新进度。
- 保留用户文件及注释。原文件名 `lesosn01.py` 虽有拼写错误，仍原样保留，使用标准入口解决运行命名问题。
- Lesson06原文件名为`lesoon06.py`，同样保留；标准`lesson06.py`通过runpy执行原课堂。用户追问tokenize True/False，一步与两步编码的区别须继续明确解释。
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


## 9. Lesson03 历史结果与当时接续

2026-10-26 已提供 Lesson03 四题参考答案，本轮按用户指令补齐文件并同步。用户新增 `lesson03.py` 已实际执行，其全部原始字节、注释和无句号输入保留；同时保留并同步用户对 Lesson02 两行参数的缩进整理，不改动既有历史报告。

固定使用同一 Qwen 模型与 revision，CPU float32；隐藏维度 896，词表输出维度 151936。用户文本“请统计最近一个月各部门的订单总金额”T=9，hidden `[1,9,896]`、logits `[1,9,151936]`；手动输出头差异为 0，最高分编号 8997（`'。\n'`）。作业三条带句号文本 T=[2,5,10]；完整词表 float64 概率和约 1，top5 仍是原分布中的概率。

31 项检查通过，包括课堂运行、输出头、概率、候选、作业数值与因果对照。不同长度前缀最大差异约 2.18e-4，按记录的 float32 容差处理；同长度改变后续 Token，前两个位置差异为 0。参考答案模块单独运行成功。来源和边界见验收文件；没有训练或执行多步生成。

- [用户课堂](../hugging_learn/lesson03.py)、[可运行答案](../hugging_learn/homework/lesson03_answers.py)
- [讲义](C03-Lesson03-notes.md)、[作业与答案](C03-Lesson03-homework.md)
- [验收与来源](C03-Lesson03-acceptance.md)、[实际运行 JSON](C03-Lesson03-verification.json)
- [Lesson03→04 交接](C03-Lesson03-handoff.md)、[复核入口](../verify_lesson03.py)

运行：`python -m hugging_learn.lesson03`、`python -m hugging_learn.homework.lesson03_answers`、`python verify_lesson03.py --report /tmp/c03-lesson03-verification.json`。

Lesson03 当时接续至 Lesson04 手写自回归循环。该课现已授课并完成参考答案与运行复核，最新结果见第 10 节。


## 10. Lesson04 最新结果与 Lesson05 接续

2026-10-27 已讲授 Lesson04 并给出四题答案，本轮按用户明确要求补齐本地文件并同步 GitHub。先保存用户尚未落盘的课堂修改，随后保留当前 `lesson04.py` 的全部字节：无句号输入、三步设置、导入兼容及“cat是concatenate”注释。完整四题参考单独保存，不用参考实现覆盖课堂练习。

沿用固定 Qwen 模型、CPU float32、eval、inference_mode、单句无 PAD、`use_cache=False`。课堂输入 T=9，新增 `[8997,100012,100364]`，续写 `'。\n为了帮助'`，最终长度 12，最后一次 logits 的 T=11。三条带句号作业文本 T=[2,5,10]，各新增 8 个 Token，最终长度 [10,13,18]。零上限不前向，一步上限只生成 8997。

35 项检查通过，用户课堂和四题共享模型，共 31 次真实 Qwen 前向。完整答案独立直接运行成功，结果与报告一致；课堂直接脚本入口和答案模块帮助入口也通过。真实样例按新增上限结束，受控假模型的 EOS 停止检查、手工 EOS 解码示例均单独标明来源。

- [用户课堂](../hugging_learn/lesson04.py)、[完整四题代码](../hugging_learn/homework/lesson04_answers.py)
- [讲义](C03-Lesson04-notes.md)、[四题答案](C03-Lesson04-homework.md)
- [验收与来源](C03-Lesson04-acceptance.md)、[实际运行 JSON](C03-Lesson04-verification.json)
- [Lesson04→05 交接](C03-Lesson04-handoff.md)、[验证器](../verify_lesson04.py)

运行：`python -m hugging_learn.lesson04`、`python -m hugging_learn.homework.lesson04_answers`、`python verify_lesson04.py --report /tmp/c03-lesson04-verification.json`。

Lesson04 当时接续 Lesson05，后者现已授课并复核，最新结果见第11节。Lesson04 历史范围没有训练、调用 generate 或评估生成质量；后续 generate 实验不反写为 Lesson04 已完成。

## 11. Lesson05 最新结果与 Lesson06 接续

2026-10-27 已讲授 Lesson05。用户明确说“完成了”，要求给答案、检查文件并同步GitHub。当前lesson05.py运行通过，连同用户新增的字典展开注释逐字节保留；完整四题参考、讲义、验证器、验收与交接为助手补充。源码哈希与来源在运行JSON中登记，未额外验收用户独立能力。

沿用固定Qwen与CPU float32、eval、inference_mode、单句无PAD、use_cache=False。课堂T=9，贪心八步与手写编号一致，总长度17；第一题三步新增[8997,100012,100364]，完整[1,12]。两条输入T=[2,9]，新增4/8得到完整长度[6,10,13,17]；同一max_length=12实测新增[10,3]。模型自带max_new_tokens=2048，改用总上限时显式清除；本版本新增上限0报ValueError。

采样固定temperature=0.8、top_k=20、top_p=0.9，上限8。两次重设42逐编号一致，43本次不同；只设一次42的连续三轮本次产生不同结果。教学logits[3,2,1,0,-1]的温度、过滤和归一化另外执行，不当作模型实际候选。共38项检查通过，真实生成均按长度停止，没有生成EOS；没有训练、质量评估或性能结论。

- [用户课堂](../hugging_learn/lesson05.py)、[完整四题代码](../hugging_learn/homework/lesson05_answers.py)
- [详细讲义](C03-Lesson05-notes.md)、[四题答案](C03-Lesson05-homework.md)
- [验收与来源](C03-Lesson05-acceptance.md)、[实际运行JSON](C03-Lesson05-verification.json)
- [Lesson05→06交接](C03-Lesson05-handoff.md)、[验证器](../verify_lesson05.py)

运行：`python -m hugging_learn.lesson05`、`python -m hugging_learn.homework.lesson05_answers`、`python verify_lesson05.py --report /tmp/c03-lesson05-verification.json`。

Lesson05当时接续Lesson06；后者现已授课与复核，最新结果见第12节。Lesson05历史范围不包含后续批量和模板实验。KVcache细节仍留Lesson07。

## 12. Lesson06最新结果与Lesson07接续

2026-10-28已讲授Lesson06并解释tokenize True/False。用户随后要求给出答案、检查文件、同步GitHub，并特别要求更新Markdown。本轮保留原课堂`lesoon06.py`全部字节与原名，助手新增标准入口`lesson06.py`，四题完整代码、讲义、题解、验证器、验收与交接单独补充。用户完成确认、助手参考与运行证据分别登记，未另验收独立能力。

固定Qwen基础模型与revision、CPU float32、eval、inference_mode、左补齐、use_cache=False。课堂普通输入T=[2,5,9]、聊天T=[27,43]；作业普通输入T=[1,3,10]、左PAD=[9,7,0]，四步生成[3,10]→[3,14]，批量与逐条编号相同。错误按真实长度切片得到“订单价为10”，按共同宽度切片才是“价为10”。

模板助手开头增加3个token；两段聊天长度24/40→27/43，批量[2,43]→[2,51]。一步tokenize=True和两步tokenize=False再编码逐编号相同。手动追加实际短回复与追问后消息2→4，下一轮输入[1,50]，本轮仅编码、未生成下一轮回复。PAD/EOS=151643，模板im_end=151645，分别读取含义。

40项固定检查通过；完整答案直接脚本独立运行结果与验证器一致。真实样例均按新增上限结束，没有EOS；手工EOS/PAD示例单列来源。不训练、不评价Instruct权重或短续写质量，不测性能或KVcache。

- [用户原课堂](../hugging_learn/lesoon06.py)、[标准入口](../hugging_learn/lesson06.py)、[完整四题代码](../hugging_learn/homework/lesson06_answers.py)
- [详细讲义](C03-Lesson06-notes.md)、[四题题解](C03-Lesson06-homework.md)
- [验收与来源](C03-Lesson06-acceptance.md)、[实际运行JSON](C03-Lesson06-verification.json)
- [Lesson06→07交接](C03-Lesson06-handoff.md)、[验证器](../verify_lesson06.py)

运行：`python -m hugging_learn.lesson06`、`python -m hugging_learn.homework.lesson06_answers`、`python verify_lesson06.py --report /tmp/c03-lesson06-verification.json`。

下一课Lesson07讲推理资源与复现：上下文预算、KVcache缓存内容与形状、cache开关对照、内存、耗时、版本和实验记录。Lesson07–08未开始。继续按用户要求详细推导、逐段逐行讲解，最后统一四题；只有明确要求同步才更新GitHub。
