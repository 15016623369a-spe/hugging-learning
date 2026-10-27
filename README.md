# Hugging Learning

从真实 Tokenizer 接续 Transformer 基础，学习 Hugging Face 模型推理与文本生成。

更新日期：2026-10-27（Asia/Shanghai）。C03 Lesson01–05 已授课、参考答案已给出，固定样例复核通过；Lesson05 用户已明确确认完成。Lesson06–08 尚未开始，下一课为 Lesson06。

## 当前课程：Lesson05

将手写循环与 `model.generate` 的贪心结果逐编号对照，再比较采样、temperature、top-k/top-p、两种长度上限和随机种子。

- [详细讲义与代码解释](docs/C03-Lesson05-notes.md)
- [四道作业与详细参考答案](docs/C03-Lesson05-homework.md)
- [可运行完整四题代码](hugging_learn/homework/lesson05_answers.py)
- [验收与来源](docs/C03-Lesson05-acceptance.md)
- [实际运行记录](docs/C03-Lesson05-verification.json)
- [Lesson05→06 交接](docs/C03-Lesson05-handoff.md)

用户 `hugging_learn/lesson05.py` 及字典展开学习注释逐字节保留。38 项检查通过；三步贪心新增 `[8997,100012,100364]`，与手写循环一致。相同种子重设、只设一次种子的连续采样、不同输入长度、教学概率过滤均实际执行，真实生成与手工分数分别记录。完整答案另存，不覆盖课堂练习。

## Lesson04 资料

手写贪心循环：取最后位置 → 选一个 Token → 追加编号与 Mask → 再次前向 → EOS 或新增上限停止 → 截取续写。

- [讲义](docs/C03-Lesson04-notes.md)
- [四道作业与完整参考答案](docs/C03-Lesson04-homework.md)
- [可运行完整四题代码](hugging_learn/homework/lesson04_answers.py)
- [验收与来源](docs/C03-Lesson04-acceptance.md)
- [实际运行记录](docs/C03-Lesson04-verification.json)
- [Lesson04→05 历史交接](docs/C03-Lesson04-handoff.md)

用户课堂 `hugging_learn/lesson04.py` 保留三步练习和全部现有注释，包含编辑器直接运行的导入兼容。助手参考代码单独保存，兼容直接启动和模块运行。35 项检查通过；课堂 T=9，新增 3 个 Token，最终 T=12。完整答案还包含三条输入各生成八步、零步／一步边界和两种错误说明。

## Lesson03 资料

最终隐藏表示 `[B,T,D]` → 词表头 → logits `[B,T,V]` → 最后位置 → 全词表概率 → top5 与最高分编号。

- [讲义与课堂说明](docs/C03-Lesson03-notes.md)
- [四道作业及完整参考答案](docs/C03-Lesson03-homework.md)
- [验收与来源说明](docs/C03-Lesson03-acceptance.md)
- [真实运行记录](docs/C03-Lesson03-verification.json)
- [Lesson03→04 历史交接](docs/C03-Lesson03-handoff.md)
- [通用课程交接](docs/handoff.md)
- [C03 八节课表](docs/C03-syllabus.md)

用户课堂文件 `hugging_learn/lesson03.py` 的内容和注释逐字节保留。其输入没有句号，T=9；带句号作业版本 T=10，两组结果分别记录。四题参考、讲义和验证器为助手补充。31 项检查通过，手动输出头与模型 logits 差异为 0。

## Lesson02 资料

配置与 Tokenizer → 预训练权重 → 设备和类型 → eval 与推理模式 → 一次前向。

- [讲义](docs/C03-Lesson02-notes.md)
- [四题答案](docs/C03-Lesson02-homework.md)
- [验收](docs/C03-Lesson02-acceptance.md)
- [运行记录](docs/C03-Lesson02-verification.json)
- [Lesson02→03 历史交接](docs/C03-Lesson02-handoff.md)

Lesson03 更新时已同步用户对 `lesson02.py` 两行参数的缩进整理，行为不变；历史验收记录保留。

## Lesson01 资料

真实 Tokenizer、特殊 Token、批量编码、补齐截断与 Mask。

- [讲义](docs/C03-Lesson01-notes.md)
- [四道作业及参考答案](docs/C03-Lesson01-homework.md)
- [验收](docs/C03-Lesson01-acceptance.md)
- [运行记录](docs/C03-Lesson01-verification.json)
- [2026-10-24 历史交接](docs/C03-Lesson01-handoff.md)

`hugging_learn/lesosn01.py` 为用户原课堂文件，内容与注释逐字节保留。标准入口 `hugging_learn/lesson01.py` 执行同一原文件。

## 环境与运行

沿用 Conda `fde`。本次复核：Python 3.11.16、PyTorch 2.14.0、Transformers 5.20.2、huggingface_hub 2.2.0、tokenizers 0.23.3；CPU。未重建或升级环境。

从仓库根目录运行：

```sh
conda activate fde
python -m hugging_learn.lesson01
python -m hugging_learn.homework.lesson01_answers
python verify.py --report /tmp/c03-lesson01-verification.json
python -m hugging_learn.lesson02
python -m hugging_learn.homework.lesson02_answers
python verify_lesson02.py --report /tmp/c03-lesson02-verification.json
python -m hugging_learn.lesson03
python -m hugging_learn.homework.lesson03_answers
python verify_lesson03.py --report /tmp/c03-lesson03-verification.json
python -m hugging_learn.lesson04
python -m hugging_learn.homework.lesson04_answers
python verify_lesson04.py --report /tmp/c03-lesson04-verification.json
python -m hugging_learn.lesson05
python -m hugging_learn.homework.lesson05_answers
python verify_lesson05.py --report /tmp/c03-lesson05-verification.json
```

课堂与作业也支持编辑器直接运行；例如 `python hugging_learn/homework/lesson05_answers.py`。

当前机器已有 Lesson01–05 所需缓存，以上入口只读本地缓存；验证器按参数写入结果 JSON。新机器首次使用时，先核对解释器和依赖，按需要安装，再联网准备：

```sh
python -m pip install -r requirements.txt
python -m hugging_learn.download_tokenizer
python -m hugging_learn.download_model
```

`download_tokenizer` 仅获取 Lesson01 BERT 分词文件；`download_model` 获取 Lesson02 Qwen 配置、分词文件及约 1 GB 权重。模型文件保存在 Hugging Face 缓存，不提交到仓库。仅验证上述环境，不声称支持所有依赖组合。

## 固定版本与验收

- Lesson01：`google-bert/bert-base-chinese`，revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea`，右侧补齐和截断。历史 27 项离线检查通过，仅涉及分词和张量。
- Lesson02：`Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，使用其配套分词器，CPU float32。26 项检查通过；课堂输入 `[1,5]`，logits `[1,5,151936]`。作业三条文本 T 为 2、5、10。
- Lesson03：沿用 Lesson02 的固定模型；31 项检查通过，用户课堂 T=9，作业 T=[2,5,10]；float64 softmax 仅用于观察概率。
- Lesson04：沿用同一固定模型；35 项检查通过，已实际执行手写贪心生成循环。真实样例按长度停止，EOS 分支的受控检查单独记录。
- Lesson05：沿用同一固定模型；38 项检查通过，已实际执行 generate 贪心与采样。max_length 实验显式清除模型自带 max_new_tokens；温度与过滤的教学分数另列。

截至 Lesson05 已执行手写循环、generate 贪心和采样；没有训练或生成质量评估。助手参考代码运行通过、用户完成确认与独立实现分别记录。

聊天为主要课堂。用户要求更详细：先回顾与说明动机，再用具体数字推导、标明张量形状、逐段逐行解释代码、解读实测结果与常见错误，最后统一给出作业。后续课程和 GitHub 同步按用户指令推进。

本仓库只收录 C03 学习内容，不搬迁 C02 历史代码。环境、密钥、缓存、模型权重与编辑器配置均排除在 Git 之外。
