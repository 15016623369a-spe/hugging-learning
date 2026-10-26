# Hugging Learning

从真实 Tokenizer 接续 Transformer 基础，学习 Hugging Face 模型推理与文本生成。

更新日期：2026-10-26（Asia/Shanghai）。C03 Lesson01–03 已授课、参考答案已给出，固定样例复核通过；索要答案按约定接受为私下完成确认。Lesson04–08 尚未开始，下一课为 Lesson04。

## 当前课程：Lesson03

最终隐藏表示 `[B,T,D]` → 词表头 → logits `[B,T,V]` → 最后位置 → 全词表概率 → top5 与最高分编号。

- [讲义与课堂说明](docs/C03-Lesson03-notes.md)
- [四道作业及完整参考答案](docs/C03-Lesson03-homework.md)
- [验收与来源说明](docs/C03-Lesson03-acceptance.md)
- [真实运行记录](docs/C03-Lesson03-verification.json)
- [Lesson03→04 交接](docs/C03-Lesson03-handoff.md)
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

本轮一并同步用户对 `lesson02.py` 两行参数的缩进整理，行为不变；历史验收记录保留。

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
```

当前机器已有 Lesson01–03 所需缓存，以上入口只读本地文件。新机器首次使用时，先核对解释器和依赖，按需要安装，再联网准备：

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

截至 Lesson03 已实际加载模型并前向、分析候选 Token；没有训练、生成循环或生成效果评估。助手参考代码运行通过与用户独立实现分别记录。

聊天为主要课堂，采用知识点、教师代码与解释、易错点、统一作业的节奏。后续课程和 GitHub 同步按用户指令推进。

本仓库只收录 C03 学习内容，不搬迁 C02 历史代码。环境、密钥、缓存、模型权重与编辑器配置均排除在 Git 之外。
