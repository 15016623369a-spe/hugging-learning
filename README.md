# Hugging Learning

从真实 Tokenizer 接续 Transformer 基础，学习 Hugging Face 模型推理与文本生成。

更新日期：2026-10-26（Asia/Shanghai）。C03 Lesson01–02 已授课、参考答案已给出，固定样例复核通过；索要答案按约定接受为私下完成确认。Lesson03–08 尚未开始，下一课为 Lesson03。

## 当前课程：Lesson02

配置与配套 Tokenizer → 预训练权重 → 设备和类型 → eval 与推理模式 → 一次前向 → logits `[B,T,V]`。

- [讲义与教师代码说明](docs/C03-Lesson02-notes.md)
- [四道作业及完整参考答案](docs/C03-Lesson02-homework.md)
- [验收与来源说明](docs/C03-Lesson02-acceptance.md)
- [真实运行记录](docs/C03-Lesson02-verification.json)
- [Lesson02→03 交接](docs/C03-Lesson02-handoff.md)
- [通用课程交接](docs/handoff.md)
- [C03 八节课表](docs/C03-syllabus.md)

用户课堂文件 `hugging_learn/lesson02.py` 在本轮修改前已成功完成前向；仅规范模型名及两处打印文字，保留注释和代码结构。参考答案、辅助入口、讲义与验证器为助手补充，分别记录来源。

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
```

当前机器已有两课缓存，以上入口只读本地文件。新机器首次使用时，先核对解释器和依赖，按需要安装，再联网准备：

```sh
python -m pip install -r requirements.txt
python -m hugging_learn.download_tokenizer
python -m hugging_learn.download_model
```

`download_tokenizer` 仅获取 Lesson01 BERT 分词文件；`download_model` 获取 Lesson02 Qwen 配置、分词文件及约 1 GB 权重。模型文件保存在 Hugging Face 缓存，不提交到仓库。仅验证上述环境，不声称支持所有依赖组合。

## 固定版本与验收

- Lesson01：`google-bert/bert-base-chinese`，revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea`，右侧补齐和截断。历史 27 项离线检查通过，仅涉及分词和张量。
- Lesson02：`Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，使用其配套分词器，CPU float32。26 项检查通过；课堂输入 `[1,5]`，logits `[1,5,151936]`。作业三条文本 T 为 2、5、10。

Lesson02 已实际加载模型并前向；没有训练、逐 Token 生成或生成效果评估。助手参考代码运行通过与用户独立实现分别记录。

聊天为主要课堂，采用知识点、教师代码与解释、易错点、统一作业的节奏。后续课程和 GitHub 同步按用户指令推进。

本仓库只收录 C03 学习内容，不搬迁 C02 历史代码。环境、密钥、缓存、模型权重与编辑器配置均排除在 Git 之外。
