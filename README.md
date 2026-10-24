# Hugging Learning

从真实 Tokenizer 接续 Transformer 基础，学习 Hugging Face 模型推理与文本生成。

更新日期：2026-10-24（Asia/Shanghai）。当前完成 C03 Lesson01 的授课、参考答案整理和本地固定样例复核；Lesson02–08 尚未开始。

## 当前课程

**C03 Lesson01：真实 Tokenizer 与模型输入编码**

文本 → 分词 → Token ID → 特殊 Token → padding/truncation → input_ids / attention_mask。

- [知识点与教师代码说明](docs/C03-Lesson01-notes.md)
- [四道作业及参考答案](docs/C03-Lesson01-homework.md)
- [验收与来源说明](docs/C03-Lesson01-acceptance.md)
- [机器可读复核结果](docs/C03-Lesson01-verification.json)
- [C03 八节课表](docs/C03-syllabus.md)
- [下一课交接](docs/C03-Lesson01-handoff.md)

`hugging_learn/lesosn01.py` 是用户原有课堂文件，内容和注释逐字节保留。`hugging_learn/lesson01.py` 是本次增加的标准拼写入口，运行同一个原文件。参考答案、辅助入口、讲义和复核脚本是助手补充，分别标明来源。

## 环境与运行

沿用已有 Conda `fde` 环境。本次复核版本：Python 3.11.16、PyTorch 2.14.0、Transformers 5.20.2、huggingface_hub 2.2.0、tokenizers 0.23.3；CPU。

从仓库根目录运行：

```bash
conda activate fde
python -m hugging_learn.lesson01
python -m hugging_learn.homework.lesson01_answers
python verify.py --report /tmp/c03-lesson01-verification.json
```

原文件也可直接运行：

```bash
python hugging_learn/lesosn01.py
```

当前机器已有缓存，以上入口离线加载。其他机器首次使用时，先核对解释器和依赖，按需要安装 `requirements.txt`，再联网下载固定版本的分词文件：

```bash
python -m pip install -r requirements.txt
python -m hugging_learn.download_tokenizer
```

下载入口只调用 AutoTokenizer，不加载模型权重。缓存由 Hugging Face 管理，不提交到仓库。本文只验证本次记录的环境，未测试所有系统或依赖组合。

## 复现约定

Tokenizer：`google-bert/bert-base-chinese`，固定 revision：`8f23c25b06e129b6c986331a13d8d025a92cf0ea`。采用右侧 padding 和右侧 truncation。它是 BERT 配套 Tokenizer；后续生成实验将选用因果语言模型及其自身 Tokenizer。

本课 27 项离线检查通过，覆盖原课堂代码、标准入口、四题真实编码结果、形状、有效长度、特殊 Token、截断和 Mask 语义。不包含模型前向推理、训练或生成效果评估。

用户索要答案按约定接受为私下完成确认；助手参考代码运行通过不等于用户独立实现证明。聊天为主要课堂，先知识点、教师代码与解释、易错点，最后统一作业。后续推进与 GitHub 同步按用户明确指令分别执行。

本仓库只收录 C03 学习内容，不搬迁既有 C02 历史代码。环境、分词缓存、模型权重、密钥和编辑器配置均排除在 Git 之外。
