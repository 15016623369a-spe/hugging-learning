# C03 Lesson01 → Lesson02 交接

更新日期：2026-10-24（Asia/Shanghai）。

## 当前进度

C02 Lesson01–05 按此前交接已完成，本轮不重复验收旧模块。C03 Lesson01 已在聊天授课，用户索要四题答案按约定接受为私下完成确认；本轮已复核用户课堂文件、补充参考答案与说明，并通过 27 项固定样例检查。用户原文件、参考答案与运行证据分别保留来源。C03 Lesson02–08 未开始。

## 本课掌握范围

真实 Tokenizer、WordPiece 子词、encode/decode、特殊 Token、input_ids、attention_mask、BERT token_type_ids、批量 padding/truncation、[B,T]、有效长度和不同 API 的 Mask 语义。

尚未加载模型权重，没有模型推理、词表预测分数或生成循环。BERT 的 [MASK] Token 与 attention_mask 已区分；HF 有效位置 Mask 不等于因果 Mask，也不保证 PAD 输出清零。

## 文件与环境

- 仓库：[hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。
- 本地根目录：`/Users/jace/agent学习/hugging-learning`，课程文件在 `hugging_learn/`。
- 2026-10-28 统一为单一课堂文件 `lesson01.py`，原代码与注释完整保留；运行 `python -m hugging_learn.lesson01`。本交接的教学进度仍为历史快照。
- 参考答案为 `python -m hugging_learn.homework.lesson01_answers`。
- 复核为 `python verify.py --report /tmp/c03-lesson01-verification.json`。
- 沿用 fde，Python 3.11.16、PyTorch 2.14.0、Transformers 5.20.2、huggingface_hub 2.2.0、tokenizers 0.23.3。
- 分词文件版本锁定在 common.py，日常运行只读缓存；其他机器首次运行使用下载入口。

## 下一课

**C03 Lesson02：加载预训练模型与基础推理。** 实施前重新核对环境与可用资源，选择合适的小型因果语言模型并加载配套 Tokenizer，讲解配置、权重、设备、数据类型、eval 和推理模式，完成一次前向推理。词表分数的详细解释放在 Lesson03，手写生成循环放在 Lesson04。

教学继续采用：先完整知识点，再教师代码和逐段解释、易错点，最后统一作业。索要答案接受为私下完成，不要求截图。下一课只触发教学；之后明确要求同步时再提交 GitHub。不自动创建新聊天，不把未来课登记为完成。
