# C03 Lesson03 → Lesson04 交接

更新日期：2026-10-26（Asia/Shanghai）。

## 当前位置

C02 Lesson01–05 按此前交接已完成。C03 Lesson01、Lesson02 已授课，历史固定样例分别通过 27、26 项检查。Lesson03 已在聊天讲授、四题答案已给出，索要答案按既有约定接受为私下完成确认；本轮课堂与参考实现完成 31 项检查。用户原文件与助手补充来源分开记录。

下一次用户说“下一课”时进入 **Lesson04：手写自回归生成循环**。Lesson04–08 尚未开始。旧 Lesson01→02、Lesson02→03 文件保留为历史快照，当前入口以本文件和 [通用交接](handoff.md) 为准。

## Lesson03 已覆盖

`[B,T,D]` 最终隐藏表示、词表输出头、`[B,T,V]` logits；因果模型每个位置预测下一位置；单条无 PAD 输入的最后位置；softmax 沿词表维归一化；原词表概率与 top5 概率和；argmax 贪心选择；Token ID、raw token 与字节片段解码。

手动输出头与模型 logits 的差异为 0；用户无句号文本 T=9，最高分 ID=8997，单项解码 `'。\n'`。教师示例 T=5、作业带句号文本 T=10，各自结果按输入区分，不能把用户文本改掉以匹配教师输出。不同长度前缀有 float32 舍入差异；同长度后缀替换对照差异为 0。

没有追加 Token 的生成循环，没有训练或生成质量评估。不要将“选出一个编号”登记为“已完成多步生成”。

## 文件入口

- 用户课堂：[lesson03.py](../hugging_learn/lesson03.py)，字节与注释保留。
- 助手参考：[lesson03_answers.py](../hugging_learn/homework/lesson03_answers.py)。
- [讲义](C03-Lesson03-notes.md)、[作业与答案](C03-Lesson03-homework.md)、[验收与来源](C03-Lesson03-acceptance.md)、[真实运行记录](C03-Lesson03-verification.json)。
- [验证器](../verify_lesson03.py)：`python verify_lesson03.py --report /tmp/c03-lesson03-verification.json`。
- 课堂：`python -m hugging_learn.lesson03`；答案：`python -m hugging_learn.homework.lesson03_answers`。
- [C03 课表](C03-syllabus.md)。

本地根目录 `/Users/jace/agent学习/hugging-learning`，公开仓库 [hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。

沿用 Conda `fde`，Python 3.11.16、PyTorch 2.14.0、Transformers 5.20.2、huggingface_hub 2.2.0、tokenizers 0.23.3。固定模型 `Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU float32、eval、inference_mode，日常只读本地缓存。加载入口为 [causal_common.py](../hugging_learn/causal_common.py)。

## Lesson04 教学入口

从已理解的 `next_logits = logits[:,-1,:]` 和 `next_id = argmax(...)` 开始：

1. 先用单条无补齐文本，显式设置较小的 `max_new_tokens`。
2. 将 `[B]` 的 next_id 变成 `[B,1]`，追加到 `[B,T]` 的 input_ids；同步给 attention_mask 追加有效位置 1。
3. 在 `eval` 和推理模式下，以更新后的完整序列再次前向，先用 `use_cache=False` 理解重复计算。
4. 每步记录新编号和长度，区分完整输入加续写、仅新增 Token；最后对完整新增编号序列统一解码，避免字节碎片误读。
5. 从实际 Tokenizer/配置读取 EOS，判断结束编号；另保留最大新增长度上限，说明 EOS 不保证出现。

先讲完整知识点，再给教师代码并逐段解释，说明常见错误，最后统一布置作业。采样与 `generate` 留给 Lesson05；批量补齐与聊天模板留给 Lesson06；KV cache 详解留给 Lesson07。

用户索要答案接受为私下完成，不要求截图。保留用户已有文件和注释，区分授课、用户确认、助手补充和运行证据。用户要求“下一课”时继续教学；GitHub 同步按明确指令执行，不自动新建聊天。
