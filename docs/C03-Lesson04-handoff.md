# C03 Lesson04 → Lesson05 交接

更新日期：2026-10-27（Asia/Shanghai）。

## 当前位置

C02 Lesson01–05 按此前交接已完成。C03 Lesson01–03 已授课、答案已给出，历史固定样例分别通过 27、26、31 项检查。Lesson04 已在本聊天讲授，四题答案已给出；用户索要答案按既有约定接受为私下完成确认。本轮课堂与参考代码通过 35 项检查。

下一次用户说“下一课”时进入 **Lesson05：generate 与生成参数**。Lesson05–08 尚未开始。此前各课 handoff 保留为历史快照，当前接续以本文件和 [通用交接](handoff.md) 为准。

## 已覆盖与实际代码

已覆盖最后位置取分数、贪心选择、`keepdim=True`、追加 Token ID 与 Mask、完整上下文再次前向、EOS 与最大新增上限、原输入与续写切分、统一解码、零步和一步边界。

用户 [lesson04.py](../hugging_learn/lesson04.py) 保留三步设置与“cat是concatenate”注释，兼容编辑器直接运行和模块运行。函数导入不加括号，调用加括号；编辑后须保存再运行。

完整四题参考单独位于 [lesson04_answers.py](../hugging_learn/homework/lesson04_answers.py)，支持直接运行与模块运行。三条输入分别重新编码并复用一份模型。参考实现不登记为用户原有独立实现。

## 固定结果与范围

模型 `Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU float32、eval、inference_mode、单句无 PAD、`use_cache=False`。原有 `fde` 环境、公共加载器和缓存沿用。

课堂无句号输入 T=9，上限 3，新增 `[8997,100012,100364]`，续写 `'。\n为了帮助'`，最终长度 12；最后一次 logits 的 T=11。三条带句号作业文本原始 T=[2,5,10]，各新增 8 个 Token，最终 T=[10,13,18]。上限 0 不前向，上限 1 只新增 8997。

真实模型样例均按长度结束。EOS 分支另用受控假模型检查，解码另用手工编号说明，不把受控样例当成自然生成。已运行手写贪心循环，没有调用 generate、采样、训练或评估生成质量。

环境：Python 3.11.16、PyTorch 2.14.0、Transformers 5.20.2、huggingface_hub 2.2.0、tokenizers 0.23.3。不默认重建环境或升级依赖。

## 接续 Lesson05

1. 复习手写循环，使用同一输入比较 `model.generate` 的明确贪心设置与手写编号。
2. 明确设置 `max_new_tokens`、`do_sample=False`，说明 EOS 和输出包含原始输入，检查模型生成配置的默认值。
3. 讲解采样与贪心的区别，再进入 temperature、top-k、top-p，说明参数适用条件及随机种子。
4. 记录版本、输入、参数、编号与结果；不将单次续写效果当成稳定质量评估。

批量补齐、聊天模板与指令模型留到 Lesson06；KV cache 详解留到 Lesson07。继续按“完整知识点→教师代码与逐段解释→易错点→统一作业”的顺序教学。

## 入口

- [讲义](C03-Lesson04-notes.md)、[四题答案](C03-Lesson04-homework.md)、[验收](C03-Lesson04-acceptance.md)、[运行记录](C03-Lesson04-verification.json)。
- `python -m hugging_learn.lesson04`
- `python -m hugging_learn.homework.lesson04_answers`
- `python verify_lesson04.py --report /tmp/c03-lesson04-verification.json`
- [C03 课表](C03-syllabus.md)。

本地根目录 `/Users/jace/agent学习/hugging-learning`；公开仓库 [hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。

用户原文件及注释保留；区分授课、私下完成确认、助手补充与实际运行。不要求截图，不自动创建新聊天。“下一课”继续教学，GitHub 同步按用户明确指令执行。
