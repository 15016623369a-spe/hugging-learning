# C03 Lesson02 → Lesson03 交接

更新日期：2026-10-26（Asia/Shanghai）。

## 当前进度

C02 Lesson01–05 按此前交接已完成。C03 Lesson01 已授课、参考答案已给出，历史 27 项检查通过，原文件保留。Lesson02 已于本聊天讲授，本轮提供四题答案；用户索要答案按约定接受为私下完成确认。用户课堂脚本、参考答案与实际运行证据分别记录；本轮 Lesson02 的 26 项检查通过。

Lesson03–08 尚未开始。下一次用户要求“下一节课”时，从 **Lesson03：从隐藏表示到下一个 Token** 开始。

## 已覆盖范围

配置、Tokenizer、模型权重职责；from_pretrained 与 from_config；CausalLM 及输出头；固定 revision 和本地文件缓存；设备、模型参数 dtype 与整数 Token ID；eval 与 inference_mode；输入字典展开；一次前向及 `[B,T,V]` 输出。

没有训练、生成循环或生成效果评估。logits 尚未按 Lesson03 的深度讲解；不能据此登记为掌握了候选 Token 筛选、采样或手写生成。

## 文件与运行

本地根目录 `/Users/jace/agent学习/hugging-learning`，公开仓库 [hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。

- 用户课堂：[lesson02.py](../hugging_learn/lesson02.py)，仅规范模型名及两处输出提示；注释与结构保留。
- 参考代码：[lesson02_answers.py](../hugging_learn/homework/lesson02_answers.py)。
- [讲义](C03-Lesson02-notes.md)、[四题答案](C03-Lesson02-homework.md)、[验收](C03-Lesson02-acceptance.md)、[真实运行 JSON](C03-Lesson02-verification.json)。
- 复核：`python verify_lesson02.py --report /tmp/c03-lesson02-verification.json`。
- 课堂：`python -m hugging_learn.lesson02`；答案：`python -m hugging_learn.homework.lesson02_answers`。
- 新机器首次联网准备：`python -m hugging_learn.download_model`，日常只读取本地缓存。

环境沿用 `fde`：Python 3.11.16、PyTorch 2.14.0、Transformers 5.20.2、huggingface_hub 2.2.0、tokenizers 0.23.3。运行前确认当前解释器与资源，不默认升级依赖。

## 固定实验结果

模型 `Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU float32；Qwen2ForCausalLM；hidden_size 896、24 层、输出词表 151936、参数量 494032768。

`len(tokenizer)=151665`，与模型输出头维度不相等，不应强制断言二者相等。课堂输入“我想查询订单金额。”为 `[[104100,51154,50410,80094,1773]]`，输入 `[1,5]`，logits `[1,5,151936]`；输出均有限、`requires_grad=False`。

三条作业文本“你好。”“我想查询订单金额。”“请统计最近一个月各部门的订单总金额。”分别得到 T 为 2、5、10。Linear 实验中 eval 后普通前向仍记录梯度，inference_mode 内不记录。

## Lesson03 教学入口

沿用同一基础模型与配套 Tokenizer，先复习 `[B,T]` 的输入和 `[B,T,V]` 的输出，再连接 C02 中 `[B,T,D]` 的隐藏表示与词表输出头。

重点解释：每个位置的词表分数、因果模型的位置含义、最后一个有效位置的下一 Token 分数、logits 与 softmax 概率、候选 Token 与 ID、选出一个 Token 后如何解码。本课先处理单条无补齐文本，批量补齐和聊天模板按后续课表展开。

手写生成循环留给 Lesson04；generate 参数与采样留给 Lesson05；KV cache 详解留给 Lesson07。

继续采用“完整知识点→教师代码及解释→易错点→统一作业”的节奏。索要答案接受为私下完成，不要求截图。下一节课只继续教学；GitHub 同步按用户明确指令执行，不自动创建新聊天。
