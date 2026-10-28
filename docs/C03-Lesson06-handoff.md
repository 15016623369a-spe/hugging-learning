# C03 Lesson06 → Lesson07 交接

更新日期：2026-10-28（Asia/Shanghai）。本文件保留Lesson06→07历史内容；当前Lesson07已完成题解和检查，最新接续见[Lesson07→08交接](C03-Lesson07-handoff.md)。

## 当前接续

C02 Lesson01–05按历史交接已完成。C03 Lesson01–05已授课、题解与历史检查已归档；Lesson06在本聊天已讲授，用户追问了tokenize True/False，随后要求四题答案、检查文件、同步GitHub，特别要求更新Markdown。本轮给出详细答案、补齐文件并通过40项固定检查。

后续已在本聊天讲授 **Lesson07：推理资源与复现记录**，作业尚未核对；用户再说“下一课”时进入 Lesson08。此前handoff为历史快照，当前以本文件和[通用交接](handoff.md)为准。

## 教学约定

- 聊天为主要课堂，不能只给文档链接代替讲解。
- 用户明确要求详细：先回顾前课和动机，逐项解释术语；用具体数字推导，标出张量形状；解释每段代码目的和关键行、参数、输入输出；解读实测结果、常见错误，最后统一四题作业。
- 用户索要答案按既有约定登记私下完成确认，不另要求截图。用户课堂、助手参考、实际运行证据和独立能力分别登记。
- 保留用户代码与全部注释，每课只维护一个标准课堂文件；“下一课”只教学，只有明确要求同步才更新GitHub；不自动创建新聊天。

## 文件与命名

按用户要求，课堂已合并为单一 [lesson06.py](../hugging_learn/lesson06.py)，代码与注释逐字节保留，SHA256 `51db515e0433de6dc2246e2b0bf1da3cf6e899f26f5a5235c6917dfb187ca184`。每课只维护一个标准课堂文件，取消转发入口；详见[文件整理记录](C03-file-organization.md)。

完整四题代码[lesson06_answers.py](../hugging_learn/homework/lesson06_answers.py)另存，提供describe_inputs、generate_batch和collect_answers。入口：`python -m hugging_learn.lesson06`、`python -m hugging_learn.homework.lesson06_answers`、`python verify_lesson06.py --report /tmp/c03-lesson06-verification.json`。

[详细讲义](C03-Lesson06-notes.md)、[四题题解](C03-Lesson06-homework.md)、[验收](C03-Lesson06-acceptance.md)、[运行JSON](C03-Lesson06-verification.json)、[验证器](../verify_lesson06.py)已补齐。README、课表和通用交接更新到Lesson06，历史Lesson01–05保留。

## 已覆盖与固定结果

Qwen/Qwen2.5-0.5B基础版，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU float32、eval、inference_mode、左补齐、use_cache=False。沿用fde：Python3.11.16、torch2.14.0、transformers5.20.2、huggingface_hub2.2.0、tokenizers0.23.3。不默认升级环境或下载权重。

课堂普通文本T=[2,5,9]、输入[3,9]，各新增8；聊天T=[27,43]、输入[2,43]，各新增8。作业文本T=[1,3,10]，左PAD=[9,7,0]、输入[3,10]、新增[3,4]、完整[3,14]。批量与逐条编号相同；错误切片得到“订单价为10”，正确为“价为10”。

聊天模板True添加`<|im_start|>assistant\n`共3个token。tokenize=True一步编码，与tokenize=False先返回文本再tokenizer(...,add_special_tokens=False)编码逐编号一致。add_special_tokens=False不关闭分词、不删除模板已有标记。当前API显式使用return_dict=True，避免默认返回类型差异。

两段聊天无生成开头长24/40，加后27/43；批量[2,43]→[2,51]，新增[2,8]。手动追加首段实际短回复和追问后，消息2→4，下一轮输入[1,50]；未继续生成下一轮回答。模型不自动保存Python历史。

本地基础模型Tokenizer有chat_template，但不代表Instruct权重。模板只组织输入，不改变训练阶段。基础版EOS/PAD=151643，消息im_end=151645；停止配置与消息边界不能自动等同。实际八token输出可能不完整且概念不准确，不能当作术语定义或质量验收。

本轮自然生成全部按长度停止，没有EOS。手工EOS/PAD示意另列，存储5列、含EOS生成3个、尾部补齐2列，不冒充自然轨迹。40项检查通过；没有训练、Instruct质量评估、性能或KVcache测量。

## Lesson07 建议接续

1. 从Lesson06中共同宽度W、真实长度T_i与新增上限N出发，区分存储长度、有效内容和上下文预算；先读取模型配置，不仅看名称或宣传值。
2. 解释use_cache=False为什么重复计算历史，逐步讲KVcache中缓存哪些张量、每步形状如何增长、位置与Mask如何处理。
3. 同一输入和贪心参数比较cache开/关，先核对编号再测耗时；不将一次运行当作可靠性能结论。
4. 讲内存、设备、dtype、batch、输入长度与输出长度的关系，用本地实际资源决定实验规模。CPU时间与GPU异步计时的差别按实际使用范围说明。
5. 记录模型ID/revision、版本、设备、dtype、输入、种子、参数、预热与重复次数；明确未测试项目。

本地仓库 `/Users/jace/agent学习/hugging-learning`；公开GitHub：[hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。
