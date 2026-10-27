# C03 Lesson05 → Lesson06 交接

更新日期：2026-10-27（Asia/Shanghai）。

## 当前进度

C02 Lesson01–05按此前交接已完成。C03 Lesson01–04历史固定检查分别27、26、31、35项。本聊天已讲授Lesson05；用户明确说“完成了”，随后请求四题答案、检查文件、同步GitHub。本轮给出详细答案并完成38项固定样例检查。

下一次用户说“下一课”进入 **Lesson06：批量生成与对话输入**。Lesson06–08尚未开始。早期handoff保留历史快照，接续以本文件和[通用交接](handoff.md)为准。

## 必须延续的教学偏好

用户明确要求每节课讲解详细。先衔接上一课、说明知识动机，再逐个解释术语；涉及数学用具体数字推导，涉及张量标明每步形状。教师代码先分段讲目的，再解释关键行、参数和输入输出，最后给完整代码。解释真实结果、修改参数后的可能变化和常见错误，最后统一布置四道作业。

聊天是主要课堂；不以文档链接代替完整授课。用户确认完成、助手参考与真实执行分别记录，不额外要求截图。保留用户文件和学习注释。“下一课”只继续教学；GitHub同步须用户明确要求。不自动创建新聊天。

## 现有文件

- 用户[lesson05.py](../hugging_learn/lesson05.py)逐字节保留，含字典展开示例；SHA256 `fdb29b74c3c2536321a120b1d667b978e31e31c521bdc037204100d06e7577c2`。
- 助手[四题参考](../hugging_learn/homework/lesson05_answers.py)单独保存，提供generate_once、教学过滤算例与collect_answers。
- [详细讲义](C03-Lesson05-notes.md)、[四题答案](C03-Lesson05-homework.md)、[验收](C03-Lesson05-acceptance.md)、[真实运行](C03-Lesson05-verification.json)、[验证器](../verify_lesson05.py)。
- 保留Lesson01–04源码和历史结果。课堂与答案兼容直接运行与模块入口。

## 固定模型与结果

Qwen/Qwen2.5-0.5B，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，CPU float32、eval、inference_mode、单句无PAD、use_cache=False。沿用fde与本地缓存，不默认升级依赖或下载新模型。

Python3.11.16、torch2.14.0、transformers5.20.2、huggingface_hub2.2.0、tokenizers0.23.3。模型自带max_new_tokens=2048，部分生成字段为None；课堂显式设小上限。仅设置max_length时要清除已有max_new_tokens。本版本max_new_tokens=0报ValueError。

课堂无句号输入T=9，贪心八步编号 `[8997,100012,100364,87026,100787,104044,105033,106852]`，续写 `'。\n为了帮助您统计最近一个月各部门'`，完整 `[1,17]`，与手写循环逐编号一致。三步都为 `[8997,100012,100364]`，完整 `[1,12]`。

两条独立输入T=[2,9]；新增上限4/8使完整长度为[6,10,13,17]；总上限12时新增[10,3]。当前自然样例均按长度结束，没有EOS。

采样do_sample=True、temperature=0.8、top_k=20、top_p=0.9、max_new_tokens=8。每次重设42的编号一致；43本次不同。只设一次42的连续三轮不同，完整序列可在同条件下复现。不得把固定种子解释成跨平台绝对保证，也不承诺不同种子必然不同。

温度与过滤使用独立手工logits `[3,2,1,0,-1]`；这些算例与真实Qwen轨迹明确分开。没有训练、质量评估或性能结论。

## Lesson06 的接续建议

1. 从Lesson05的单条编码与生成输出切片接续，比较不同长度输入如何组成一个批量张量。
2. 讲清真实长度、补齐后的共同宽度和attention_mask；验证decoder-only批量生成的补齐方向，不能沿用Lesson01 BERT的右补齐结论。
3. 明确批量生成切分通常依据输入张量共同宽度，不能盲目逐行用mask和切原输出前缀；先实测再解释。
4. 区分基础模型与指令模型；聊天模板决定角色文本如何编码。不要未经核对就对当前基础模型调用模板或声称有配套模板。
5. 如果需要新指令模型或额外权重，先检查已有缓存、配置和资源，再按本次授权与可用条件推进；不把未运行的示例登记为已验证。
6. EOS、PAD及批量停止记录清楚；KVcache完整原理和资源测量留Lesson07。

本地仓库 `/Users/jace/agent学习/hugging-learning`，公开GitHub：[hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。
