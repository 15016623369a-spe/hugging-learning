# C03 Lesson07：推理资源与复现记录

日期：2026-10-28（Asia/Shanghai）。课堂是[lesson07.py](../hugging_learn/lesson07.py)，四题参考在[homework](../hugging_learn/homework/lesson07_answers.py)，详细题解见[作业答案](C03-Lesson07-homework.md)。每课只维护一个课堂文件。

## 接续Lesson06

继续区分真实长度T、补齐宽度W、Batch大小B和新增上限N。聊天角色及模板标记也占token。本课按实际模型配置32768检查 `W+N` 预算；通过长度检查仍需考虑设备内存。稠密批量缓存中的补齐位置也会占存储列。

## Prefill与逐步Decode

不开缓存时，生成每一步会重新计算已给出的历史。开启KV Cache后，第一次完整处理提示词（Prefill），保存每层的K/V；后续每次只送一个尚未处理的新token，复用历史缓存（Decode）。旧位置在因果注意力中不能访问未来，所以其历史K/V可复用。新位置仍需关注历史，不能把单步成本理解为与上下文无关。

参数权重、KV Cache、Python中的messages分别承担训练参数、当前序列中间结果、应用聊天历史的用途。缓存不会自动维护messages。每次独立生成应从干净缓存开始。

手动前向必须配合完整Mask：`[B,历史缓存长度+本次新输入长度]`。本次输入长度可以为1，完整Mask仍然增长。输入完整历史缓存后，再传完整提示词会重复处理前缀。

## 用“订单金额”理解张量

“订单金额”在当前Qwen词表中为两个token：[50410,80094]。三次本次输入形状为 `[1,2]→[1,1]→[1,1]`；Mask为 `[1,2]→[1,3]→[1,4]`；缓存长度为2→3→4。

`current_ids=next_id` 是第二次只输入一个token的直接原因；历史K/V通过 `past_key_values=cache` 传入。第一次选出100381“超过”，第二次输入即为 `[[100381]]`。“超过”有两个汉字但只占一个token。

K和V每层各自形状为 `[B,KV头数,S,head_dim]`，本模型分别为 `[1,2,S,64]`。14个Query头与2个KV头不同；K/V中的“2个头”和第一次“2个输入token”只是数值相同。

选出最后一个token后，它可能还未再次前向，因此已选择的可见长度可能比缓存长度多1。第一题最后为可见5、缓存4。

## 资源计算

模型24层、每个头64维，float32元素4字节。KV张量字节数为 `2×24×B×2×S×64×4`，每条每token为24KiB。参数实际494,032,768个，参数张量约1.840GiB；KV估算不包含临时激活、logits、工作区、运行环境等开销。

## 测量与复现

同一文本、同一模型与revision、同样生成参数，仅切换缓存；先核对编号，再比较耗时。加载与编码放在计时区间外，各条件预热1次、运行5次、取中位数并保存原始值，交替先后顺序。CPU计时使用perf_counter；其他设备的异步执行需另行处理，本课没有测量GPU/MPS。

保存实际设备/dtype、线程数、版本、输入编号、长度、完整生成条件、EOS/PAD、缓存开关、输出编号、计时范围和预热/重复次数。模型配置标注的dtype不一定等于实际加载dtype，应读取参数本身。

官方原理参考：[Hugging Face缓存解释](https://huggingface.co/docs/transformers/cache_explanation)、[Python perf_counter](https://docs.python.org/3/library/time.html#time.perf_counter)。本仓库运行证据以固定本地环境的JSON为准。

## 当前课堂与运行修复

用户当前课堂改用attention_mask解释题和新增上限8；三次缓存长度9/10/11，课堂还比较原提示词与重复输入的耗时。四题参考独立记录“订单金额”的2/3/4轨迹，二者输入不同。

保留原代码及所有注释，仅将上下文预算中的固定 `+16` 改为 `+generation_configs[True].max_new_tokens`，使其随实际新增上限联动。直接运行兼容代码包括Path、sys导入，并在 `not __package__` 时加入项目根目录；避免此前ModuleNotFoundError和Path未导入错误。

```sh
python -m hugging_learn.lesson07
python hugging_learn/lesson07.py
```
