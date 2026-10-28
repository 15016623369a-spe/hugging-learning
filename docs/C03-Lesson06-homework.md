# C03 Lesson06 四道作业与详细参考答案

更新日期：2026-10-28（Asia/Shanghai）。完整代码：[lesson06_answers.py](../hugging_learn/homework/lesson06_answers.py)。答案由助手补充；用户索要答案按既有约定登记私下完成确认，未另验收独立能力。原课堂与注释保留。

沿用Qwen/Qwen2.5-0.5B固定revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，CPU float32、eval、inference_mode、左补齐、use_cache=False；同一份模型复用。所有编号和参数见[运行JSON](C03-Lesson06-verification.json)。

## 第1题：批量编码、Mask和补齐数量

题目输入：

```python
texts = ["订单", "查询订单金额", "请统计最近一个月各部门的订单总金额。"]
tokenizer.padding_side = "left"
inputs = tokenizer(texts, padding=True, return_tensors="pt")
real_lengths = inputs["attention_mask"].sum(dim=1)
width = inputs["input_ids"].shape[1]
```

| 输入 | 真实长度 | 共同宽度 | 左侧PAD数量 |
|---|---:|---:|---:|
| 订单 | 1 | 10 | 9 |
| 查询订单金额 | 3 | 10 | 7 |
| 请统计最近一个月各部门的订单总金额。 | 10 | 10 | 0 |

input_ids与Mask形状均为 `[3,10]`。Mask为：

```python
[
    [0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
]
```

正文编号（去掉左PAD后）分别是：

```python
[50410]
[51154, 50410, 80094]
[14880, 100787, 104044, 105033, 106852, 9370, 50410, 59743, 80094, 1773]
```

PAD编号为151643，按上表数量填在每行左边。`attention_mask.sum(dim=1)`沿每行累加得到[1,3,10]；共同宽度由张量列数得到10。PAD数为共同宽度减有效长度。第三条题目带句号，和课堂无句号T=9不同，不能改文本来匹配课堂长度。

## 第2题：四步批量生成与正确切片

```python
width = inputs["input_ids"].shape[1]
with torch.inference_mode():
    outputs = model.generate(
        **inputs, max_new_tokens=4, do_sample=False, num_beams=1,
        use_cache=False, eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=False,
    )
new_ids = outputs[:, width:]
```

实际输出 `[3,14]`，新增 `[3,4]`，三条均未生成EOS、达到上限后停止。

| 输入 | 新增编号 | 续写repr | 与逐条生成相同 |
|---|---|---|---|
| 订单 | `[50292,17714,16,15]` | `'价为10'` | True |
| 查询订单金额 | `[198,51154,50410,89656]` | `'\n查询订单详情'` | True |
| 长输入 | `[220,100012,100364,87026]` | `' 为了帮助您'` | True |

这些是基础模型的续写，不是数据库查询结果。相同编号是本次固定模型与CPU环境的实测，不保证任意模型或跨设备都逐编号相同。

最短输入真实长度1，但输出第一行布局是：

```text
9个PAD + 1个正文“订单” + 4个新增token
└────── 共同输入宽度10 ──────┘
```

错误切法 `outputs[0,1:]` 仍包含若干PAD与正文。跳过特殊符号后，PAD看不见，正文却还在：`'订单价为10'`。

正确切法 `outputs[0,10:]` 或统一的 `outputs[:,width:]` 得到`'价为10'`。真实长度用于解释有效内容，共同宽度用于定位当前批量输出中的新增起点，二者不能互换。

## 第3题：聊天模板、角色开头和批量编码

这道题可有不同的聊天内容。参考答案使用与课堂相同的两个独立聊天，便于逐项复核：第一段2条消息，第二段含一轮历史，共4条。

```python
chats = [
    [
        {"role": "system", "content": "你是一位耐心的中文老师。"},
        {"role": "user", "content": "请用一句话解释什么是 Token。"},
    ],
    [
        {"role": "system", "content": "你是一位耐心的中文老师。"},
        {"role": "user", "content": "你好。"},
        {"role": "assistant", "content": "你好，我们一起学习。"},
        {"role": "user", "content": "attention_mask 有什么作用？"},
    ],
]
```

外层列表的每项是一个独立聊天；内层列表按顺序保存同一聊天的消息；字典的role表示角色，content表示正文。把两段聊天的消息直接放在同一个内层列表，会变成一段共享历史。

### add_generation_prompt的差别

分别运行 `apply_chat_template(chat,tokenize=False,add_generation_prompt=False/True)`。第一段False时为：

```text
<|im_start|>system
你是一位耐心的中文老师。<|im_end|>
<|im_start|>user
请用一句话解释什么是 Token。<|im_end|>
```

True在末尾再加：

```text
<|im_start|>assistant
```

本地模板中这段开头对应3个token，所以两段聊天的编码长度分别为：

| 聊天 | 不加助手开头 | 加助手开头 |
|---|---:|---:|
| 第1段 | 24 | 27 |
| 第2段 | 40 | 43 |

开头只是准备输入，不产生回答。然后用：

```python
chat_inputs = tokenizer.apply_chat_template(
    chats, tokenize=True, padding=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
```

真实长度[27,43]，PAD数[16,0]，输入 `[2,43]`。本次各生成8个token，完整输出 `[2,51]`，新增 `[2,8]`。续写是：

```python
[
    "Token 是中文中用于表示一个单词",
    "attention_mask 是一个用于在模型中",
]
```

这些是基础模型的八token短续写，不是完整、可靠的术语定义。token不限定为中文单词，可以是词、子词、标点或字符片段。

### 用户追问：为什么tokenize上面True、下面False

一步写法 `apply_chat_template(...,tokenize=True,return_dict=True,return_tensors='pt')` 同时完成套模板和编码。

两步写法先 `formatted=apply_chat_template(...,tokenize=False)` 得到字符串，再 `tokenizer(formatted,add_special_tokens=False,return_tensors='pt')` 编码。False只表示第一步暂不编码，不表示整个流程不编码。

`add_special_tokens=False`控制不额外自动添加特殊标记，避免和模板已有标记重复；不关闭分词、不删除模板标记。本轮分别验证两段聊天的一步和两步路径，编号完全相同。`tokenizer`是对象名，`tokenize`是参数名。

## 第4题：手动历史、模型类型与EOS/PAD计数

```python
history = [dict(message) for message in chats[0]]
history.append({"role": "assistant", "content": replies[0]})
history.append({"role": "user", "content": "再举一个例子。"})
next_inputs = tokenizer.apply_chat_template(
    history, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
```

角色顺序从 `[system,user]` 变成 `[system,user,assistant,user]`。参考中保存的是第3题实际得到的8个token短续写，消息数2→4，重新编码后长度27→50，形状 `[1,50]`。这一步只编码下一轮，没有生成其回答。

历史由程序列表保存，再作为输入送回模型，不是模型自动记住了上次Python调用。角色和边界标记也会占token，不能只用新增文字的字符数估计长度。

模板改变消息的排列与编码，不改变模型权重。当前实际加载Qwen2.5-0.5B基础版；Instruct版经过指令相关后训练，不能通过套模板、设置system或调用eval把基础版转成Instruct。本轮没有下载或验证Instruct版的回答质量。

手工序列：

```text
[token1, token2, EOS, PAD, PAD]
```

存储5列；包含EOS的实际生成数3；尾部补齐2。EOS本身也占一个生成位置。本课PAD与EOS均为151643，手工编号可写成 `[8997,100012,151643,151643,151643]`。解码不跳过特殊标记时为`'。\n为了<|endoftext|><|endoftext|><|endoftext|>'`；跳过后为`'。\n为了'`。

这个示意的前三个位置由题目给定为真实生成，后两个由题目给定为补齐；不能只看同号标记出现次数判断生成步数。原输入Mask也不能代替生成停止记录。手工示意与本次自然生成分开归档；本次真实样例均未生成EOS。

## 完整运行入口

```sh
conda activate fde
python -m hugging_learn.lesson06
python -m hugging_learn.homework.lesson06_answers
python -m hugging_learn.homework.lesson06_answers --report /tmp/c03-lesson06-answers.json
python verify_lesson06.py --report /tmp/c03-lesson06-verification.json
```

标准课堂入口执行原名lesoon06.py；原文件也可直接运行。答案支持直接脚本或模块运行。40项固定检查通过，详见[验收与来源](C03-Lesson06-acceptance.md)。
