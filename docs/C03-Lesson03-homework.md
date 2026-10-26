# C03 Lesson03 课后习题与参考答案

更新日期：2026-10-26（Asia/Shanghai）。答案为助手补充，用户索要答案按教学约定接受为私下完成确认；不将参考代码登记成用户原有独立实现。

[可运行答案](../hugging_learn/homework/lesson03_answers.py) 共用一次模型加载，依次处理三个单句；[运行 JSON](C03-Lesson03-verification.json) 保存完整精度、编号及来源。

## 第 1 题：预测并检查形状

将文本改为“你好。”，预测输入、隐藏状态、logits、最后位置分数、前五候选和最高分编号的形状，再实际运行。

本固定 Tokenizer 编码为 `[[108386,1773]]`，所以 `B=1,T=2,D=896,V=151936`：

| 张量 | 形状 |
|---|---|
| input_ids | `[1,2]` |
| hidden | `[1,2,896]` |
| logits | `[1,2,151936]` |
| next_logits | `[1,151936]` |
| top_ids | `[1,5]` |
| top_probs | `[1,5]` |
| next_ids | `[1]` |

候选数量 `k=5` 与输入长度 `T=2` 是不同的概念。对最后位置索引后，时间维消失；最高分编号按词表维取最大值后，词表维也消失。

## 第 2 题：三条文本的前五个候选

模型只加载一次，分别处理“你好。”“我想查询订单金额。”“请统计最近一个月各部门的订单总金额。”，记录 Token 数、前五候选 ID、单项解码、概率及前五项概率和。

以下均为 CPU float32 前向、float64 softmax 的本机实际输出，概率列四舍五入；输入包含末尾句号。

### 你好。

T=2，input_ids 为 `[[108386, 1773]]`。

| 排名 | ID | 单项解码（repr） | 原词表概率 |
|---:|---:|---|---:|
| 1 | 35946 | `'我'` | 0.31689039 |
| 2 | 109194 | `'请问'` | 0.06239531 |
| 3 | 104100 | `'我想'` | 0.05514854 |
| 4 | 103998 | `'医生'` | 0.04243862 |
| 5 | 97611 | `'我的'` | 0.03194399 |

完整词表概率和约 1；前五项合计 `0.5088168597`，约 50.88%。

### 我想查询订单金额。

T=5，input_ids 为 `[[104100, 51154, 50410, 80094, 1773]]`。

| 排名 | ID | 单项解码（repr） | 原词表概率 |
|---:|---:|---|---:|
| 1 | 50410 | `'订单'` | 0.11492907 |
| 2 | 35946 | `'我'` | 0.03597498 |
| 3 | 14880 | `'请'` | 0.02946816 |
| 4 | 18493 | `'在'` | 0.02555683 |
| 5 | 220 | `' '` | 0.02485930 |

完整词表概率和约 1；前五项合计 `0.2307883432`，约 23.08%。

### 请统计最近一个月各部门的订单总金额。

T=10，input_ids 为 `[[14880, 100787, 104044, 105033, 106852, 9370, 50410, 59743, 80094, 1773]]`。

| 排名 | ID | 单项解码（repr） | 原词表概率 |
|---:|---:|---|---:|
| 1 | 220 | `' '` | 0.25975629 |
| 2 | 84897 | `' 当'` | 0.09363907 |
| 3 | 4891 | `' �'` | 0.06548042 |
| 4 | 6567 | `' �'` | 0.05549307 |
| 5 | 69058 | `' 输入'` | 0.04108295 |

完整词表概率和约 1；前五项合计 `0.5154518143`，约 51.55%。

前五项概率和不等于 1，因为剩余词表项也有概率。这里保留原分布概率，没有对五项重新做 softmax。

第三条的单项解码 `' �'` 对应两个不同的字节片段：ID `4891` 的 raw token 为 `Ġå`，ID `6567` 为 `Ġæ`。一个候选 Token 不保证包含完整 UTF-8 字符，因此独立解码可能显示替换字符；并非两项相同，也不能据此断言为 UNK。

用户课堂文件的第三条文本没有句号，T=9，最高分是 `8997`（`'。\n'`），与这里 T=10 的结果不同。原文件保持不变。

核心循环使用：

```python
config, tokenizer, model = load_causal_lm()
for text in TEXTS:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, output_hidden_states=True)
        next_logits = outputs.logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1, dtype=torch.float64)
        top_probs, top_ids = probs.topk(5, dim=-1)
```

完整代码另记录各形状、raw token、dtype、有限值与最高分编号。本课固定 CPU；完整入口仍显式把输入搬到模型设备。

## 第 3 题：手算 softmax 与前两项重新归一化

给定 `scores=[3,1,-1]`，求 softmax、最大值索引和原始前两项概率和；再比较只对 `[3,1]` 做 softmax。

完整分布的共同分母为 `exp(3)+exp(1)+exp(-1)`，结果是：

```text
完整分布       [0.8668133322, 0.1173104278, 0.0158762400]
最高分索引     0
原前两项之和   0.9841237600
只对 [3,1] 算  [0.8807970780, 0.1192029220]
重新归一化之和 约 1
```

只保留两项后，分母变成 `exp(3)+exp(1)`，概率重新分配。它是两个候选之间的条件分布，不能冒充原词表上的概率。这也解释了为什么不能把 top5 的 logits 单独 softmax 后当成原分布的 top5 概率。[softmax 定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)。

## 第 4 题：诊断三个常见错误

| 写法 | 问题 | 修正 |
|---|---|---|
| `outputs.logits[:,0,:]` | 取的是第一个位置，预测第一个 Token 之后；没有使用完整输入的最后位置 | 本课无 PAD，用 `[:,-1,:]` |
| `softmax(outputs.logits,dim=1)` | 在 `[B,T,V]` 的时间维归一化，对每个词表编号跨位置求和 | 沿词表维 `dim=-1` |
| `outputs.logits.argmax()` | 全张量找最大值，返回展平后的索引 | 先取最后位置，再 `argmax(dim=-1)` |

```python
next_logits = outputs.logits[:, -1, :]
probs = torch.softmax(next_logits, dim=-1, dtype=torch.float64)
next_ids = next_logits.argmax(dim=-1)
token_id = next_ids.item()  # 仅适用于 B=1
```

用户课堂的全局 `argmax()` 实际返回 `1143646`，还原位置是 `[batch=0,time=7,vocab=80094]`；它与最后位置的候选 `8997` 不同。全局结果不是可直接解码的 Token ID。[argmax 接口](https://docs.pytorch.org/docs/2.14/generated/torch.argmax.html)。

运行：`python -m hugging_learn.homework.lesson03_answers`；可加 `--report /tmp/c03-lesson03-answers.json` 保存结果。复核全部课堂与答案：`python verify_lesson03.py --report /tmp/c03-lesson03-verification.json`。
