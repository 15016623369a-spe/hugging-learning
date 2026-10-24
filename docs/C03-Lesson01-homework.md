# C03 Lesson01：四道作业及参考答案

参考答案为 2026-10-24 助手补充，结果来自固定版本 Tokenizer 的真实运行。用户索要答案按授课约定接受为私下完成确认；此文件和参考代码不冒充用户原有独立作业。

可运行完整参考代码：`python -m hugging_learn.homework.lesson01_answers`。加载缓存、输出每题 Token、ID、Mask、shape 与解码结果，不下载模型权重。

## 作业 1：观察真实分词

题目：对 `订单金额`、`playing`、`Transformer`、`transformer`、`🧠` 打印 Token 和不自动加入特殊标记的 ID，解释拆分与未知 Token。

| 输入 | Token | ID |
|---|---|---|
| 订单金额 | 订、单、金、额 | [6370,1296,7032,7583] |
| playing | play、##ing | [8942,8221] |
| Transformer | [UNK] | [100] |
| transformer | t、##ran、##s、##form、##er | [162,10477,8118,12725,8196] |
| 🧠 | [UNK] | [100] |

`playing` 和小写 `transformer` 被拆成子词；`订单金额` 在本词表中逐字拆分。大写 `Transformer` 和 emoji 在当前规则下变成 `[UNK]`。本配置保留大小写，不能推断所有 Tokenizer 都会如此。

Token 数分别为 4、2、1、5、1。它由分词规则和词表决定，不能用字符数或单词数代替。`[UNK]` 的 attention_mask 为 1，依然占有效位置。

核心代码：

```python
for text in ["订单金额", "playing", "Transformer", "transformer", "🧠"]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    print(text, tokenizer.convert_ids_to_tokens(ids), ids)
```

## 作业 2：预测批量形状

输入：`订单`、`查询订单`、`请查询本月订单金额`。设置 `padding=True, return_tensors="pt"`，默认加入特殊 Token。

三条内容分别有 2、4、9 个 Token，加上 `[CLS]` 和 `[SEP]` 后，有效长度为 **[4,6,11]**。因此 B=3、T=11，input_ids 和 attention_mask 的形状均为 **[3,11]**。

```text
input_ids =
[[101,6370,1296,102,0,0,0,0,0,0,0],
 [101,3389,6418,6370,1296,102,0,0,0,0,0],
 [101,6435,3389,6418,3315,3299,6370,1296,7032,7583,102]]

attention_mask =
[[1,1,1,1,0,0,0,0,0,0,0],
 [1,1,1,1,1,1,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,1,1,1]]
```

```python
texts = ["订单", "查询订单", "请查询本月订单金额"]
batch = tokenizer(texts, padding=True, return_tensors="pt")
print(batch["input_ids"])
print(batch["attention_mask"])
print(batch["input_ids"].shape)
print(batch["attention_mask"].sum(dim=1))
```

## 作业 3：对比长度策略

继续使用作业 2 的输入。

| 策略 | shape | 有效长度 | 内容变化 |
|---|---|---|---|
| padding=True | [3,11] | [4,6,11] | 无 |
| padding="max_length", max_length=12 | [3,12] | [4,6,11] | 无，仅增加补齐位置 |
| padding="max_length", truncation=True, max_length=8 | [3,8] | [4,6,8] | 第三条截断 |

前两组解码结果相同：

```text
订 单
查 询 订 单
请 查 询 本 月 订 单 金 额
```

第三组解码结果：

```text
订 单
查 询 订 单
请 查 询 本 月 订
```

长度 8 包含两个边界 Token，所以第三条只保留前六个内容 Token：`[CLS] 请 查 询 本 月 订 [SEP]`。它丢失了 `单 金 额`，还截断了“订单”这个词的语义。

```python
strategies = [
    {"padding": True},
    {"padding": "max_length", "max_length": 12},
    {"padding": "max_length", "truncation": True, "max_length": 8},
]
for kwargs in strategies:
    encoded = tokenizer(texts, return_tensors="pt", **kwargs)
    print(encoded["input_ids"].shape)
    print(encoded["attention_mask"].sum(dim=1))
    print(tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True))
```

`**kwargs` 把字典中的键值展开为本次函数调用的关键字参数。

## 作业 4：解释 Mask

给定 `[1,1,1,1,1,1,0,0]`：

1. 有效长度为 **6**。有效的特殊 Token 也计入该长度，不能把它直接当作正文 Token 数。
2. MHA 的布尔 key_padding_mask 为 **[False,False,False,False,False,False,True,True]**。True 表示对应 Key 被屏蔽。
3. 它本身没有表达因果限制。它标记有效和补齐位置；禁止读取未来位置还需要因果访问规则。
4. 屏蔽 PAD Key 只防止其他位置读取它。PAD Query 仍可能读取有效 Key，残差、FFN bias 或 LayerNorm 仿射项也可能产生非零表示。因此不能据此断言 PAD 位置的输出为零。

```python
mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
print(mask.sum(dim=1))
print(~mask.bool())
```
