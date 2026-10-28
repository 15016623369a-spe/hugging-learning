# C03 Lesson01：真实 Tokenizer 与模型输入编码

本讲义由助手根据本聊天课堂整理。教师代码见 `hugging_learn/lesson01.py`；四题参考代码另存于 `hugging_learn/homework/lesson01_answers.py`。

## 1. 从 C02 接续

C02 用空格 split 和小词表解释 Token/ID/Embedding。真实 Tokenizer 按预训练时保存的规则处理原始文本：

```text
文本 → 预处理和分词 → Token → Token ID
     → 特殊 Token、补齐与截断 → 模型输入字段
```

Tokenizer 输出编号与输入字段；模型内部再用配套 Embedding 表把 `[B,T]` 的整数编号变为 `[B,T,D]`。Token ID 只有词表索引意义，数字大小不代表语义强弱。不同 Tokenizer 的相同编号也可能对应不同内容。

本课用 `google-bert/bert-base-chinese`，固定 revision 为 `8f23c25b06e129b6c986331a13d8d025a92cf0ea`。词表大小 21128。生成课将加载因果模型及它配套的 Tokenizer。

## 2. 子词与真实输出

WordPiece 用子词表示内容，`##` 标记接在词内前段后面的片段，不是原文中的井号。

| 输入 | 本次真实 Token |
|---|---|
| 查询订单 | 查、询、订、单 |
| playing | play、##ing |
| agent | agent |
| transformer | t、##ran、##s、##form、##er |
| Transformer | [UNK] |

字符数、单词数、Token 数应分别计算。这里的 Tokenizer 保留大小写；大小写和未知内容的处理依赖实际配置，不能由模型名称猜测。

## 3. 特殊 Token

| Token | ID | 含义 |
|---|---:|---|
| [PAD] | 0 | 补齐位置 |
| [UNK] | 100 | 无法被当前规则和词表表示的内容，仍是有效位置 |
| [CLS] | 101 | BERT 开头的特殊标记，常用于分类任务 |
| [SEP] | 102 | BERT 片段边界或末尾标记 |
| [MASK] | 103 | BERT 掩码预测的占位符 |

`查询订单` 加入特殊 Token 后是 `[CLS] 查 询 订 单 [SEP]`，总长度为 6。特殊 Token 占用长度。`[MASK]` 是词表条目，`attention_mask` 是输入位置规则；二者不同。

## 4. 常用接口与 Python 用法

| 调用 | 用途 |
|---|---|
| AutoTokenizer.from_pretrained(...) | 根据配置加载已保存的分词器 |
| tokenizer.tokenize(text) | 查看 Token 字符串 |
| tokenizer.encode(text, add_special_tokens=False) | 获取不自动添加边界标记的 ID |
| tokenizer(text, return_tensors="pt") | 获取 PyTorch 格式的多个输入字段 |
| tokenizer.convert_ids_to_tokens(ids) | 按编号查看 Token |
| tokenizer.decode(ids) | 将一条 ID 序列解码 |
| tokenizer.batch_decode(ids) | 对 batch 中每条序列解码 |

`from_pretrained` 在本课加载的是分词文件，不运行训练。`revision` 固定文件版本；`local_files_only=True` 只查本地缓存。`return_tensors="pt"` 要求返回 PyTorch 张量。

`batch["input_ids"]` 从类似字典的对象取出编号字段；`.shape` 查看维度；`.dtype` 查看类型；`.sum(dim=1)` 按行累加；`.tolist()` 转为普通 Python 列表，便于打印。

## 5. 输入字段与 shape

| 字段 | 含义 | 张量形状 |
|---|---|---|
| input_ids | Token ID，整数，本次为 torch.int64 | [B,T] |
| attention_mask | 本课中 1=有效，0=补齐 | [B,T] |
| token_type_ids | BERT 输入片段编号，单段输入全为 0 | [B,T] |

单条文本转张量也保留 batch 维。`查询订单` 得到 `[1,6]`。有些模型不需要 `token_type_ids`，也不使用 BERT 这套特殊标记。

## 6. Padding 与 truncation

课堂两条输入是 `查询订单` 和 `我想查询最近一个月的订单金额`。加入特殊标记后长度为 6、16。

- `padding=True`：补到当前 batch 最长长度，本例为 `[2,16]`。
- `padding="max_length", max_length=10`：把短序列补到 10；单独使用不会截断超长输入。
- `truncation=True, max_length=10`：启用长度上限，截断超长输入。
- 两者同时启用：本例为 `[2,10]`，有效长度为 `[6,10]`。

本课按右侧补齐、右侧截断执行。长度上限包括自动加入的特殊 Token。第二条截断后的内容为 `我 想 查 询 最 近 一 个`；它满足张量长度要求，但问题信息已丢失。

## 7. Mask 衔接

| 接口或变量 | 允许/有效 | 屏蔽 |
|---|---|---|
| 本课 HF attention_mask | 1 | 0 |
| C02 手工 allowed | True | False |
| PyTorch SDPA 的布尔 attn_mask | True | False |
| PyTorch MHA 的布尔 key_padding_mask | False | True |

HF 的 `[B,T]` 有效位置标记本身不表达因果限制。模型是否允许读取未来位置，还由对应架构的因果规则控制。

```python
valid_positions = batch["attention_mask"].bool()       # [B,T]
key_allowed = valid_positions[:, None, None, :]       # [B,1,1,T]
mha_key_padding_mask = ~valid_positions               # [B,T]
```

`None` 插入长度为 1 的维度，供 Key 规则广播到注意力分数的 `[B,H,T,T]`。上例还没有加入 Causal Mask。先转布尔再使用 `~`，使含义明确。

有效长度是每行 attention_mask 的和，包含 `[CLS]`、`[SEP]` 和 `[UNK]`。屏蔽 PAD Key 限制它被读取，不保证 PAD Query 的输出清零；后者还涉及注意力、残差、FFN、LayerNorm 等计算。

## 8. Decode 与常见错误

解码不保证逐字恢复原始字符串：规范化、空格、未知 Token 都可能改变结果。`skip_special_tokens=True` 会移除本课的特殊 Token，也包括 `[UNK]`；它不会找回丢失的内容。

易错点：按字数预算 Token；混用 Tokenizer 和模型；误以为 max_length 自动补齐并截断；不计算特殊 Token；把 `[UNK]` 当 PAD；混用 HF 和 PyTorch Mask；把分词成功当作模型理解或生成成功。

## 官方资料

- [Tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers)
- [Padding and truncation](https://huggingface.co/docs/transformers/pad_truncation)
- [Attention mask](https://huggingface.co/docs/transformers/glossary#attention-mask)
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
