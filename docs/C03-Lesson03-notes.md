# C03 Lesson03：从隐藏表示到下一个 Token

更新日期：2026-10-26（Asia/Shanghai）。本讲义为助手整理；[用户课堂文件](../hugging_learn/lesson03.py) 的内容与注释原样保留。

## 1. 本课连接什么

```text
文本 → Token ID [B,T] → Transformer → 最终隐藏表示 [B,T,D]
     → 词表输出头 → logits [B,T,V]
     → 最后位置 [B,V] → softmax → 候选 Token → 选出一个编号
```

`B` 是批大小，`T` 是输入 Token 数，`D` 是隐藏维度，`V` 是模型输出词表维度。一次前向同时给出每个输入位置的下一 Token 分数，本课只选择一个候选；追加并重复推理在 Lesson04 展开。

固定使用 `Qwen/Qwen2.5-0.5B` 基础模型，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU、float32、eval，前向置于 `torch.inference_mode()`。沿用配套 Tokenizer 和本地缓存。[公共加载入口](../hugging_learn/causal_common.py) 不变。

## 2. 隐藏表示与词表头

```python
with torch.inference_mode():
    outputs = model(
        **inputs, use_cache=False,
        output_hidden_states=True, return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    head = model.get_output_embeddings()
    rebuilt_logits = head(hidden)
```

本模型 `D=896`、24 层，`hidden_states` 返回 25 组张量，最后一组是供输出头使用的最终表示。输出头权重 `[151936,896]`，没有 bias；计算相当于 `hidden @ head.weight.T`，把每个位置的 896 维表示转换成 151936 个候选编号的分数。本机手动计算与 `outputs.logits` 最大绝对差异为 0。

这是当前固定模型的结果。其他架构可能有额外处理，应使用其正式输出头并检查结果。[Qwen2 官方接口](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2ForCausalLM) 说明了隐藏状态与词表分数的返回形状。

## 3. 每个位置预测哪里

对因果语言模型，索引 `t` 的输出根据到该位置为止的上下文，给出位置 `t+1` 的 Token 分数。输入已有的中间位置可以与其后已有 Token 对照；最后位置才对应整个输入之后的新 Token。

```python
next_logits = outputs.logits[:, -1, :]  # [B,V]
```

`-1` 指输入最后位置。本课每次只处理一条无 PAD 的文本，最后位置就是最后有效位置；遇到批量补齐时不能不加判断地照搬，Lesson06 再处理。

因果性对照：将用户文本第 3 个及以后的 Token 替换、保持张量长度与 Mask 不变，前两个位置的 logits 最大差异为 0。另将前两个 Token 独立前向，与长输入相应位置比较，float32/SDPA 最大差异约 `2.18e-4`，按 `rtol=1e-4, atol=5e-4` 检查通过。这两类对照的运算形状不同，不要求逐位相等；[PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) 解释了浮点计算与切片计算可能出现差异。

## 4. logits、概率与选择

```python
probs = torch.softmax(next_logits, dim=-1, dtype=torch.float64)
top_probs, top_ids = torch.topk(probs, k=5, dim=-1)
next_ids = next_logits.argmax(dim=-1)
```

logits 是未归一化分数，可以为负。softmax 沿最后的词表维度计算 `p_i = exp(z_i) / sum_j exp(z_j)`，完整词表概率和在浮点容差内为 1。此处仅观察用 softmax 转为 float64，模型和 logits 仍为 float32。接口参数见 [PyTorch softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)。

`topk` 取出原分布中最大的五项，不会自动将它们重新归一化。`argmax(dim=-1)` 取得每条输入的最高分编号，形状 `[B]`；贪心选择直接对 logits 取最大值即可。只有 `B=1` 时才能用 `.item()` 得到单个整数。省略 `dim` 会在整个张量上返回全局索引，见 [PyTorch argmax](https://docs.pytorch.org/docs/2.14/generated/torch.argmax.html)。

## 5. 文本不同，长度和候选也不同

| 来源 | 文本 | T | 隐藏表示 | logits |
|---|---|---:|---|---|
| 用户课堂文件 | 请统计最近一个月各部门的订单总金额 | 9 | `[1,9,896]` | `[1,9,151936]` |
| 教师示例／作业第二条 | 我想查询订单金额。 | 5 | `[1,5,896]` | `[1,5,151936]` |
| 作业第一条 | 你好。 | 2 | `[1,2,896]` | `[1,2,151936]` |
| 作业第三条（有句号） | 请统计最近一个月各部门的订单总金额。 | 10 | `[1,10,896]` | `[1,10,151936]` |

不要为了匹配教师输出改掉用户输入。用户无句号版本的最高分候选是 ID `8997`，单项解码为 `'。\n'`；带句号作业版本的最高分候选是 ID `220`，解码为空格。二者上下文不同。

Token 是分词单元，不保证对应一个汉字。Qwen 的字节级分词可能把一个字符分到不同 Token，单独解码某个候选时出现 `�`；这不等于模型输出了未知 Token。作业中 ID `4891` 与 `6567` 单项解码看起来相同，但 raw token 不同。查看编号、`convert_ids_to_tokens` 与整段 `decode`，不要用单项显示文本判断两个 Token 相同。

## 6. 运行与后续

从仓库根目录、`fde` 环境运行：

```sh
python -m hugging_learn.lesson03
python -m hugging_learn.homework.lesson03_answers
python verify_lesson03.py --report /tmp/c03-lesson03-verification.json
```

配套资料：[四题答案](C03-Lesson03-homework.md)、[本地验收](C03-Lesson03-acceptance.md)、[真实运行记录](C03-Lesson03-verification.json)、[Lesson03→04 交接](C03-Lesson03-handoff.md)。下一课手写自回归循环；本课尚未训练、运行生成循环或评估生成效果。
