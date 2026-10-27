# C03 Lesson04：手写自回归生成循环

更新日期：2026-10-27（Asia/Shanghai）。讲义与四题参考为助手整理；[用户课堂文件](../hugging_learn/lesson04.py) 的现有内容、三步设置与注释保留。

## 1. 从一次选择到连续生成

```text
原始输入 → 选出 A
原始输入 + A → 选出 B
原始输入 + A + B → 选出 C
```

每一步使用前面的输入与已生成编号作为上下文，这就是本课的自回归过程。一次前向的 `[B,T,V]` 包含当前输入各位置的词表分数，不能把这些位置直接当成尚未生成的未来多个 Token。

沿用 `Qwen/Qwen2.5-0.5B` 基础模型，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU float32、eval、inference_mode，单条无补齐输入，`use_cache=False`。每一步重新计算完整序列；缓存详解留到 Lesson07。

## 2. 取最后位置并保留维度

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=False,
    return_dict=True,
)
next_logits = outputs.logits[:, -1, :]
next_token = next_logits.argmax(dim=-1, keepdim=True)
```

`next_logits` 为 `[1,151936]`，`next_token` 为 `[1,1]`。这里直接对 logits 贪心取最大值，无需先计算 softmax。`keepdim=True` 保留长度为 1 的维度，方便拼接；也可使用 `argmax(dim=-1).unsqueeze(-1)`。

本课 `B=1`，所以 `.item()` 能得到单个编号。批量输入不能直接照搬这一写法。

## 3. 同时追加编号与 Mask

```python
input_ids = torch.cat([input_ids, next_token], dim=-1)
attention_mask = torch.cat([
    attention_mask,
    torch.ones_like(next_token, dtype=attention_mask.dtype),
], dim=-1)
```

`[1,T]` 与 `[1,1]` 沿最后一维连接成 `[1,T+1]`，其余维度匹配。新生成的 Token 是有效输入，Mask 对应位置追加 1。`torch.cat` 的规则见 [PyTorch 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.cat.html)。

使用更新后的张量再前向，模型才能读到刚生成的内容。本课没有使用缓存，因此不能把 input_ids 直接替换为新 Token；否则会丢失完整上下文。Mask 表示有效位置，因果限制由模型内部处理。

## 4. EOS 与长度上限

```python
eos = model.generation_config.eos_token_id
if eos is None:
    eos = tokenizer.eos_token_id
```

配置可能是整数、整数列表或 None，先整理为集合。当前固定模型的 EOS 集合为 `{151643}`，不应把该编号硬编码成所有模型的规则。

循环由 `range(max_new_tokens)` 限制最多新增多少个 Token。每步先追加编号，再检查是否为 EOS：

```python
if token_id in eos_ids:
    stop_reason = "eos_token"
    break
```

句号、换行不一定是 EOS；EOS 也不保证在预算内出现。`max_new_tokens` 不包含原始输入长度；相关参数定义见 [Hugging Face 生成配置](https://huggingface.co/docs/transformers/main_classes/text_generation)。本课手动实现循环，没有调用 `generate`。

`skip_special_tokens=True` 只影响解码显示，不负责停止循环。当前代码保留生成的 EOS 编号，显示文本时可以隐藏它。

## 5. 截取新增部分

```python
prompt_length = input_ids.shape[1]  # 循环开始前记录
# ……生成循环……
new_ids = input_ids[:, prompt_length:]
continuation = tokenizer.decode(new_ids[0], skip_special_tokens=True)
```

切片以原始 Token 数为准，不能使用汉字数量。最后统一解码完整新增编号序列，避免单项解码时的字节碎片影响显示。

当前用户输入没有末尾句号，原始 T=9，上限为 3：新增 `[8997,100012,100364]`，续写 `'。\n为了帮助'`，最终输入 `[1,12]`，最后一次前向的 logits 仍为 `[1,11,151936]`。最后一个 Token 是前向后才追加的。

最初教师八步示例生成 `[8997,100012,100364,87026,100787,104044,105033,106852]`，续写 `'。\n为了帮助您统计最近一个月各部门'`；当前用户文件保留三步设置，不改回八步。作业第三条输入有句号，T=10，其第一步为空格编号 220，二者不能混用。

## 6. 本轮处理的运行问题

导入函数时不加调用括号，调用时才加：

```python
from hugging_learn.causal_common import load_causal_lm
config, tokenizer, model = load_causal_lm()
```

编辑器里修改后要保存，再启动程序。课堂文件已兼容按包运行和编辑器直接运行；作业入口也根据自身文件位置处理直接运行时的导入路径。没有安装另一个同名包、修改系统 Python 路径或改变环境。

## 7. 文件入口

- [用户课堂](../hugging_learn/lesson04.py)：`python -m hugging_learn.lesson04`。
- [完整四题代码](../hugging_learn/homework/lesson04_answers.py)：`python -m hugging_learn.homework.lesson04_answers`。
- [作业与解释](C03-Lesson04-homework.md)、[验收](C03-Lesson04-acceptance.md)、[运行记录](C03-Lesson04-verification.json)。
- [验证器](../verify_lesson04.py)：`python verify_lesson04.py --report /tmp/c03-lesson04-verification.json`。
- [Lesson04→05 交接](C03-Lesson04-handoff.md)。

从仓库根目录、Conda `fde` 环境运行。当前已有模型缓存，无需重新下载。本课不训练，不评估生成质量；Lesson05 再进入 `generate` 与采样。
