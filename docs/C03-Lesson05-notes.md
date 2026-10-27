# C03 Lesson05：generate 与生成参数

更新日期：2026-10-27（Asia/Shanghai）。本讲义由助手整理；用户 [lesson05.py](../hugging_learn/lesson05.py) 的原代码与学习注释逐字节保留。

## 1. 从上一课的手写循环接续

上一课每轮先用完整上下文前向，取 `logits[:, -1, :]`，用 `argmax` 选下一个编号，然后同时追加编号和 Mask，最后判断 EOS 或新增上限。下一轮的输入已经包含刚刚生成的编号。

`model.generate()` 将这些步骤封装起来，并增加采样、过滤、缓存等配置。它不是一次前向直接得到未来八个编号，也不会因为调用生成接口而训练模型。

本课继续使用 `Qwen/Qwen2.5-0.5B` 基础模型、revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，CPU float32、eval、inference_mode，单条输入无 PAD。显式设置 `use_cache=False`，与上一课的计算方式对齐；缓存原理留到 Lesson07。

## 2. 输出为什么还包含输入

本课是 decoder-only 因果语言模型，且 `return_dict_in_generate=False`，生成返回的张量是“输入编号 + 新增编号”。设输入长度为 T、实际新增 N，则完整输出为 `[1,T+N]`：

```python
prompt_length = inputs["input_ids"].shape[1]
output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
new_ids = output_ids[:, prompt_length:]
```

`:` 保留所有样本行；`prompt_length:` 从原始输入之后切到结尾。索引的单位是 token 位置，不能用 `len(text)` 的字符数代替。

课堂输入“请统计最近一个月各部门的订单总金额”没有末尾句号，编码 T=9；实际新增八个编号，输出 `[1,17]`，新增 `[1,8]`。输入加句号会改变编码和续写，不为匹配结果修改用户文本。

## 3. 长度上限与模型默认配置

`max_new_tokens=N` 限制新生成数量；`max_length=L` 限制输入与生成的总长度。在本课单句条件下，前者允许最多 N 个新增，后者只剩 `L-T` 个位置。EOS 可以提前停止，因此上限并不承诺一定生成这么多。

当前模型的 `generation_config.max_new_tokens` 实际为 2048。课堂明确设为 8；不能把任何教程中的默认值直接套到具体模型。当前配置中部分字段为 `None`，表示没有在该字段显式给值，不能把它解释成算法完全没有默认行为。

作业另实测两条输入都使用 `max_length=12`。必须同时传 `max_new_tokens=None`，清除模型自带的 2048，否则新增上限可能覆盖总长度设置：

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=None,
    max_length=12,
    do_sample=False,
    num_beams=1,
    use_cache=False,
    eos_token_id=model.generation_config.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    return_dict_in_generate=False,
)
```

当前 Transformers 5.20.2 的 `max_new_tokens=0` 实测抛出 ValueError。上一课手写的 `range(0)` 可以不进入循环，两种接口的边界约束不同。

## 4. 贪心与采样的区别

`do_sample=False, num_beams=1` 使用贪心：每一步选择当前分数最高的编号。本课实际与上一课手写循环逐编号一致；这项结论以相同输入和生成配置为条件。额外的重复惩罚、禁用编号等处理规则可能改变候选，不能认为任意模型的 generate 都等于未经处理的 argmax。

贪心是逐步的局部选择，不保证整段序列的联合概率最高，也不保证文本质量最好。

`do_sample=True, num_beams=1` 使用采样：根据处理后的概率分布抽取编号。采样不是所有候选等概率抽签；概率高的候选更容易被抽到。后续上下文会受本步选择影响，因此两次输出可能很快分叉。

## 5. temperature：缩放分数后再计算概率

设五个教学分数为 `[3,2,1,0,-1]`，记为 z，温度为 τ>0。计算 `softmax(z/τ)`，不是用原概率除以温度。

以 τ=0.5 为例，缩放分数变成 `[6,4,2,0,-2]`。相邻差距从 1 扩大到 2，经指数运算后高分项优势扩大。τ=2 时分数变成 `[1.5,1,0.5,0,-0.5]`，差距缩小，分布更平缓。

为稳定计算，可先减去最大缩放分数；softmax 的结果不变。τ=1 时可计算 `exp([0,-1,-2,-3,-4])`，再除以这些数的和。

| 温度 | A | B | C | D | E |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 86.4704% | 11.7025% | 1.5838% | 0.2143% | 0.0290% |
| 1.0 | 63.6409% | 23.4122% | 8.6129% | 3.1685% | 1.1656% |
| 2.0 | 42.8656% | 25.9993% | 15.7694% | 9.5646% | 5.8012% |

正温度保留原分数的大小顺序，所以这三组的最高项都是 A，但采样概率不同。温度更高不代表更聪明，温度更低也不保证内容正确。使用贪心直接设 `do_sample=False`，不要把温度设为 0。

## 6. top-k 与 top-p：过滤之后重新归一化

`top_k=2` 在这个没有同分的教学例子中保留 A、B，其余概率设为零。A、B 原概率之和约为 0.8705303，重新归一化：

```text
A：0.6364086 / 0.8705303 ≈ 0.7310586
B：0.2341217 / 0.8705303 ≈ 0.2689414
```

现在抽样只在 A、B 中进行，二者加起来必须为 1。`top_k=0` 关闭该过滤。存在同分边界时实际算子可能保留更多同分项，因此不要把“恰好 k 个”当成所有情况的严格承诺。

`top_p=0.9` 则按概率降序累计：A 为 0.6364086；A+B 为 0.8705303，尚未达到 0.9；加 C 后为 0.9566588，所以保留 A、B、C。重新归一化后约为 `[0.6652410,0.2447285,0.0900306,0,0]`。

top-k 用候选排名限制范围；top-p 按累计概率决定范围，候选数量会随分布变化。top-p 不是筛选“单个概率大于 p”的编号。`top_p=1.0` 关闭这项过滤。

两项可以一起使用。在本课配置下按温度、top-k、top-p 的顺序处理；top-p 应根据 top-k 之后剩余候选归一化的概率判断，而不是继续使用被过滤前的概率总和。例子中 k=2 后 A 占 0.7311，尚不足 0.9，因此再加 B，两项都保留。

## 7. 随机种子是随机状态的起点

```python
for seed in [42, 42, 43]:
    set_seed(seed)
    # 从这里开始本轮生成
```

两次 42 都重置起点，所以在本机相同条件下得到相同编号。若只在循环之前调用一次 `set_seed(42)`，第一轮会消耗随机状态，第二轮继续往后使用，不会自动回到第一轮。

每次重新运行整个“种子只设一次”的三轮实验，在相同环境条件下可以重复该三轮序列；这与“三轮内部彼此相同”是不同问题。换种子也可能偶然抽到相同结果。跨版本、设备与平台的逐编号复现不能只靠种子承诺。

`set_seed()` 不会代替 `model.eval()` 或 `torch.inference_mode()`：前者控制随机数起点，eval 切换模型模块的推理行为，inference_mode 关闭本次计算中的梯度记录等开销。

## 8. 读懂课堂的 Python 语法

```python
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
```

`items()` 依次给出键和值；`to(device)` 将张量放到模型所在设备；字典推导式保留字段名，组成新的输入字典。

```python
model.generate(**inputs, **common, do_sample=True)
```

`**` 在调用位置将字典展开成关键字参数。例如 `inputs` 的两个键会成为 `input_ids=...`、`attention_mask=...`。它展开的是字典，不是拆解模型本身。多个字典含有重复参数名时会报重复传参错误，应先合并或修改字典。

`repr(text)` 让换行显示为 `\n`，便于比较结果；`skip_special_tokens=True` 控制显示时是否隐藏特殊 token，不控制生成循环停止。停止需要 EOS 或长度等规则。

## 9. 代码、题解与下一课

用户课堂保留，完整四题参考单独在 [lesson05_answers.py](../hugging_learn/homework/lesson05_answers.py)。详解见 [四题答案](C03-Lesson05-homework.md)，38 项检查、环境、编号、源码哈希见 [运行记录](C03-Lesson05-verification.json)，范围见 [验收说明](C03-Lesson05-acceptance.md)。

下一课 Lesson06 学习批量生成、补齐方向、输出截取、基础模型与指令模型、聊天模板。本课没有运行这些后续内容，也没有训练或评估生成质量。

## 官方资料

- [Hugging Face Generation 参数](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [PyTorch 2.14 随机性与复现](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

参数语义以官方资料为依据；本文具体默认配置、数值和生成轨迹来自本机固定版本的实际执行，不从单次结果推断普遍生成质量。
