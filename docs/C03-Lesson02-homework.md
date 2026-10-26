# C03 Lesson02：四道作业及参考答案

更新日期：2026-10-26（Asia/Shanghai）。本文件和参考代码由助手补充；结果来自固定版本模型在本机的真实运行。用户索要答案按教学约定接受为私下完成确认。

完整代码：[lesson02_answers.py](../hugging_learn/homework/lesson02_answers.py)。在仓库根目录、`fde` 环境运行：

```sh
python -m hugging_learn.homework.lesson02_answers
```

## 作业 1：检查配置、模型与参数

题目：打印配置与模型类型、第一个参数的设备和 dtype、训练模式标记，解释 Config、Tokenizer、Model 各自做什么。

参考代码：

```python
from hugging_learn.causal_common import load_causal_lm

config, tokenizer, model = load_causal_lm()
parameter = next(model.parameters())
print(type(config).__name__, type(model).__name__)
print(config.model_type, config.hidden_size)
print(config.num_hidden_layers, config.vocab_size)
print(parameter.device, parameter.dtype, model.training)
```

本次实际结果：

| 项目 | 结果 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B`（基础模型） |
| revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| 配置类 | `Qwen2Config` |
| 模型类 | `Qwen2ForCausalLM` |
| Tokenizer 类 | `Qwen2Tokenizer` |
| 架构标识 | `qwen2` |
| hidden_size | 896 |
| 层数 | 24 |
| 输出词表维度 | 151936 |
| 第一个参数的设备 | `cpu` |
| 第一个参数的 dtype | `torch.float32` |
| model.training | `False` |
| 参数量 | 494032768 |
| 参数张量存储 | 约 1.84 GiB，不等于进程总内存 |

解释：

- **Config** 描述模型结构和配置，例如层数、表示宽度、输出词表维度；只读取配置不会得到模型权重。
- **Tokenizer** 把文本转换成配套编号和输入字段，也负责解码；它本身不运行模型前向。
- **Model** 包含结构和实际参数。这里通过 `from_pretrained` 加载预训练权重；`ForCausalLM` 还包含把隐藏表示映射到词表分数的输出头。

`len(tokenizer)` 本次是 151665，与模型输出维度 151936 不相等。不要把两者必须相等写成通用断言；输入 Token ID 必须能被模型 Embedding 表索引。输出维度的进一步解释放在 Lesson03。

## 作业 2：同一模型依次处理三条文本

题目：模型只加载一次，依次处理下列三条文本，记录 Token 数、输入形状、logits 形状以及是否记录梯度。

接续作业 1 已加载的 `tokenizer` 和 `model`：

```python
import torch

texts = [
    "你好。",
    "我想查询订单金额。",
    "请统计最近一个月各部门的订单总金额。",
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {name: tensor.to(parameter.device)
              for name, tensor in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    print(text, inputs["input_ids"].shape[1])
    print(inputs["input_ids"].shape, outputs.logits.shape)
    print(outputs.logits.requires_grad)
```

真实结果如下，每次只输入一条，因此 `B=1`，不需要批量补齐：

| 文本 | T | input_ids 形状 | logits 形状 | requires_grad |
|---|---:|---|---|---|
| 你好。 | 2 | `[1,2]` | `[1,2,151936]` | `False` |
| 我想查询订单金额。 | 5 | `[1,5]` | `[1,5,151936]` | `False` |
| 请统计最近一个月各部门的订单总金额。 | 10 | `[1,10]` | `[1,10,151936]` | `False` |

对应的真实编号：

```text
你好。
[108386, 1773]

我想查询订单金额。
[104100, 51154, 50410, 80094, 1773]

请统计最近一个月各部门的订单总金额。
[14880, 100787, 104044, 105033, 106852, 9370, 50410, 59743, 80094, 1773]
```

所有 `input_ids` 都是 `torch.int64`、CPU；三条 `attention_mask` 分别是 2、5、10 个 1；输出为 CPU、`torch.float32`，全部为有限值。

这里的 `T` 由当前分词器决定，不能按字数估计，也不能照搬 Lesson01 的 BERT 分词结果。Qwen 分词器打印出的内部 Token 字符串可能是字节级表示，不必等同于可直接阅读的汉字片段。

`logits` 是 `[B,T,V]` 的原始词表分数，不是已经生成的回答，也不是已经归一化的概率。一次前向没有执行逐 Token 生成循环。

## 作业 3：eval 与 inference_mode

题目：用 `Linear(2,2)` 对比 `eval()` 后的普通前向和推理模式前向。

```python
import torch

layer = torch.nn.Linear(2, 2)
layer.eval()
x = torch.ones(1, 2)

with torch.enable_grad():
    ordinary = layer(x)
with torch.inference_mode():
    inference = layer(x)

print(layer.training)                 # False
print(x.requires_grad)                # False
print(layer.weight.requires_grad)     # True
print(ordinary.requires_grad)         # True
print(inference.requires_grad)        # False
print(ordinary.grad_fn)               # 非 None，本次为 AddmmBackward0
print(inference.grad_fn)              # None
```

结论：`eval()` 切换模块的训练/评估行为；它不关闭 autograd。输入 `x` 不需要梯度，但 Linear 的权重默认需要梯度，普通前向仍建立计算图。`inference_mode()` 关闭该上下文中的梯度记录，但不会替你调用 `eval()`。正式推理通常二者配合使用。本次 Linear 两种方式计算的数值相同；具体随机数值不是验收要求。

## 作业 4：诊断四种错误

| 错误写法或想法 | 原因 | 修正 |
|---|---|---|
| `inputs["input_ids"].float()` | 编号是 Embedding 查询索引，不能随模型参数转成浮点数 | 保持 `torch.int64`；移设备用 `.to(device)` |
| `model(inputs)` | 把整个字典作为一个位置参数传入 | 使用 `model(**inputs, use_cache=False)` |
| 想加载预训练权重，却调用 `from_config(config)` | 按配置建立并初始化模型，不读取预训练权重 | 使用 `from_pretrained` 并固定版本 |
| 只调用 `eval()`，期待自动不记录梯度 | 评估模式与梯度记录是两个机制 | `eval()` 配合 `torch.inference_mode()` |

正确的基本调用方式：

```python
model.eval()
inputs = tokenizer("我想查询订单金额。", return_tensors="pt")
inputs = {name: tensor.to(next(model.parameters()).device)
          for name, tensor in inputs.items()}
with torch.inference_mode():
    outputs = model(**inputs, use_cache=False)
```

`**inputs` 相当于把字典中的 `input_ids=...`、`attention_mask=...` 等字段作为命名参数传入。本次验证实际触发了浮点编号和未展开字典两种错误，详情见 [真实运行记录](C03-Lesson02-verification.json)。

## 资料与验收

本次在既有 `fde` 环境、CPU float32、本地固定缓存上通过 26 项检查；验证代码见 [verify_lesson02.py](../verify_lesson02.py)。没有训练、文本生成或生成质量评估。

上述接口语义核对自 [Hugging Face Auto classes](https://huggingface.co/docs/transformers/model_doc/auto)、[模型输出](https://huggingface.co/docs/transformers/main_classes/output)、[PyTorch eval](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval) 与 [inference_mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html)。
