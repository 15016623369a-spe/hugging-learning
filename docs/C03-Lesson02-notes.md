# C03 Lesson02：加载预训练模型与基础推理

整理日期：2026-10-26（Asia/Shanghai）。依据本聊天 2026-10-25 的授课和本轮实际运行整理，代码见 [用户课堂文件](../hugging_learn/lesson02.py)。

## 1. 本课链路

```text
配置文件 → 确认模型结构
Tokenizer 文件 → 文本 → 整数输入 [B,T]
预训练权重 + 结构 → CausalLM → 前向 → logits [B,T,V]
```

本课完成模型加载与一次前向，建立可检查的基础链路。词表分数与下一 Token 在 Lesson03 展开，生成循环在 Lesson04 展开。

## 2. 固定模型与版本

使用基础模型 `Qwen/Qwen2.5-0.5B`，revision 为 `060db6499f32faf8b98477b0a26969ef7d8b9987`。不要与 Instruct 版本混用，也不要沿用 BERT 分词器。

- `AutoConfig.from_pretrained` 读取架构配置；实际为 Qwen2Config，24 层、hidden_size 896。
- `AutoTokenizer.from_pretrained` 读取同版本的配套分词器。
- `AutoModelForCausalLM.from_pretrained` 建立带词表输出头的模型并读取预训练权重，实际为 Qwen2ForCausalLM。
- `AutoModelForCausalLM.from_config` 仅按配置建立并初始化模型，不能替代加载已有权重。
- `revision` 固定文件版本，`local_files_only=True` 使日常入口只查缓存。

模型标称 0.5B，本次实际参数量为 494032768。float32 参数张量存储约 1.84 GiB；运行时还有激活、输出和框架等额外占用。磁盘权重文件大小与内存中的参数大小也不是同一个数。

## 3. 设备与类型

本轮明确使用 CPU 和 `torch.float32`，不依赖 GPU。模型参数与输入张量需要放在兼容的设备上。

```python
model.to(torch.device("cpu"))
inputs = tokenizer(text, return_tensors="pt")
inputs = {name: tensor.to("cpu") for name, tensor in inputs.items()}
```

模型参数是浮点数；`input_ids` 是整数编号，本次为 `torch.int64`。把编号转为 float32 不能帮助模型“匹配 dtype”，反而会导致 Embedding 查表失败。不要笼统地把输入字典里每个张量都转为模型参数的 dtype。

## 4. 评估模式和梯度

```python
model.eval()
with torch.inference_mode():
    outputs = model(**inputs, use_cache=False)
```

`eval()` 让受训练模式影响的模块按评估规则工作，不是关闭梯度的开关。`inference_mode()` 控制该次计算的梯度记录及相关开销，不切换模型的训练/评估状态。

`use_cache=False` 在本课关闭用于后续生成的 KV cache 返回，不代表关闭 Hugging Face 文件缓存。当前任务只观察完整输入的一次前向，KV cache 在 Lesson07 详讲。

## 5. 输入与输出

`model(**inputs)` 将字典展开成命名参数。不要写成 `model(inputs)`，也不要为了只传一个字段而遗漏需要的输入规则。

课堂输入“我想查询订单金额。”得到：

```text
input_ids = [[104100, 51154, 50410, 80094, 1773]]
input_ids shape = [1,5]，int64，cpu
attention_mask shape = [1,5]，全 1
outputs 类型 = CausalLMOutputWithPast
logits shape = [1,5,151936]，float32，cpu
logits.requires_grad = False
```

`B=1` 表示本次一个样本；`T=5` 表示当前输入的五个 Token；`V=151936` 表示模型词表输出维度。分数并未转换成概率，也没有自动成为文字回答。不要把 hidden_size 896 当作 logits 的最后一维。

当前 `len(tokenizer)=151665`，不要求恰好等于输出头维度；使用配套分词器并检查编号范围。详细候选 Token、最后位置和预测目标留给 Lesson03。

## 6. 复现与下一步

```sh
conda activate fde
python -m hugging_learn.lesson02
python -m hugging_learn.homework.lesson02_answers
python verify_lesson02.py --report /tmp/c03-lesson02-verification.json
```

新机器首次使用时，安装仓库已有依赖后联网运行 `python -m hugging_learn.download_model`。下载器仅获取固定版本所需文件，权重保存在 Hugging Face 缓存中；日常入口不联网。

四题完整答案见 [Lesson02 作业](C03-Lesson02-homework.md)，验收见 [来源说明](C03-Lesson02-acceptance.md) 和 [运行 JSON](C03-Lesson02-verification.json)。下一课为 Lesson03：从隐藏表示到下一个 Token。

官方资料：[固定模型配置](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json)、[Auto classes](https://huggingface.co/docs/transformers/model_doc/auto)、[Model outputs](https://huggingface.co/docs/transformers/main_classes/output)、[PyTorch eval](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval)、[inference_mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html)。
