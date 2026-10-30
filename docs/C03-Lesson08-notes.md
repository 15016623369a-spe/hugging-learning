# C03 Lesson08：可复现的文本生成程序

更新日期：2026-10-30（Asia/Shanghai）。[课堂代码](../hugging_learn/lesson08.py)、[用户保存的课堂结果](../hugging_learn/lesson08_results.json)、[本轮检查记录](C03-Lesson08-verification.json)。

## 从前七课接续

本课把文本编码、预训练模型前向、隐藏表示、词表分数、逐Token生成、生成参数、缓存与运行环境记录串起来。共享加载器沿用Qwen/Qwen2.5-0.5B基础模型及固定revision，CPU float32、4计算线程；模型只加载一次，随后对每条文本分别执行实验。

`load_causal_lm()`返回config、tokenizer、model三个对象。`_, tokenizer, model = load_causal_lm()`中的下划线仍然是普通变量，约定表示这段代码暂时不使用config；它不是减号，也不是特殊的丢弃语法。

## 数据链路与实际形状

对一条长度T的输入，链路为：文本→`input_ids`/Mask `[1,T]`→最后隐藏表示`[1,T,896]`→logits `[1,T,151936]`→最后位置`[1,151936]`→贪心编号`[1]`。`argmax(dim=-1)`沿词表维选择最高分编号；它不是概率。

| 本轮文本 | 输入 | 完整输出 | 新增部分 | 第一步贪心编号 |
|---|---|---|---|---:|
| 订单金额 | `[1, 2]` | `[1, 10]` | `[1, 8]` | 100381 |
| 查询订单金额 | `[1, 3]` | `[1, 11]` | `[1, 8]` | 198 |
| 请统计最近一个月各部门的订单总金额。 | `[1, 10]` | `[1, 18]` | `[1, 8]` | 220 |

三句话分别运行，所以batch size始终是1；本轮没有把三句话合为一个批次。句号本身也会影响编码，第三句带句号时T=10，不应拿前课无句号的T=9替换本轮记录。

## run_one的职责

`run_one(model, tokenizer, text, *, ...)`接受一条字符串；星号后的参数通过名称传入。它拒绝空白输入，以及零、负数、非整数的新增上限；后续检查输入宽度与新增上限之和不超过模型上下文预算。

函数先确定参数所在设备，将编码得到的所有输入张量移到同一设备。`model.eval()`控制评估行为，`torch.inference_mode()`关闭梯度记录；二者用途不同。

`GenerationConfig`显式固定max_new_tokens=8、num_beams=1、缓存开关和停止编号。本课只把配置对象传入generate，不再同时传入重复的生成参数。贪心组不设置采样温度；采样组使用temperature=0.8、top_k=20、top_p=0.9，并在每次generate前重新设置seed=42。

生成结果包含输入前缀。先核对前缀逐编号相同，再使用`output_ids[:, width:]`取新增部分。`decode(..., skip_special_tokens=True)`用于阅读；编号列表是核对依据。停止原因根据新增编号里是否出现EOS判断，只适用于本课单条序列、无其他停止条件的设置。本轮所有组均新增8个Token后停止，没有自然EOS。

## main中的循环和四组实验

模型加载位于循环外，`for index, text in enumerate(texts, start=1)`依次取出字符串。每条先做一次前向探针，然后执行greedy、no_cache、sample_a、sample_b四组，最后将text、probe、checks、runs追加到samples。

每条的三项课堂核对分别是：第一次贪心选择与探针一致；缓存开关的贪心编号一致；两次同种子采样编号一致。三条共12次生成、9项课堂核对；本轮验证器额外检查入口、输入边界、形状、切片、解码、配置与记录，共68项。

当前模型是基础模型，短续写不是指令回答质量验收。真实采样若包含换行、空格或替换字符，结果照实保存，不为美化显示而改动编号。相同随机种子的复现结论仅限本次固定环境。

## 三条输入曾经报错的原因

最初把text改成列表并设置padding=True，编码形状变成`[3,10]`。logits为`[3,10,151936]`，最后一列为`[3,151936]`，argmax得到`[3]`，因此`.item()`会报“3 elements cannot be converted to Scalar”。padding只补齐长度，不把三条输入变成一条。

若真正批量处理，多个编号应使用`.tolist()`，但还必须同步改造run_one的字符串校验、每条输出解码和核对。仅改item会接着在`text.strip()`遇到list没有strip的错误；`new_ids[0]`也只读取第一条。

当前默认右补齐，短句最后一列是PAD。`attention_mask`不会把`[:, -1, :]`自动移动到正文末尾。decoder-only批量生成应左补齐，并在自行前向时核对位置编号；续写从共同输入宽度W截取。当前作业1选择逐条调用run_one，所以无需跨句补齐，`.item()`仍然适用。

## 保存、运行与来源

`Tensor.tolist()`把张量变成可写入JSON的列表；`ensure_ascii=False`保留中文显示。记录保存到脚本旁的lesson08_results.json，包含实际版本、设备、dtype、线程、注意力实现、模型版本和三条samples。

```sh
python -m hugging_learn.lesson08
python hugging_learn/lesson08.py
python verify_lesson08.py --report /tmp/c03-lesson08-verification.json
```

课堂自身运行会刷新它旁边的结果文件；验证器在临时目录复制源码，实际运行直接文件与模块入口，不覆盖原结果。直接文件方式也从项目外目录执行过，导入兼容通过。

参考：[PyTorch item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html)、[HF生成配置](https://huggingface.co/docs/transformers/main_classes/text_generation)、[HF补齐方向](https://huggingface.co/docs/transformers/llm_tutorial#padding-side)、[PyTorch随机复现](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。

作业与实际检查范围见[四题记录](C03-Lesson08-homework.md)和[验收来源](C03-Lesson08-acceptance.md)。
