# C03 Lesson07 四道作业与详细参考答案

日期：2026-10-28（Asia/Shanghai）。本页结果来自本轮 [实际验收 JSON](C03-Lesson07-verification.json)。完整可运行参考代码：[lesson07_answers.py](../hugging_learn/homework/lesson07_answers.py)。

用户索要答案按既有约定登记私下完成确认；本页代码与整理为助手参考，实际运行证据单独记录。

## 第一题：观察“订单金额”的三次缓存增长

题目：换成 `text="订单金额"`，执行三次前向，记录输入、Mask、logits、K/V、缓存长度及 next_id，并解释为什么只输入新 token 也能使用历史。

当前固定 Qwen Tokenizer 的真实编码为 `[50410,80094]`，分别解码为“订单”“金额”。四个汉字在这里是两个 token。

| 前向 | 本次输入形状 | 完整 Mask | logits | 第一层 K、V（各自） | 缓存长度 | 选出的下一个 token |
|---|---|---|---|---|---:|---|
| 1 | `[1, 2]` | `[1, 2]` | `[1, 2, 151936]` | `[1, 2, 2, 64]` | 2 | `100381`（超过） |
| 2 | `[1, 1]` | `[1, 3]` | `[1, 1, 151936]` | `[1, 2, 3, 64]` | 3 | `2130`（____） |
| 3 | `[1, 1]` | `[1, 4]` | `[1, 1, 151936]` | `[1, 2, 4, 64]` | 4 | `23305`（元） |

新增编号：`[100381,2130,23305]`，新增文本：`超过____元`。四个下划线在这里组成一个 token，不能按可见字符数推断 token 数。

逐步解释：

1. 第一次完整读入两个提示词 token，缓存长度变为2，利用最后位置选出“超过”。
2. 执行 `current_ids=next_id` 后，第二次输入只有“超过”一个 token；通过 `past_key_values=cache` 复用前两个位置的 K/V，缓存长度变为3。
3. 第三次输入编号2130，缓存长度变为4，再选出“元”。选出的“元”尚未再次送入模型，所以此时可见序列有5个 token，而缓存只有4个位置。

输入 `[1,1]` 表示本次新计算一个位置；完整 Mask 覆盖历史缓存加新位置。K/V 中的维度顺序为 `[batch, KV头数, 缓存长度, 每头维度]`，因此第一次 K/V 形状里的两个2分别表示头数和长度。

关键代码：

```python
outputs = model(
    input_ids=current_ids,
    attention_mask=mask,
    past_key_values=cache,
    use_cache=True,
)
cache = outputs.past_key_values
next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
current_ids = next_id
mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)
```

每次独立实验从 `cache=None` 开始；本题是单条未补齐输入的三步观察。已有完整缓存后，下一次只送尚未处理的 token。

## 第二题：核对缓存开关的输出

题目：输入“请用一句话解释 attention_mask 的作用。”，固定新增上限8、贪心、单束，分别开启和关闭缓存，比较新增编号与文本。

该输入实际长9个 token，形状 `[1,9]`。两组完整输出均为 `[1,17]`，新增部分均为 `[1,8]`，实际新增编号完全相同：

```text
[6529, 9999, 54851, 46944, 100751, 18493, 104949, 15946]
```

解码文本（保留开头空格）：

```text
 attention_mask 是一个用于在模型中
```

这是八个 token 的截断续写。本课使用基础模型，该片段不作为完整概念定义。此题核对的是相同条件下的编号；本次两组都按新增长度结束，没有出现 EOS。

参考实现分别构造两个 `GenerationConfig`，其中 `max_new_tokens=8`、`do_sample=False`、`num_beams=1`、EOS/PAD相同，仅 `use_cache` 不同；调用时只传配置对象，避免混用配置对象与额外生成参数。

```python
off = model.generate(**encoded, generation_config=configs[False])
on = model.generate(**encoded, generation_config=configs[True])
print(torch.equal(off, on))  # 本次为 True
new_ids = on[:, encoded["input_ids"].shape[1]:]
```

## 第三题：计算 K/V 张量大小

题目：CPU float32 下，计算 B=1/S=512、B=1/S=2048、B=2/S=2048 的缓存大小，并说明为什么使用2个KV头。

模型实际配置：24层、14个Query头、2个KV头、每头64维。当前实际 dtype=float32，每个元素4字节。

```text
KV字节数 = 2 × 层数 × B × KV头数 × S × 每头维度 × 元素字节数
         = 2 × 24 × B × 2 × S × 64 × 4
         = 24,576 × B × S
```

前面的2表示K和V两份，头数中的2表示KV头数。

| B | S | 字节数 | MiB（除以1024²） |
|---:|---:|---:|---:|
| 1 | 512 | 12,582,912 | 12 |
| 1 | 2048 | 50,331,648 | 48 |
| 2 | 2048 | 100,663,296 | 96 |

本模型使用GQA，14个Query头共享2组K/V，缓存实际保存的形状使用2个KV头。误用14会把此项放大7倍。

第一题实际读取24层K/V张量并累加 `numel()*element_size()`，三次得到49,152、73,728、98,304字节，与缓存长度2/3/4代入公式一致。

参数张量另占1,976,131,072字节，约1.840GiB。参数、KV、临时结果和运行开销共同影响内存；本题的K/V大小不是进程总内存，也不是峰值测量。

## 第四题：8/16 token 的可复现耗时记录

题目：选定同一输入，分别新增8/16个token；每种长度下缓存开关各预热1次，正式运行5次，记录中位数及环境。

本次固定使用第二题文本，输入长度始终为9；每个条件重新开始生成。模型只加载一次，编码、预热、断言、打印和解码均在计时区间外。

环境：Python3.11.16、torch2.14.0、transformers5.20.2、huggingface_hub2.2.0、tokenizers0.23.3；CPU、float32、4个计算线程，eval和inference_mode；模型`Qwen/Qwen2.5-0.5B`，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`。

计时区间是 `model.generate`，包含提示词处理和后续解码；使用 `time.perf_counter`，每轮交替两组的执行顺序。本次贪心无随机采样，种子记录为None。各组预热1次不计入下列5次。

| 新增上限 | 关闭缓存中位数/秒 | 开启缓存中位数/秒 | 本次速度比 |
|---:|---:|---:|---:|
| 8 | 0.506892 | 0.283099 | 1.79 |
| 16 | 1.016260 | 0.537403 | 1.89 |

原始五次耗时：

- N=8，cache=False：`[0.508175, 0.50507, 0.504961, 0.506892, 0.509748]` 秒。
- N=8，cache=True：`[0.282563, 0.282529, 0.283099, 0.285333, 0.283364]` 秒。
- N=16，cache=False：`[1.014075, 1.015506, 1.021426, 1.01626, 1.06139]` 秒。
- N=16，cache=True：`[0.534866, 0.537879, 0.536414, 0.54139, 0.537403]` 秒。

每种新增长度下，缓存开关的编号完全一致，各次重复也一致。8-token完整输出为 `[1,17]`，16-token完整输出为 `[1,25]`；16-token续写的前8个编号与第二题相同。

本次开启缓存较快，新增16的总耗时高于新增8。数值只对应上述环境与输入，不设固定加速门槛，也不据此推断其他设备速度。没有单独测量首token延迟、GPU/MPS或进程峰值内存。

## 完整运行

从仓库根目录执行：

```sh
conda activate fde
python -m hugging_learn.homework.lesson07_answers --report /tmp/c03-lesson07-answers.json
python verify_lesson07.py --report /tmp/c03-lesson07-verification.json
```

答案也支持 `python hugging_learn/homework/lesson07_answers.py` 直接运行；每次实际计时会波动，不能用本页某个耗时值作为逐次相等断言。
