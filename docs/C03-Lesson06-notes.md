# C03 Lesson06：批量生成与对话输入

更新日期：2026-10-28（Asia/Shanghai）。聊天为主要课堂，本讲义为助手整理。[课堂 lesson06.py](../hugging_learn/lesson06.py) 为合并后的单一实现，原代码与注释逐字节保留。

## 1. 从单条生成到批量生成

Lesson05输入为 `[1,T]`，返回原输入与续写。本节将若干条独立输入排列为 `[B,W]`，B是样本数量，W是补齐后的共同宽度。各行是独立样本，不会把另一行的文本作为自己的上下文。

课堂三条文本的实际长度为 `[2,5,9]`：

```python
texts = ["你好。", "我想查询订单金额。", "请统计最近一个月各部门的订单总金额"]
tokenizer.padding_side = "left"
inputs = tokenizer(texts, padding=True, return_tensors="pt")
```

`padding=True`补到当前批次最长输入，得到input_ids和attention_mask均为 `[3,9]`。补齐与截断是独立操作；不能认为padding会自动截短超长输入。

## 2. 共同宽度与有效长度

```text
第一行：P P P P P P P A B
第二行：P P P P C D E F G
第三行：H I J K L M N O Q
```

字母为示意token，P为补齐。对应Mask：

```python
[
    [0,0,0,0,0,0,0,1,1],
    [0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
]
```

`inputs['input_ids'].shape[1]`得到共同宽度9；`attention_mask.sum(dim=1)`沿每行累加1，得到有效长度[2,5,9]。本课Mask的0表示PAD位置，1表示有效输入，并不直接等于完整的因果注意力矩阵。

## 3. 为什么这条生成路径使用左补齐

当前Transformers常规生成路径读取 `outputs.logits[:, -1, :]`。短输入右补齐时最后位置是PAD；左补齐时最后位置落在正文末尾。

```text
右补齐：A B P P P    最后位置不是正文B
左补齐：P P P A B    最后位置是正文B
```

Mask控制补齐位置是否参与有效上下文，不会自动将`-1`索引改为每行最后一个有效位置，也不保证PAD位置的最终输出为零。因此本课使用左补齐并显式提供Mask。该规则针对本课decoder-only批量generate路径，不机械推广到BERT编码或所有训练任务。

当前生成代码还依据Mask准备position_ids，核心是`attention_mask.long().cumsum(-1)-1`，再处理PAD位置。例如Mask[0,0,0,1,1]的正文位置编号仍为0、1。调用generate与自行忽略位置处理的手写批量循环不能不加核对地互换。

授课阶段另做过右补齐诊断：课堂两条短输入的编号变化，最长无PAD行不变；该历史教师示例留在聊天工作区。本轮仓库验收执行用户左补齐课堂和四题，不将此前诊断冒充本轮新测项目。

## 4. 新增内容从共同宽度开始截取

```python
width = inputs["input_ids"].shape[1]
outputs = model.generate(**inputs, **common)
new_ids = outputs[:, width:]
replies = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
```

课堂实际输出 `[3,17]`，新增 `[3,8]`。第一行存储的是7个PAD、2个正文、8个新增，生成部分从列9开始，而不是从有效长度2开始。

作业使用另一组三条文本，其长度[1,3,10]、共同宽度10。最短输入从列1截取会得到`'订单价为10'`，从列10截取才得到`'价为10'`。即使解码隐藏PAD，原输入“订单”仍会留在错误切片中。

`decode`处理一条序列，`batch_decode`处理多行并返回字符串列表。`skip_special_tokens=True`控制显示，不负责生成停止，也不能修复错误的切片起点。

## 5. PAD与EOS可以同号，但含义不同

本课Tokenizer PAD与EOS均为151643。左侧输入PAD由Mask标为0，不会仅因输入中含有这些编号就马上结束生成。生成过程新产生EOS才触发对应的结束规则。

批量中一行先结束、其他行继续时，已结束行可能继续填PAD以保持张量规则形状。所以新增区域的共同列数，不一定等于每行实际生成数量。

手工示意`[token1,token2,EOS,PAD,PAD]`有5列，生成到EOS共3个token，尾部补齐2列。EOS本身占一个生成位置。PAD与EOS同号时不能把所有这个编号都删除后再推断生成步数；须结合顺序和停止规则。本轮自然生成均按上限结束，没有生成EOS；手工示例另列来源。

## 6. 对话消息怎样变成token

一段聊天是消息字典的列表，每条包含role与content；多个独立聊天是消息列表的列表。外层区分样本，内层保存同一段对话的历史。

```python
messages = [
    {"role": "system", "content": "你是一位耐心的中文老师。"},
    {"role": "user", "content": "请用一句话解释什么是 Token。"},
]
```

system描述任务或回答方式，user提供用户输入，assistant保存助手回复。模型最终接收编号，而非Python字典；聊天模板把这些角色和内容序列化为模型配套的格式。

本地固定Qwen基础模型Tokenizer实际带有chat_template。第一段聊天格式为：

```text
<|im_start|>system
你是一位耐心的中文老师。<|im_end|>
<|im_start|>user
请用一句话解释什么是 Token。<|im_end|>
<|im_start|>assistant
```

模板中的消息结束标记im_end为151645，而当前基础模型的生成EOS为151643。消息格式边界和生成停止配置分别查看，不能自动等同。

## 7. 专项解释：为什么上面tokenize=True，下面False

`tokenizer`是负责处理输入的对象；`tokenize`是apply_chat_template的参数，控制套模板之后是否立刻编码。

一步路径：

```python
chat_inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
```

执行“消息列表→模板文本→编号→张量”。`return_dict=True`明确返回input_ids和attention_mask等字段；`return_tensors='pt'`决定字段值是PyTorch张量。当前版本返回字典式BatchEncoding，不要照搬旧示例默认返回列表的假设。

两步路径：

```python
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
chat_inputs = tokenizer(
    formatted, add_special_tokens=False, return_tensors="pt",
)
```

第一步False只是暂时停在字符串，方便打印检查。第二步调用tokenizer(formatted)才执行编码。两条路径本次逐编号相同。

`add_special_tokens=False`控制另一件事：编码时不额外自动添加特殊边界符，避免与模板已有标记重复。它不会关闭分词，也不会删除模板中的im_start/im_end；已有标记仍转换为编号。

## 8. add_generation_prompt只准备助手开头

在当前Qwen模板中，True比False多出`<|im_start|>assistant\n`，对应3个token。两段参考聊天的长度分别从24变27、40变43。它只追加输入提示，答案仍由model.generate产生。

推荐直接用tokenize=True一步准备输入；想先检查文本则拆成两步。不同模型模板可能不同，应读取各自的配置，不手工套用当前Qwen标记。

## 9. 基础模型、指令模型与历史

本节实际模型是Qwen/Qwen2.5-0.5B基础版，不是Qwen/Qwen2.5-0.5B-Instruct。模板决定输入格式，指令后训练影响权重与任务响应能力；套模板、设system或调用eval都不会把基础模型训练成指令模型。本节没有下载或验证Instruct权重。

聊天批次真实长度[27,43]，左补齐后 `[2,43]`，新增8列后 `[2,51]`。短续写`'Token 是中文中用于表示一个单词'`与`'attention_mask 是一个用于在模型中'`只是基础模型受八token上限限制的原始输出。前者不能当作token定义；token可为词、子词、标点或字符片段，不限中文单词。

程序需要手动把回复追加为assistant，再追加user追问，重新应用模板。参考中消息数从2条变4条，输入从27变50个token；新角色和边界标记也占位置，长度按实际编码读取。本轮只编码下一轮，没有生成其回答。模型不会自动保存Python列表中的历史。

## 10. 入口与验证

```sh
conda activate fde
python -m hugging_learn.lesson06
python -m hugging_learn.homework.lesson06_answers
python verify_lesson06.py --report /tmp/c03-lesson06-verification.json
```

详细答案见[四题题解](C03-Lesson06-homework.md)，完整代码见[lesson06_answers.py](../hugging_learn/homework/lesson06_answers.py)，40项检查与版本/源码哈希见[运行记录](C03-Lesson06-verification.json)，来源见[验收](C03-Lesson06-acceptance.md)。下一课Lesson07讲上下文、KVcache、内存、耗时与复现。

官方参考：[批量生成](https://huggingface.co/docs/transformers/llm_tutorial)、[聊天模板](https://huggingface.co/docs/transformers/main/chat_templating)。具体编号、默认值和形状以本机固定版本实测为依据。
