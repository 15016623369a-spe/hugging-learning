from pathlib import Path
import sys

import torch

# 兼容编辑器直接运行和模块运行。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hugging_learn.causal_common import load_causal_lm


# 1. 加载现有模型
config, tokenizer, model = load_causal_lm()
model.eval()

device = next(model.parameters()).device

# 本课的批量生成使用左补齐。
tokenizer.padding_side = "left"

if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise ValueError("需要先确认模型的 PAD 配置。")
    tokenizer.pad_token = tokenizer.eos_token

eos = model.generation_config.eos_token_id
if eos is None:
    eos = tokenizer.eos_token_id

common = dict(
    max_new_tokens=8,
    do_sample=False,
    num_beams=1,
    use_cache=False,
    eos_token_id=eos,
    pad_token_id=tokenizer.pad_token_id,
    return_dict_in_generate=False,
)


# 2. 两类输入共用同一个生成函数
@torch.inference_mode()
def generate_batch(encoded, title):
    encoded = {
        name: tensor.to(device)
        for name, tensor in encoded.items()
    }

    input_width = encoded["input_ids"].shape[1]
    real_lengths = encoded["attention_mask"].sum(dim=1)

    output_ids = model.generate(
        **encoded,
        **common,
    )

    # 验证输出前缀仍是完整的补齐后输入。
    assert torch.equal(
        output_ids[:, :input_width],
        encoded["input_ids"],
    )

    # 从整个批次的共同宽度之后开始取。
    new_ids = output_ids[:, input_width:]

    replies = tokenizer.batch_decode(
        new_ids,
        skip_special_tokens=True,
    )

    print(f"\n{title}")
    print("输入形状：", encoded["input_ids"].shape)
    print("Mask：", encoded["attention_mask"].tolist())
    print("真实长度：", real_lengths.tolist())
    print("完整输出形状：", output_ids.shape)
    print("新增形状：", new_ids.shape)

    for index, reply in enumerate(replies, start=1):
        print(f"第{index}条续写：", repr(reply))

    return replies


# 3. 普通文本批量生成
texts = [
    "你好。",
    "我想查询订单金额。",
    "请统计最近一个月各部门的订单总金额",
]

text_inputs = tokenizer(
    texts,
    padding=True,
    return_tensors="pt",
)

text_replies = generate_batch(
    text_inputs,
    "普通文本批量生成",
)


# 4. 两个独立聊天
chats = [
    [
        {"role": "system", "content": "你是一位耐心的中文老师。"},
        {"role": "user", "content": "请用一句话解释什么是 Token。"},
    ],
    [
        {"role": "system", "content": "你是一位耐心的中文老师。"},
        {"role": "user", "content": "你好。"},
        {"role": "assistant", "content": "你好，我们一起学习。"},
        {"role": "user", "content": "attention_mask 有什么作用？"},
    ],
]

if not tokenizer.chat_template:
    raise ValueError("当前 Tokenizer 没有配置聊天模板。")

formatted = tokenizer.apply_chat_template(
    chats[0],
    tokenize=False,
    add_generation_prompt=True,
)

print("\n第一个聊天的格式化文本：")
print(formatted)

chat_inputs = tokenizer.apply_chat_template(
    chats,
    tokenize=True,
    padding=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

chat_replies = generate_batch(
    chat_inputs,
    "基础模型的聊天格式续写",
)


# 5. 手动追加第一段聊天的历史
history = [dict(message) for message in chats[0]]

history.append({
    "role": "assistant",
    "content": chat_replies[0],
})

history.append({
    "role": "user",
    "content": "再举一个例子。",
})

# 下一轮需要把更新后的 history 再次应用模板。
next_inputs = tokenizer.apply_chat_template(
    history,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

print("\n下一轮输入形状：", next_inputs["input_ids"].shape)