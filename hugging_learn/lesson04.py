import torch

# 兼容包方式运行，以及编辑器直接运行本文件。
if __package__:
    from hugging_learn.causal_common import load_causal_lm
else:
    from causal_common import load_causal_lm

config, tokenizer, model = load_causal_lm()
model.eval()

# 1. 编码原始输入
text = "请统计最近一个月各部门的订单总金额"
inputs = tokenizer(text, return_tensors="pt")

device = next(model.parameters()).device
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

prompt_length = input_ids.shape[1]
max_new_tokens = 3

# 2. 读取 EOS，统一整理成集合
eos = model.generation_config.eos_token_id

if eos is None:
    eos = tokenizer.eos_token_id

if eos is None:
    eos_ids = set()
elif isinstance(eos, int):
    eos_ids = {eos}
else:
    eos_ids = set(eos)

stop_reason = "max_new_tokens"

# 3. 逐 Token 生成
with torch.inference_mode():
    for step in range(max_new_tokens):
        length_before = input_ids.shape[1]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # 取最后位置，贪心选择一个编号
        next_logits = outputs.logits[:, -1, :]
        next_token = next_logits.argmax(
            dim=-1,
            keepdim=True,
        )

        token_id = next_token.item()  # 本课固定 B=1

        # 把新编号追加到序列末尾，cat是concatenate
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=-1,
        )

        # 新 Token 是有效位置，Mask 追加 1
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(
                    next_token,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=-1,
        )

        print(
            f"第 {step + 1} 步："
            f"长度 {length_before} -> {input_ids.shape[1]}，"
            f"ID={token_id}，"
            f"单项解码={tokenizer.decode([token_id])!r}"
        )

        if token_id in eos_ids:
            stop_reason = "eos_token"
            break

# 4. 截取新增部分，并统一解码
new_ids = input_ids[:, prompt_length:]

print("停止原因：", stop_reason)
print("新增编号：", new_ids[0].tolist())
print("新增 Token 数：", new_ids.shape[1])

print(
    "仅新增文本：",
    repr(tokenizer.decode(
        new_ids[0],
        skip_special_tokens=True,
    )),
)

print(
    "输入加续写：",
    repr(tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )),
)

