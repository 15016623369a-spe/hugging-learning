import torch

from hugging_learn.causal_common import load_causal_lm

# 1. 加载模型并编码输入
config, tokenizer, model = load_causal_lm()

text = "请统计最近一个月各部门的订单总金额"
inputs = tokenizer(text, return_tensors="pt")

device = next(model.parameters()).device
inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

with torch.inference_mode():
    # 2. 前向，并额外返回各层隐藏表示
    outputs = model(
        **inputs,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )

    # 3. 取最终隐藏表示，手动经过词表输出头
    hidden = outputs.hidden_states[-1]
    head = model.get_output_embeddings()
    rebuilt_logits = head(hidden)

    # 4. 取最后位置，计算整个词表上的概率
    next_logits = outputs.logits[:, -1, :]

    # 仅观察用概率采用 float64，减少归一化舍入误差
    probs = torch.softmax(
        next_logits,
        dim=-1,
        dtype=torch.float64,
    )

    # 5. 查看前五个候选，选出最高分编号
    top_probs, top_ids = torch.topk(
        probs, k=5, dim=-1
    )
    next_id = next_logits.argmax(dim=-1)

print("输入形状：", tuple(inputs["input_ids"].shape))
print("最终隐藏表示：", tuple(hidden.shape))
print("输出头权重：", tuple(head.weight.shape))
print("logits：", tuple(outputs.logits.shape))
print("最后位置分数：", tuple(next_logits.shape))

difference = (
    rebuilt_logits - outputs.logits
).abs().max().item()

print("手动输出头与模型输出的最大差异：", difference)
print("完整词表概率和：", probs.sum(dim=-1).tolist())

for token_id, probability in zip(
    top_ids[0].tolist(),
    top_probs[0].tolist(),
):
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
    )
    print(
        "ID =", token_id,
        "文本 =", repr(decoded),
        "概率 =", round(probability, 6),
    )

chosen_id = next_id.item()
print("最高分编号：", chosen_id)
print("对应文本：", repr(tokenizer.decode([chosen_id])))