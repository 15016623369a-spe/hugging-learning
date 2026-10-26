"""助手补充：Lesson02 四题参考答案；同一模型依次处理三条文本。"""
import json

import torch

from hugging_learn.causal_common import MODEL_ID, REVISION, load_causal_lm

TEXTS = [
    "你好。",
    "我想查询订单金额。",
    "请统计最近一个月各部门的订单总金额。",
]


def compare_grad_modes():
    """eval 控制模块行为；inference_mode 控制这次计算的梯度记录。"""
    layer = torch.nn.Linear(2, 2)
    layer.eval()
    x = torch.ones(1, 2)
    # 显式启用梯度，避免调用方的外部 no_grad 改变本题条件。
    with torch.enable_grad():
        ordinary = layer(x)
    with torch.inference_mode():
        inference = layer(x)
    return {
        "layer_training": layer.training,
        "input_requires_grad": x.requires_grad,
        "weight_requires_grad": layer.weight.requires_grad,
        "ordinary_requires_grad": ordinary.requires_grad,
        "ordinary_grad_fn": type(ordinary.grad_fn).__name__,
        "inference_requires_grad": inference.requires_grad,
        "inference_grad_fn": None if inference.grad_fn is None else type(inference.grad_fn).__name__,
        "outputs_equal": torch.equal(ordinary.detach(), inference),
    }


def collect_answers(*, config=None, tokenizer=None, model=None):
    if config is None and tokenizer is None and model is None:
        config, tokenizer, model = load_causal_lm()
    elif config is None or tokenizer is None or model is None:
        raise ValueError("复用模型时，config、tokenizer、model 需要同时提供。")

    parameter = next(model.parameters())
    q1 = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "tokenizer_length": len(tokenizer),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_storage_gib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "training": model.training,
    }

    q2 = []
    # 不做批量补齐，不在循环内重新加载模型。
    for text in TEXTS:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {name: tensor.to(parameter.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False)
        logits = outputs.logits
        q2.append({
            "text": text,
            "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()),
            "input_ids": inputs["input_ids"].tolist(),
            "token_count": inputs["input_ids"].shape[1],
            "input_shape": list(inputs["input_ids"].shape),
            "attention_mask": inputs["attention_mask"].tolist(),
            "input_dtype": str(inputs["input_ids"].dtype),
            "input_device": str(inputs["input_ids"].device),
            "output_class": type(outputs).__name__,
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "logits_device": str(logits.device),
            "requires_grad": logits.requires_grad,
            "finite": bool(torch.isfinite(logits).all()),
        })

    q4 = [
        {
            "mistake": "把 input_ids 转成 float32",
            "reason": "input_ids 是 Embedding 查询的整数索引，不是模型浮点参数。",
            "correction": 'inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)',
        },
        {
            "mistake": "model(inputs)",
            "reason": "把整个字典作为第一个位置参数传入，不能自动映射各输入字段。",
            "correction": "outputs = model(**inputs, use_cache=False)",
        },
        {
            "mistake": "想加载预训练权重，却使用 from_config(config)",
            "reason": "from_config 根据结构建立模型并初始化参数，不会读取预训练权重。",
            "correction": "使用 AutoModelForCausalLM.from_pretrained，固定 MODEL_ID 和 REVISION。",
        },
        {
            "mistake": "只调用 eval()，希望自动关闭梯度记录",
            "reason": "eval 切换训练/评估行为，不改变 autograd 的开关。",
            "correction": "先 model.eval()，再在 with torch.inference_mode(): 中执行前向。",
        },
    ]
    return {"q1": q1, "q2": q2, "q3": compare_grad_modes(), "q4": q4}


def main():
    print(json.dumps(collect_answers(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
