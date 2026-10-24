"""C03 Lesson01 四题参考答案（助手补充，非用户独立提交）。"""
import json

import torch

from hugging_learn.common import load_tokenizer

SAMPLES = ["订单金额", "playing", "Transformer", "transformer", "🧠"]
TEXTS = ["订单", "查询订单", "请查询本月订单金额"]


def collect_answers():
    tokenizer = load_tokenizer()
    q1 = []
    for text in SAMPLES:
        encoded = tokenizer(text, add_special_tokens=False)
        q1.append({
            "text": text,
            "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })

    batch = tokenizer(TEXTS, padding=True, return_tensors="pt")
    q2 = {
        "texts": TEXTS,
        "shape": list(batch["input_ids"].shape),
        "input_ids": batch["input_ids"].tolist(),
        "attention_mask": batch["attention_mask"].tolist(),
        "valid_lengths": batch["attention_mask"].sum(dim=1).tolist(),
    }

    strategies = [
        ("按当前 batch 最长序列补齐", {"padding": True}),
        ("补齐到 12", {"padding": "max_length", "max_length": 12}),
        ("补齐并截断到 8", {"padding": "max_length", "truncation": True, "max_length": 8}),
    ]
    q3 = []
    for name, kwargs in strategies:
        encoded = tokenizer(TEXTS, return_tensors="pt", **kwargs)
        q3.append({
            "strategy": name,
            "shape": list(encoded["input_ids"].shape),
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
            "valid_lengths": encoded["attention_mask"].sum(dim=1).tolist(),
            "decoded": tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True),
        })

    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    q4 = {
        "valid_length": mask.sum(dim=1).item(),
        "mha_key_padding_mask": (~mask.bool()).tolist()[0],
        "includes_special_tokens": True,
        "is_causal_mask": False,
        "explanation": "屏蔽 PAD Key 只控制读取规则，不能保证 PAD Query 的输出为零。",
    }
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}


def main():
    print(json.dumps(collect_answers(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
