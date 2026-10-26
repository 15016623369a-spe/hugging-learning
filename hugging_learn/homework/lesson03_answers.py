"""C03 Lesson03 四题参考答案：助手补充，使用固定模型本地缓存。"""
import argparse
import json
from pathlib import Path

import torch
from hugging_learn.causal_common import MODEL_ID, REVISION, load_causal_lm

TEXTS = [
    "你好。",
    "我想查询订单金额。",
    "请统计最近一个月各部门的订单总金额。",
]


def collect_answers(*, config=None, tokenizer=None, model=None):
    if config is None and tokenizer is None and model is None:
        config, tokenizer, model = load_causal_lm()
    elif config is None or tokenizer is None or model is None:
        raise ValueError("复用模型时，config、tokenizer、model 需要同时提供。")
    device = next(model.parameters()).device

    q2 = []
    # 三条文本共享一份模型，分别编码，不做批量补齐。
    for text in TEXTS:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            outputs = model(
                **inputs, use_cache=False, output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            next_logits = outputs.logits[:, -1, :]
            # 与教师示例一致：仅观察用 softmax 采用 float64。
            probs = torch.softmax(next_logits, dim=-1, dtype=torch.float64)
            top_probs, top_ids = torch.topk(probs, k=5, dim=-1)
            next_ids = next_logits.argmax(dim=-1)

        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, dtype=torch.float64))
        assert torch.equal(next_ids, top_ids[:, 0])
        candidates = []
        for token_id, probability in zip(top_ids[0].tolist(), top_probs[0].tolist()):
            candidates.append({
                "token_id": token_id,
                "raw_token": tokenizer.convert_ids_to_tokens(token_id),
                "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
                "probability": probability,
            })
        q2.append({
            "text": text,
            "input_ids": inputs["input_ids"].tolist(),
            "token_count": inputs["input_ids"].shape[1],
            "input_shape": list(inputs["input_ids"].shape),
            "hidden_shape": list(hidden.shape),
            "logits_shape": list(outputs.logits.shape),
            "next_logits_shape": list(next_logits.shape),
            "top_ids_shape": list(top_ids.shape),
            "top_probs_shape": list(top_probs.shape),
            "next_ids_shape": list(next_ids.shape),
            "input_dtype": str(inputs["input_ids"].dtype),
            "logits_dtype": str(outputs.logits.dtype),
            "probability_dtype": str(probs.dtype),
            "finite_logits": bool(torch.isfinite(outputs.logits).all()),
            "logits_requires_grad": outputs.logits.requires_grad,
            "probability_sum": probs.sum().item(),
            "top5_probability_sum": top_probs.sum().item(),
            "top5": candidates,
        })

    q1 = {name: q2[0][name] for name in (
        "text", "token_count", "input_shape", "hidden_shape", "logits_shape",
        "next_logits_shape", "top_ids_shape", "top_probs_shape", "next_ids_shape",
    )}
    expected_shapes = {
        "input_shape": [1, 2], "hidden_shape": [1, 2, 896],
        "logits_shape": [1, 2, 151936], "next_logits_shape": [1, 151936],
        "top_ids_shape": [1, 5], "top_probs_shape": [1, 5],
        "next_ids_shape": [1],
    }
    for name, expected in expected_shapes.items():
        assert q1[name] == expected, (name, q1[name])
    assert [row["token_count"] for row in q2] == [2, 5, 10]

    scores = torch.tensor([3.0, 1.0, -1.0], dtype=torch.float64)
    full_probs = torch.softmax(scores, dim=-1)
    first_two_probs = torch.softmax(scores[:2], dim=-1)
    torch.testing.assert_close(full_probs.sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(first_two_probs.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert scores.argmax().item() == full_probs.argmax().item() == 0
    q3 = {
        "scores": scores.tolist(),
        "full_probabilities": full_probs.tolist(),
        "argmax_index": scores.argmax().item(),
        "top2_original_probability_sum": full_probs[:2].sum().item(),
        "softmax_of_first_two_scores": first_two_probs.tolist(),
        "renormalized_sum": first_two_probs.sum().item(),
    }
    q4 = {
        "first_position": "outputs.logits[:, 0, :] 使用第一个位置，而非完整输入的最后位置。",
        "time_softmax": "对 [B,T,V] 使用 dim=1 沿位置归一化，应沿词表维度 -1。",
        "flat_argmax": "不指定 dim 会在整个张量展平后找最大值，返回扁平索引。",
        "correct_code": [
            "next_logits = outputs.logits[:, -1, :]",
            "probs = torch.softmax(next_logits, dim=-1, dtype=torch.float64)",
            "next_ids = next_logits.argmax(dim=-1)",
        ],
    }
    return {
        "scope": "Lesson03 四题助手参考；实际离线前向与数值计算，不训练，不执行生成循环",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "device": str(device), "model_dtype": str(next(model.parameters()).dtype),
        "probability_dtype": "torch.float64",
        "hidden_size": config.hidden_size, "vocab_size": config.vocab_size,
        "q1": q1, "q2": q2, "q3": q3, "q4": q4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = collect_answers()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
