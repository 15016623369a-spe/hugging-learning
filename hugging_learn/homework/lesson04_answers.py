"""C03 Lesson04 四题助手参考：同一份模型、单句、无 PAD、手写贪心生成。"""
import argparse
import json
from pathlib import Path

import torch

# 编辑器直接运行作业文件时，使用文件位置确定仓库根目录。
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hugging_learn.causal_common import MODEL_ID, REVISION, load_causal_lm

CLASSROOM_TEXT = "请统计最近一个月各部门的订单总金额"
TEXTS = ["你好。", "我想查询订单金额。", CLASSROOM_TEXT + "。"]


@torch.inference_mode()
def greedy_generate(model, tokenizer, text, max_new_tokens=8):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数")
    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = input_ids.shape[1]
    original_ids = input_ids[0].tolist()

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        eos_ids = set()
    elif isinstance(eos, int):
        eos_ids = {eos}
    else:
        eos_ids = set(eos)

    steps = []
    stop_reason = "max_new_tokens"
    for step in range(max_new_tokens):
        length_before = input_ids.shape[1]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        next_logits = outputs.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        token_id = next_token.item()
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(next_token, dtype=attention_mask.dtype),
        ], dim=-1)
        steps.append({
            "step": step + 1,
            "input_shape_before": [1, length_before],
            "logits_shape": list(outputs.logits.shape),
            "next_logits_shape": list(next_logits.shape),
            "next_token_shape": list(next_token.shape),
            "input_shape_after": list(input_ids.shape),
            "mask_shape_after": list(attention_mask.shape),
            "mask_all_ones": bool((attention_mask == 1).all()),
            "finite_logits": bool(torch.isfinite(outputs.logits).all()),
            "logits_requires_grad": outputs.logits.requires_grad,
            "token_id": token_id,
            "single_decoded": tokenizer.decode([token_id]),
        })
        if token_id in eos_ids:
            stop_reason = "eos_token"
            break

    new_ids = input_ids[:, prompt_length:]
    return {
        "text": text,
        "prompt_length": prompt_length,
        "original_ids": original_ids,
        "max_new_tokens": max_new_tokens,
        "forward_count": len(steps),
        "steps": steps,
        "new_ids": new_ids[0].tolist(),
        "new_ids_shape": list(new_ids.shape),
        "final_shape": list(input_ids.shape),
        "stop_reason": stop_reason,
        "continuation": tokenizer.decode(new_ids[0], skip_special_tokens=True),
        "full_text": tokenizer.decode(input_ids[0], skip_special_tokens=True),
        "eos_ids": sorted(eos_ids),
    }


def collect_answers(*, tokenizer=None, model=None):
    if tokenizer is None and model is None:
        _, tokenizer, model = load_causal_lm()
    elif tokenizer is None or model is None:
        raise ValueError("复用模型时，tokenizer 和 model 需要同时提供。")
    model.eval()
    q1 = greedy_generate(model, tokenizer, CLASSROOM_TEXT, 3)
    q2 = [greedy_generate(model, tokenizer, text, 8) for text in TEXTS]
    q4 = {str(limit): greedy_generate(model, tokenizer, CLASSROOM_TEXT, limit) for limit in [0, 1]}

    # 第3题第二段：真实执行维数不一致的 cat，保留报错供解释。
    wrong_next = torch.tensor([8997])
    try:
        torch.cat([torch.zeros((1, 9), dtype=torch.int64), wrong_next], dim=-1)
    except RuntimeError as exc:
        rank_error = str(exc)
    else:
        raise AssertionError("预期 cat 因张量维数不一致而报错")
    q3 = {
        "replace_input": "input_ids = next_token 会丢失完整上下文；本课不使用缓存，必须追加。",
        "rank_mismatch": {"input_shape": [1, 9], "wrong_next_shape": list(wrong_next.shape), "actual_error": rank_error},
        "fixes": [
            "next_token = next_logits.argmax(dim=-1, keepdim=True)",
            "input_ids = torch.cat([input_ids, next_token], dim=-1)",
            "也可先 argmax(dim=-1)，再 unsqueeze(-1)，并同步追加 attention_mask。",
        ],
    }

    # 手工构造，仅演示解码如何隐藏 EOS，不冒充模型自然生成了 EOS。
    eos = q1["eos_ids"][0]
    sample_ids = tokenizer.encode("一个月", add_special_tokens=False) + [eos]
    synthetic = {
        "provenance": "手工构造的解码示例，不是模型生成轨迹",
        "ids": sample_ids,
        "decoded_with_special": tokenizer.decode(sample_ids, skip_special_tokens=False),
        "decoded_without_special": tokenizer.decode(sample_ids, skip_special_tokens=True),
        "id_count": len(sample_ids),
        "visible_python_character_count": len(tokenizer.decode(sample_ids, skip_special_tokens=True)),
    }

    # 核对实际生成轨迹与循环的长度、形状和停止条件。
    for row in [q1, *q2, *q4.values()]:
        assert row["forward_count"] == len(row["new_ids"]) <= row["max_new_tokens"]
        assert row["final_shape"] == [1, row["prompt_length"] + len(row["new_ids"])]
        for index, step in enumerate(row["steps"]):
            length = row["prompt_length"] + index
            assert step["input_shape_before"] == [1, length]
            assert step["logits_shape"] == [1, length, 151936]
            assert step["next_logits_shape"] == [1, 151936]
            assert step["next_token_shape"] == [1, 1]
            assert step["input_shape_after"] == step["mask_shape_after"] == [1, length + 1]
            assert step["mask_all_ones"] and step["finite_logits"] and not step["logits_requires_grad"]
        if row["stop_reason"] == "eos_token":
            assert row["new_ids"][-1] in row["eos_ids"]
        else:
            assert row["forward_count"] == row["max_new_tokens"]
            assert not set(row["new_ids"]).intersection(row["eos_ids"])
    assert q1["prompt_length"] == 9 and q1["new_ids"] == [8997, 100012, 100364]
    assert [row["prompt_length"] for row in q2] == [2, 5, 10]
    assert q4["0"]["new_ids_shape"] == [1, 0] and q4["0"]["full_text"] == CLASSROOM_TEXT

    return {
        "lesson_date": "2026-10-27", "timezone": "Asia/Shanghai",
        "provenance": "四题助手参考；用户索要答案按约定接受为私下完成确认，与独立能力验证分开记录",
        "scope": "单句、无 PAD、CPU float32、手写贪心循环、use_cache=False；同一份模型复用；不训练",
        "model_id": MODEL_ID, "revision": REVISION,
        "q1": q1, "q2": q2, "q3": q3,
        "q4": {"boundary_runs": q4, "synthetic_eos_decode": synthetic},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = collect_answers()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
