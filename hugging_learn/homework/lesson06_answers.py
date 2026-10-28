"""C03 Lesson06 四题助手参考：批量、切片、模板与手动历史。"""
import argparse
import json
from pathlib import Path

import torch

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hugging_learn.causal_common import MODEL_ID, REVISION, load_causal_lm

TEXTS = ["订单", "查询订单金额", "请统计最近一个月各部门的订单总金额。"]
CHATS = [
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


def describe_inputs(encoded):
    mask = encoded["attention_mask"]
    width = encoded["input_ids"].shape[1]
    lengths = mask.sum(dim=1).tolist()
    return {
        "input_shape": list(encoded["input_ids"].shape),
        "input_ids": encoded["input_ids"].tolist(), "attention_mask": mask.tolist(),
        "input_width": width, "real_lengths": lengths,
        "padding_counts": [width - length for length in lengths],
    }


@torch.inference_mode()
def generate_batch(model, tokenizer, encoded, *, max_new_tokens=4):
    """单次普通批量生成；本课无其他停止规则，EOS 后可能填 PAD。"""
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in encoded.items()}
    info = describe_inputs(inputs)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    options = dict(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=False,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
    )
    output_ids = model.generate(**inputs, **options)
    width = info["input_width"]
    assert torch.equal(output_ids[:, :width], inputs["input_ids"])
    new_ids = output_ids[:, width:]
    counts, stops = [], []
    for row in new_ids.tolist():
        first_eos = next((i for i, token in enumerate(row) if token in eos_ids), None)
        counts.append(len(row) if first_eos is None else first_eos + 1)
        stops.append("length_limit" if first_eos is None else "eos_token")
    return {
        **info, "options": options, "eos_ids": eos_ids, "prefix_preserved": True,
        "output_shape": list(output_ids.shape), "output_ids": output_ids.tolist(),
        "new_shape": list(new_ids.shape), "new_ids": new_ids.tolist(),
        "continuations": tokenizer.batch_decode(new_ids, skip_special_tokens=True),
        "generated_counts_including_eos": counts,
        "trailing_padding_counts": [new_ids.shape[1] - count for count in counts],
        "stop_reasons": stops,
    }


def collect_answers(*, tokenizer=None, model=None):
    if tokenizer is None and model is None:
        _, tokenizer, model = load_causal_lm()
    elif tokenizer is None or model is None:
        raise ValueError("复用模型时，tokenizer 和 model 必须同时提供。")
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("需先确定模型的 PAD 配置。")
        tokenizer.pad_token = tokenizer.eos_token

    # 第1题：三条文本左补齐，真实长度与共同宽度分别记录。
    encoded = tokenizer(TEXTS, padding=True, return_tensors="pt")
    q1 = {"texts": TEXTS, "padding_side": tokenizer.padding_side, **describe_inputs(encoded)}

    # 第2题：同一模型逐条运行，对照批量编号；另记录错误切片。
    batch = generate_batch(model, tokenizer, encoded, max_new_tokens=4)
    individual = [
        generate_batch(model, tokenizer, tokenizer(text, return_tensors="pt"), max_new_tokens=4)
        for text in TEXTS
    ]
    equal = [row["new_ids"][0] == batch["new_ids"][i] for i, row in enumerate(individual)]
    shortest = min(range(len(TEXTS)), key=lambda i: q1["real_lengths"][i])
    wrong = batch["output_ids"][shortest][q1["real_lengths"][shortest]:]
    q2 = {
        "batch": batch, "individual": individual, "ids_equal": equal,
        "shortest_row": shortest,
        "wrong_slice": {"start": q1["real_lengths"][shortest], "ids": wrong,
                        "decoded": tokenizer.decode(wrong, skip_special_tokens=True)},
        "correct_slice": {"start": q1["input_width"], "ids": batch["new_ids"][shortest],
                          "decoded": batch["continuations"][shortest]},
    }

    # 第3题：消息结构、是否添加 assistant 开头、一步/两步编码。
    if not tokenizer.chat_template:
        raise ValueError("此 tokenizer 没有 chat_template，不能猜测聊天格式。")
    comparisons = []
    for chat in CHATS:
        without = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        with_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        without_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False, return_dict=True,
        )["input_ids"]
        one_step = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_dict=True,
        )["input_ids"]
        two_step = tokenizer(with_prompt, add_special_tokens=False)["input_ids"]
        comparisons.append({
            "message_count": len(chat), "without_generation_prompt": without,
            "with_generation_prompt": with_prompt,
            "length_without_prompt": len(without_ids), "length_with_prompt": len(one_step),
            "appended_text": with_prompt[len(without):],
            "one_step_ids": one_step, "two_step_ids": two_step,
            "two_paths_equal": one_step == two_step,
        })
    chat_inputs = tokenizer.apply_chat_template(
        CHATS, tokenize=True, padding=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    chat_batch = generate_batch(model, tokenizer, chat_inputs, max_new_tokens=8)
    q3 = {"chats": CHATS, "template_comparisons": comparisons, "batch": chat_batch}

    # 第4题：保存实际短续写再追加追问，只编码下一轮，不声称已生成新回答。
    history = [dict(message) for message in CHATS[0]]
    history.append({"role": "assistant", "content": chat_batch["continuations"][0]})
    history.append({"role": "user", "content": "再举一个例子。"})
    next_inputs = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    eos = chat_batch["eos_ids"][0]
    synthetic = [8997, 100012, eos, tokenizer.pad_token_id, tokenizer.pad_token_id]
    q4 = {
        "history": history, "original_message_count": len(CHATS[0]),
        "new_message_count": len(history),
        "original_input_length": comparisons[0]["length_with_prompt"],
        "next_input": describe_inputs(next_inputs),
        "next_turn_generated": False,
        "synthetic_eos_example": {
            "provenance": "手工编号示意，非 Qwen 自然生成轨迹",
            "ids": synthetic, "stored_columns": len(synthetic),
            "generated_count_including_eos": 3, "trailing_padding_count": 2,
            "decode_with_special": tokenizer.decode(synthetic, skip_special_tokens=False),
            "decode_without_special": tokenizer.decode(synthetic, skip_special_tokens=True),
        },
    }
    return {
        "provenance": "四题助手参考；用户索要答案按既有约定登记私下完成确认，未另验收独立能力",
        "scope": "固定基础模型、CPU float32、eval、inference_mode、左补齐、use_cache=False；批量与模板格式；不训练，不评价指令质量",
        "model_id": MODEL_ID, "revision": REVISION,
        "pad_token_id": tokenizer.pad_token_id, "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "special_token_ids": {token: tokenizer.convert_tokens_to_ids(token) for token in ["<|endoftext|>","<|im_start|>","<|im_end|>"]},
        "q1": q1, "q2": q2, "q3": q3, "q4": q4,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = collect_answers()
    rendered = json.dumps(result,ensure_ascii=False,indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(rendered,encoding="utf-8")
    print(rendered,end="")
