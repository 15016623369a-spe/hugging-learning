"""C03 Lesson05 四题助手参考：贪心对照、长度、种子与教学概率。"""
import argparse
import json
from pathlib import Path

import torch
from transformers import set_seed

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hugging_learn.causal_common import MODEL_ID, REVISION, load_causal_lm
from hugging_learn.homework.lesson04_answers import CLASSROOM_TEXT, greedy_generate

SAMPLING_OPTIONS = dict(temperature=0.8, top_k=20, top_p=0.9)


@torch.inference_mode()
def generate_once(model, tokenizer, text, *, max_new_tokens=8, max_length=None,
                  do_sample=False, **sampling_options):
    """单句无补齐；每次重新编码，不沿用前一次生成的编号。"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos if isinstance(eos, int) else (eos[0] if eos else None)
    options = dict(
        num_beams=1, use_cache=False, do_sample=do_sample,
        eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False,
    )
    if max_length is None:
        options["max_new_tokens"] = max_new_tokens
    else:
        # 当前模型自带 max_new_tokens=2048；改用总长度时必须显式清除它。
        options.update(max_new_tokens=None, max_length=max_length)
    options.update(sampling_options)
    output_ids = model.generate(**inputs, **options)
    new_ids = output_ids[:, prompt_length:]
    assert torch.equal(output_ids[:, :prompt_length], inputs["input_ids"])
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    ids = new_ids[0].tolist()
    return {
        "text": text, "prompt_length": prompt_length,
        "input_ids": inputs["input_ids"][0].tolist(),
        "input_shape": list(inputs["input_ids"].shape),
        "output_shape": list(output_ids.shape), "new_shape": list(new_ids.shape),
        "new_ids": ids, "continuation": tokenizer.decode(ids, skip_special_tokens=True),
        "full_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
        "prefix_preserved": True, "options": options, "eos_ids": eos_ids,
        "stop_reason": "eos_token" if ids and ids[-1] in eos_ids else "length_limit",
    }


def toy_distribution(logits, *, temperature=1.0, top_k=0, top_p=1.0):
    """教学数值，不来自模型；使用当前 Transformers 的实际温度与过滤算子。"""
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
    )
    scores = torch.tensor([logits], dtype=torch.float64)
    dummy_ids = torch.zeros((1, 1), dtype=torch.int64)
    scaled = TemperatureLogitsWarper(float(temperature))(dummy_ids, scores)
    probabilities_before_filter = scaled.softmax(dim=-1)
    filtered = scaled
    if top_k:
        filtered = TopKLogitsWarper(top_k)(dummy_ids, filtered)
    if top_p < 1:
        filtered = TopPLogitsWarper(top_p)(dummy_ids, filtered)
    final = filtered.softmax(dim=-1)
    return {
        "provenance": "手工教学 logits；不是 Qwen 候选分数或实际生成轨迹",
        "logits": logits, "temperature": temperature, "top_k": top_k, "top_p": top_p,
        "probabilities_before_filter": probabilities_before_filter[0].tolist(),
        "retained_indices": torch.isfinite(filtered[0]).nonzero().flatten().tolist(),
        "final_probabilities": final[0].tolist(),
    }


def collect_answers(*, tokenizer=None, model=None):
    if tokenizer is None and model is None:
        _, tokenizer, model = load_causal_lm()
    elif tokenizer is None or model is None:
        raise ValueError("复用模型时，tokenizer 和 model 必须同时提供。")
    model.eval()

    # 第1题：相同输入、相同三步上限，逐编号对照。
    manual = greedy_generate(model, tokenizer, CLASSROOM_TEXT, 3)
    generated = generate_once(model, tokenizer, CLASSROOM_TEXT, max_new_tokens=3)
    assert manual["new_ids"] == generated["new_ids"]
    q1 = {"manual": manual, "generate": generated, "ids_equal": True}

    # 第2题：每条独立编码；另实测同一总长度上限的不同新增预算。
    q2 = {
        "new_token_limits": [
            generate_once(model, tokenizer, text, max_new_tokens=limit)
            for text in ["你好。", CLASSROOM_TEXT] for limit in [4, 8]
        ],
        "total_length_limits": [
            generate_once(model, tokenizer, text, max_length=12)
            for text in ["你好。", CLASSROOM_TEXT]
        ],
    }

    # 第3题：重设种子后应从相同随机状态起步；只设一次则继续消耗状态。
    reset_each_time = []
    for seed in [42, 42, 43]:
        set_seed(seed)
        row = generate_once(model, tokenizer, CLASSROOM_TEXT, do_sample=True, **SAMPLING_OPTIONS)
        reset_each_time.append({"seed": seed, **row})
    set_seed(42)
    seed_once = []
    for index in range(3):
        row = generate_once(model, tokenizer, CLASSROOM_TEXT, do_sample=True, **SAMPLING_OPTIONS)
        seed_once.append({"call": index + 1, "seed_set_before_sequence": 42, **row})
    q3 = {"reset_each_time": reset_each_time, "seed_once": seed_once}

    # 第4题：先看温度，再分别、组合观察 top-k 与 top-p。
    logits = [3, 2, 1, 0, -1]
    q4 = {
        "temperature_runs": [toy_distribution(logits, temperature=t) for t in [0.5, 1.0, 2.0]],
        "top_k_2": toy_distribution(logits, top_k=2),
        "top_p_09": toy_distribution(logits, top_p=0.9),
        "combined_k2_p09": toy_distribution(logits, top_k=2, top_p=0.9),
        "user_selected_controls": toy_distribution(logits, temperature=0.85, top_k=4, top_p=0.9),
    }
    try:
        generate_once(model, tokenizer, CLASSROOM_TEXT, max_new_tokens=0)
    except ValueError as exc:
        zero_observation = {"type": type(exc).__name__, "message": str(exc)}
    else:
        zero_observation = {"type": "returned"}

    return {
        "provenance": "助手四题参考；用户已在聊天明确确认完成，独立能力未另行验收",
        "scope": "固定 Qwen、CPU float32、eval、inference_mode、单句无 PAD、use_cache=False；教学概率另列；不训练",
        "model_id": MODEL_ID, "revision": REVISION,
        "sampling_options": SAMPLING_OPTIONS,
        "q1": q1, "q2": q2, "q3": q3, "q4": q4,
        "zero_new_tokens_observation": zero_observation,
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
