"""C03 Lesson07 四题助手参考：缓存轨迹、开关对照、内存与可复现计时。"""
import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from zoneinfo import ZoneInfo

import torch
from transformers import GenerationConfig

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hugging_learn.causal_common import MODEL_ID, REVISION, load_causal_lm

Q1_TEXT = "订单金额"
Q2_TEXT = "请用一句话解释 attention_mask 的作用。"


def make_generation_config(model, tokenizer, max_new_tokens, use_cache):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=use_cache,
    )


def describe_generation(model, tokenizer, encoded, output, generation):
    width = encoded["input_ids"].shape[1]
    new_ids = output[0, width:].tolist()
    eos = generation.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    first_eos = next((i for i, value in enumerate(new_ids) if value in eos_ids), None)
    return {
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
        "input_shape": list(encoded["input_ids"].shape),
        "output_ids": output.tolist(), "output_shape": list(output.shape),
        "new_ids": new_ids, "new_shape": [1, len(new_ids)],
        "new_text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "prefix_preserved": torch.equal(output[:, :width], encoded["input_ids"]),
        "stop_reason": "length_limit" if first_eos is None else "eos_token",
        "generated_count_including_eos": len(new_ids) if first_eos is None else first_eos + 1,
        "options": {key: getattr(generation, key) for key in [
            "max_new_tokens", "do_sample", "num_beams", "use_cache", "eos_token_id", "pad_token_id"]},
    }


@torch.inference_mode()
def trace_cache(model, tokenizer, text=Q1_TEXT, steps=3):
    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt").to(device)
    current_ids = encoded["input_ids"]
    mask = encoded["attention_mask"]
    cache = None
    rows, selected = [], []
    for step in range(steps):
        before = 0 if cache is None else cache.get_seq_length()
        outputs = model(
            input_ids=current_ids, attention_mask=mask,
            past_key_values=cache, use_cache=True,
        )
        cache = outputs.past_key_values
        pairs = [(layer[0], layer[1]) for layer in cache]
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        selected.append(next_id.item())
        rows.append({
            "forward": step + 1, "input_ids": current_ids.tolist(),
            "input_shape": list(current_ids.shape), "mask_shape": list(mask.shape),
            "logits_shape": list(outputs.logits.shape),
            "first_K_shape": list(pairs[0][0].shape),
            "first_V_shape": list(pairs[0][1].shape),
            "cache_type": type(cache).__name__, "cache_layers": len(pairs),
            "cache_length_before": before, "cache_length_after": cache.get_seq_length(),
            "cache_dtype": str(pairs[0][0].dtype),
            "cache_tensor_bytes": sum(t.numel() * t.element_size() for pair in pairs for t in pair),
            "next_id": next_id.item(), "next_text": tokenizer.decode([next_id.item()]),
            "visible_sequence_length_after_selection": encoded["input_ids"].shape[1] + len(selected),
        })
        current_ids = next_id
        mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)
    return {
        "text": text, "prompt_ids": encoded["input_ids"][0].tolist(),
        "prompt_tokens_decoded": [tokenizer.decode([value]) for value in encoded["input_ids"][0].tolist()],
        "steps": rows, "selected_ids": selected,
        "new_text": tokenizer.decode(selected, skip_special_tokens=True),
        "cache_length_at_end": cache.get_seq_length(),
        "visible_sequence_length_at_end": encoded["input_ids"].shape[1] + len(selected),
    }


@torch.inference_mode()
def compare_cache(model, tokenizer, text=Q2_TEXT, max_new_tokens=8):
    encoded = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    rows = {}
    for flag in (False, True):
        generation = make_generation_config(model, tokenizer, max_new_tokens, flag)
        output = model.generate(**encoded, generation_config=generation)
        rows[str(flag)] = describe_generation(model, tokenizer, encoded, output, generation)
    return {"text": text, "runs": rows, "ids_equal": rows["False"]["output_ids"] == rows["True"]["output_ids"]}


def cache_memory(model):
    config = model.config
    parameter = next(model.parameters())
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * parameter.element_size()
    cases = []
    for batch, length in [(1, 512), (1, 2048), (2, 2048)]:
        size = per_token * batch * length
        cases.append({"batch_size": batch, "cache_length": length, "bytes": size, "MiB": size / 1024**2})
    parameter_bytes = sum(value.numel() * value.element_size() for value in model.parameters())
    return {
        "layers": config.num_hidden_layers, "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads, "head_dim": head_dim,
        "dtype": str(parameter.dtype), "bytes_per_value": parameter.element_size(),
        "bytes_per_token_per_batch_item": per_token,
        "cases": cases, "parameter_count": sum(value.numel() for value in model.parameters()),
        "parameter_bytes": parameter_bytes, "parameter_GiB": parameter_bytes / 1024**3,
        "scope": "仅参数张量及普通稠密KV张量自身；不是进程总内存或峰值内存",
    }


@torch.inference_mode()
def benchmark_cache(model, tokenizer, text=Q2_TEXT, max_new_tokens=8, repeats=5):
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise ValueError("本课计时限定 CPU；其他设备需另行处理同步和测量方法。")
    encoded = tokenizer(text, return_tensors="pt").to(device)
    width = encoded["input_ids"].shape[1]
    if width + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("输入宽度与新增上限超过本课使用的上下文预算。")
    configs = {flag: make_generation_config(model, tokenizer, max_new_tokens, flag) for flag in (False, True)}
    references = {flag: model.generate(**encoded, generation_config=configs[flag]) for flag in (False, True)}
    if not torch.equal(references[False], references[True]):
        raise RuntimeError("本次缓存开关的输出编号不同，应先核对结果再比较速度。")
    seconds = {False: [], True: []}
    for repeat in range(repeats):
        for flag in ((False, True) if repeat % 2 == 0 else (True, False)):
            start = time.perf_counter()
            output = model.generate(**encoded, generation_config=configs[flag])
            elapsed = time.perf_counter() - start
            if not torch.equal(output, references[flag]):
                raise RuntimeError("同一条件的重复运行输出不同，停止计时比较。")
            seconds[flag].append(elapsed)
    runs = {}
    for flag in (False, True):
        runs[str(flag)] = {
            **describe_generation(model, tokenizer, encoded, references[flag], configs[flag]),
            "seconds": seconds[flag], "median_seconds": statistics.median(seconds[flag]),
        }
    return {
        "text": text, "max_new_tokens": max_new_tokens,
        "warmup_per_condition": 1, "repeats": repeats,
        "clock": "time.perf_counter", "timed_region": "model.generate only; CPU; prefill + decode",
        "order": "alternating False/True then True/False", "ids_equal": True, "all_repeats_equal": True,
        "runs": runs, "observed_speedup": runs["False"]["median_seconds"] / runs["True"]["median_seconds"],
    }


def environment(model):
    parameter = next(model.parameters())
    return {
        "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model_id": MODEL_ID, "revision": REVISION,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ["torch", "transformers", "huggingface_hub", "tokenizers"]},
        "machine": platform.machine(), "device": str(parameter.device), "dtype": str(parameter.dtype),
        "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
        "attention_implementation": model.config._attn_implementation,
        "max_position_embeddings": model.config.max_position_embeddings,
        "eval": not model.training, "inference_mode": torch.is_inference_mode_enabled(),
        "seed": None, "seed_reason": "本次使用贪心，无随机采样",
    }


@torch.inference_mode()
def collect_answers(*, tokenizer=None, model=None):
    torch.set_num_threads(4)
    if tokenizer is None and model is None:
        _, tokenizer, model = load_causal_lm()
    elif tokenizer is None or model is None:
        raise ValueError("复用模型时，tokenizer 与 model 必须同时提供。")
    model.eval()
    return {
        "scope": "C03 Lesson07 四题；固定基础模型，本地缓存、CPU float32；不训练，不测GPU/MPS或进程峰值内存",
        "environment": environment(model),
        "q1": trace_cache(model, tokenizer),
        "q2": compare_cache(model, tokenizer),
        "q3": cache_memory(model),
        "q4": [benchmark_cache(model, tokenizer, max_new_tokens=limit) for limit in (8, 16)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="可选：保存完整 JSON 运行记录")
    args = parser.parse_args()
    answers = collect_answers()
    text = json.dumps(answers, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
