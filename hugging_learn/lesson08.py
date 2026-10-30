"""C03 Lesson08：可运行、可核对、可保存记录的文本生成程序。"""
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo

import torch
from transformers import GenerationConfig

# 编辑器直接运行时，先让 Python 找到项目根目录。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hugging_learn.causal_common import MODEL_ID, REVISION, load_causal_lm


@torch.inference_mode()
def run_one(model, tokenizer, text, *, max_new_tokens=8, use_cache=True,
            do_sample=False, seed=None, temperature=0.8):
    """单条普通文本；新增上限必须是正整数。"""
    if not text.strip():
        raise ValueError("输入文本不能为空。")

    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须是正整数。")

    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    width = inputs["input_ids"].shape[1]

    if width + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("输入宽度与新增上限超过上下文预算。")

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id

    options = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=1,
        use_cache=use_cache,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=False,
    )

    if do_sample:
        options.update(
            temperature=temperature,
            top_k=20,
            top_p=0.9,
        )

        if seed is not None:
            torch.manual_seed(seed)

    generation = GenerationConfig(**options)

    output_ids = model.generate(
        **inputs,
        generation_config=generation,
    )

    # 完整输出应保留输入前缀。
    assert torch.equal(
        output_ids[:, :width],
        inputs["input_ids"],
    )

    # 只取输入之后的新增部分。
    new_ids = output_ids[:, width:]
    ids = new_ids[0].tolist()

    eos_ids = (
        [] if eos is None
        else [eos] if isinstance(eos, int)
        else list(eos)
    )
    has_eos = any(value in eos_ids for value in ids)

    return {
        "text": text,
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist(),
        "input_shape": list(inputs["input_ids"].shape),
        "output_ids": output_ids.tolist(),
        "output_shape": list(output_ids.shape),
        "new_ids": ids,
        "new_shape": list(new_ids.shape),
        "new_text": tokenizer.decode(
            ids,
            skip_special_tokens=True,
        ),
        "stop_reason": "eos_token" if has_eos else "max_new_tokens",
        "options": options,
        "seed": seed if do_sample else None,
    }


def main():
    torch.set_num_threads(4)

    # 模型只加载一次，后面各组共用。
    _, tokenizer, model = load_causal_lm()
    model.eval()

    parameter = next(model.parameters())
    texts = [
        "订单金额",
        "查询订单金额",
        "请统计最近一个月各部门的订单总金额。",
    ]

    samples = []

    for index, text in enumerate(texts, start=1):
        # 先观察隐藏表示、logits 和第一次贪心选择。
        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(parameter.device)

        with torch.inference_mode():
            out = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            last_logits = out.logits[:, -1, :]

            probe = {
                "hidden_shape": list(out.hidden_states[-1].shape),
                "logits_shape": list(out.logits.shape),
                "last_logits_shape": list(last_logits.shape),
                "greedy_first_id": last_logits.argmax(dim=-1).item(),
            }

        del out, last_logits

        # 四组实验使用同一模型和同一条输入。
        runs = {
            "greedy": run_one(model, tokenizer, text),
            "no_cache": run_one(
                model, tokenizer, text,
                use_cache=False,
            ),
            "sample_a": run_one(
                model, tokenizer, text,
                do_sample=True,
                seed=42,
            ),
            "sample_b": run_one(
                model, tokenizer, text,
                do_sample=True,
                seed=42,
            ),
        }

        checks = {
            "first_id_matches_logits": (
                runs["greedy"]["new_ids"][0] == probe["greedy_first_id"]
            ),
            "cache_ids_equal": (
                runs["greedy"]["new_ids"] == runs["no_cache"]["new_ids"]
            ),
            "seeded_sample_ids_equal": (
                runs["sample_a"]["new_ids"] == runs["sample_b"]["new_ids"]
            ),
        }

        assert all(checks.values()), checks

        samples.append({
            "text": text,
            "probe": probe,
            "checks": checks,
            "runs": runs,
        })

        print(f"\n第 {index} 条输入：{text}")
        print("前向形状与第一步选择：", probe)

        for name, row in runs.items():
            print(f"\n{name}")
            print(
                "输入/完整输出/新增：",
                row["input_shape"],
                row["output_shape"],
                row["new_shape"],
            )
            print("新增编号：", row["new_ids"])
            print("新增文本：", repr(row["new_text"]))
            print("停止原因：", row["stop_reason"])

        print("\n核对结果：", checks)

    record = {
        "recorded_at": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in [
                "torch", "transformers",
                "huggingface_hub", "tokenizers",
            ]
        },
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "num_threads": torch.get_num_threads(),
        "attention_implementation": model.config._attn_implementation,
        "max_position_embeddings": model.config.max_position_embeddings,
        "samples": samples,
    }

    report_path = Path(__file__).with_name("lesson08_results.json")

    report_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n记录文件：", report_path)


if __name__ == "__main__":
    main()