"""助手补充：离线执行用户课堂文件和 Lesson02 作业，记录真实结果。"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import runpy
from zoneinfo import ZoneInfo

import torch

from hugging_learn.causal_common import MODEL_ID, REVISION
from hugging_learn.homework.lesson02_answers import collect_answers

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="可选的 JSON 输出路径")
    args = parser.parse_args()
    checks = []

    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    source = ROOT / "hugging_learn/lesson02.py"
    captured = io.StringIO()
    with redirect_stdout(captured):
        scope = runpy.run_path(str(source), run_name="__main__")
    model, config, tokenizer = scope["model"], scope["config"], scope["tokenizer"]
    check("用户课堂文件完成实际前向", "logits shape" in captured.getvalue())
    check("课堂固定模型和 revision", scope["MODEL_ID"] == MODEL_ID and scope["REVISION"] == REVISION)
    check("课堂文本 ID 与固定版本一致", scope["inputs"]["input_ids"].tolist() == [[104100, 51154, 50410, 80094, 1773]])
    check("课堂 logits 为 [1,5,151936]", list(scope["outputs"].logits.shape) == [1, 5, 151936])
    check("课堂输出有限且未记录梯度", bool(torch.isfinite(scope["outputs"].logits).all()) and not scope["outputs"].logits.requires_grad)
    answers = collect_answers(config=config, tokenizer=tokenizer, model=model)
    q1 = answers["q1"]
    check("模型架构与输出头正确", q1["model_class"] == "Qwen2ForCausalLM" and q1["config_class"] == "Qwen2Config")
    check("结构配置符合固定权重", [q1[k] for k in ("hidden_size", "num_hidden_layers", "vocab_size")] == [896, 24, 151936])
    check("参数量符合实际加载", q1["parameter_count"] == 494032768)
    check("CPU float32 与 eval 状态正确", q1["device"] == "cpu" and q1["dtype"] == "torch.float32" and not q1["training"])
    check("分词器编号空间可被输出词表覆盖", q1["tokenizer_length"] == 151665 and q1["tokenizer_length"] <= q1["vocab_size"])
    # 固定样例的 T 是独立预期，不只检查两处 shape 相互一致。
    expected_lengths = [2, 5, 10]
    check("三条文本真实 Token 长度符合预期", [x["token_count"] for x in answers["q2"]] == expected_lengths)
    for index, (row, length) in enumerate(zip(answers["q2"], expected_lengths), start=1):
        check(f"文本{index}：输入与输出形状", row["input_shape"] == [1, length] and row["logits_shape"] == [1, length, 151936])
        check(f"文本{index}：编号、Mask 与设备类型", row["input_dtype"] == "torch.int64" and row["input_device"] == "cpu" and row["attention_mask"] == [[1] * length] and all(0 <= token < q1["vocab_size"] for token in row["input_ids"][0]))
        check(f"文本{index}：实际输出有限且无梯度", row["finite"] and not row["requires_grad"] and row["logits_dtype"] == "torch.float32" and row["logits_device"] == "cpu")
    q3 = answers["q3"]
    check("eval 保留参数梯度开关", not q3["layer_training"] and not q3["input_requires_grad"] and q3["weight_requires_grad"])
    check("eval 普通前向仍记录梯度", q3["ordinary_requires_grad"] and q3["ordinary_grad_fn"] != "NoneType")
    check("inference_mode 禁用输出梯度记录", not q3["inference_requires_grad"] and q3["inference_grad_fn"] is None)
    check("Linear 两种模式数值一致", q3["outputs_equal"])

    # 两个常见错误在现有模型上实际触发，不加载第二份大模型。
    inputs = scope["inputs"]
    bad = dict(inputs)
    bad["input_ids"] = bad["input_ids"].float()
    try:
        with torch.inference_mode():
            model(**bad, use_cache=False)
    except (RuntimeError, TypeError) as exc:
        float_error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("浮点 input_ids 应被拒绝")
    check("浮点 input_ids 实际被拒绝", bool(float_error))
    try:
        with torch.inference_mode():
            model(inputs, use_cache=False)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        dict_error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("未展开的输入字典应被拒绝")
    check("未展开字典实际被拒绝", bool(dict_error))

    report = {
        "verification_date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "timezone": "Asia/Shanghai",
        "scope": "C03 Lesson02；本地缓存、CPU float32、用户课堂代码与四题参考；不训练、不生成、不评估生成质量",
        "provenance": "用户课堂文件仅规范模型名及两处打印文字；参考答案与验证器为助手补充；索要答案按约定接受为私下完成确认",
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "transformers", "huggingface_hub", "tokenizers")},
        "model_id": MODEL_ID,
        "revision": REVISION,
        "lesson02_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checks_passed": len(checks),
        "checks": checks,
        "classroom": {
            "input_shape": list(scope["inputs"]["input_ids"].shape),
            "logits_shape": list(scope["outputs"].logits.shape),
            "seconds_before_first_forward": scope["forward_start"] - scope["load_start"],
            "forward_seconds": scope["forward_seconds"],
            "stdout": captured.getvalue(),
        },
        "answers": answers,
        "demonstrated_errors": {"float_input_ids": float_error, "unexpanded_inputs": dict_error},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"C03 Lesson02：{len(checks)} 项检查通过；全程使用本地缓存，模型只加载一次。")


if __name__ == "__main__":
    main()
