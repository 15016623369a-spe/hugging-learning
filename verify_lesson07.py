"""C03 Lesson07：实际执行用户课堂，复用同一模型复核四题与运行记录。"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path
import platform
import runpy
import statistics
from zoneinfo import ZoneInfo

import torch

from hugging_learn.causal_common import MODEL_ID, REVISION
from hugging_learn.homework.lesson07_answers import Q2_TEXT, collect_answers

ROOT = Path(__file__).resolve().parent
CLASSROOM_SHA256 = "b628be3eba3db1a3715fc9fc518d5b0c0a88f033d6168facbfdc5e893145f69d"
EXPECTED_Q2 = [6529, 9999, 54851, 46944, 100751, 18493, 104949, 15946]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--date", default=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat())
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    source = ROOT / "hugging_learn/lesson07.py"
    check("Lesson07仅有一个课堂文件且源码符合本轮检查快照", sorted((ROOT/"hugging_learn").glob("*07.py")) == [source] and hashlib.sha256(source.read_bytes()).hexdigest() == CLASSROOM_SHA256)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        scope = runpy.run_module("hugging_learn.lesson07", run_name="__main__")
    model, tokenizer = scope["model"], scope["tokenizer"]
    parameter = next(model.parameters())
    check("用户课堂实际运行且保留当前attention_mask题目", scope["text"] == Q2_TEXT and "第 3 次前向" in stdout.getvalue() and "较长输入" in stdout.getvalue())
    check("沿用固定Qwen基础模型CPU float32与eval", type(model).__name__ == "Qwen2ForCausalLM" and parameter.device.type == "cpu" and parameter.dtype == torch.float32 and not model.training)
    check("课堂新增上限均为8且显式对照缓存", all(scope["generation_configs"][flag].max_new_tokens == 8 and scope["generation_configs"][flag].use_cache == flag for flag in (False, True)))
    check("课堂原始提示词9token且模型上下文配置32768", list(scope["inputs"]["input_ids"].shape) == [1,9] and model.config.max_position_embeddings == 32768)
    check("课堂两组重复计时实际完成", all(len(scope["samples"][flag]) == 5 for flag in (False, True)) and stdout.getvalue().count("五次耗时 =") == 4)

    answers = collect_answers(tokenizer=tokenizer, model=model)
    q1, q2, q3, q4 = (answers[key] for key in ["q1", "q2", "q3", "q4"])
    check("第一题分成订单和金额两个token", q1["prompt_ids"] == [50410,80094] and q1["prompt_tokens_decoded"] == ["订单","金额"])
    check("第一题三次选出的编号与实际续写", q1["selected_ids"] == [100381,2130,23305] and q1["new_text"] == "超过____元")
    expected_inputs = [[[50410,80094]], [[100381]], [[2130]]]
    for index, row in enumerate(q1["steps"]):
        width = 2 if index == 0 else 1
        length = 2 + index
        check(f"第一题第{index+1}次：仅输入尚未处理的token与对应logits", row["input_ids"] == expected_inputs[index] and row["input_shape"] == [1,width] and row["logits_shape"] == [1,width,151936])
        check(f"第一题第{index+1}次：Mask与缓存历史连续增长", row["mask_shape"] == [1,length] and row["cache_length_before"] == (0 if index == 0 else length-1) and row["cache_length_after"] == length)
        check(f"第一题第{index+1}次：24层真实KV形状与字节", row["cache_type"] == "DynamicCache" and row["cache_layers"] == 24 and row["first_K_shape"] == row["first_V_shape"] == [1,2,length,64] and row["cache_dtype"] == "torch.float32" and row["cache_tensor_bytes"] == 24576*length)
    check("第一题最后已选择5个位置但仅4个位置进入缓存", q1["visible_sequence_length_at_end"] == 5 and q1["cache_length_at_end"] == 4)

    check("第二题缓存开关的完整编号逐项一致", q2["ids_equal"] and q2["runs"]["False"]["output_ids"] == q2["runs"]["True"]["output_ids"])
    for flag, row in q2["runs"].items():
        check(f"第二题cache={flag}：固定新增编号与形状", row["input_shape"] == [1,9] and row["output_shape"] == [1,17] and row["new_shape"] == [1,8] and row["new_ids"] == EXPECTED_Q2)
        check(f"第二题cache={flag}：贪心、前缀保留与真实停止原因", row["prefix_preserved"] and row["options"]["use_cache"] == (flag == "True") and row["options"]["do_sample"] is False and row["options"]["num_beams"] == 1 and row["stop_reason"] == "length_limit" and row["generated_count_including_eos"] == 8)

    check("第三题使用2个KV头而非14个Q头", (q3["layers"],q3["query_heads"],q3["kv_heads"],q3["head_dim"],q3["bytes_per_value"]) == (24,14,2,64,4))
    check("第三题每条每token占24KiB", q3["bytes_per_token_per_batch_item"] == 24576)
    check("第三题三个答案为12/48/96MiB", [row["MiB"] for row in q3["cases"]] == [12,48,96] and [row["bytes"] for row in q3["cases"]] == [12582912,50331648,100663296])
    check("第三题估算与第一题实际KV张量一致", all(row["cache_tensor_bytes"] == q3["bytes_per_token_per_batch_item"]*row["cache_length_after"] for row in q1["steps"]))
    check("参数张量实际数量与字节统计", q3["parameter_count"] == 494032768 and q3["parameter_bytes"] == 1976131072)

    check("第四题覆盖同一输入的8和16token上限", [item["max_new_tokens"] for item in q4] == [8,16] and all(item["text"] == Q2_TEXT for item in q4))
    for item in q4:
        limit = item["max_new_tokens"]
        check(f"第四题N={limit}：每组预热1次、重复5次、编号一致", item["warmup_per_condition"] == 1 and item["repeats"] == 5 and item["ids_equal"] and item["all_repeats_equal"])
        for flag, row in item["runs"].items():
            check(f"第四题N={limit},cache={flag}：原始耗时与中位数", len(row["seconds"]) == 5 and all(math.isfinite(value) and value > 0 for value in row["seconds"]) and row["median_seconds"] == statistics.median(row["seconds"]))
            check(f"第四题N={limit},cache={flag}：输出编号和长度", row["input_ids"] == q2["runs"][flag]["input_ids"] and row["prefix_preserved"] and row["new_ids"][:8] == EXPECTED_Q2 and len(row["new_ids"]) == limit and row["output_shape"] == [1,9+limit] and row["stop_reason"] == "length_limit")
        check(f"第四题N={limit}：速度比按本次中位数计算", item["observed_speedup"] == item["runs"]["False"]["median_seconds"] / item["runs"]["True"]["median_seconds"])
    env = answers["environment"]
    check("复现记录包含实际模型版本、CPU、精度与线程", env["model_id"] == MODEL_ID and env["revision"] == REVISION and env["device"] == "cpu" and env["dtype"] == "torch.float32" and env["num_threads"] == 4 and env["eval"] and env["inference_mode"])
    check("所有参数没有产生梯度", all(value.grad is None for value in model.parameters()))

    paths = ["hugging_learn/lesson07.py", "hugging_learn/homework/lesson07_answers.py", "verify_lesson07.py", "hugging_learn/causal_common.py"]
    report = {
        "verification_date": args.date, "timezone": "Asia/Shanghai",
        "provenance": "保留用户课堂代码与全部注释，仅将写死16的上下文预算改为实际生成上限；四题参考、验证器和文档为助手补充；索要答案按既有约定登记私下完成确认，未另验收独立能力",
        "model_id": MODEL_ID, "revision": REVISION,
        "python": platform.python_version(),
        "versions": {name:version(name) for name in ["torch","transformers","huggingface_hub","tokenizers"]},
        "source_sha256": {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
        "checks_passed": len(checks), "checks": checks,
        "classroom": {"file":"hugging_learn/lesson07.py", "text":scope["text"], "max_new_tokens":8, "stdout":stdout.getvalue()},
        "answers": answers,
        "limits": "仅CPU单条输入、固定贪心；不把缓存K/V估算当进程峰值，不对GPU/MPS、训练或答案质量作结论；耗时不设速度门槛",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"C03 Lesson07：{len(checks)} 项检查通过；课堂与四题复用一份本地模型。")


if __name__ == "__main__":
    main()
