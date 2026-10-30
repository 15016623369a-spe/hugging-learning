"""C03 Lesson08：隔离运行两种入口，核对三条输入和四组实验，不覆盖课堂记录。"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer

from hugging_learn.causal_common import MODEL_ID, REVISION

ROOT = Path(__file__).resolve().parent
TEXTS = ["订单金额", "查询订单金额", "请统计最近一个月各部门的订单总金额。"]


def semantic_record(record):
    """两次运行时间不同，其余记录应相同。"""
    return {key: value for key, value in record.items() if key != "recorded_at"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    source = ROOT / "hugging_learn/lesson08.py"
    check("每课仅一个标准课堂文件", all(
        sorted((ROOT / "hugging_learn").glob(f"*{index:02d}.py"))
        == [ROOT / f"hugging_learn/lesson{index:02d}.py"]
        for index in range(1, 9)
    ))

    spec = importlib.util.spec_from_file_location("checked_lesson08", source)
    classroom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classroom)
    check("导入课堂可复用run_one而不执行main", callable(classroom.run_one) and callable(classroom.main))
    for label, text, options in [
        ("空字符串", "", {}), ("纯空白", "  ", {}),
        ("零上限", "订单金额", {"max_new_tokens": 0}),
        ("负上限", "订单金额", {"max_new_tokens": -1}),
        ("非整数上限", "订单金额", {"max_new_tokens": 1.5}),
    ]:
        try:
            classroom.run_one(None, None, text, **options)
        except ValueError:
            check(f"{label}在加载模型前得到明确错误", True)
        else:
            raise AssertionError(label)

    records, executions = {}, {}
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="c03-lesson08-") as temporary:
        work = Path(temporary)
        package = work / "hugging_learn"
        package.mkdir()
        for name in ["__init__.py", "causal_common.py", "lesson08.py"]:
            shutil.copy2(ROOT / "hugging_learn" / name, package / name)
        elsewhere = work / "elsewhere"
        elsewhere.mkdir()
        commands = {
            "direct_file_outside_project": ([sys.executable, str(package / "lesson08.py")], elsewhere),
            "module_from_project_root": ([sys.executable, "-m", "hugging_learn.lesson08"], work),
        }
        for mode, (command, cwd) in commands.items():
            completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=240)
            if completed.returncode:
                raise RuntimeError(f"{mode}\n{completed.stdout}\n{completed.stderr}")
            check(f"{mode}实际执行成功", True)
            records[mode] = json.loads((package / "lesson08_results.json").read_text(encoding="utf-8"))
            executions[mode] = {
                "returncode": completed.returncode,
                "recorded_at": records[mode]["recorded_at"],
                "stdout": completed.stdout,
                "semantic_sha256": hashlib.sha256(json.dumps(
                    semantic_record(records[mode]), ensure_ascii=False, sort_keys=True
                ).encode()).hexdigest(),
            }

    record = records["direct_file_outside_project"]
    check("直接运行与模块运行全部语义记录一致", semantic_record(record) == semantic_record(records["module_from_project_root"]))
    saved_path = ROOT / "hugging_learn/lesson08_results.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    check("用户已有JSON可解析且与本轮真实运行相符", semantic_record(saved) == semantic_record(record))
    check("沿用固定模型CPU float32四线程", record["model_id"] == MODEL_ID and record["revision"] == REVISION
          and record["device"] == "cpu" and record["dtype"] == "torch.float32" and record["num_threads"] == 4)
    check("上下文与版本记录齐全", record["max_position_embeddings"] == 32768 and record["attention_implementation"] == "sdpa"
          and all(record["versions"].get(name) for name in ["torch", "transformers", "huggingface_hub", "tokenizers"]))
    check("三条课堂输入完整保留", [sample["text"] for sample in record["samples"]] == TEXTS)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True, token=False
    )
    for index, (sample, length) in enumerate(zip(record["samples"], [2, 3, 10], strict=True), start=1):
        probe, runs = sample["probe"], sample["runs"]
        check(f"第{index}条隐藏表示和logits形状", probe["hidden_shape"] == [1, length, 896]
              and probe["logits_shape"] == [1, length, 151936] and probe["last_logits_shape"] == [1, 151936])
        check(f"第{index}条第一步编号与原始logits一致", runs["greedy"]["new_ids"][0] == probe["greedy_first_id"])
        check(f"第{index}条缓存开关编号一致", runs["greedy"]["new_ids"] == runs["no_cache"]["new_ids"])
        check(f"第{index}条每次重设种子后采样一致", runs["sample_a"]["new_ids"] == runs["sample_b"]["new_ids"])
        check(f"第{index}条课堂三项核对真实通过", len(sample["checks"]) == 3 and all(sample["checks"].values()))
        check(f"第{index}条包含四组实验", set(runs) == {"greedy", "no_cache", "sample_a", "sample_b"})
        for name, row in runs.items():
            check(f"第{index}条{name}前缀与新增切片", row["input_shape"] == [1, length]
                  and row["attention_mask"] == [[1] * length]
                  and row["output_ids"][0][:length] == row["input_ids"][0]
                  and row["output_ids"][0][length:] == row["new_ids"]
                  and row["output_shape"] == [1, length + len(row["new_ids"])] and row["new_shape"] == [1, len(row["new_ids"])])
            options = row["options"]
            sampling = name.startswith("sample")
            check(f"第{index}条{name}生成参数受控", options["max_new_tokens"] == 8 and options["num_beams"] == 1
                  and options["use_cache"] == (name != "no_cache") and options["do_sample"] == sampling
                  and options["return_dict_in_generate"] is False
                  and row["seed"] == (42 if sampling else None)
                  and (not sampling or (options["temperature"], options["top_k"], options["top_p"]) == (0.8, 20, 0.9)))
            check(f"第{index}条{name}解码与实际停止原因", row["new_text"] == tokenizer.decode(row["new_ids"], skip_special_tokens=True)
                  and len(row["new_ids"]) == 8 and options["eos_token_id"] not in row["new_ids"]
                  and row["stop_reason"] == "max_new_tokens")

    report = {
        "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai",
        "provenance": "用户修改后的Lesson08课堂及已有结果逐字节保留；助手在隔离临时目录实际执行两种入口，补充验证器和Markdown。用户说ok了，不登记为四道作业或模块独立验收全部完成。",
        "checks_passed": len(checks), "checks": checks,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            "hugging_learn/lesson08.py", "hugging_learn/lesson08_results.json", "hugging_learn/causal_common.py", "verify_lesson08.py"
        ]},
        "executions": executions,
        "classroom": record,
        "limits": "固定CPU与已缓存基础模型；三条文本依次运行，每条四组实验。未验证作业2的4/8上限全组合、作业3的0.5/1.2温度对照、作业4的聊天批量；不作跨环境随机一致性、生成质量或模块独立能力结论。",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"C03 Lesson08：{len(checks)}项检查通过；两种入口、三条文本、每条四组实验。")


if __name__ == "__main__":
    main()
