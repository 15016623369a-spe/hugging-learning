"""助手补充：离线复核原课堂代码、标准入口及四题参考结果。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from importlib.metadata import version

import torch

from hugging_learn.common import MODEL_ID, REVISION, load_tokenizer
from hugging_learn.homework.lesson01_answers import collect_answers

ROOT = Path(__file__).resolve().parent
ORIGINAL_SHA256 = "07206df58b7dcd8415b11664025c51ded13c3d6c1fca4eaa33ef5d803db07c81"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="可选的 JSON 报告输出路径")
    args = parser.parse_args()
    checks = []

    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    source = ROOT / "hugging_learn" / "lesosn01.py"
    check("用户原课堂文件逐字节保留", hashlib.sha256(source.read_bytes()).hexdigest() == ORIGINAL_SHA256)
    output = io.StringIO()
    with redirect_stdout(output):
        scope = runpy.run_path(str(source), run_name="__main__")
    check("原课堂脚本执行通过", bool(output.getvalue()))
    check("课堂单条输入为整数 [1,6]", tuple(scope["single"]["input_ids"].shape) == (1, 6) and scope["single"]["input_ids"].dtype == torch.long)
    batch, fixed = scope["batch"], scope["fixed"]
    check("课堂批量输入与 Mask 均为 [2,16]", tuple(batch["input_ids"].shape) == tuple(batch["attention_mask"].shape) == (2, 16))
    check("课堂有效长度为 [6,16]", batch["attention_mask"].sum(dim=1).tolist() == [6, 16])
    check("课堂固定长度为 [2,10]", tuple(fixed["input_ids"].shape) == tuple(fixed["attention_mask"].shape) == (2, 10))
    check("课堂截断后有效长度为 [6,10]", fixed["attention_mask"].sum(dim=1).tolist() == [6, 10])
    check("Key 允许规则广播形状为 [2,1,1,16]", tuple(scope["key_allowed"].shape) == (2, 1, 1, 16))
    check("MHA 规则屏蔽且只屏蔽补齐位置", torch.equal(scope["mha_key_padding_mask"], batch["attention_mask"].eq(0)))

    tokenizer = load_tokenizer()
    check("分词版本词表与五个特殊 ID 符合课堂约定", tokenizer.vocab_size == 21128 and [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id] == [0, 100, 101, 102, 103])
    answers = collect_answers()
    check("作业1五条分词结果符合真实词表", [x["tokens"] for x in answers["q1"]] == [["订", "单", "金", "额"], ["play", "##ing"], ["[UNK]"], ["t", "##ran", "##s", "##form", "##er"], ["[UNK]"]])
    check("作业1五条 ID 精确匹配", [x["input_ids"] for x in answers["q1"]] == [[6370, 1296, 7032, 7583], [8942, 8221], [100], [162, 10477, 8118, 12725, 8196], [100]])
    check("UNK 是有效位置", answers["q1"][2]["attention_mask"] == answers["q1"][4]["attention_mask"] == [1])
    check("作业2 batch 形状为 [3,11]", answers["q2"]["shape"] == [3, 11])
    check("作业2有效长度为 [4,6,11]", answers["q2"]["valid_lengths"] == [4, 6, 11])
    check("作业3三种形状正确", [x["shape"] for x in answers["q3"]] == [[3, 11], [3, 12], [3, 8]])
    check("作业3三种有效长度正确", [x["valid_lengths"] for x in answers["q3"]] == [[4, 6, 11], [4, 6, 11], [4, 6, 8]])
    check("只补齐不改变文本内容", answers["q3"][0]["decoded"] == answers["q3"][1]["decoded"])
    check("截断保留第三条前六个内容 Token", answers["q3"][2]["decoded"] == ["订 单", "查 询 订 单", "请 查 询 本 月 订"])
    for result in [answers["q2"], *answers["q3"]]:
        ids = torch.tensor(result["input_ids"])
        mask = torch.tensor(result["attention_mask"])
        check("补齐、边界和编号有效：" + str(result["shape"]), bool(torch.all(ids[mask == 0] == tokenizer.pad_token_id)) and bool(torch.all(ids[:, 0] == tokenizer.cls_token_id)) and all(ids[i, n-1].item() == tokenizer.sep_token_id for i, n in enumerate(result["valid_lengths"])) and bool(torch.all((ids >= 0) & (ids < len(tokenizer)))))
    check("作业4有效长度与 MHA 布尔语义正确", answers["q4"]["valid_length"] == 6 and answers["q4"]["mha_key_padding_mask"] == [False]*6 + [True]*2 and not answers["q4"]["is_causal_mask"])

    try:
        tokenizer(["订单", "请查询本月订单金额"], return_tensors="pt")
    except ValueError:
        ragged_rejected = True
    else:
        ragged_rejected = False
    check("不等长序列未补齐时拒绝组成普通二维张量", ragged_rejected)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    standard = subprocess.run([sys.executable, "-m", "hugging_learn.lesson01"], cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=60)
    check("标准入口保留原课堂输出", standard.stdout == output.getvalue())
    homework = subprocess.run([sys.executable, "-m", "hugging_learn.homework.lesson01_answers"], cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=60)
    check("参考答案入口可独立运行", json.loads(homework.stdout) == answers)

    report = {
        "verification_date": "2026-10-24",
        "timezone": "Asia/Shanghai",
        "scope": "C03 Lesson01 固定 Tokenizer 与 CPU 样例；不加载模型、不训练、不调用付费 API",
        "provenance": "原课堂文件保留；作业参考与复核脚本为助手补充；索要答案按约定接受为用户私下完成确认，运行通过不等同于独立实现证明",
        "python": platform.python_version(),
        "versions": {n: version(n) for n in ["torch", "transformers", "huggingface_hub", "tokenizers"]},
        "model_id": MODEL_ID,
        "revision": REVISION,
        "original_sha256": ORIGINAL_SHA256,
        "checks_passed": len(checks),
        "checks": checks,
        "answers": answers,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"C03 Lesson01：{len(checks)} 项检查通过；全程使用本地分词缓存。")


if __name__ == "__main__":
    main()
