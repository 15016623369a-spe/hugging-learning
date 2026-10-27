"""助手补充：Lesson04 固定样例复核；用户课堂与四题参考共用一份模型。"""
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
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import torch

from hugging_learn.causal_common import MODEL_ID, REVISION
from hugging_learn.homework.lesson04_answers import collect_answers, greedy_generate

ROOT = Path(__file__).resolve().parent


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

    captured = io.StringIO()
    with redirect_stdout(captured):
        scope = runpy.run_module("hugging_learn.lesson04", run_name="__main__")
    model, tokenizer = scope["model"], scope["tokenizer"]
    parameter = next(model.parameters())
    check("用户课堂入口实际运行", "新增 Token 数： 3" in captured.getvalue())
    check("固定模型结构与配置", type(model).__name__ == "Qwen2ForCausalLM" and scope["config"].hidden_size == 896 and scope["config"].vocab_size == 151936)
    check("模型为 CPU float32 且已 eval", str(parameter.device) == "cpu" and parameter.dtype == torch.float32 and not model.training)
    check("课堂输入九个 Token 且上限为三", scope["prompt_length"] == 9 and scope["max_new_tokens"] == 3 and scope["inputs"]["input_ids"].tolist() == [[14880,100787,104044,105033,106852,9370,50410,59743,80094]])
    check("课堂新增编号与最终长度", scope["new_ids"].tolist() == [[8997,100012,100364]] and list(scope["input_ids"].shape) == [1,12])
    check("课堂 Mask 同步增长且均为有效位置", list(scope["attention_mask"].shape) == [1,12] and bool((scope["attention_mask"] == 1).all()))
    logits = scope["outputs"].logits
    check("最后一次前向使用追加前的长度", list(logits.shape) == [1,11,151936] and list(scope["next_logits"].shape) == [1,151936] and list(scope["next_token"].shape) == [1,1])
    check("课堂输出有限且不记录梯度", bool(torch.isfinite(logits).all()) and not logits.requires_grad)
    check("课堂按新增上限停止且 EOS 来自配置", scope["stop_reason"] == "max_new_tokens" and scope["eos_ids"] == {151643})

    answers = collect_answers(tokenizer=tokenizer, model=model)
    check("第一题与用户课堂实际生成一致", answers["q1"]["new_ids"] == scope["new_ids"][0].tolist() and answers["q1"]["final_shape"] == [1,12] and answers["q1"]["continuation"] == "。\n为了帮助")
    check("第二题三条文本分别重新编码", [r["prompt_length"] for r in answers["q2"]] == [2,5,10])

    rows = [("第一题",answers["q1"])]
    rows += [(f"第二题文本{i}",row) for i,row in enumerate(answers["q2"],1)]
    rows += [(f"第四题上限{limit}",row) for limit,row in answers["q4"]["boundary_runs"].items()]
    for label,row in rows:
        check(label+"：累计长度、次数与新增上限", row["forward_count"] == len(row["new_ids"]) == row["max_new_tokens"] and row["final_shape"] == [1,row["prompt_length"]+len(row["new_ids"])])
        check(label+"：逐步形状、Mask 与输出状态", all(
            step["input_shape_before"] == [1,row["prompt_length"]+index]
            and step["logits_shape"] == [1,row["prompt_length"]+index,151936]
            and step["next_logits_shape"] == [1,151936]
            and step["next_token_shape"] == [1,1]
            and step["input_shape_after"] == step["mask_shape_after"] == [1,row["prompt_length"]+index+1]
            and step["mask_all_ones"] and step["finite_logits"] and not step["logits_requires_grad"]
            for index,step in enumerate(row["steps"])))
        check(label+"：真实样例按长度停止且未生成 EOS", row["stop_reason"] == "max_new_tokens" and not set(row["new_ids"]).intersection(row["eos_ids"]))

    expected = [
        [35946,104044,104014,100681,116280,3837,101885,104854],
        [50410,17992,20412,16,17,18,19,20],
        [220,100012,100364,87026,100787,104044,105033,106852],
    ]
    check("第二题三组固定生成编号", [r["new_ids"] for r in answers["q2"]] == expected)
    check("第三题真实复现 cat 的维数错误", "same number of dimensions" in answers["q3"]["rank_mismatch"]["actual_error"])
    zero = answers["q4"]["boundary_runs"]["0"]
    check("第四题零上限不前向且续写为空", zero["forward_count"] == 0 and zero["new_ids_shape"] == [1,0] and zero["continuation"] == "" and zero["full_text"] == zero["text"])
    synthetic = answers["q4"]["synthetic_eos_decode"]
    check("手工 EOS 解码样例与字符数", synthetic["ids"] == [105033,151643] and synthetic["decoded_without_special"] == "一个月" and synthetic["id_count"] == 2 and synthetic["visible_python_character_count"] == 3)

    # 受控假模型仅用于检查 EOS 分支；不计入 Qwen 自然生成结果。
    class ControlledModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.generation_config = SimpleNamespace(eos_token_id=151643)
            self.calls = 0

        def forward(self, input_ids, attention_mask, **kwargs):
            token = [8997,151643][self.calls]
            self.calls += 1
            scores = torch.full((1,input_ids.shape[1],151936),-1000.0)
            scores[:,-1,token] = 1000.0
            return SimpleNamespace(logits=scores)

    controlled_model = ControlledModel().eval()
    controlled = greedy_generate(controlled_model, tokenizer, "你好。", 8)
    check("受控 EOS：追加结束编号后立即停止", controlled["new_ids"] == [8997,151643] and controlled["stop_reason"] == "eos_token" and controlled_model.calls == 2 and controlled["final_shape"] == [1,4])
    check("受控 EOS：显示时隐藏特殊标记", controlled["continuation"] == "。\n")

    source_paths = ["hugging_learn/lesson04.py","hugging_learn/homework/lesson04_answers.py","hugging_learn/lesson03.py","hugging_learn/lesson02.py","hugging_learn/lesosn01.py"]
    report = {
        "verification_date":args.date,"timezone":"Asia/Shanghai",
        "scope":"固定 Qwen 单句无 PAD、CPU float32、手写贪心循环与四题；另列受控 EOS 分支和手工解码，不训练、不调用 generate",
        "provenance":"用户课堂文件及新增注释保留；助手补充四题参考、验证器和说明；索要答案按约定接受为私下完成确认",
        "model_id":MODEL_ID,"revision":REVISION,
        "python":platform.python_version(),
        "versions":{name:version(name) for name in ["torch","transformers","huggingface_hub","tokenizers"]},
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths},
        "checks_passed":len(checks),"checks":checks,
        "qwen_forward_count":scope["max_new_tokens"]+sum(row["forward_count"] for _,row in rows),
        "classroom":{"text":scope["text"],"prompt_length":scope["prompt_length"],"max_new_tokens":scope["max_new_tokens"],"new_ids":scope["new_ids"][0].tolist(),"final_shape":list(scope["input_ids"].shape),"last_logits_shape":list(logits.shape),"stop_reason":scope["stop_reason"],"stdout":captured.getvalue()},
        "answers":answers,
        "controlled_eos_case":{"provenance":"受控假模型给定两个候选；不是 Qwen 自然生成轨迹","result":controlled},
    }
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"C03 Lesson04：{len(checks)} 项检查通过；课堂与四题共享模型；实际 Qwen 前向 {report['qwen_forward_count']} 次。")


if __name__ == "__main__":
    main()
