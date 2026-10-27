"""助手补充：实际执行用户课堂，再复用模型检查 Lesson05 四题。"""
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
from hugging_learn.homework.lesson05_answers import collect_answers

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
        scope = runpy.run_module("hugging_learn.lesson05", run_name="__main__")
    model, tokenizer = scope["model"], scope["tokenizer"]
    parameter = next(model.parameters())
    check("用户课堂实际运行且原有断言通过", scope["same"] and "相同种子的两次结果一致： True" in captured.getvalue())
    check("固定模型、设备精度与 eval", type(model).__name__ == "Qwen2ForCausalLM" and parameter.device.type == "cpu" and parameter.dtype == torch.float32 and not model.training)
    check("课堂输入九个 token，显式上限八且关闭缓存", scope["prompt_length"] == 9 and scope["common"]["max_new_tokens"] == 8 and not scope["common"]["use_cache"])
    check("课堂模型自带新增上限为2048", model.generation_config.max_new_tokens == 2048)
    expected_greedy = [8997,100012,100364,87026,100787,104044,105033,106852]
    check("课堂贪心编号与手写循环逐编号相同", scope["new_ids"][0].tolist() == scope["manual"]["new_ids"] == expected_greedy)
    check("课堂完整输出包含原输入且为17个token", list(scope["output_ids"].shape) == [1,17] and torch.equal(scope["output_ids"][:,:9],scope["inputs"]["input_ids"]))
    check("课堂两次seed42相同", scope["sample_results"][0] == scope["sample_results"][1])
    answers = collect_answers(tokenizer=tokenizer, model=model)
    q1, q2, q3, q4 = (answers[name] for name in ["q1","q2","q3","q4"])
    check("第一题三步手写与generate编号一致", q1["ids_equal"] and q1["generate"]["new_ids"] == [8997,100012,100364])
    check("第一题输入/新增/总形状", q1["generate"]["input_shape"] == [1,9] and q1["generate"]["new_shape"] == [1,3] and q1["generate"]["output_shape"] == [1,12])
    check("第二题各文本独立编码且长度正确", [row["prompt_length"] for row in q2["new_token_limits"]] == [2,2,9,9])
    check("第二题新增上限分别为4和8", [len(row["new_ids"]) for row in q2["new_token_limits"]] == [4,8,4,8])
    check("第二题完整长度随输入变化", [row["output_shape"][1] for row in q2["new_token_limits"]] == [6,10,13,17])
    check("第二题同一max_length12对应新增10和3", [len(row["new_ids"]) for row in q2["total_length_limits"]] == [10,3] and all(row["output_shape"] == [1,12] and row["options"]["max_new_tokens"] is None for row in q2["total_length_limits"]))
    expected42 = [8997,30534,100787,104044,105033,106852,9370,50410]
    expected43 = [3837,23031,99667,17714,75317,198,103942,73670]
    check("第三题重设种子的结果与课堂一致", [row["new_ids"] for row in q3["reset_each_time"]] == scope["sample_results"] == [expected42,expected42,expected43])
    check("第三题只设一次种子的首轮与重设首轮一致", q3["seed_once"][0]["new_ids"] == expected42)
    check("第三题只设一次的后续两轮本次结果发生变化", all(row["new_ids"] != expected42 for row in q3["seed_once"][1:]))

    generated_rows = [q1["generate"], *q2["new_token_limits"], *q2["total_length_limits"], *q3["reset_each_time"], *q3["seed_once"]]
    for index, row in enumerate(generated_rows, 1):
        budget = row["options"]["max_new_tokens"]
        if budget is None:
            budget = row["options"]["max_length"] - row["prompt_length"]
        check(f"实际生成{index}：前缀、形状与停止条件", row["prefix_preserved"] and row["output_shape"] == [1,row["prompt_length"]+len(row["new_ids"])] and row["new_shape"] == [1,len(row["new_ids"])] and len(row["new_ids"]) <= budget and row["stop_reason"] == "length_limit" and not set(row["new_ids"]).intersection(row["eos_ids"]))

    temps = q4["temperature_runs"]
    check("第四题温度升高使首项概率降低", temps[0]["final_probabilities"][0] > temps[1]["final_probabilities"][0] > temps[2]["final_probabilities"][0])
    check("第四题所有正温度保留最高分编号", all(max(range(5),key=lambda i:row["final_probabilities"][i]) == 0 for row in temps))
    check("第四题top_k2保留A/B", q4["top_k_2"]["retained_indices"] == [0,1])
    check("第四题top_p09保留A/B/C", q4["top_p_09"]["retained_indices"] == [0,1,2])
    check("第四题组合k2与p09保留A/B", q4["combined_k2_p09"]["retained_indices"] == [0,1])
    check("第四题用户所选0.85/4/0.9保留A/B", q4["user_selected_controls"]["retained_indices"] == [0,1])
    toy_rows = temps + [q4[name] for name in ["top_k_2","top_p_09","combined_k2_p09","user_selected_controls"]]
    check("第四题过滤后概率和为1且被过滤项为0", all(abs(sum(row["final_probabilities"])-1) < 1e-12 and all(p == 0 for i,p in enumerate(row["final_probabilities"]) if i not in row["retained_indices"]) for row in toy_rows))
    check("当前generate的零上限真实抛出ValueError", answers["zero_new_tokens_observation"]["type"] == "ValueError" and "greater than 0" in answers["zero_new_tokens_observation"]["message"])
    classroom_path = ROOT/"hugging_learn/lesson05.py"
    check("用户课堂注释及全部字节保留", hashlib.sha256(classroom_path.read_bytes()).hexdigest() == "fdb29b74c3c2536321a120b1d667b978e31e31c521bdc037204100d06e7577c2")
    sources = ["hugging_learn/lesson05.py","hugging_learn/homework/lesson05_answers.py","verify_lesson05.py","hugging_learn/lesson04.py","hugging_learn/causal_common.py"]
    report = {
        "verification_date":args.date, "timezone":"Asia/Shanghai",
        "provenance":"用户明确确认完成；原课堂文件逐字节保留；助手补充四题参考、验证器与文档；未另行验证用户独立能力",
        "scope":answers["scope"], "model_id":MODEL_ID, "revision":REVISION,
        "python":platform.python_version(),
        "versions":{name:version(name) for name in ["torch","transformers","huggingface_hub","tokenizers"]},
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        "checks_passed":len(checks), "checks":checks,
        "classroom":{"text":scope["text"],"prompt_length":scope["prompt_length"],"common_options":scope["common"],"greedy_new_ids":scope["new_ids"][0].tolist(),"sample_results":scope["sample_results"],"stdout":captured.getvalue()},
        "answers":answers,
    }
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"C03 Lesson05：{len(checks)} 项检查通过；课堂与四题共用一份模型。")


if __name__ == "__main__":
    main()
