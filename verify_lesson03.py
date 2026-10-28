"""助手补充：离线复核 Lesson03 用户课堂文件与四道参考答案。"""
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
from hugging_learn.homework.lesson03_answers import collect_answers

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--date", default=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(), help="报告日期，默认使用运行时上海日期")
    args = parser.parse_args()
    checks = []

    def check(name, passed):
        if not bool(passed):
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    source = ROOT / "hugging_learn/lesson03.py"
    captured = io.StringIO()
    with redirect_stdout(captured):
        scope = runpy.run_path(str(source), run_name="__main__")
    config, tokenizer, model = scope["config"], scope["tokenizer"], scope["model"]
    inputs, outputs = scope["inputs"], scope["outputs"]
    logits = outputs.logits
    check("用户课堂入口实际执行", "手动输出头与模型输出的最大差异" in captured.getvalue())
    check("模型结构符合固定版本", type(model).__name__ == "Qwen2ForCausalLM" and config.hidden_size == 896 and config.vocab_size == 151936)
    parameter = next(model.parameters())
    check("课堂模型为 CPU float32 且已 eval", str(parameter.device) == "cpu" and parameter.dtype == torch.float32 and not model.training)
    check("无句号课堂文本编码为九个固定编号", scope["text"] == "请统计最近一个月各部门的订单总金额" and inputs["input_ids"].dtype == torch.int64 and inputs["input_ids"].tolist() == [[14880,100787,104044,105033,106852,9370,50410,59743,80094]])
    check("课堂输入与 Mask 为 [1,9] 且没有 PAD", list(inputs["input_ids"].shape) == list(inputs["attention_mask"].shape) == [1,9] and inputs["attention_mask"].tolist() == [[1]*9])
    check("最终隐藏表示与层数正确", list(scope["hidden"].shape) == [1,9,896] and len(outputs.hidden_states) == 25)
    check("词表输出头形状与 bias 正确", list(scope["head"].weight.shape) == [151936,896] and scope["head"].bias is None)
    check("手动输出头还原实际 logits", torch.allclose(scope["rebuilt_logits"], logits, rtol=1e-5, atol=1e-5))
    check("课堂 logits 形状、类型和有限值正确", list(logits.shape) == [1,9,151936] and logits.dtype == torch.float32 and torch.isfinite(logits).all() and not logits.requires_grad)
    check("最后位置切片正确", list(scope["next_logits"].shape) == [1,151936] and torch.equal(scope["next_logits"],logits[:,-1,:]))
    probs = scope["probs"]
    check("完整词表 float64 概率归一化", list(probs.shape) == [1,151936] and probs.dtype == torch.float64 and torch.isfinite(probs).all() and torch.all(probs >= 0) and torch.allclose(probs.sum(dim=-1),torch.ones(1,dtype=torch.float64),rtol=1e-10,atol=1e-10))
    check("前五候选对应原词表概率且有序", list(scope["top_ids"].shape) == [1,5] and torch.equal(probs.gather(-1,scope["top_ids"]),scope["top_probs"]) and torch.all(scope["top_probs"][:,1:] <= scope["top_probs"][:,:-1]))
    check("logits、概率与 top1 的最高分编号一致", scope["next_id"].shape == (1,) and torch.equal(scope["next_id"],probs.argmax(dim=-1)) and torch.equal(scope["next_id"],scope["top_ids"][:,0]))

    with torch.inference_mode():
        prefix = {name: tensor[:,:2] for name,tensor in inputs.items()}
        prefix_logits = model(**prefix,use_cache=False).logits[:,-1,:]
    prefix_difference = (prefix_logits-logits[:,1,:]).abs().max().item()
    # T=2 与 T=9 的 float32/SDPA 运算形状不同；实际最大误差约 2.18e-4。
    check("不同长度前缀结果在 float32 容差内一致", torch.allclose(prefix_logits,logits[:,1,:],rtol=1e-4,atol=5e-4))
    with torch.inference_mode():
        changed_tail = {name:tensor.clone() for name,tensor in inputs.items()}
        changed_tail["input_ids"][:,2:] = tokenizer.encode("你好。",add_special_tokens=False)[0]
        changed_logits = model(**changed_tail,use_cache=False).logits[:,:2,:]
    suffix_difference = (changed_logits-logits[:,:2,:]).abs().max().item()
    check("同长度改变后续 Token 不影响前两个位置", torch.allclose(changed_logits,logits[:,:2,:],rtol=1e-5,atol=1e-5))

    answers = collect_answers(config=config,tokenizer=tokenizer,model=model)
    q1=answers["q1"]
    check("作业1的七组形状正确", all(q1[name]==expected for name,expected in {
        "input_shape":[1,2],"hidden_shape":[1,2,896],"logits_shape":[1,2,151936],
        "next_logits_shape":[1,151936],"top_ids_shape":[1,5],"top_probs_shape":[1,5],"next_ids_shape":[1],
    }.items()))
    check("作业2三条文本长度为 2、5、10", [r["token_count"] for r in answers["q2"]] == [2,5,10])
    for index,(row,length) in enumerate(zip(answers["q2"],[2,5,10]),start=1):
        check(f"作业2文本{index}：输入、隐藏和输出形状", row["input_shape"]==[1,length] and row["hidden_shape"]==[1,length,896] and row["logits_shape"]==[1,length,151936])
        check(f"作业2文本{index}：整数输入、有限输出与无梯度", row["input_dtype"]=="torch.int64" and row["logits_dtype"]=="torch.float32" and row["finite_logits"] and not row["logits_requires_grad"])
        check(f"作业2文本{index}：全词表与前五项概率", abs(row["probability_sum"]-1)<1e-10 and 0<row["top5_probability_sum"]<1 and abs(sum(c["probability"] for c in row["top5"])-row["top5_probability_sum"])<1e-12)
    check("作业2最高分编号符合三个固定样例", [r["top5"][0]["token_id"] for r in answers["q2"]]==[35946,50410,220])
    q3=answers["q3"]
    check("作业3完整 softmax 数值正确", torch.allclose(torch.tensor(q3["full_probabilities"],dtype=torch.float64),torch.tensor([0.8668133321973347,0.11731042782619835,0.015876239976466762],dtype=torch.float64),rtol=1e-10,atol=1e-12))
    check("作业3最高分索引为 0", q3["argmax_index"]==0)
    check("作业3原始前两项概率合计正确", abs(q3["top2_original_probability_sum"]-0.9841237600235331)<1e-12)
    check("作业3重新归一化的结果与原始概率不同", torch.allclose(torch.tensor(q3["softmax_of_first_two_scores"],dtype=torch.float64),torch.tensor([0.8807970779778823,0.11920292202211755],dtype=torch.float64),rtol=1e-10,atol=1e-12) and abs(q3["renormalized_sum"]-1)<1e-12)

    top5 = [{"token_id":i,"raw_token":tokenizer.convert_ids_to_tokens(i),"decoded":tokenizer.decode([i],skip_special_tokens=False),"probability":p} for i,p in zip(scope["top_ids"][0].tolist(),scope["top_probs"][0].tolist())]
    flat_index=logits.argmax().item()
    flat_position=[int(i) for i in torch.unravel_index(torch.tensor(flat_index),logits.shape)]
    report={
        "verification_date":args.date,"timezone":"Asia/Shanghai",
        "scope":"C03 Lesson03 用户课堂、前缀对照与四题参考；本地缓存、CPU；不训练、不执行生成循环",
        "provenance":"用户 Lesson03 文件逐字节保留；Lesson02 用户缩进改动保留；参考答案、验证器和说明为助手补充；索要答案按约定接受为私下完成确认",
        "python":platform.python_version(),
        "versions":{n:version(n) for n in ["torch","transformers","huggingface_hub","tokenizers"]},
        "model_id":MODEL_ID,"revision":REVISION,
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,ROOT/"hugging_learn/lesson02.py",ROOT/"hugging_learn/lesson01.py"]},
        "checks_passed":len(checks),"checks":checks,
        "classroom":{
            "text":scope["text"],"input_ids":inputs["input_ids"].tolist(),
            "input_shape":list(inputs["input_ids"].shape),"hidden_shape":list(scope["hidden"].shape),
            "head_weight_shape":list(scope["head"].weight.shape),"logits_shape":list(logits.shape),
            "head_max_abs_difference":scope["difference"],"probability_sum":probs.sum().item(),
            "top5_probability_sum":scope["top_probs"].sum().item(),"top5":top5,
            "greedy_id":scope["chosen_id"],"prefix_max_abs_difference":prefix_difference,
            "prefix_tolerance":{"rtol":1e-4,"atol":5e-4},
            "changed_suffix_max_abs_difference":suffix_difference,
            "flat_argmax_index":flat_index,"flat_argmax_position_b_t_v":flat_position,
            "stdout":captured.getvalue(),
        },
        "answers":answers,
    }
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"C03 Lesson03：{len(checks)} 项检查通过；课堂 T=9，作业 T=[2,5,10]；同一模型复用。")


if __name__=="__main__":
    main()
