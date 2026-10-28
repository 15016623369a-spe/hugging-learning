"""助手补充：执行标准课堂入口，并复用同一模型复核 Lesson06 四题。"""
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
from hugging_learn.homework.lesson06_answers import collect_answers

ROOT = Path(__file__).resolve().parent
CLASSROOM_HASH = "51db515e0433de6dc2246e2b0bf1da3cf6e899f26f5a5235c6917dfb187ca184"


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

    output = io.StringIO()
    with redirect_stdout(output):
        scope = runpy.run_module("hugging_learn.lesson06", run_name="__main__")
    model, tokenizer = scope["model"], scope["tokenizer"]
    parameter = next(model.parameters())
    check("Lesson06单一课堂文件实际运行", scope["__file__"] == str(ROOT/"hugging_learn/lesson06.py") and sorted((ROOT/"hugging_learn").glob("*06.py")) == [ROOT/"hugging_learn/lesson06.py"] and "普通文本批量生成" in output.getvalue())
    check("用户课堂字节和注释保留", hashlib.sha256((ROOT/"hugging_learn/lesson06.py").read_bytes()).hexdigest() == CLASSROOM_HASH)
    check("沿用固定Qwen基础模型和CPU float32", type(model).__name__ == "Qwen2ForCausalLM" and scope["config"].hidden_size == 896 and parameter.device.type == "cpu" and parameter.dtype == torch.float32)
    check("课堂处于eval并显式贪心/无缓存", not model.training and scope["common"]["do_sample"] is False and scope["common"]["use_cache"] is False and scope["common"]["max_new_tokens"] == 8)
    check("课堂左补齐、PAD和EOS配置", tokenizer.padding_side == "left" and tokenizer.pad_token_id == tokenizer.eos_token_id == scope["common"]["eos_token_id"] == 151643)
    check("课堂普通批次形状与有效长度", list(scope["text_inputs"]["input_ids"].shape) == [3,9] and scope["text_inputs"]["attention_mask"].sum(dim=1).tolist() == [2,5,9])
    check("课堂普通批次固定续写", scope["text_replies"] == ["我最近总是感觉头晕，而且有时候","订单号是12345","。\n为了帮助您统计最近一个月各部门"])
    check("课堂聊天批次与有效长度", list(scope["chat_inputs"]["input_ids"].shape) == [2,43] and scope["chat_inputs"]["attention_mask"].sum(dim=1).tolist() == [27,43])
    check("课堂实际编码下一轮历史", list(scope["next_inputs"]["input_ids"].shape) == [1,50] and [message["role"] for message in scope["history"]] == ["system","user","assistant","user"])

    answers = collect_answers(tokenizer=tokenizer, model=model)
    q1,q2,q3,q4 = (answers[key] for key in ["q1","q2","q3","q4"])
    check("第一题真实长度与共同宽度", q1["real_lengths"] == [1,3,10] and q1["input_width"] == 10 and q1["input_shape"] == [3,10])
    check("第一题各行补齐数量", q1["padding_counts"] == [9,7,0])
    check("第一题Mask为连续左0右1", all(mask == [0]*pads+[1]*length for mask,pads,length in zip(q1["attention_mask"],q1["padding_counts"],q1["real_lengths"])))
    check("第一题只有被标为PAD的位置使用补齐且正文可还原", all(all(token == tokenizer.pad_token_id for token,mask in zip(ids,masks) if not mask) and tokenizer.decode([token for token,mask in zip(ids,masks) if mask],skip_special_tokens=True) == text for ids,masks,text in zip(q1["input_ids"],q1["attention_mask"],q1["texts"])))
    check("第二题批量与逐条编号逐项相同", q2["ids_equal"] == [True,True,True])
    check("第二题四步新增与完整形状", q2["batch"]["new_shape"] == [3,4] and q2["batch"]["output_shape"] == [3,14])
    expected = [[50292,17714,16,15],[198,51154,50410,89656],[220,100012,100364,87026]]
    check("第二题固定新增编号", q2["batch"]["new_ids"] == expected)
    check("第二题错误切片保留订单正文", q2["wrong_slice"]["start"] == 1 and q2["wrong_slice"]["decoded"] == "订单价为10" and q2["correct_slice"]["start"] == 10 and q2["correct_slice"]["decoded"] == "价为10")

    for label,row in [("第二题批量",q2["batch"]), *[(f"第二题逐条{i}",row) for i,row in enumerate(q2["individual"],1)], ("第三题聊天",q3["batch"])]:
        width = row["input_width"]
        check(label+"：完整输出保留共同宽度的输入前缀", row["prefix_preserved"] and all(complete[:width] == ids and complete[width:] == new for complete,ids,new in zip(row["output_ids"],row["input_ids"],row["new_ids"])))
        check(label+"：本次均按长度停止且没有EOS", all(reason == "length_limit" for reason in row["stop_reasons"]) and all(len(ids) == row["options"]["max_new_tokens"] and not set(ids).intersection(row["eos_ids"]) for ids in row["new_ids"]) and row["generated_counts_including_eos"] == [row["options"]["max_new_tokens"]]*len(row["new_ids"]))

    comparisons = q3["template_comparisons"]
    check("第三题外层两个聊天、内层分别2/4条消息", len(q3["chats"]) == 2 and [len(chat) for chat in q3["chats"]] == [2,4])
    check("第三题生成开头恰好多3个token", [(row["length_without_prompt"],row["length_with_prompt"]) for row in comparisons] == [(24,27),(40,43)] and all(row["appended_text"] == "<|im_start|>assistant\n" for row in comparisons))
    check("第三题True一步与False后单独编码逐编号一致", all(row["two_paths_equal"] and row["one_step_ids"] == row["two_step_ids"] for row in comparisons))
    check("第三题批量补齐后有效编号等于逐条模板编码", all([token for token,mask in zip(ids,masks) if mask] == comparison["one_step_ids"] for ids,masks,comparison in zip(q3["batch"]["input_ids"],q3["batch"]["attention_mask"],comparisons)))
    check("第三题批量形状/新增/完整宽度", q3["batch"]["input_shape"] == [2,43] and q3["batch"]["padding_counts"] == [16,0] and q3["batch"]["new_shape"] == [2,8] and q3["batch"]["output_shape"] == [2,51])
    check("第三题聊天续写与课堂相同", q3["batch"]["continuations"] == scope["chat_replies"])
    check("第四题追加角色并保留原来的两条消息", q4["original_message_count"] == 2 and q4["new_message_count"] == 4 and q4["history"][:2] == q3["chats"][0] and [message["role"] for message in q4["history"]] == ["system","user","assistant","user"])
    check("第四题历史保存实际短续写", q4["history"][2]["content"] == q3["batch"]["continuations"][0] and q4["history"][3]["content"] == "再举一个例子。")
    check("第四题下一轮仅编码且输入从27增长到50", q4["original_input_length"] == 27 and q4["next_input"]["input_shape"] == [1,50] and q4["next_input"]["real_lengths"] == [50] and q4["next_turn_generated"] is False)
    example = q4["synthetic_eos_example"]
    check("第四题手工EOS与PAD的存储计数", example["ids"] == [8997,100012,151643,151643,151643] and example["stored_columns"] == 5 and example["generated_count_including_eos"] == 3 and example["trailing_padding_count"] == 2)
    check("第四题解码只影响显示", example["decode_without_special"] == "。\n为了" and example["decode_with_special"].count("<|endoftext|>") == 3)
    check("模板消息结束编号与模型EOS不同", answers["special_token_ids"]["<|im_end|>"] == 151645 and model.generation_config.eos_token_id == 151643)
    check("所有模型参数未产生梯度", all(parameter.grad is None for parameter in model.parameters()))

    sources = ["hugging_learn/lesson06.py","hugging_learn/homework/lesson06_answers.py","verify_lesson06.py","hugging_learn/lesson05.py","hugging_learn/causal_common.py"]
    report = {
        "verification_date":args.date, "timezone":"Asia/Shanghai",
        "provenance":"课堂统一为标准文件名，原代码与注释逐字节保留；答案、验证器与文档为助手补充；索要答案按约定登记私下完成确认，未另验收独立能力",
        "scope":answers["scope"],"model_id":MODEL_ID,"revision":REVISION,
        "python":platform.python_version(),
        "versions":{name:version(name) for name in ["torch","transformers","huggingface_hub","tokenizers"]},
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        "checks_passed":len(checks),"checks":checks,
        "classroom":{"file":"hugging_learn/lesson06.py","options":scope["common"],"text_input_shape":list(scope["text_inputs"]["input_ids"].shape),"text_replies":scope["text_replies"],"chat_input_shape":list(scope["chat_inputs"]["input_ids"].shape),"chat_replies":scope["chat_replies"],"next_input_shape":list(scope["next_inputs"]["input_ids"].shape),"stdout":output.getvalue()},
        "answers":answers,
    }
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"C03 Lesson06：{len(checks)} 项检查通过；标准入口与四题共用一份模型。")


if __name__ == "__main__":
    main()
