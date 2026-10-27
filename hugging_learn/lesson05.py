from pathlib import Path
import sys

import torch
from transformers import set_seed

# 兼容编辑器直接运行和按模块运行。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hugging_learn.causal_common import load_causal_lm
from hugging_learn.homework.lesson04_answers import greedy_generate


# 1. 加载模型和编码输入
config, tokenizer, model = load_causal_lm()
model.eval()

text = "请统计最近一个月各部门的订单总金额"
inputs = tokenizer(text, return_tensors="pt")

device = next(model.parameters()).device
inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

prompt_length = inputs["input_ids"].shape[1]

print("输入形状：", inputs["input_ids"].shape)
print("模型自带生成配置：", model.generation_config)


# 2. 读取停止标记和补齐标记
eos = model.generation_config.eos_token_id
if eos is None:
    eos = tokenizer.eos_token_id

pad = tokenizer.pad_token_id
if pad is None:
    pad = eos if isinstance(eos, int) else eos[0]


# 3. 两种生成方式共用的配置
common = dict(
    max_new_tokens=8,
    num_beams=1,
    use_cache=False,
    eos_token_id=eos,
    pad_token_id=pad,
    return_dict_in_generate=False,
)


# 4. 上一课的手写贪心循环
manual = greedy_generate(
    model,
    tokenizer,
    text,
    max_new_tokens=8,
)


# 5. generate 的贪心生成
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        **common,
        do_sample=False,
    )

new_ids = output_ids[:, prompt_length:]

assert torch.equal(
    output_ids[:, :prompt_length],
    inputs["input_ids"],
)

same = new_ids[0].tolist() == manual["new_ids"]
assert same

print("\n与手写循环一致：", same)
print("完整输出形状：", output_ids.shape)
print("新增编号：", new_ids[0].tolist())
print(
    "新增文本：",
    repr(tokenizer.decode(new_ids[0], skip_special_tokens=True)),
)


# 6. 采样：比较相同种子与不同种子
sample_results = []

for seed in [42, 42, 43]:
    # 每次生成前重新设置种子。
    set_seed(seed)
    #这个**是模型解包的意思
    #eg：my_dict = {"name": "小明", "age": 18}
    #def introduce(name, age):
    #print(f"我是{name}，今年{age}岁")
    #直接传字典introduce(my_dict)会报错
    #但可以introduce(**my_dict)
    with torch.inference_mode():
        sampled_ids = model.generate(
            **inputs,#相当于把这俩字典里的内容直接“平铺”写进了 generate 函数里。
            **common,
            do_sample=True,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
        )

    sampled_new_ids = sampled_ids[:, prompt_length:]
    token_ids = sampled_new_ids[0].tolist()
    sample_results.append(token_ids)

    print(f"\n采样 seed={seed}")
    print("新增编号：", token_ids)
    print(
        "新增文本：",
        repr(
            tokenizer.decode(
                sampled_new_ids[0],
                skip_special_tokens=True,
            )
        ),
    )

print(
    "\n相同种子的两次结果一致：",
    sample_results[0] == sample_results[1],
)