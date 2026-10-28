# 3. 生成参数保持一致，仅改变缓存开关
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers import GenerationConfig

# 兼容编辑器直接运行文件和 python -m 模块运行。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hugging_learn.causal_common import load_causal_lm


# 1. 固定 CPU 计算线程数，加载模型
torch.set_num_threads(4)

config, tokenizer, model = load_causal_lm()
model.eval()

device = next(model.parameters()).device
assert device.type == "cpu"

text = "请用一句话解释 attention_mask 的作用。"
inputs = tokenizer(
    text,
    return_tensors="pt",
).to(device)


# 2. 第一次输入完整提示词
current_ids = inputs["input_ids"]
mask = inputs["attention_mask"]
cache = None

with torch.inference_mode():
    for step in range(3):
        outputs = model(
            input_ids=current_ids,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
        )

        cache = outputs.past_key_values

        # 当前环境的缓存接口：
        # 每层前两项分别是 K、V
        first_layer = next(iter(cache))
        k, v = first_layer[:2]

        print(f"\n第 {step + 1} 次前向")
        print("本次输入：", current_ids.shape)
        print("完整 Mask：", mask.shape)
        print("logits：", outputs.logits.shape)
        print("第一层 K：", k.shape)
        print("第一层 V：", v.shape)
        print("缓存长度：", cache.get_seq_length())

        # 取最后一个位置，选择分数最大的编号
        next_id = outputs.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        print("选出的编号：", next_id.item())

        # 下一次只输入新选出的一个 token
        current_ids = next_id

        # Mask 则继续覆盖完整上下文
        mask = torch.cat(
            [mask, torch.ones_like(next_id)],
            dim=1,
        )

del outputs, cache, first_layer, k, v

eos = model.generation_config.eos_token_id
if eos is None:
    eos = tokenizer.eos_token_id

generation_configs = {
    flag: GenerationConfig(
        max_new_tokens=8,
        do_sample=False,
        num_beams=1,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=flag,
    )
    for flag in (False, True)
}


@torch.inference_mode()
def generate(encoded, use_cache):
    return model.generate(
        **encoded,
        generation_config=generation_configs[use_cache],
    )


# 4. 两种输入长度分别比较
cases = [
    ("短输入", text),
    ("较长输入", (text + "。") * 8),
]

for name, prompt in cases:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    width = encoded["input_ids"].shape[1]

    assert width + generation_configs[True].max_new_tokens <= config.max_position_embeddings

    # 各预热一次，预热不计入正式耗时
    references = {
        flag: generate(encoded, flag)
        for flag in (False, True)
    }

    # 先确认本次缓存开关没有改变输出编号
    assert torch.equal(
        references[False],
        references[True],
    )

    samples = {False: [], True: []}

    # 每种条件正式测量五次
    for repeat in range(5):
        order = (
            (False, True)
            if repeat % 2 == 0
            else (True, False)
        )

        for flag in order:
            start = time.perf_counter()
            output = generate(encoded, flag)
            elapsed = time.perf_counter() - start

            assert torch.equal(output, references[flag])
            samples[flag].append(elapsed)

    print(f"\n{name}，输入长度：{width}")

    for flag in (False, True):
        print(
            "use_cache =", flag,
            "五次耗时 =", samples[flag],
            "中位数 =", statistics.median(samples[flag]),
        )

    new_ids = references[True][0, width:]
    print("新增编号：", new_ids.tolist())
    print(
        "新增文本：",
        tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        ),
    )
