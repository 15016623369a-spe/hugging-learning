#明确模型、版本和运行配置

import time

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

DEVICE = torch.device("cpu")
DTYPE = torch.float32

#读取配置，加载tokenizer和模型

config = AutoConfig.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    local_files_only=True,
)

print("架构类型：", config.model_type)
print("表示宽度", config.hidden_size)
print("层数", config.num_hidden_layers)
print("输出词表维度", config.vocab_size)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
revision=REVISION,
local_files_only=True,
)

load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    local_files_only= True,
    dtype=DTYPE,
    use_safetensors=True,
)

model.to(DEVICE)
model.eval()

print("模型类：", type(model).__name__)
print("训练模式标记：", model.training)
print("加载耗时：", time.perf_counter() - load_start)

#检查模型实际存放的位置和类型
first_parameter = next(model.parameters())

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

parameter_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in model.parameters()
)

print("第一个参数的设备：", first_parameter.device)
print("第一个参数的类型：", first_parameter.dtype)
print("参数总数：", parameter_count)
print("参数存储 GiB：", parameter_bytes / 1024**3)

#编码文本，并移动输出
text = "我想查询订单金额。"

inputs = tokenizer(
    text,
    return_tensors="pt",
)

inputs = {
    name: tensor.to(DEVICE)
    for name, tensor in inputs.items()
}

print("input_ids：", inputs["input_ids"])

for name, tensor in inputs.items():
    print(
        name,
        "shape =", tuple(tensor.shape),
        "dtype =", tensor.dtype,
        "device =", tensor.device,
    )


#完成一次向前推理
forward_start = time.perf_counter()

with torch.inference_mode():
    outputs = model(
        **inputs,
        use_cache=False,
    )

forward_seconds = time.perf_counter() - forward_start

print("输出对象类型：", type(outputs).__name__)
print("logits shape：", tuple(outputs.logits.shape))
print("logits dtype：", outputs.logits.dtype)
print("logits device：", outputs.logits.device)
print("是否记录梯度：", outputs.logits.requires_grad)

print(
    "输出是否全部为有限值：",
    torch.isfinite(outputs.logits).all().item(),
)

print("前向耗时：", forward_seconds)