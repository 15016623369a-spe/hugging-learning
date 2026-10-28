#加载tokenizer
from transformers import AutoTokenizer

MODEL_ID = "google-bert/bert-base-chinese"
REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    local_files_only=True,
)

tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

print("词表大小：", tokenizer.vocab_size)
print("特殊 Token：", tokenizer.special_tokens_map)

#观察token、id、解码
text = "查询订单"

tokens = tokenizer.tokenize(text)
plain_ids = tokenizer.encode(text, add_special_tokens=False)
input_ids = tokenizer.encode(text, add_special_tokens=True)

print("Token：", tokens)
print("未加特殊标记的 ID：", plain_ids)
print("加入特殊标记的 ID：", input_ids)

print("ID 对应的 Token：",
      tokenizer.convert_ids_to_tokens(input_ids))

print("保留特殊标记：",
      tokenizer.decode(input_ids, skip_special_tokens=False))

print("移除特殊标记：",
      tokenizer.decode(input_ids, skip_special_tokens=True))

for word in ["playing", "transformer", "Transformer"]:
    print(word, "→", tokenizer.tokenize(word))

#单条输入和批量补齐
single = tokenizer("查询订单", return_tensors="pt")

print("单条 input_ids：", single["input_ids"])
print("单条 shape：", single["input_ids"].shape)
print("单条 dtype：", single["input_ids"].dtype)

texts = [
    "查询订单",
    "我想查询最近一个月的订单金额",
]

batch = tokenizer(
    texts,
    padding=True,
    return_tensors="pt",
)

print("批量 input_ids：", batch["input_ids"])
print("批量 attention_mask：", batch["attention_mask"])
print("批量 shape：", batch["input_ids"].shape)

valid_lengths = batch["attention_mask"].sum(dim=1)
print("有效长度：", valid_lengths.tolist())

#固定长度，并观察阶段损失
fixed = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=10,
    return_tensors="pt",
)

print("shape：", fixed["input_ids"].shape)
print("Mask：", fixed["attention_mask"])

print(
    "有效长度：",
    fixed["attention_mask"].sum(dim=1).tolist(),
)

decoded = tokenizer.batch_decode(
    fixed["input_ids"],
    skip_special_tokens=True,
)

for text in decoded:
    print("解码结果：", text)

#把有效位置标记接回c02
mask = batch["attention_mask"]       # [B,T]

valid_positions = mask.bool()        # 1 → True，0 → False

# 提供 Key 维的允许规则，供注意力分数广播使用
key_allowed = valid_positions[:, None, None, :]

# PyTorch MHA 的 key_padding_mask：True 表示屏蔽
mha_key_padding_mask = ~valid_positions

print("有效位置 shape：", valid_positions.shape)
print("Key 允许规则 shape：", key_allowed.shape)
print("MHA 屏蔽规则第一行：",
      mha_key_padding_mask[0].tolist())