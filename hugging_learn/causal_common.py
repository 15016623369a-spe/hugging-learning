"""助手补充：Lesson02 的固定模型配置；默认只读本地缓存。"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DEVICE = torch.device("cpu")
DTYPE = torch.float32


def load_causal_lm():
    """加载同一 revision 的配置、分词器和预训练权重，进入评估模式。"""
    options = dict(revision=REVISION, local_files_only=True, token=False)
    try:
        config = AutoConfig.from_pretrained(MODEL_ID, **options)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=DTYPE, use_safetensors=True, **options
        )
    except OSError as exc:
        raise RuntimeError(
            "缺少本课模型缓存。首次请联网运行："
            "python -m hugging_learn.download_model"
        ) from exc
    model.to(DEVICE)
    model.eval()
    return config, tokenizer, model
