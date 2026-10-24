"""助手补充：统一 Tokenizer 配置；日常运行默认只读本地缓存。"""
from transformers import AutoTokenizer

MODEL_ID = "google-bert/bert-base-chinese"
REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"


def load_tokenizer(*, local_files_only=True):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            local_files_only=local_files_only,
            token=False,
        )
    except OSError as exc:
        if local_files_only:
            raise RuntimeError(
                "未找到本课分词缓存。请在仓库根目录先运行："
                "python -m hugging_learn.download_tokenizer"
            ) from exc
        raise
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer
