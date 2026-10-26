"""助手补充：首次联网准备 Lesson02 固定版本文件，约 1 GB 权重。"""
import os

# 本机此前使用普通 HTTPS 完成下载；在导入 Hub 之前设置。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from hugging_learn.causal_common import MODEL_ID, REVISION


def main():
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        token=False,
        allow_patterns=[
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt",
            "model.safetensors",
        ],
    )
    print("Lesson02 模型文件已准备：", path)
    print("日常课堂与作业入口只读取缓存，不重新下载。")


if __name__ == "__main__":
    main()
