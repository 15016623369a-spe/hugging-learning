"""助手补充：首次联网缓存分词文件，不加载模型权重。"""
from hugging_learn.common import MODEL_ID, REVISION, load_tokenizer


def main():
    tokenizer = load_tokenizer(local_files_only=False)
    print("Tokenizer 已缓存：", MODEL_ID)
    print("固定版本：", REVISION)
    print("词表大小：", tokenizer.vocab_size)


if __name__ == "__main__":
    main()
