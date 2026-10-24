"""助手补充的标准拼写入口；原课堂文件 lesosn01.py 保留原样。"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("lesosn01.py")), run_name="__main__")
