"""标准拼写入口；执行用户原课堂 lesoon06.py，保留其原名与全部注释。"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    classroom = runpy.run_path(str(Path(__file__).with_name("lesoon06.py")), run_name="__main__")
