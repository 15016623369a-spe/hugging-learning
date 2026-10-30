# C03 课堂文件整理

日期：2026-10-28（Asia/Shanghai）。用户要求合并重复文件。

| 整理前的课堂文件 | 整理后的唯一课堂文件 |
|---|---|
| `hugging_learn/lesosn01.py` 加转发入口 | [lesson01.py](../hugging_learn/lesson01.py) |
| `hugging_learn/lesoon06.py` 加转发入口 | [lesson06.py](../hugging_learn/lesson06.py) |

完整课堂代码直接放在标准文件中，两个拼写错误的旧文件已移除；原代码与所有注释逐字节保留。作业参考仍单独保存在 homework/。

Lesson01 SHA256：`07206df58b7dcd8415b11664025c51ded13c3d6c1fca4eaa33ef5d803db07c81`。

Lesson06 SHA256：`51db515e0433de6dc2246e2b0bf1da3cf6e899f26f5a5235c6917dfb187ca184`。

README、讲义、验收、交接、课表和验证脚本已更新路径；Lesson06 验证器直接执行课堂文件，取消对转发字典的依赖。Lesson01 的 27 项与 Lesson06 的 40 项检查重新运行，结果写入各课运行 JSON。Lesson03/04 验证器的源码路径同步更新；其已有运行 JSON 是历史证据，其中旧 source_sha256 路径按上表映射读取，原日期和结果不改写。

课堂同时支持 `python -m hugging_learn.lesson01`、`python -m hugging_learn.lesson06` 与编辑器直接运行。后续每课只维护一个 lessonXX.py，新增练习放在对应作业目录，避免重复入口。

## 2026-10-30复核

Lesson01–08各仅一个标准lessonXX.py，课堂中直接包含实现。Lesson08的结果JSON与根目录verify_lesson08.py分别用于记录和检查，不是重复课堂；验证器仅在运行时创建并清理临时目录。此次没有新增Lesson08作业答案入口。
