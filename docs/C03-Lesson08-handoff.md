# C03 Lesson08 当前交接

更新日期：2026-10-30（Asia/Shanghai）。当前进度以本文件和[通用交接](handoff.md)为准，早期交接为历史快照。

## 当前进度

C02 Lesson01–05按此前交接已完成。C03 Lesson01–07已授课、给过四题参考答案并有固定运行记录；Lesson08已在聊天详细讲授，用户修正三条文本的循环后确认ok，本轮课堂68项检查通过。

用户当前lesson08.py与已有结果逐字节保留。每课仅一个lessonXX.py；没有新增课堂转发入口。基础模型、fde环境和缓存沿用，未训练或切换Instruct权重。

## 本轮关键结果与易错点

三条文本逐条处理，长度2/3/10，每条新增8，完整长度10/11/18；每条做greedy、no_cache、sample_a、sample_b。两种启动方式均通过，缓存开关贪心编号一致，seed=42重设后的采样一致。JSON使用samples保存三条结果。

用户把text改成列表、padding=True后，argmax变为3元素，item报错；只改tolist会接着遇到list.strip。已通过texts列表加for循环解决，run_one保持单条字符串契约。注意右补齐的最后一列可能为PAD，Mask不自动改变最后位置索引。用户也追问过`_, tokenizer, model`的下划线：它接收暂时不用的config，仍是普通变量。

源码SHA256：`48439c36e7093c75845d2fa1864cf00731f6c0ab47cf24e18ca78b9c07b060dd`。本轮不改动代码和注释，用户已有运行结果与助手复核证据分别保存。

## 后续工作

课堂运行已检查；四题要求见[练习记录](C03-Lesson08-homework.md)。第一题三条输入已在课堂执行，验证器补充了错误边界。作业2的4-token缓存组合、作业3的0.5/1.2温度对照、作业4的聊天批量未在本轮执行。后续根据用户请求继续题解、补充实验或模块回顾，不因八节课编号结束就登记独立验收通过，也不自动虚构C04课表。

继续保持详细中文教学：前课动机、术语、具体数字、逐步形状、代码关键行与参数、真实结果、常见错误，最后统一四道作业。用户索要答案按既有约定登记私下完成确认；不另要求截图证明。教师代码从一开始支持编辑器直接启动和模块运行。

用户要求“下一课”只教学；明确要求同步时才更新GitHub，不自动新建聊天。保留用户代码及所有注释，每课仅一份课堂文件。

## 文件与运行

- [课堂](../hugging_learn/lesson08.py)、[用户保存结果](../hugging_learn/lesson08_results.json)、[验证器](../verify_lesson08.py)
- [详细讲义](C03-Lesson08-notes.md)、[四题与范围](C03-Lesson08-homework.md)、[验收与来源](C03-Lesson08-acceptance.md)、[本轮实际检查](C03-Lesson08-verification.json)

```sh
python -m hugging_learn.lesson08
python hugging_learn/lesson08.py
python verify_lesson08.py --report /tmp/c03-lesson08-verification.json
```

仓库：[hugging-learning](https://github.com/15016623369a-spe/hugging-learning)。
