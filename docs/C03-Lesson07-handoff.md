# C03 Lesson07 → Lesson08 交接

更新日期：2026-10-28（Asia/Shanghai）。当前接续以本文件与[通用交接](handoff.md)为准，早期交接为历史快照。

## 当前进度

C02 Lesson01–05按此前交接已完成。C03 Lesson01–07已授课、四题答案已给出并有各课运行证据；用户索要答案按既有约定登记私下完成确认，未另要求截图。Lesson07本轮43项固定检查通过。

下一次用户说“下一课”进入 **Lesson08：综合实验与模块验收**。Lesson08尚未开始。

## 用户约定

- 聊天是主要课堂。讲解详细：接续前课与动机、解释术语、具体数字推导、逐步张量形状、每段目的、关键代码和参数、实际输出与易错点，最后统一四道作业。
- 每课仅保留一个lessonXX.py；作业参考在homework/，不再添加同课转发入口。用户原代码与全部注释保留，必要修复明确记录。
- 教师代码从一开始就兼容编辑器直接启动与模块运行：Path/sys先导入，直接运行时把项目根目录加入sys.path；不再漏掉Lesson07曾出现的包路径兼容。
- “下一课”只教学；用户明确要求同步时再更新代码与GitHub，不自动新建聊天。
- 用户完成确认、助手参考、实际运行与独立实现分别登记。

## Lesson07关键实测

模型Qwen/Qwen2.5-0.5B基础版，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；CPU float32、4计算线程、eval、inference_mode，沿用fde与本地缓存。

用户课堂现为attention_mask解释题，9-token输入，新增8；另有72-token重复输入计时。唯一修复为上下文预算从固定+16改为读取实际新增上限，原注释全部保留。

第一题“订单金额”=[50410,80094]，三次本次输入[1,2]/[1,1]/[1,1]，Mask长度2/3/4，K/V每层[1,2,S,64]，24层缓存长度2/3/4。选出[100381,2130,23305]即“超过____元”。最后选出的“元”还未前向，所以可见长度5、缓存4。用户已追问过第二次为什么只输入一个token，要继续明确current_ids=next_id与past_key_values的配合。

第二题attention_mask提示词9token，缓存开关新增8编号一致；完整[1,17]。第三题每条每token24KiB，三组12/48/96MiB；参数约1.840GiB，实际dtype为float32，14Q/2KV使用GQA。

第四题同一9-token输入分别新增8/16，缓存各预热1次、重复5次、交替顺序、perf_counter计时generate，保存原始耗时、中位数与输出。完整[1,17]/[1,25]；本次都按长度停止，未出现EOS。速度只对本次CPU环境成立，不设固定加速断言；没有测GPU/MPS或峰值内存。

## 文件与运行

- [课堂](../hugging_learn/lesson07.py)、[四题完整代码](../hugging_learn/homework/lesson07_answers.py)、[验证器](../verify_lesson07.py)。
- [讲义](C03-Lesson07-notes.md)、[详细题解](C03-Lesson07-homework.md)、[验收](C03-Lesson07-acceptance.md)、[实际运行JSON](C03-Lesson07-verification.json)。

```sh
python -m hugging_learn.lesson07
python -m hugging_learn.homework.lesson07_answers --report /tmp/c03-lesson07-answers.json
python verify_lesson07.py --report /tmp/c03-lesson07-verification.json
```

本地 `/Users/jace/agent学习/hugging-learning`；公开[GitHub仓库](https://github.com/15016623369a-spe/hugging-learning)。

## Lesson08建议接续

把文本编码、模型加载、设备和dtype、生成参数、停止条件、续写切片、缓存选择及运行记录串成一个最小可复现程序。先明确实验目标与输入输出，再逐段实现；至少包含贪心基线、一个受控对照、输出编号核对和可保存记录。

继续区分输入宽度W与真实长度、完整输出与续写、EOS与PAD、模板格式与基础/指令权重、缓存中的历史与应用messages。沿用固定基础模型，不默认升级环境或下载Instruct权重。最后完成C03模块回顾，结合用户实际表现验收，不因课程编号直接登记独立掌握。
