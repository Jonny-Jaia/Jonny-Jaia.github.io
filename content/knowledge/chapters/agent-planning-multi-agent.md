# Agent 规划执行与 Multi-Agent：任务分解、反思和协作控制

## 当前定位

Planning、Reflection 和 Multi-Agent 解决的是 Agent 在复杂任务中的控制问题：如何拆解目标、选择工具、执行步骤、观察结果、修正计划，并在必要时引入多个角色协作。面试中要避免把它讲成“多开几个模型”，而要讲清 **状态、预算、检查点、失败恢复和角色边界**。

> **面试抓手**：规划让 Agent 知道先做什么、后做什么；反思让 Agent 在失败后修正；Multi-Agent 让不同角色并行或互审。但这些能力都会增加成本和不确定性，必须配合预算、停止条件和评估机制。

```archify
Agent Runtime Architecture|assets/diagrams/html/agent-runtime-architecture.html
```

## 一、Plan-and-Execute 怎么讲

```mermaid
flowchart LR
  n1["Goal"] --> n2["Draft plan"]
  n2 --> n3["Execute step"]
  n3 --> n4["Observe result"]
  n4 --> n5["Update state"]
  n5 --> n6["Replan or stop"]
```

| 元素 | 说明 | 常见 bug |
|---|---|---|
| task decomposition | 把目标拆成子任务 | 子任务太粗，无法验证 |
| dependency | 哪些步骤依赖前置结果 | 顺序错导致重复工作 |
| tool assignment | 每步需要什么工具 | 工具选择错误或过度调用 |
| success criteria | 如何判断这步完成 | 没有检查点，任务越走越偏 |
| stop condition | 什么时候停止 | 无限循环或过早结束 |
| budget | token、时间、工具调用次数 | 成本不可控 |

**结论**：Planning 的价值不是让模型“想得更长”，而是把复杂任务变成可检查、可回滚、可预算控制的执行链路。

## 二、常见规划范式

| 范式 | 适用场景 | 优势 | 局限 |
|---|---|---|---|
| ReAct | 工具反馈密集任务 | 边做边观察，适合搜索和调试 | 推理链容易冗长，错误 observation 会污染后续 |
| Plan-and-Execute | 中等复杂任务 | 结构清晰，便于检查状态 | 初始计划可能不准确 |
| Replanning | 环境动态变化 | 失败后能修正路径 | 需要维护任务状态和变更原因 |
| Tree of Thoughts | 多路径搜索、推理题 | 可比较候选路径 | 成本高，树宽和深度难控 |
| Least-to-Most | 复杂问题逐步拆解 | 降低一次性推理压力 | 子问题划分质量决定上限 |
| Reflexion | 有可验证反馈的任务 | 可以把失败经验转成下一轮提示 | 自评不可靠，最好有外部验证器 |

面试中可以把这些范式统一到一个框架：**状态管理 + 候选生成 + 工具执行 + 结果评估 + 策略更新**。

## 三、Reflection 如何避免自嗨

Reflection 有效的前提是有可靠反馈。只让模型自我批评，容易生成看似合理但不可验证的解释。更稳的做法是引入外部反馈：

| 反馈来源 | 例子 | 价值 |
|---|---|---|
| 单元测试 / 编译 | 代码 Agent | 明确 pass/fail |
| 检索证据 | RAG / Web Agent | 检查事实依据 |
| 环境状态 | 浏览器、数据库、机器人仿真 | 判断动作是否真的生效 |
| 规则验证器 | 格式、权限、安全规则 | 稳定、低成本 |
| 人工或 LLM judge | 主观质量评估 | 灵活但成本高、有偏差 |

Reflection 的输出最好结构化：失败类型、证据、下一步修正、是否需要回滚、是否值得继续消耗预算。

## 四、Multi-Agent 什么时候值得用

| 模式 | 角色 | 适合任务 | 风险 |
|---|---|---|---|
| Planner-Executor | planner 拆任务，executor 执行 | 长任务、工具调用 | planner 和 executor 状态不一致 |
| Specialist Team | coder、retriever、reviewer、tester | 软件工程、研究分析 | 协调成本高 |
| Debate / Critic | 多个候选互评 | 开放问题、方案选择 | 可能互相强化错误 |
| Supervisor-Workers | supervisor 分配任务并汇总 | 并行搜索、批量处理 | supervisor 成为瓶颈 |
| Verifier Agent | 独立检查结果 | AppDev、Web、代码修复 | 验证成本高，judge 偏差 |

**使用判断**：如果任务可并行、需要不同专业能力、需要独立审查，Multi-Agent 值得考虑；如果只是简单问答或状态共享困难，多 Agent 只会增加延迟和成本。

## 五、工程控制点

| 控制点 | 为什么重要 | 实现方式 |
|---|---|---|
| shared state | 多角色必须看到一致事实 | state store、event log、artifact store |
| message protocol | 防止角色间传递噪声 | typed message、任务状态字段 |
| budget controller | 控制 token、工具、时间成本 | max turns、max tool calls、deadline |
| progress check | 判断是否真的推进 | 每步记录 changed files、new evidence、test result |
| conflict resolution | 多个角色结论冲突 | priority、verifier、human escalation |
| audit log | 复盘失败和训练数据回流 | trace、tool args、observation、decision |

## 六、和 Agentic RL 的连接

| 推理执行概念 | RL 训练对应物 |
|---|---|
| plan step | step state / action |
| tool call | policy action + environment transition |
| observation | environment feedback，通常不进 policy loss |
| reflection | critic signal 或下一轮 prompt state |
| verifier result | final reward / step reward |
| budget cost | cost penalty |

因此 Agentic RL 不是另一个孤立章节，而是把 planning、tool use、memory、verifier 和 reward loop 统一成可训练系统。详细训练链路见 [Agentic RL 系统链路](#knowledge/agentic-rl-system)。

## 面试 QA

**Q：Planning 和 ReAct 的关系是什么？**

A：Planning 更偏全局任务分解，ReAct 更偏局部执行循环。复杂 Agent 往往先做 plan，再在每个步骤内用 ReAct + tools 执行，并根据 observation 更新状态或 replan。

**Q：Reflection 为什么不一定有效？**

A：如果只有模型自评，没有外部证据、测试或环境反馈，reflection 可能只是生成看似合理的解释。有效 reflection 通常需要可验证反馈，并把失败原因、修正计划和停止条件结构化。

**Q：Multi-Agent 什么时候值得用？**

A：当任务需要多角色、多视角、并行探索或独立审查时值得用，比如代码开发、研究综述、复杂 RAG、AppDev 验证。简单问答、状态共享困难或成本敏感任务不适合默认多 Agent。

**Q：如何控制 Agent 无限循环？**

A：设置 max turns、工具调用预算、时间预算、progress check、重复动作检测和 stop condition。每一步都应该能回答“新获得了什么证据或状态变化”。没有进展就应该停止、回滚或交给人工。

## 知识索引引用

| 知识点 | 主要来源 | 本页使用方式 |
|---|---|---|
| ReAct | https://arxiv.org/abs/2210.03629 | 用于解释推理-行动-观察循环 |
| Reflexion | https://arxiv.org/abs/2303.11366 | 用于解释 verbal reinforcement / 失败反思机制 |
| Tree of Thoughts | https://arxiv.org/abs/2305.10601 | 用于解释树搜索式规划和候选路径评估 |
| AutoGen | https://arxiv.org/abs/2308.08155 | 用于解释 multi-agent conversation 和角色协作 |
| AgentGuide 面试目录 | https://github.com/adongwanai/AgentGuide/tree/main/docs/04-interview | 用于补充规划、多 Agent、评估类面试问法 |

## 相关章节

| 章节 | 关系 |
|---|---|
| [Agent 系统](#knowledge/agent) | Agent 总览入口 |
| [Agent RAG 系统](#knowledge/agent-rag-systems) | 规划过程中的证据来源 |
| [Agent Memory 与工具调用](#knowledge/agent-memory-tooling) | 规划执行依赖的状态和工具协议 |
| [Agentic RL 系统链路](#knowledge/agentic-rl-system) | 多步轨迹如何进入训练闭环 |
