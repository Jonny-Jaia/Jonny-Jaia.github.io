# Agent 系统：关键技术与前沿范式

## 当前定位

Agent 是把 LLM 从单轮文本生成器扩展成任务执行系统的一组工程范式。它通常需要维护任务状态、检索外部知识、调用工具、观察环境反馈、更新计划，并在必要时使用反思、评估器或多智能体协作完成复杂任务。

> **面试抓手**：Agent 不是单独算法，而是一套系统工程。回答时优先讲清楚主循环、状态、工具、RAG、Memory、Planning、Evaluation 和 Safety，而不是只说 ReAct。

```archify
Agent Runtime Architecture|assets/diagrams/html/agent-runtime-architecture.html
```

## 一、Agent 和 Chatbot 的区别

| 维度 | Chatbot | Agent |
|---|---|---|
| 目标 | 回答当前问题 | 完成可分解任务 |
| 状态 | 主要依赖上下文窗口 | 显式维护任务状态、memory、工具结果 |
| 外部能力 | 少量或无工具 | 检索、代码执行、浏览器、数据库、业务 API |
| 控制流 | 单轮或多轮对话 | 规划、执行、观察、修正、停止 |
| 评估 | 文本质量 | 任务成功率、工具正确率、成本、安全性 |

**一句话结论**：Chatbot 更像会说话的模型，Agent 更像由 LLM 驱动的任务执行系统。

## 二、拆分专题阅读入口

Agent 总览页只保留整体地图和关键结论，细节优先沉淀到下面三个子专题，避免单页过厚、目录过长：

| 子专题 | 解决什么问题 | 建议阅读时机 |
|---|---|---|
| [Agent RAG 系统](#knowledge/agent-rag-systems) | 检索增强、chunk、embedding/reranker、hybrid search、引用与评估 | 准备 RAG / 知识库 / 企业问答面试时 |
| [Agent Memory 与工具调用](#knowledge/agent-memory-tooling) | memory schema、Tool Registry、权限、错误恢复、response_mask 边界 | 准备 Function Calling / Tool Use / Agent 训练时 |
| [Agent 规划执行与 Multi-Agent](#knowledge/agent-planning-multi-agent) | plan-and-execute、reflection、multi-agent、预算和停止条件 | 准备复杂 Agent 系统设计题时 |

## 三、关键技术总览

| 技术 | 解决的问题 | 核心机制 | 主要风险 |
|---|---|---|---|
| ReAct | 推理和行动耦合 | Reason + Act + Observe 循环 | 推理冗长、工具错误传播 |
| Function / Tool Calling | 工具接口稳定性 | schema、参数生成、执行回填 | 参数幻觉、权限越界、错误恢复差 |
| RAG | 外部知识和可追溯证据 | retrieve、rerank、context packing、citation | 召回失败、上下文污染、引用不可信 |
| Memory | 长期偏好和任务历史 | extraction、store、retrieve、decay | 记忆污染、隐私风险、过期信息 |
| Planning | 复杂任务分解 | plan、execute、check、replan | 计划不可执行、停止条件差 |
| Reflection | 失败后修正 | evaluator、critique、retry | 自评不可靠、成本高 |
| Multi-Agent | 多角色协作 | planner、executor、critic、specialist | 协调成本、状态不一致、责任边界不清 |

## 四、典型工程架构

一个可维护 Agent 通常包含：

- **Orchestrator**：控制主循环、预算、状态和停止条件。
- **Planner**：拆任务、安排工具、设置检查点。
- **Memory Store**：维护长期状态、偏好和经验。
- **RAG Pipeline**：提供外部知识检索、rerank 和引用。
- **Tool Registry**：维护工具 schema、权限、执行和审计。
- **Executor**：执行每一步并收集 observation。
- **Evaluator / Verifier**：检查结果是否满足目标。
- **Safety Layer**：处理权限、隔离、提示注入防护和敏感操作确认。

## 五、和 Agentic RL / VLA 的连接

Agentic RL 把 Agent 的多轮轨迹变成可训练样本，重点是 trace schema、response_mask、reward loop 和 credit assignment。VLA 则可以看作多模态 Agent 的具身版本：action 不再只是工具调用，也可能是机器人动作、自动驾驶轨迹或连续控制信号。

| 关联章节 | 关系 |
|---|---|
| [Agentic RL 系统链路](#knowledge/agentic-rl-system) | 多轮工具轨迹如何进入 RL 后训练 |
| [VeRL 框架](#knowledge/verl-rl-framework) | DataProto、Agent Loop、Reward Loop 的系统实现视角 |
| [VLA 与自动驾驶具身智能](#knowledge/vla-embodied-driving) | 从文本工具 action 扩展到视觉-语言-动作 |
| [具身智能与机器人 VLA](#knowledge/embodied-ai-robotics-vla) | 机器人动作表示、Sim-to-Real 和闭环评估 |

## 面试 QA

**Q：Agent 和普通 Chatbot 最大区别是什么？**

A：Chatbot 主要回答当前上下文里的问题；Agent 要维护任务状态、调用工具、观察环境、更新计划，并在必要时使用 memory、RAG、reflection 或 verifier 完成多步任务。

**Q：Agent 系统设计面试应该先讲什么？**

A：先讲任务主循环和状态，再讲工具协议、RAG、memory、规划执行、评估和安全边界。不要一上来只说 ReAct，因为 ReAct 只是执行范式之一。

**Q：为什么 Agent 评估不能只看最终文本？**

A：Agent 的质量还取决于工具调用是否正确、是否越权、是否真的完成任务、成本是否可控、失败后能否恢复，以及过程是否可复现。最终文本看起来对，不代表中间轨迹可靠。

## 知识索引引用

| 知识点 | 来源 | 本页使用方式 |
|---|---|---|
| ReAct / Tool Use / Planning 总览 | https://arxiv.org/abs/2210.03629 | 用于解释 Agent 的推理-行动-观察循环 |
| Memory / Reflection / Multi-Agent 总览 | https://github.com/datawhalechina/hello-agents/tree/main/Extra-Chapter | 用于补充 Agent 面试问答和工程实践 |
| AgentGuide 面试目录 | https://github.com/adongwanai/AgentGuide/tree/main/docs/04-interview | 用于整理 RAG、工具调用、规划和评估的高频问法 |
| VeRL Agent Loop / Reward Loop | https://verl.readthedocs.io/en/latest/advance/agent_loop.html | 用于连接 Agent 轨迹和后训练系统 |

## 后续补全计划

- 为 RAG 子专题继续补 embedding/reranker 训练与评估细节。
- 为 Memory 与工具调用子专题补权限分级、sandbox 和 typed result 示例。
- 为 Planning 与 Multi-Agent 子专题补 planner-executor、critic-reviewer、verifier agent 的案例。
