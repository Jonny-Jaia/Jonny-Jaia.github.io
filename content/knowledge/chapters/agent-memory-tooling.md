# Agent Memory 与工具调用：状态、权限和可复现执行

## 当前定位

Memory 和 Tool Calling 是 Agent 从“会聊天”走向“能做事”的两个关键能力。Memory 让系统在单次上下文窗口之外维护任务状态和历史经验；Tool Calling 让模型把意图转成结构化调用，由系统执行真实工具并返回 observation。

> **面试抓手**：Memory 不是把所有历史塞回 prompt，Tool Calling 也不是让模型真的执行函数。前者是数据治理和检索问题，后者是 schema、权限、执行、审计和错误恢复问题。

```archify
Agent Memory Lifecycle|assets/diagrams/html/memory-lifecycle.html
```

## 一、Memory 机制怎么讲

| 类型 | 保存什么 | 典型用途 | 风险 |
|---|---|---|---|
| Short-term Memory | 当前任务状态、最近对话、临时变量 | 多轮任务连续性 | 上下文过长、噪声积累 |
| Long-term Memory | 稳定偏好、长期项目背景、反复确认的事实 | 个性化、跨会话协作 | 过期信息、隐私风险 |
| Episodic Memory | 某次任务过程、失败原因、操作结果 | 复盘、经验复用 | 错误经验被长期继承 |
| Semantic Memory | 抽象规则、总结、知识片段 | 决策一致性 | 抽象过度、丢失条件 |

Memory 的核心不是“存更多”，而是 **写入准入、结构化存储、相关性检索、过期机制、冲突解决和隐私控制**。

### 写入策略

```python
memory_item = {
    "subject": "用户/项目/任务对象",
    "content": "可复用的事实或偏好",
    "source": "来自哪次对话或哪份文档",
    "timestamp": "写入时间",
    "confidence": 0.86,
    "expires_at": "可选过期时间",
    "privacy_level": "public / private / sensitive",
}
```

写入前要问三件事：是否稳定、是否未来可复用、是否有明确来源。临时情绪、单次尝试、未验证猜测不应该直接写成长记忆。

### 检索策略

Memory retrieval 通常不是只看向量相似度，还要结合 recency、importance、task match、confidence 和 conflict check。

```python
def memory_score(similarity: float, importance: float, recency: float, confidence: float) -> float:
    """一个可面试手写的记忆打分骨架。"""
    return 0.45 * similarity + 0.25 * importance + 0.20 * recency + 0.10 * confidence
```

如果新旧记忆冲突，优先级通常是：用户显式最新确认 > 高可信外部来源 > 旧记忆 > 模型自动总结。

## 二、Tool Registry 怎么讲

```archify
Tool Calling Safety Flow|assets/diagrams/html/tool-calling-safety.html
```

| 字段 | 作用 | 面试追问 |
|---|---|---|
| tool name | 工具唯一标识 | 命名不清会导致误调用 |
| description | 什么时候用、什么时候不用 | 描述过宽会导致模型滥用工具 |
| schema | 参数类型、必填字段、枚举、默认值 | schema 是约束模型输出的关键 |
| permission | 是否需要用户确认、是否有写权限 | 敏感工具必须有权限门控 |
| timeout / retry | 超时和重试策略 | 避免无限循环和阻塞 |
| result contract | 工具返回结构 | 区分 success、error、partial |
| audit log | 参数、结果、失败原因、成本 | 便于复盘和安全审计 |

### 最小代码骨架

```python
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    func: Callable[..., Any]
    requires_confirmation: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        tool = self._tools[name]
        if tool.requires_confirmation:
            raise PermissionError(f"Tool {name} requires confirmation")
        return tool.func(**args)
```

真实系统还需要 schema validation、sandbox、权限分级、dry-run、幂等性、错误通道和审计日志。

## 三、工具调用的安全边界

| 风险 | 例子 | 工程处理 |
|---|---|---|
| 参数幻觉 | 编造文件路径、表名、API 参数 | schema validate、候选约束、dry-run |
| 权限越界 | 删除文件、发邮件、下单、访问敏感数据 | permission gate、human confirmation |
| Prompt Injection | 检索文档诱导模型泄漏系统信息 | instruction hierarchy、tool isolation |
| 结果误读 | 工具失败被当作成功 | typed result、error channel、retry policy |
| 循环调用 | 反复调用同一工具无进展 | max turns、budget、progress check |

**结论**：Function Calling / Tool Calling 的难点不在“输出 JSON”，而在 **工具选择、参数可靠性、权限控制、错误恢复、审计和线上一致性**。

## 四、Memory 与 Tool 的关系

| 关系 | 说明 |
|---|---|
| 工具结果可以写入 memory | 例如数据库查询结果、文件修改结果、用户确认的偏好 |
| memory 可以影响工具选择 | 例如记住用户偏好的代码风格或项目目录 |
| 工具 observation 不等于模型输出 | 训练时必须用 mask 区分，不应直接进入 policy loss |
| memory 写入必须可解释 | 不能把工具返回的噪声自动写成长记忆 |

这也是 Agentic RL 中 `response_mask` 重要的原因：assistant token 是 policy action，tool observation 是环境返回。把两者混在一起会污染训练目标。

## 五、排障清单

| 症状 | 可能原因 | 排查方式 |
|---|---|---|
| Agent 频繁调用错工具 | 工具描述太像、schema 太松、缺少负例 | 加工具选择评估集、收紧 description、增加 few-shot |
| 工具参数经常错 | schema 不完整、字段含义不明确 | 加枚举、范围、默认值、validation error feedback |
| Agent 记住错误信息 | 写入准入太宽、没有过期和冲突处理 | 增加 confidence、expires_at、source、人工确认 |
| 长任务越做越偏 | memory 和 tool output 噪声堆积 | context compression、状态快照、progress check |
| 线上和训练不一致 | tool template / observation 格式不同 | 固定工具协议，保留原始 trace 和 mask |

## 面试 QA

**Q：Memory 和 RAG 有什么区别？**

A：RAG 通常面向外部知识库和事实证据，Memory 面向用户、任务和交互历史。RAG 更像查资料，Memory 更像维护任务状态和长期偏好。两者都需要检索，但数据来源、更新频率、隐私风险和过期策略不同。

**Q：为什么 Memory 不是存越多越好？**

A：存得越多，噪声、冲突、隐私和过期信息越多。好的 memory 系统要有写入准入、结构化 schema、相关性检索、过期机制、冲突解决和删除能力。

**Q：Tool Calling 的核心难点是什么？**

A：不是让模型输出函数名，而是工具 schema 设计、参数校验、权限控制、错误恢复、审计日志和结果解释。模型只生成结构化调用意图，真正执行、校验和授权应该由系统完成。

**Q：工具返回为什么不能直接参与 policy loss？**

A：工具返回是环境 observation，不是模型 action。如果把 observation 也算进 loss，模型会被训练去模仿工具结果，导致工具边界被破坏，也会污染 logprob、advantage 和 response mask。

## 知识索引引用

| 知识点 | 主要来源 | 本页使用方式 |
|---|---|---|
| ReAct 与工具观察循环 | https://arxiv.org/abs/2210.03629 | 用于解释 Thought / Action / Observation 的工具调用范式 |
| Generative Agents Memory | https://arxiv.org/abs/2304.03442 | 用于整理 memory store、reflection 和长期记忆思想 |
| VeRL Agent Loop | https://verl.readthedocs.io/en/latest/advance/agent_loop.html | 用于解释 response_mask 与 tool observation 的训练边界 |
| OpenAI Function Calling / Tool Calling | https://platform.openai.com/docs/guides/function-calling | 用于补充 schema、structured output 和工具调用工程边界 |

## 相关章节

| 章节 | 关系 |
|---|---|
| [Agent 系统](#knowledge/agent) | Agent 总览入口 |
| [Agent RAG 系统](#knowledge/agent-rag-systems) | 外部知识检索和证据增强 |
| [Agentic RL 系统链路](#knowledge/agentic-rl-system) | 工具轨迹如何进入 RL 训练 |
| [VeRL 框架](#knowledge/verl-rl-framework) | DataProto、Agent Loop、Reward Loop 的系统实现视角 |
