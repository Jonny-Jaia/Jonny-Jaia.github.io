# 强化学习基础

## 当前定位

这章用于沉淀强化学习基础：MDP、价值函数、策略梯度、Actor-Critic、TRPO/PPO、离线 RL。它是理解 GRPO、RLHF、RLVR 的前置知识。

> **面试抓手**：强化学习问题要先讲清状态、动作、奖励、策略和目标，再讨论 value-based / policy-based / actor-critic 的方法差异。

```flow
MDP -> Value function -> Policy gradient -> Actor critic -> PPO and GRPO
```

## ReadyBlog 来源

- ReadyBlog: `深度学习/强化学习/RL面试问题.md`
- ReadyBlog: `深度学习/强化学习/主流模型.md`
- ReadyBlog: `杂/3-论文综述/基于RL的推荐系统综述学习笔记.md`
- ReadyBlog: `杂/4-大模型实习/基于RL的指令微调范式.md`

## 知识画像

| 主题 | 必须掌握 | 面试表达重点 |
|---|---|---|
| MDP | state、action、reward、transition、discount | RL 建模的五元组 |
| Value-based | $V(s)$、$Q(s,a)$、DQN | 学价值再导出策略 |
| Policy-based | policy gradient、REINFORCE | 直接优化策略期望回报 |
| Actor-Critic | actor、critic、advantage | 降低方差和提升训练效率 |
| PPO/TRPO | trust region、clip、KL | 稳定策略更新 |
| Offline RL | distribution shift、CQL | 离线数据和策略外推风险 |

## 核心公式

回报：

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

动作价值函数：

$$
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
$$

优势函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

策略梯度直觉：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\nabla_\theta \log \pi_\theta(a\mid s) A^\pi(s,a)
\right]
$$

## 核心结论

- **Value-based 方法先学评价**：适合离散动作和可以稳定估计 Q 值的场景。
- **Policy-based 方法直接学策略**：适合连续动作或随机策略，但梯度方差更大。
- **Actor-Critic 是折中**：actor 决定怎么做，critic 估计做得好不好。
- **PPO/GRPO 都是在解决稳定更新问题**：PPO 用 value/advantage + clip，GRPO 在 LLM 场景用组内 reward 构造 advantage。
- **LLM RL 的特殊点是动作空间巨大**：token 序列极长，reward 往往稀疏，所以 reward 设计和 KL 约束非常关键。

## 和后训练的连接

| RL 基础概念 | 在 LLM 后训练中的对应 |
|---|---|
| state | prompt + 已生成上下文 |
| action | 下一个 token 或完整回答 |
| reward | reward model、规则验证器、单元测试、人工偏好 |
| policy | 当前语言模型 |
| reference policy | SFT model / frozen base model |
| advantage | PPO critic 估计或 GRPO group-relative advantage |

## 面试 QA

**Q：Value-based 和 Policy-based 的区别是什么？**

A：Value-based 学 $V$ 或 $Q$，再根据价值选择动作；Policy-based 直接参数化策略并最大化期望回报。前者通常样本效率高但对连续动作不方便，后者表达灵活但梯度方差大。

**Q：Actor-Critic 为什么能降低方差？**

A：critic 提供 value / advantage 估计，相当于给 policy gradient 加 baseline，减少回报估计的方差；actor 仍然负责更新策略方向。

**Q：PPO 的 clip 在解决什么问题？**

A：clip 限制新旧策略概率比，避免单次更新让 policy 偏离太远，提升训练稳定性。GRPO 继承了类似稳定更新思想，但 advantage 构造方式不同。

## 后续补全计划

- 从 ReadyBlog 主流模型笔记中拆 TRPO/PPO、DDPG/TD3、SAC、CQL 卡片。
- 给 GRPO 章节增加从 policy gradient 到 group advantage 的推导桥。
- 将 PPO/GAE 的最小代码加入“原理代码”。
