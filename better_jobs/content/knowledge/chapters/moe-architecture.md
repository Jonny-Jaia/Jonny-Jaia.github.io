# MoE 模型架构演进

## 当前定位

MoE（Mixture-of-Experts）是大模型扩展中非常重要的一条路线：它通过 **条件计算** 让模型拥有很大的总参数量，但每个 token 只激活少量专家，从而控制单次前向计算成本。面试里要能讲清楚：MoE 不是“免费变大”，它把 dense 模型的计算瓶颈，部分转换成 **路由、负载均衡、通信、专家利用率和训练稳定性** 问题。

> **面试抓手**：Dense 模型是“每个 token 都过同一套 FFN”；MoE 是“每个 token 由 router 选择少数专家 FFN”。优势是参数容量变大，代价是路由不均衡、专家通信、训练不稳定和推理部署复杂。

```mermaid
flowchart LR
  n1["Token hidden states"] --> n2["Router logits"]
  n2["Router logits"] --> n3["top-k routing"]
  n3["top-k routing"] --> n4["Expert FFNs"]
  n4["Expert FFNs"] --> n5["Weighted combine"]
  n5["Weighted combine"] --> n6["Next transformer block"]
```

## 一句话演进线

| 阶段 | 代表方向 | 关键变化 | 核心问题 |
|---|---|---|---|
| Dense Transformer | GPT/BERT dense FFN | 每层所有 token 使用同一个 FFN | 参数与计算同步增长 |
| Sparse MoE | GShard / Switch Transformer | 每个 token 只路由到少数专家 | 路由、容量、负载均衡 |
| 大规模 MoE LM | GLaM / Mixtral | MoE 成为开源和工业模型主路线 | 训练稳定与推理吞吐 |
| 专家细分 | DeepSeekMoE | shared experts + routed experts + fine-grained expert segmentation | 专家专门化与知识冗余 |
| 稳定训练 | DeepSeek-V3 等 | auxiliary-loss-free load balancing、node-limited routing | 负载均衡与主任务损失冲突 |
| 模型族工程化 | Qwen MoE / Mixtral / DeepSeek | dense/MoE 组合、多语言、多任务、推理部署 | 生态、量化、并行和 serving |

## 基本结构

Transformer 里的 MoE 通常替换 FFN 层，而不是替换 attention 层。普通 FFN 是：

$$
\mathrm{FFN}(x)=W_2 \sigma(W_1 x)
$$

MoE 则有多个专家 $E_i$，router 根据 token hidden state $x$ 输出专家概率：

$$
p_i(x)=\mathrm{softmax}(W_r x)_i
$$

top-k routing 选择概率最高的 $k$ 个专家：

$$
\mathrm{MoE}(x)=\sum_{i \in \mathrm{TopK}(p(x), k)} p_i(x) E_i(x)
$$

```mermaid
flowchart LR
  n1["Transformer block"] --> n2["Attention"]
  n2["Attention"] --> n3["MoE FFN layer"]
  n3["MoE FFN layer"] --> n4["Router"]
  n4["Router"] --> n5["Expert parallel"]
  n5["Expert parallel"] --> n6["Combine"]
  n6["Combine"] --> n7["Residual output"]
```

**核心结论**：MoE 扩大的主要是 **总参数量**，不是每个 token 的激活参数量。比如总共有很多专家，但每个 token 只经过 top-1 或 top-2 个专家。

## top-k routing

top-k routing 是 MoE 的核心机制。router 为每个 token 计算对所有专家的打分，然后选择 top-k 专家执行。

| routing 方式 | 特点 | 优势 | 风险 |
|---|---|---|---|
| top-1 routing | 每个 token 只进一个专家 | 计算和通信更省 | 容易路由不稳定，表达能力受限 |
| top-2 routing | 每个 token 进两个专家 | 表达更强，训练更平滑 | 计算和通信更高 |
| group-limited routing | 先限制专家组/节点，再选专家 | 降低跨节点通信 | 可能限制全局最优专家选择 |
| expert-choice routing | 专家选择 token | 更容易控制专家容量 | 实现和排序更复杂 |

**工程注意**：

- router logits 通常要加噪声或正则，避免早期专家塌缩。
- top-k 概率要重新归一化，否则组合权重尺度不稳定。
- 推理部署时，token 会被按专家重排，涉及 all-to-all 通信。

## load balancing

MoE 最典型的问题是 load balancing：如果 router 总是把 token 发给少数专家，热门专家拥堵，冷门专家学不到东西。

经典 auxiliary load balancing loss 会鼓励 token 分布更均匀。例如令 $f_i$ 表示实际分配到专家 $i$ 的 token 比例，$P_i$ 表示 router 给专家 $i$ 的平均概率，可以构造：

$$
\mathcal{L}_{balance}
= N \sum_{i=1}^{N} f_i P_i
$$

其中 $N$ 是专家数。直觉上，它惩罚“概率高但实际过载”的专家。

**问题在于**：辅助损失会和主语言模型损失存在 trade-off。强行均衡可能损害语义最优路由，所以后续模型开始探索更细的专家划分、shared experts、routing bias 或 auxiliary-loss-free load balancing。

## shared experts

shared experts 的思想是：有些知识是通用的，不应该被稀疏路由完全切开。因此一部分专家对所有 token 共享激活，另一部分专家才由 router 选择。

```mermaid
flowchart LR
  n1["Token"] --> n2["Shared experts always active"]
  n2["Shared experts always active"] --> n3["Routed experts selected by router"]
  n3["Routed experts selected by router"] --> n4["Merge shared and routed outputs"]
```

**为什么需要 shared experts**：

- 降低 routed experts 学重复通用知识的压力。
- 给所有 token 一个稳定的 shared capacity。
- 缓解专家过度碎片化导致的训练不稳定。

**面试表达**：shared experts 可以理解为 MoE 里的“公共基础能力”，routed experts 则承担更细分的模式、领域或 token 类型。

## fine-grained experts

fine-grained experts 是 DeepSeekMoE 等路线强调的方向：把专家切得更细，同时激活更多细粒度专家，让专家更容易专门化。

相比少量大专家，细粒度专家的潜在优势是：

- 专家职责更细，减少知识冗余。
- router 有更灵活的组合空间。
- 可以在激活参数量相近时提高专家组合表达能力。

代价是：

- router 选择空间更大，训练更难。
- all-to-all 通信和调度更复杂。
- 专家负载均衡更敏感。

## auxiliary-loss-free load balancing

DeepSeek-V3 技术报告中强调了 auxiliary-loss-free load balancing 思路：不再强依赖辅助损失直接压 router，而是通过可学习或动态调整的 bias 影响专家选择，尽量减少负载均衡目标对主任务 loss 的干扰。

可以把它理解成：

```mermaid
flowchart LR
  n1["Router score"] --> n2["Add expert bias"]
  n2["Add expert bias"] --> n3["Select top-k experts"]
  n3["Select top-k experts"] --> n4["Observe expert load"]
  n4["Observe expert load"] --> n5["Update bias"]
  n5["Update bias"] --> n6["Keep load balanced"]
```

**面试结论**：辅助均衡损失简单直接，但可能和语言建模目标冲突；auxiliary-loss-free 方法希望把“负载均衡控制”从主 loss 中拆出来，让 router 更专注语义选择。

## MoE 并行与通信

MoE 的训练/推理复杂度主要来自专家并行。一个 batch 的 token 会被分发到不同专家，通常需要 token dispatch 和 combine：

```mermaid
flowchart LR
  n1["Tokens on each GPU"] --> n2["Router decides expert ids"]
  n2["Router decides expert ids"] --> n3["All-to-all dispatch"]
  n3["All-to-all dispatch"] --> n4["Expert FFN compute"]
  n4["Expert FFN compute"] --> n5["All-to-all combine"]
  n5["All-to-all combine"] --> n6["Original token order"]
```

| 并行维度 | 作用 | MoE 中的问题 |
|---|---|---|
| Data Parallel | 复制模型处理不同 batch | router 负载统计要跨卡 |
| Tensor Parallel | 切分单个矩阵 | expert FFN 内部可能继续切 |
| Pipeline Parallel | 切分层 | MoE 层和 dense 层调度要协调 |
| Expert Parallel | 专家分布在不同设备 | all-to-all 通信成为瓶颈 |
| Sequence Parallel | 切分序列维度 | 与 expert dispatch 组合复杂 |

**工程结论**：MoE 模型是否高效，不只看激活参数量，还要看专家分布、通信拓扑、batch size、capacity factor、token dropping 和推理调度。

## 代表模型路线

| 模型/论文 | MoE 特点 | 值得记的点 |
|---|---|---|
| GShard | 早期大规模 conditional computation | 自动分片、top-2 gating、大规模翻译 |
| Switch Transformer | top-1 routing 简化 MoE | 简化路由，提高可扩展性 |
| GLaM | 稀疏激活语言模型 | 用更少激活计算获得强性能 |
| Mixtral | 开源 sparse MoE | 每层多个专家，top-2 routing，推理生态影响大 |
| DeepSeekMoE | shared experts + routed experts + fine-grained experts | 专家专门化、减少知识冗余 |
| DeepSeek-V3 | auxiliary-loss-free load balancing、node-limited routing | MoE 训练稳定与工程效率 |
| Qwen MoE | dense/MoE 模型族路线 | 工程生态、模型尺寸组合、开源适配 |

## 与 Dense 模型的对比

| 维度 | Dense | MoE |
|---|---|---|
| 参数利用 | 每个 token 用全部 FFN 参数 | 每个 token 只用少数专家 |
| 训练稳定 | 相对简单 | router、负载、通信更复杂 |
| 扩展方式 | 参数量和计算量一起涨 | 总参数可涨，激活计算可控 |
| 推理部署 | 规则简单 | 专家并行、动态路由、batch 调度复杂 |
| 表达能力 | 所有 token 共享容量 | 不同 token 可使用不同专家组合 |
| 典型风险 | 计算成本高 | 专家塌缩、负载不均、通信瓶颈 |

## 代码理解：top-k routing

```python
import torch
import torch.nn.functional as F


def moe_topk_router(hidden_states: torch.Tensor, router_weight: torch.Tensor, k: int = 2):
    logits = hidden_states @ router_weight
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    return topk_ids, topk_probs
```

这个骨架只展示 router 的选择逻辑。真实 MoE 还需要 token dispatch、expert capacity、load balancing loss、all-to-all、expert output combine 和反向传播通信。

## 代码理解：负载统计

```python
import torch


def expert_load(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    flat_ids = topk_ids.reshape(-1)
    counts = torch.bincount(flat_ids, minlength=num_experts).float()
    return counts / counts.sum().clamp_min(1.0)
```

面试里可以用这个函数解释：如果少数专家的 load 接近 1，大部分专家接近 0，就说明 router 发生了严重不均衡。

## 面试 QA

**Q：MoE 为什么能扩大参数量但不同比例增加计算量？**

A：因为每个 token 只激活 top-k 个专家，而不是经过全部专家。总参数量可以很大，但激活参数量和 FLOPs 主要由 top-k 专家决定。

**Q：MoE 的最大工程难点是什么？**

A：不是专家 FFN 本身，而是 routing、load balancing、token dispatch/combine、all-to-all 通信、capacity 控制和推理调度。

**Q：top-1 routing 和 top-2 routing 怎么取舍？**

A：top-1 更省计算和通信，但可能更不稳定、表达能力更弱；top-2 更平滑、组合能力更强，但计算和通信成本更高。

**Q：为什么需要 load balancing loss？**

A：router 可能把大量 token 发给少数专家，导致热门专家过载、冷门专家训练不足。load balancing loss 让专家使用更均匀。

**Q：load balancing loss 有什么副作用？**

A：它可能和语言建模主目标冲突。强行均匀不一定是语义上最优的路由，所以后续出现 auxiliary-loss-free 或 bias-based balancing。

**Q：shared experts 解决什么问题？**

A：它让通用知识由共享专家承担，减少 routed experts 重复学习公共能力，使稀疏专家更专注细分模式。

**Q：fine-grained experts 的价值是什么？**

A：把专家切得更细，可以提高专家专门化和组合灵活性，但会增加 routing、负载均衡和通信调度难度。

**Q：MoE 和模型并行是什么关系？**

A：MoE 通常需要 expert parallel，把不同专家放在不同设备上；同时也可以和 data parallel、tensor parallel、pipeline parallel 组合。

## 与后训练和推理的关系

- **SFT**：MoE 的专家可能在 instruction 数据上发生路由偏移，需要关注专家利用率和负载变化。
- **DPO / RLHF / GRPO**：偏好优化或 RL 会改变 token 分布和生成分布，可能导致专家负载重新分配。
- **推理服务**：动态路由会让 batch 内 token 去不同专家，影响吞吐、显存和通信；量化与 expert parallel 也更复杂。
- **模型家族比较**：国产模型中 DeepSeek、Qwen、Kimi、MiniMax 等路线都可能涉及 MoE、稀疏 attention、长上下文和后训练组合，需要分清架构创新和训练策略创新。

## 后续补全计划

- 补 DeepSeekMoE 与 DeepSeek-V3 MoE 结构的独立精读卡。
- 补 Switch Transformer / GShard / GLaM / Mixtral 的论文卡片。
- 补 MoE capacity factor、token dropping、router z-loss、expert parallel 的代码 demo。
- 补一张 Dense FFN 与 MoE FFN 的 SVG 结构图。

## 参考资料

- GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding, arXiv:2006.16668。
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, arXiv:2101.03961。
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts, arXiv:2112.06905。
- Mixtral of Experts, arXiv:2401.04088。
- DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models, arXiv:2401.06066。
- DeepSeek-V3 Technical Report, arXiv:2412.19437。
