# 训练推理框架

## 当前定位

训练推理框架这一章的目标不是背工具名，而是建立一张 **LLM 系统工程地图**：训练框架解决“模型、梯度、优化器状态和通信怎么放得下、跑得快”；推理框架解决“KV cache、调度、吞吐和延迟怎么控制”；RL 后训练框架解决“rollout、reward、reference、advantage、trainer 怎么闭环”。

> 面试抓手：DeepSpeed / Megatron-LM 更偏训练侧，vLLM / SGLang 更偏推理与 rollout 侧，VeRL / Slime / SWIFT 更偏后训练编排和算法落地。回答时要先讲清楚系统边界，再讲具体机制。

```archify
LLM Framework Landscape|assets/diagrams/html/framework-landscape.html
```

## 一、六类框架横向定位

| 框架 | 主要阶段 | 核心问题 | 面试关键词 |
|---|---|---|---|
| DeepSpeed | 预训练、SFT、全参微调 | 降低训练显存和通信压力 | ZeRO、ZeRO-Offload、ZeRO-Infinity、optimizer states shard |
| Megatron-LM | 大规模 Transformer 训练 | 模型并行与高性能训练内核 | TP、PP、DP、EP、CP、pipeline bubble |
| VeRL | RLHF / RLVR / Agentic RL | 把 rollout、reward、reference、trainer 编排成 RL dataflow | HybridFlow、WorkerGroup、DataProto、GRPO/PPO/DAPO/OPD |
| Slime | 大规模 RL scaling | 连接 Megatron 训练端与 SGLang rollout 端 | Megatron + SGLang、weight sync、data buffer、agentic rollout |
| vLLM | 推理服务、RL rollout | 用高效 KV cache 和调度提高吞吐 | PagedAttention、continuous batching、prefix cache、OpenAI API |
| SGLang | Agent/RAG/结构化生成服务 | 复用前缀、执行多轮和约束生成程序 | RadixAttention、structured output、tool loop、multi-turn serving |
| MS-SWIFT | 算法应用框架 | 把 SFT、GRPO、GKD、DPO、采样、评测、部署串成可复现实验 | reward_funcs、teacher server、EvalScope、Megatron-SWIFT |

### 三层边界

**训练框架** 的瓶颈是参数、梯度、优化器状态、激活值和跨卡通信。DeepSpeed 更像“训练显存与状态管理工具箱”，Megatron-LM 更像“大规模 Transformer 并行训练骨架”。

**推理框架** 的瓶颈是 KV cache、prefill/decode 调度、批处理、前缀复用和结构化输出。vLLM 更偏通用高吞吐 serving，SGLang 更偏结构化 LLM 程序、Agent、RAG 和多轮调用。

**RL 后训练框架** 的难点是系统闭环：policy 要 rollout，rollout 要推理引擎，训练要 logprob/reference/reward/advantage，权重要在训练端和生成端同步。VeRL 更偏通用 RLHF/RLVR dataflow，Slime 更偏 Megatron + SGLang 的大规模 RL 链路，SWIFT 更偏算法配置和训练应用入口。

## 二、DeepSpeed：ZeRO 怎么讲

DeepSpeed 最常见的面试入口是 **ZeRO**。普通数据并行中，每张卡都保存完整参数、梯度和优化器状态；ZeRO 的核心思想是把这些冗余状态按数据并行 rank 切分。

```archify
DeepSpeed ZeRO Stages|assets/diagrams/html/deepspeed-zero-stages.html
```

普通数据并行的单卡状态近似包含：

$$
M_{DP} \approx P + G + O + A
$$

其中 $P$ 是参数，$G$ 是梯度，$O$ 是优化器状态，$A$ 是激活值。ZeRO 的理想目标是把可分片的部分按并行度 $N$ 摊薄：

$$
M_{ZeRO} \approx \frac{P + G + O}{N} + A + M_{comm}
$$

| 阶段 | 切分对象 | 直观理解 | 代价 |
|---|---|---|---|
| ZeRO-1 | optimizer states | Adam 的一阶、二阶动量不再每卡完整保存 | 通信变化较小 |
| ZeRO-2 | optimizer states + gradients | 梯度也做 reduce-scatter / all-gather 式管理 | 梯度同步更复杂 |
| ZeRO-3 | optimizer states + gradients + parameters | 参数也分片，计算时按需 gather | 通信和调试复杂度更高 |
| ZeRO-Offload / Infinity | CPU / NVMe 分层存储 | 用更慢但更大的存储换 GPU 显存 | 容易受 PCIe/NVMe 带宽限制 |

**面试结论**：ZeRO 不是“让计算变少”，而是减少数据并行中的状态冗余，使更大的模型或 batch 能放进显存。ZeRO-3 越省显存，通信和调度成本越高；offload 能救显存，但要警惕带宽瓶颈。

## 三、Megatron-LM：并行维度怎么讲

Megatron-LM 的核心不是单个技巧，而是把 Transformer 训练拆到多个并行维度中。回答时建议从“切 batch、切层、切矩阵、切专家、切序列”五个角度讲。

```archify
Megatron Parallelism Map|assets/diagrams/html/megatron-parallelism.html
```

| 并行方式 | 切分对象 | 解决问题 | 主要代价 |
|---|---|---|---|
| Data Parallel | batch | 提升吞吐 | 梯度同步 |
| Tensor Parallel | attention head / MLP hidden / matmul | 单层太大放不下或算不快 | 层内 all-reduce / reduce-scatter |
| Pipeline Parallel | Transformer layers | 层数太多放不下 | pipeline bubble、micro-batch 调度 |
| Expert Parallel | MoE experts | 稀疏专家扩展 | token routing、all-to-all、负载均衡 |
| Context / Sequence Parallel | 序列维度 | 长上下文显存和 attention 压力 | attention 通信和实现复杂度 |

**面试结论**：DeepSpeed 和 Megatron 不冲突。Megatron 负责更结构化的模型并行与高性能训练，DeepSpeed/FSDP 一类方案负责状态切分与显存优化。大规模训练通常是多种并行维度叠加，而不是只开一个参数。

## 四、vLLM：PagedAttention 怎么讲

vLLM 的核心抓手是 **PagedAttention**。自回归生成中，每一步都要复用历史 token 的 key/value：

$$
\operatorname{Attention}(q_t, K_{\le t}, V_{\le t})
= \operatorname{softmax}\left(\frac{q_t K_{\le t}^{\top}}{\sqrt d}\right)V_{\le t}
$$

KV cache 会随 batch、序列长度和并发请求快速增长。PagedAttention 借鉴操作系统分页思想，把 KV cache 切成 block/page 管理，减少碎片和重复拷贝。

```archify
vLLM PagedAttention|assets/diagrams/html/vllm-pagedattention.html
```

| 机制 | 解决问题 | 面试说法 |
|---|---|---|
| PagedAttention | KV cache 碎片和显存利用率 | 把连续大块 KV 变成 block 级管理，提高显存复用 |
| Continuous batching | 请求动态进入和退出 | decode 阶段持续合批，提升吞吐 |
| Prefix caching | 多请求共享前缀 | RAG、系统 prompt、few-shot prompt 可复用前缀 KV |
| OpenAI-compatible server | 工程接入成本 | 方便作为服务端或 RL rollout backend |

**面试结论**：vLLM 主要解决推理吞吐和显存管理，不直接解决训练并行和 RL 算法。它常被 VeRL、Slime、SWIFT 这类框架用作 rollout 或 serving 后端。

## 五、SGLang：RadixAttention 怎么讲

SGLang 更强调“结构化 LLM 程序”的高效执行。它适合多轮 Agent、RAG、多分支采样、结构化输出和工具调用这类场景，其中很多请求共享长前缀。

```archify
SGLang RadixAttention|assets/diagrams/html/sglang-radixattention.html
```

| 机制 | 解决问题 | 面试说法 |
|---|---|---|
| RadixAttention | 多请求共享前缀复用 | 用 radix tree 组织 prompt prefix，减少重复 prefill |
| Structured output | JSON / regex / schema 约束 | 适合工具调用、结构化抽取和 Agent 动作生成 |
| Multi-turn runtime | 多轮会话与工具循环 | 可把 Agent/RAG 的控制流和推理执行放在一起优化 |
| PD disaggregation / serving 优化 | prefill/decode 资源不匹配 | 针对长上下文和高并发调度不同阶段资源 |

**面试结论**：vLLM 和 SGLang 都能做高性能推理，但关注点不同。vLLM 更像通用高吞吐引擎，SGLang 更像面向复杂 LLM 程序和 Agent 工作流的执行系统。

## 六、推理系统原理代码怎么讲

这部分代码已经沉淀到 [原理代码：KV Cache / Prefix Cache / Scheduler](#principle-code/llm-kv-cache-serving)。面试中建议按三层递进讲，不要一上来就背 vLLM 或 SGLang 的实现细节。

| 手写模块 | 对应框架机制 | 面试讲法 |
|---|---|---|
| `PagedKVCache` | vLLM PagedAttention | 每个 sequence 维护 block table，把逻辑 token 位置映射到物理 KV block；释放请求时回收 block，减少连续大块分配带来的碎片。 |
| `RadixPrefixCache` | SGLang RadixAttention | 用 radix tree 保存共享 prompt 前缀，命中前缀后只对后缀做 prefill，适合 Agent/RAG/多分支采样。 |
| `continuous_batching_step` | continuous batching | 每个 decode step 后移除完成请求，并从 waiting queue 补入新请求，让 GPU 不被长短请求差异拖空。 |

> 面试抓手：这三段代码不是复刻工业级 serving engine，而是解释“为什么推理系统需要自己的内存管理和调度层”。真实系统还要处理多层 KV 张量、prefix 引用计数、preempt、优先级、prefill/decode 分离和分布式调度。
## 七、VeRL / Slime / SWIFT 怎么放进这张图

VeRL、Slime、SWIFT 不应该和 vLLM/SGLang 混成同一种工具。它们更接近“后训练系统编排层”。

| 框架 | 更适合怎么讲 | 和本章的关系 |
|---|---|---|
| VeRL | RLHF/RLVR 的 dataflow 框架 | 详细算法和 Agent Loop 放到 [VeRL 框架](#knowledge/verl-rl-framework) 与 [Agentic RL 系统链路](#knowledge/agentic-rl-system) |
| Slime | Megatron 训练 + SGLang rollout 的 RL scaling 框架 | 用来解释大规模 RL 中训练端和推理端如何打通 |
| SWIFT | 算法应用与训练配置入口 | 用来把 GRPO、GKD、DPO、采样评测、Agent 数据格式落到可复现实验 |

面试中可以用一个总模板：

> 我会先区分训练框架、推理框架和 RL 编排框架。训练侧关注显存、通信和并行；推理侧关注 KV cache、调度和吞吐；RL 编排侧关注 rollout、reward、reference、advantage、weight sync。具体到工具，DeepSpeed 偏 ZeRO 状态切分，Megatron 偏模型并行，vLLM 偏 PagedAttention 和服务吞吐，SGLang 偏结构化程序和多轮复用，VeRL/Slime/SWIFT 负责把这些后端组合进后训练链路。

## 八、面试 QA

**Q1：DeepSpeed ZeRO 和 Megatron-LM 的区别是什么？**

60 秒回答：DeepSpeed ZeRO 主要解决数据并行中参数、梯度、优化器状态重复保存导致的显存问题；Megatron-LM 主要解决 Transformer 模型本身如何做 tensor/pipeline/expert/context 等模型并行。二者可以组合使用：一个偏状态切分，一个偏模型结构并行。

**Q2：vLLM 的 PagedAttention 解决什么问题？**

60 秒回答：LLM 推理时 KV cache 随序列长度和并发请求增长，很容易造成显存碎片和低利用率。PagedAttention 把 KV cache 做 block/page 级管理，配合 continuous batching，提高并发场景下的吞吐和显存利用率。

**Q3：SGLang 相比 vLLM 的差异是什么？**

60 秒回答：vLLM 更偏通用 serving engine，核心是 KV cache 管理和调度；SGLang 更偏结构化 LLM 程序执行，强调前缀复用、约束输出、多轮工具调用和 Agent/RAG 工作流。复杂 Agent 场景里，SGLang 的程序化抽象更容易表达控制流。

**Q4：为什么 RL 后训练框架要接推理引擎？**

60 秒回答：GRPO/PPO/RLVR 不是纯离线监督学习，训练前要不断从当前 policy 生成 rollout。rollout 的吞吐会直接影响训练效率，所以 VeRL、Slime、SWIFT 这类框架通常会接 vLLM 或 SGLang 作为生成后端，同时处理 reward、reference logprob、advantage 和权重同步。

## 九、后续补强点

- 给 DeepSpeed / FSDP / Megatron distributed optimizer 做一张“状态切分对比表”。
- 给 vLLM / SGLang 继续补充真实工程追问：prefix 引用计数、抢占/换出、prefill/decode 分离和分布式调度。
- 给 Slime 增加一页“Megatron train + SGLang rollout + verifier reward”的大规模 RL 闭环图。

## 知识索引引用

| 知识点 | 来源 | 本页使用方式 |
|---|---|---|
| DeepSpeed ZeRO | [DeepSpeed ZeRO Tutorials](https://www.deepspeed.ai/tutorials/zero/)；ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | 提炼 ZeRO-1/2/3 的状态切分逻辑和显存收益 |
| Megatron-LM | [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)；Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | 提炼 DP/TP/PP/EP/CP 的并行维度和面试边界 |
| vLLM | [vLLM Docs](https://docs.vllm.ai/en/latest/)；Efficient Memory Management for Large Language Model Serving with PagedAttention | 提炼 PagedAttention、continuous batching、prefix caching |
| SGLang | [SGLang Docs](https://docs.sglang.ai/)；SGLang: Efficient Execution of Structured Language Model Programs | 提炼 RadixAttention、结构化输出、多轮 Agent/RAG 运行时 |
| VeRL | [VeRL Docs](https://verl.readthedocs.io/en/latest/) | 用于定位 RLHF/RLVR dataflow 与 rollout 后端关系 |
| SWIFT | [MS-SWIFT Docs](https://swift.readthedocs.io/zh-cn/latest/index.html) | 用于定位算法应用、蒸馏、GRPO、采样评测和部署链路 |
| Slime | [THUDM Slime](https://github.com/THUDM/slime) | 用于定位 Megatron + SGLang 的 RL scaling 链路 |
