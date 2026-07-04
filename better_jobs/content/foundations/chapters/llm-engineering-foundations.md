# LLM 工程基础

> 本章承载长期稳定、适合反复复习的 LLM 工程底层知识。知识库专题页只保留面试主线和跳转，具体机制、公式、边界和工程取舍放在这里，避免专题章节变成过长的材料堆叠。

```flow
Transformer 结构 -> PEFT 微调 -> 分布式训练 -> 显存估算 -> 推理优化 -> 工程面试表达
```

## 复习地图

| 基础块 | 要掌握到什么程度 | 典型面试追问 |
| --- | --- | --- |
| 归一化 | 能区分 BatchNorm、LayerNorm、RMSNorm、Pre-LN、Post-LN、DeepNorm | 为什么 LLM 常用 RMSNorm？Pre-LN 为什么更稳定？ |
| PEFT | 能讲清 LoRA / AdaLoRA / QLoRA 的参数、显存和推理合并 | LoRA 到底省了什么？QLoRA 是不是 4-bit 全参训练？ |
| 并行训练 | 能从 batch、矩阵、层、序列、专家、状态分片解释 DP/TP/PP/SP/EP/ZeRO | ZeRO-3 为什么省显存但通信更复杂？ |
| 显存估算 | 能快速估参数、优化器状态、激活、KV cache 的量级 | 7B FP16 推理权重多大？长上下文为什么吃显存？ |
| 推理优化 | 能把 prefill/decode、KV cache、FlashAttention、PagedAttention、batching、speculative decoding 串起来 | FlashAttention 和 PagedAttention 优化对象有什么不同？ |

## 归一化与 Transformer 稳定性

归一化的核心目标是控制激活分布，让深层网络更容易优化。不同归一化方法的关键区别在于“沿哪个维度统计均值和方差”。

| 方法 | 统计维度 | 更常见场景 | 在 LLM 中的注意点 |
| --- | --- | --- | --- |
| BatchNorm | batch 维度 | CNN、固定 batch 视觉任务 | 依赖 batch 统计，变长序列和自回归推理不友好 |
| LayerNorm | hidden 维度 | Transformer、RNN、NLP | 不依赖 batch，适合变长序列和单样本推理 |
| RMSNorm | hidden 维度的 RMS | LLaMA 等现代 LLM | 去掉均值中心化，只保留尺度归一化，计算更轻 |
| DeepNorm | 残差缩放与初始化 | 超深 Transformer | 解决深层 Post-LN 稳定性问题，工程设计更复杂 |

LayerNorm 常见形式：

$$
LN(x)=\gamma \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm 常见形式：

$$
RMSNorm(x)=g\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
$$

**Pre-LN / Post-LN 的关键不是位置名字，而是梯度路径。** Post-LN 是子层和残差相加后再归一化，早期 Transformer 常用；Pre-LN 是进入 attention / MLP 子层前先归一化，残差路径更直接，深层训练更稳定。面试中可以这样说：

- Pre-LN 的优势是训练稳定，梯度更容易沿残差路径传播。
- Post-LN 有时被认为最终表达或收敛性质有优势，但深层网络更容易不稳定。
- DeepNorm 等方法试图让很深的 Post-LN Transformer 也能稳定训练。

**容易误解的点：** RMSNorm 不是“没有归一化”，也不是单纯为了省一点计算。它保留尺度归一化，省去 re-centering，并且通常与 Pre-Norm、RoPE、SwiGLU、合适初始化和学习率策略共同构成稳定训练方案。

## PEFT：LoRA / AdaLoRA / QLoRA

LoRA 的核心假设是：大模型针对某个任务的权重更新不一定需要满秩矩阵，可以用低秩矩阵近似。

$$
W = W_0 + \Delta W,\quad \Delta W=BA
$$

其中 $W_0$ 冻结，只训练 $A,B$。如果原权重是 $d_{out}\times d_{in}$，LoRA rank 为 $r$，新增参数量大约是：

$$
r(d_{in}+d_{out})
$$

而不是 $d_{in}d_{out}$。因此 LoRA 主要节省的是**可训练参数、梯度和优化器状态**，基础模型权重仍然需要加载。

常见工程细节：

- **target modules**：常见是 `q_proj`、`v_proj`，也会扩展到 `k_proj`、`o_proj`、MLP projection。
- **rank**：不是越大越好。rank 太小容量不足，rank 太大成本上升且可能过拟合。
- **alpha**：控制 LoRA 增量缩放，常与 rank 一起调。
- **merge**：推理时可把 $\Delta W$ 合并进 $W_0$，减少 adapter 路径带来的额外开销。

AdaLoRA 和 QLoRA 是两个不同方向：

| 方法 | 解决的问题 | 关键机制 | 面试边界 |
| --- | --- | --- | --- |
| AdaLoRA | 固定 rank 不能区分模块重要性 | 动态分配 rank 预算 | 更复杂，标准 LoRA 更常见 |
| QLoRA | 基础模型加载显存仍然高 | 4-bit NF4、double quantization、paged optimizer、LoRA adapter | 通常不是 4-bit 全参训练，而是量化基座 + 训练 adapter |

**容易误解的点：** QLoRA 的训练过程里，基础模型可以 4-bit 存储，但矩阵计算通常会反量化到较高精度；反向传播主要更新 LoRA adapter，不是直接把所有 4-bit 权重全参更新。

## 分布式训练并行策略

大模型训练的资源瓶颈主要来自参数、梯度、优化器状态、激活和通信。不同并行方式本质上是在切不同对象：

| 策略 | 切什么 | 解决什么 | 代价 |
| --- | --- | --- | --- |
| Data Parallel | batch | 提升吞吐，实现简单 | 每卡保存完整模型，梯度 all-reduce |
| Tensor Parallel | 层内矩阵 / attention head / MLP hidden | 单层太宽放不下或算不动 | 层内通信频繁，依赖高速互联 |
| Pipeline Parallel | Transformer 层 | 模型太深或参数太多 | pipeline bubble，需要 micro-batch 调度 |
| Sequence / Context Parallel | 序列或上下文维度 | 长上下文激活和 attention 压力 | attention 通信复杂 |
| Expert Parallel | MoE experts | 稀疏专家参数扩展 | token routing、all-to-all、负载均衡 |
| ZeRO / FSDP | 参数、梯度、优化器状态 | 减少数据并行冗余 | gather/scatter、通信和参数生命周期管理 |

ZeRO 的三阶段要背得非常准：

- **ZeRO-1**：分片 optimizer states。
- **ZeRO-2**：进一步分片 gradients。
- **ZeRO-3**：进一步分片 parameters，计算时按需 gather 参数。

可以用下面的显存直觉解释：

$$
M_{DP}\approx params+grads+optimizer
$$

每张卡都保留完整状态；而 ZeRO 希望让数据并行 rank 分摊状态：

$$
M_{ZeRO}\approx \frac{params+grads+optimizer}{N_{gpu}}+activation+communication
$$

**工程取舍：** ZeRO-3 更省显存，但会带来更多 all-gather / reduce-scatter；offload 可以省 GPU 显存，但 CPU / NVMe 带宽可能成为瓶颈。真实训练里常把 ZeRO/FSDP 和 TP/PP/SP 组合起来。

## 显存估算

参数显存最简单：

$$
M_{param}=N_{param}\times bytes
$$

例如 7B 参数 FP16/BF16 权重大约是：

$$
7\times10^9\times2 \approx 14GB
$$

完整训练显存至少包含：

$$
M_{train}\approx M_{param}+M_{grad}+M_{optimizer}+M_{activation}
$$

Adam / AdamW 通常保存一阶、二阶动量，优化器状态可能比参数本身还大。LoRA、QLoRA、ZeRO、activation checkpointing 都是在不同位置降低这几项开销。

KV cache 估算：

$$
M_{KV}=B\times L\times 2\times n_{layers}\times hidden\_size\times bytes
$$

其中 $B$ 是 batch size，$L$ 是序列长度，2 表示 key 和 value。长上下文和大 batch 推理时，KV cache 会快速膨胀。

**容易误解的点：**

- 推理显存不只是权重，还包括 KV cache。
- 训练显存不只是参数，还包括梯度、优化器状态和激活。
- 量化主要降低权重和部分 KV/激活开销，不能自动解决所有内存瓶颈。

## 推理优化主线

LLM 推理分为两个阶段：

| 阶段 | 主要工作 | 瓶颈特征 | 优化方向 |
| --- | --- | --- | --- |
| Prefill | 处理输入 prompt，构建 KV cache | 矩阵乘法并行度高，偏 compute-bound | FlashAttention、batching、kernel 优化 |
| Decode | 逐 token 自回归生成 | 每步读写 KV cache，偏 memory-bound | PagedAttention、KV 压缩、speculative decoding |

常见方法的边界：

| 方法 | 优化对象 | 不是在解决什么 |
| --- | --- | --- |
| FlashAttention | attention kernel 的 IO 和中间矩阵 | 不负责 KV cache 服务端分页管理 |
| PagedAttention | serving 中 KV cache 的块级管理 | 不改变 attention 数学本身 |
| Continuous batching | 动态合并请求，提高 GPU 利用率 | 不减少单个请求的理论计算量 |
| MQA / GQA | 减少 KV head，降低 KV cache | 可能影响表达能力，需要架构训练配合 |
| Speculative decoding | draft-verify 加速 decode | 不是训练目标本身，也不保证所有场景都提速 |
| MTP | 训练模型预测未来多个 token | 不等同于 speculative decoding，但可服务于多 token proposal |

**FlashAttention vs PagedAttention：**  
FlashAttention 优化 attention 计算本身，强调 exact attention 和 IO-aware tiling；PagedAttention 优化在线服务中的 KV cache 管理，减少碎片和过度预留。一个偏 kernel，一个偏 serving memory manager。

## 和知识库专题的关系

| 如果你在看 | 本章提供什么补充 |
| --- | --- |
| GRPO / DPO / OPD | 提供训练系统、显存、rollout 推理的基础 |
| MTP | 提供 speculative decoding、decode bottleneck、KV cache 基础 |
| 训练推理框架 | 提供 DeepSpeed / Megatron / vLLM / SGLang 背后的共性概念 |
| 优化器 | 提供 optimizer state 为什么占显存、为什么 ZeRO 要切 optimizer |
| Agent / RAG | 提供 serving、structured generation、多轮 KV 复用的基础 |

## 面试 QA

**Q1：为什么 LLM 更常用 LayerNorm / RMSNorm，而不是 BatchNorm？**  
A：BatchNorm 依赖 batch 统计，变长序列、自回归推理和小 batch 场景不稳定；LayerNorm / RMSNorm 在样本内部 hidden 维度归一化，不依赖 batch，更适合 Transformer。

**Q2：LoRA 到底省了什么？**  
A：省的是可训练参数、梯度和优化器状态。基础模型权重仍然要加载，但被冻结；训练时只更新低秩 adapter，推理时 adapter 可 merge。

**Q3：ZeRO-3 为什么更省显存但更复杂？**  
A：因为它连参数也分片，每卡常驻状态更少；但 forward/backward 需要按层 gather 参数，涉及通信、预取、释放和重叠调度。

**Q4：KV cache 为什么会成为长上下文推理瓶颈？**  
A：KV cache 随 batch size、序列长度、层数、hidden size 线性增长。decode 每步都要读历史 KV，所以长上下文下显存和带宽压力很大。

**Q5：FlashAttention 和 PagedAttention 的区别？**  
A：FlashAttention 是 attention kernel 优化，减少 HBM 读写；PagedAttention 是 serving 侧 KV cache 管理，减少碎片并支持更高效的动态 batching。

**Q6：MTP 和投机解码是什么关系？**  
A：MTP 是训练模型预测多个未来 token 的机制；投机解码是 draft-verify 推理加速范式。MTP 可以作为投机解码的 proposal 能力来源之一，但二者不等价。

## 知识索引引用

- [WDN LLM 面经站点](https://wdndev.github.io/llm_interview_note/#/)：用于补齐工程面试基础结构。
- [Layer Normalization 章节](https://github.com/wdndev/llm_interview_note/blob/main/02.%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/2.layer_normalization/2.layer_normalization.md)：用于归一化和 Pre-LN/Post-LN/RMSNorm/DeepNorm 复习。
- [LoRA 章节](https://github.com/wdndev/llm_interview_note/blob/main/05.%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83/4.lora/4.lora.md)：用于 PEFT 和 LoRA/QLoRA 补充。
- [分布式训练概述](https://github.com/wdndev/llm_interview_note/blob/main/04.%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/1.%E6%A6%82%E8%BF%B0/1.%E6%A6%82%E8%BF%B0.md)：用于 DP/TP/PP/ZeRO 等基础。
- [显存问题章节](https://github.com/wdndev/llm_interview_note/blob/main/04.%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/1.%E6%98%BE%E5%AD%98%E9%97%AE%E9%A2%98/1.%E6%98%BE%E5%AD%98%E9%97%AE%E9%A2%98.md)：用于显存估算和 GPU 利用率。
- [LLM 推理优化技术章节](https://github.com/wdndev/llm_interview_note/blob/main/06.%E6%8E%A8%E7%90%86/llm%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF/llm%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF.md)：用于 prefill/decode、KV cache、dynamic batching、PagedAttention、FlashAttention、speculative decoding。
- 经典论文卡片：LayerNorm、RMSNorm、DeepNorm、LoRA、AdaLoRA、QLoRA、FlashAttention、vLLM/PagedAttention。
