# LLM 工程基础面试补充

> 这一页现在定位为**知识库索引页**：负责告诉你 LLM 工程面试会追问哪些方向、每个方向应该先去哪里复习、面试时如何组织回答。长期稳定的底层机制已经迁移到 [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations)。

```mermaid
flowchart LR
  n1["面试追问"] --> n2["判断所属基础块"]
  n2["判断所属基础块"] --> n3["跳转基础知识深读"]
  n3["跳转基础知识深读"] --> n4["回到专题形成回答"]
```

## 为什么拆到基础知识

之前这一页把归一化、LoRA、分布式训练、显存估算和推理优化都展开讲，短期看完整，但长期会带来两个问题：

- **知识库专题过长**：GRPO、OPD、MTP、Agent 等专题本身已经很重，如果每个专题都塞入底层基础，会导致复习路径混乱。
- **基础概念会被重复解释**：例如 KV cache、ZeRO、LoRA、RMSNorm 会同时出现在 MTP、训练框架、SFT、推理优化里，应该有一个稳定底座统一维护。

因此现在采用两层结构：

| 层级 | 放什么 | 示例 |
| --- | --- | --- |
| 基础知识 | 长期稳定、跨专题复用的机制 | [LLM 工程基础](#foundations/llm-engineering-foundations)、深度学习基础、强化学习基础 |
| 知识库 | 面试专题、方法谱系、论文线和当前热点 | GRPO、DPO、OPD、MTP、训练推理框架、Agent |

## 面试补漏导航

| 面试追问 | 先看基础章节 | 再回到知识库专题 |
| --- | --- | --- |
| LayerNorm / RMSNorm / Pre-LN / Post-LN | [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations) | [位置编码与 Transformer 结构](#knowledge/positional-encoding)、[MoE 架构](#knowledge/moe-architecture) |
| LoRA / AdaLoRA / QLoRA | [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations) | [SFT](#knowledge/sft)、[训练推理框架](#knowledge/training-inference-frameworks) |
| DP / TP / PP / SP / ZeRO / Offload | [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations) | [训练推理框架](#knowledge/training-inference-frameworks)、[VeRL](#knowledge/verl-rl-framework) |
| 显存估算 / optimizer state / activation / KV cache | [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations) | [优化器](#knowledge/optimization-training)、[MTP](#knowledge/mtp)、[训练推理框架](#knowledge/training-inference-frameworks) |
| prefill / decode / FlashAttention / PagedAttention / speculative decoding | [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations) | [MTP](#knowledge/mtp)、[训练推理框架](#knowledge/training-inference-frameworks) |
| rollout / reward / logprob / advantage / worker 数据流 | [强化学习基础](#foundations/rl-foundations)、[LLM 工程基础](#foundations/llm-engineering-foundations) | [GRPO](#knowledge/grpo)、[VeRL](#knowledge/verl-rl-framework)、[OPD](#knowledge/opd) |

## 回答组织模板

如果面试官问一个工程基础问题，建议用四段式回答：

1. **定义**：先说它解决什么问题。
2. **机制**：讲清核心公式、数据流或系统对象。
3. **取舍**：说明它省了什么，又引入什么代价。
4. **关联**：连接到后训练、推理框架或具体论文。

示例：如果问 “PagedAttention 和 FlashAttention 的区别”，不要只回答关键词。更好的结构是：

- FlashAttention 优化 attention kernel 的 IO 和中间矩阵，是计算层优化。
- PagedAttention 优化 serving 侧 KV cache 的块级管理，是系统内存管理优化。
- 两者都提升长序列/高吞吐场景效率，但解决对象不同。
- 在 RL rollout、vLLM serving、MTP/投机解码场景中，KV cache 管理和 decode 调度会变得很关键。

## 当前内容密度策略

| 文档类型 | 展示策略 | 写作策略 |
| --- | --- | --- |
| 基础知识长文 | 保留详细公式、表格、边界和 QA | 讲清原理，避免只给结论 |
| 知识库专题 | 保留方法谱系、论文线、面试主线 | 少重复基础概念，多给跳转 |
| 论文库卡片 | 保留问题、方法、贡献、局限、面试题 | 不替代论文精读，只做索引 |
| 代码训练 | 保留题目、模板、薄弱点记录 | 与知识解释分离 |

## 面试 QA

**Q1：为什么 LLM 工程基础要放到基础知识，而不是一直放在知识库专题里？**  
A：因为归一化、LoRA、ZeRO、KV cache、FlashAttention 这类概念跨多个专题复用。放到基础知识可以统一维护，专题页只保留“如何用于 GRPO/MTP/Agent/框架面试”的上下文。

**Q2：如果我在看 MTP 时忘了 KV cache 或 speculative decoding，应该怎么办？**  
A：先跳到 [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations) 看推理优化主线，再回到 [MTP](#knowledge/mtp) 看 MTP 与投机解码、DSpark、verl MTP 支持的关系。

**Q3：如果我准备训练框架面试，应该先看哪几章？**  
A：先看 [LLM 工程基础](#foundations/llm-engineering-foundations) 建立并行训练和推理优化底座，再看 [训练推理框架](#knowledge/training-inference-frameworks) 做 DeepSpeed / Megatron / vLLM / SGLang 横向比较，最后看 [VeRL](#knowledge/verl-rl-framework) 理解 RL 后训练数据流。

## 知识索引引用

- [WDN LLM 面经站点](https://wdndev.github.io/llm_interview_note/#/)：用于确定工程基础面试补漏范围。
- [基础知识：LLM 工程基础](#foundations/llm-engineering-foundations)：承载归一化、PEFT、分布式训练、显存估算、推理优化的详细解释。
- [训练推理框架](#knowledge/training-inference-frameworks)：承载 DeepSpeed、Megatron-LM、VeRL、Slime、vLLM、SGLang 的横向比较。
- [VeRL](#knowledge/verl-rl-framework)：承载 RL 后训练 dataflow、WorkerGroup、DataProto、ResourcePool 等框架细节。
- 相关论文卡片：LayerNorm、RMSNorm、DeepNorm、LoRA、AdaLoRA、QLoRA、FlashAttention、vLLM/PagedAttention。
