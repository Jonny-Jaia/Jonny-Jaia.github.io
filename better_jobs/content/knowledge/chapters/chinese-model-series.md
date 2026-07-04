# 国产模型系列：GLM / Kimi / MiniMax / Qwen / DeepSeek

## 当前定位

本章只保留模型家族的学习入口和横向比较框架。具体论文精读、实验数字和细节结论统一放在论文库与 `papers/deep-notes/`，避免知识库正文变成论文摘要堆。

## 资料入口

- 代表论文：`papers/model-series/`
- 2026 最新论文：`papers/model-series/latest-2026/`
- 模型卡来源：`papers/model-series/model-cards/`
- 精读笔记：`papers/deep-notes/`

## 系列路线索引

### GLM

- 关注：GLM 架构传统、双语预训练、Agentic reasoning、Deep Research、最新 GLM-5.x 模型卡。
- 证据边界：GLM-5.1 / GLM-5.2 当前以模型卡为主，等待官方技术报告或论文。

### Kimi

- 关注：长上下文、RL scaling、Agentic Intelligence、多模态视觉 Agent。
- 代表入口：Kimi k1.5、Kimi K2、Kimi K2.5。

### MiniMax

- 关注：Lightning Attention、Sparse Attention、超长上下文、test-time compute。
- 代表入口：MiniMax-01、MiniMax-M1、MiniMax Sparse Attention。

### Qwen

- 关注：模型族工程生态、dense/MoE 路线、thinking/non-thinking 模式、Qwen3.6 模型卡。
- 证据边界：Qwen3.6 当前同时保留模型卡和本地推理论文，不能把本地推理论文当成官方技术报告。

### DeepSeek

- 关注：MoE、MLA、MTP、reasoning RL、GRPO/RLVR、训练效率、长上下文 KV cache。
- 代表入口：DeepSeek-V3、DeepSeek-R1、DeepSeekMath、FlashMemory-DeepSeek-V4。
- 证据边界：FlashMemory-DeepSeek-V4 是 DeepSeek-V4 相关长上下文推理论文，不是官方 DeepSeek-V4 technical report。

## 横向比较框架

| 维度 | 看什么 |
| --- | --- |
| 架构 | Dense / MoE、激活参数、路由策略、训练稳定性 |
| Attention | MLA、Lightning Attention、Sparse Attention、长上下文 KV cache |
| 后训练 | SFT、DPO/OPD、RLHF、GRPO/RLVR、蒸馏 |
| 推理能力 | thinking 模式、test-time compute、self-verification |
| Agent | 工具调用、规划、多步任务成功率、多模态交互 |
| 工程优化 | 推理吞吐、显存、并行策略、量化、部署成本 |

## 可实现算法点

- MoE top-k routing
- MLA / KV cache 压缩的 toy 版本
- Block sparse attention mask
- Memory indexer / KV chunk recall
- MTP 多 token 预测 loss
- GRPO group relative advantage

## 待交互问题

- DeepSeek-V3 的 MLA、MoE、MTP 分别解决什么问题？
- MiniMax Sparse Attention 和 FlashMemory-DeepSeek-V4 的优化目标有什么不同？
- Qwen3 thinking 模式和 reasoning RL 是同一个概念吗？
- 最新模型只有模型卡时，面试里应该如何谨慎表述？

## 面试 QA

**Q：国产模型系列应该如何横向比较？**

A：不要只按榜单分数比较，应该拆成架构、训练目标、后训练、长上下文、MoE、推理框架、Agent/工具调用、多模态和开源证据边界。比如 DeepSeek 更适合讨论 MoE、MTP、RLVR 和 serving；Qwen 更适合讨论通用开源生态、MoE 与推理部署；MiniMax 更适合讨论长上下文和 sparse attention；Kimi 更适合讨论长链路推理和 agentic intelligence；GLM 更适合讨论国产开源路线与 Agent 能力扩展。

**Q：只有 model card 没有论文时，面试中能不能引用？**

A：可以引用，但要明确证据边界。model card 可以说明模型能力、开源状态、上下文长度、部署参数和官方评测，但不能等同于论文中的方法细节。涉及架构、训练数据、损失函数和定量 ablation 时，应该回到论文、技术报告或可验证代码。

**Q：为什么这个章节不直接写成论文精读？**

A：模型家族页的作用是横向导航和面试比较，具体论文细节应该沉淀到论文库精读卡或专题章节。这样可以避免知识库变成长篇堆砌，也方便后续随模型更新替换证据。

## 知识索引引用

| 知识点 | 主要来源 | 本页使用方式 |
|---|---|---|
| GLM / Kimi / MiniMax / Qwen / DeepSeek 系列横向比较 | 本地论文库 `papers/model-series/` 与 model cards | 用于构建模型家族对比框架 |
| DeepSeek-V3 / R1 / DeepSeekMath | DeepSeek 系列论文 | 用于关联 MoE、MTP、GRPO/RLVR 与 reasoning 训练 |
| MiniMax sparse attention、Qwen3.6、GLM-5.x、Kimi K2.x | 2026 最新论文、技术报告与 model cards | 用于标注“最新信息”和证据边界 |
