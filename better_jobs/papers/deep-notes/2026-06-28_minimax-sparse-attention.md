# Deep Note - MiniMax Sparse Attention

## 0) Metadata

- **Title:** MiniMax Sparse Attention
- **Alias:** MSA
- **Authors / Org:** Xunhao Lai, Weiqi Xu, Yufeng Yang, et al.; MiniMax, Peking University, NVIDIA, Zhejiang University, HUST, Nanjing University, Hangzhou Dianzi University
- **Venue / Status:** arXiv:2606.13392
- **Date:** 2026
- **Links:**
  - Abs: https://arxiv.org/abs/2606.13392
  - PDF: https://arxiv.org/pdf/2606.13392
  - Local PDF: `papers/model-series/latest-2026/minimax-sparse-attention.pdf`
- **Tags:** MiniMax, sparse attention, long context, GQA, block sparse attention, kernel co-design
- **My rating:** 4/5
- **Read depth:** deep, PDF text extraction + paper card
- **Scoring:** Base 1 + Quality 2 + Observation 1 = **4/5**

---

## 1) Why-read

MiniMax Sparse Attention 是一条非常适合面试讲解的长上下文效率路线：它用 blockwise sparse attention + GQA group-specific top-k selection + GPU kernel co-design，在 109B MoE 多模态模型上把 1M context 的 per-token attention compute 降低 28.4x，并报告显著 prefill / decoding 加速。

---

## 2) CRGP 拆解

### C - Context

Agentic workflows、仓库级代码理解、长文档、多模态任务都需要数十万到百万级上下文。标准 softmax attention 的二次复杂度让训练和推理都变得昂贵。

### R - Related work

- Hybrid architecture：把部分 softmax attention 替换为 linear attention 或 sliding window attention。
- Sparse softmax attention：保留 softmax 语义，但只在部分 token/block 上做注意力。
- MiniMax-01：此前强调 Lightning Attention 和超长上下文扩展。
- DeepSeek / Qwen 等也在探索 compressed attention、linear attention、sparse attention。

### G - Gap

很多稀疏注意力方法理论上省计算，但落到 GPU 上不一定快：top-k selection、block gather、tensor core utilization、GQA 访问模式都会影响真实吞吐。

### P - Proposal

MSA 保留 sparse softmax attention 路线，使用 blockwise token selection。Index Branch 给 KV blocks 打分，为每个 GQA group 独立选 top-k blocks；Main Branch 对选中的 blocks 做精确 block-sparse attention。

论文强调“简单、可部署、和硬件协同”：使用 exp-free top-k selection 和 KV-outer sparse attention 来提升实际 GPU 利用率。

---

## 3) Method

核心设计：

- **Blockwise sparse attention:** 以 KV block 为选择单位，而不是单 token。
- **Index Branch:** 轻量分支为 key/value blocks 打分。
- **Group-specific selection:** 每个 GQA group 独立选择 top-k blocks。
- **Main Branch:** 对被选中的 blocks 做精确 softmax attention。
- **Kernel co-design:** 专门优化 top-k 和 sparse attention execution path。

可以用一句话讲：MSA 不是简单把 attention mask 砍稀疏，而是把“选哪些 block”和“如何在 GPU 上高效算这些 block”一起设计。

---

## 4) Experiments - Key Numbers

### Model Setup

| Item | Value |
| --- | --- |
| Backbone | 41-layer MoE |
| Total parameters | ~109B |
| Activated parameters | ~6B per token |
| Dense layers | first 3 layers |
| MoE layers | remaining 38 layers |
| Vocabulary | 200K tokens |
| Training | native multimodal text + image/video data |

### Efficiency

| Metric | Result |
| --- | --- |
| Per-token attention compute at 1M context | 28.4x reduction |
| Prefill wall-clock speedup on H800 | 14.2x |
| Decoding wall-clock speedup on H800 | 7.6x |

### Top-k Kernel

| Seq Len | Blocks | k | Ours vs torch.topk | Ours vs TileLang |
| --- | --- | --- | --- | --- |
| 128K | 1024 | 16 | 5.1x | 3.7x |
| 128K | 2048 | 32 | 2.7x | 1.8x |
| 512K | 4096 | 16 | 4.3x | 2.3x |
| 512K | 8192 | 32 | 2.7x | 1.2x |

### Limitations

- 需要精读完整 ablation 才能判断 index branch、top-k、block size 各自贡献。
- 模型是 native multimodal 109B MoE，结果不一定直接迁移到小模型或纯文本模型。
- 稀疏注意力的准确性依赖 block selection 质量，可能在需要细粒度全局依赖的任务上出现风险。

---

## 5) Why it matters

1. 和 FlashMemory-DeepSeek-V4 可以形成强对比：MSA 更偏 block sparse attention + kernel co-design；FlashMemory 更偏 predictive KV cache recall。
2. 对国产模型系列章节很有价值：MiniMax 的技术路线可以明确标为长上下文效率、稀疏注意力、test-time compute。
3. 可实现算法点清晰：top-k block selection、block sparse mask、GQA group-specific selection。

## 6) Actionable next step

- [ ] 在原理代码中完善 `sparse-block-attention`，加 top-k block selection。
- [ ] 和 MiniMax-01 Lightning Attention 做差异表。
- [ ] 做 MSA vs FlashMemory vs sliding window vs full attention 对比矩阵。

## 7) 面试问答

**Q: MiniMax Sparse Attention 的核心改动是什么？**  
A: 用 Index Branch 对 KV blocks 打分，为每个 GQA group 选择 top-k blocks，然后 Main Branch 只对选中 blocks 做精确 block-sparse attention。

**Q: MSA 和普通 sliding window 有什么不同？**  
A: sliding window 只关注最近 token；MSA 可以根据 query/group 选择远处相关 blocks，更适合长上下文任务。

**Q: 为什么论文强调 kernel co-design？**  
A: 稀疏算法理论上省 FLOPs，但如果 top-k 和 block gather 执行不高效，实际 wall-clock 不一定快。MSA 同时优化算法和 GPU 执行路径。
