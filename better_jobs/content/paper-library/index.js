window.BJ_CONTENT = window.BJ_CONTENT || {};

const BJ_DEEPSEEK_PAPERS = window.BJ_CONTENT.deepseekPapers?.papers || [];

window.BJ_CONTENT.paperLibrary = {
  id: "paper-library",
  title: "论文库",
  eyebrow: "Paper Library",
  description: "这里统一保存模型系列、DeepSeek、OPD/MTP、后训练方法等论文卡片，并通过分组区分章节。",
  papers: [
  {
    "id": "layer-normalization",
    "group": "LLM 工程基础",
    "title": "Layer Normalization",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/1607.06450",
    "linkedTopics": [
      "llm-engineering-interview-supplement"
    ],
    "problem": "BatchNorm 依赖 batch 统计，在 RNN / Transformer 等序列模型和自回归推理中不够稳定。",
    "method": "在样本内部的 hidden dimension 上做均值方差归一化，避免依赖 batch 维度统计。",
    "contributions": [
      "成为 Transformer 结构稳定训练的核心组件",
      "为 Pre-LN / Post-LN / RMSNorm 等后续结构讨论提供基础"
    ],
    "limitations": [
      "仍包含均值中心化和方差计算，极致效率场景下会被 RMSNorm 等简化变体替代"
    ],
    "interviewQuestions": [
      "为什么 LLM 通常用 LayerNorm 而不是 BatchNorm？"
    ]
  },
  {
    "id": "rmsnorm",
    "group": "LLM 工程基础",
    "title": "Root Mean Square Layer Normalization",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/1910.07467",
    "linkedTopics": [
      "llm-engineering-interview-supplement"
    ],
    "problem": "LayerNorm 的均值中心化和方差计算存在额外开销，部分场景下不是必要条件。",
    "method": "只使用 root mean square 做尺度归一化，省去 re-centering。",
    "contributions": [
      "降低归一化计算成本",
      "成为 LLaMA 等现代 LLM 架构常见组件"
    ],
    "limitations": [
      "不是所有模型都必然优于 LayerNorm，需要结合架构和训练规模判断"
    ],
    "interviewQuestions": [
      "RMSNorm 相比 LayerNorm 省掉了什么，为什么仍然有效？"
    ]
  },
  {
    "id": "deepnorm",
    "group": "LLM 工程基础",
    "title": "DeepNet / DeepNorm",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/2203.00555",
    "linkedTopics": [
      "llm-engineering-interview-supplement"
    ],
    "problem": "非常深的 Transformer 使用 Post-LN 时容易出现训练不稳定。",
    "method": "通过残差缩放和初始化策略稳定深层 Transformer。",
    "contributions": [
      "提供深层 Post-LN Transformer 稳定训练方案",
      "适合作为 Pre-LN / Post-LN 取舍的面试延伸"
    ],
    "limitations": [
      "工程实现和超参设计比标准 Pre-LN 更复杂"
    ],
    "interviewQuestions": [
      "Pre-LN 更稳定时，为什么还会有人研究 DeepNorm/Post-LN 稳定化？"
    ]
  },
  {
    "id": "lora",
    "group": "参数高效微调",
    "title": "LoRA: Low-Rank Adaptation of Large Language Models",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/2106.09685",
    "linkedTopics": [
      "llm-engineering-interview-supplement",
      "sft",
      "training-inference-frameworks"
    ],
    "problem": "全参微调大模型需要保存和更新大量参数、梯度与优化器状态。",
    "method": "冻结原权重，只训练低秩增量矩阵 BA，推理时可合并回原权重。",
    "contributions": [
      "显著降低可训练参数量和优化器状态",
      "成为 SFT / 偏好优化 / 领域适配中最常见的 PEFT 方法之一"
    ],
    "limitations": [
      "rank、target modules、alpha 和数据规模需要调参，能力上限受 adapter 容量限制"
    ],
    "interviewQuestions": [
      "LoRA 省显存主要省在哪里？为什么推理时可以 merge？"
    ]
  },
  {
    "id": "adalora",
    "group": "参数高效微调",
    "title": "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/2303.10512",
    "linkedTopics": [
      "llm-engineering-interview-supplement",
      "sft"
    ],
    "problem": "固定 rank 的 LoRA 无法区分不同权重矩阵对任务的重要性。",
    "method": "根据重要性评分动态分配 rank 预算，把参数容量分给更关键的模块。",
    "contributions": [
      "把 PEFT 从固定结构推进到自适应预算分配",
      "适合作为 LoRA 面试追问的延伸"
    ],
    "limitations": [
      "训练流程和重要性评估更复杂，工程落地不如标准 LoRA 普遍"
    ],
    "interviewQuestions": [
      "AdaLoRA 解决的是 LoRA 的什么问题？"
    ]
  },
  {
    "id": "qlora",
    "group": "参数高效微调",
    "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/2305.14314",
    "linkedTopics": [
      "llm-engineering-interview-supplement",
      "sft",
      "training-inference-frameworks"
    ],
    "problem": "在消费级或有限显存 GPU 上微调大模型仍然困难。",
    "method": "使用 4-bit NF4 存储基础模型，结合 double quantization、paged optimizer 和 LoRA adapter 训练。",
    "contributions": [
      "显著降低大模型微调显存门槛",
      "把量化与 PEFT 结合成可复用训练范式"
    ],
    "limitations": [
      "通常不是更新 4-bit 基础权重本身，而是训练 adapter；量化误差与任务难度仍需评估"
    ],
    "interviewQuestions": [
      "QLoRA 是不是 4-bit 全参训练？NF4、double quantization、paged optimizer 分别解决什么？"
    ]
  },
  {
    "id": "flashattention",
    "group": "推理与训练系统",
    "title": "FlashAttention",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/2205.14135",
    "linkedTopics": [
      "llm-engineering-interview-supplement",
      "training-inference-frameworks"
    ],
    "problem": "标准 attention 的显存访问和中间矩阵开销很高，长序列下效率受限。",
    "method": "通过 tiling 和 IO-aware 设计，在不近似 attention 的情况下减少 HBM 读写。",
    "contributions": [
      "成为长上下文训练和推理优化中的基础 kernel 思路",
      "面试中常与 PagedAttention、KV cache 优化一起比较"
    ],
    "limitations": [
      "主要优化 attention kernel 本身，不直接解决 KV cache 管理和服务端调度问题"
    ],
    "interviewQuestions": [
      "FlashAttention 为什么说是 exact attention？它主要省的是计算还是 IO？"
    ]
  },
  {
    "id": "pagedattention-vllm",
    "group": "推理与训练系统",
    "title": "vLLM / PagedAttention",
    "status": "索引完成",
    "source": "https://arxiv.org/abs/2309.06180",
    "linkedTopics": [
      "llm-engineering-interview-supplement",
      "training-inference-frameworks",
      "mtp"
    ],
    "problem": "在线推理服务中 KV cache 容易因变长请求产生碎片和过度预留，影响吞吐。",
    "method": "借鉴虚拟内存分页，把 KV cache 切成 block，通过 block table 管理请求的缓存。",
    "contributions": [
      "显著提升 LLM serving 的吞吐与显存利用率",
      "成为 vLLM 的核心技术点，也是推理系统面试高频问题"
    ],
    "limitations": [
      "解决的是 KV cache 管理与调度，不等价于模型压缩或投机解码"
    ],
    "interviewQuestions": [
      "PagedAttention 和 FlashAttention 的优化对象有什么不同？"
    ]
  },

  {
    "id": "deepseekmath",
    "group": "DeepSeek",
    "title": "DeepSeekMath",
    "status": "已下载",
    "source": "papers/model-series/deepseekmath.pdf",
    "linkedTopics": [
      "grpo",
      "sampling-evaluation-rft",
      "chinese-model-series"
    ],
    "problem": "理解 GRPO 在数学推理 RLVR 路线中的原始应用背景。",
    "method": "围绕数学推理数据、RL 训练和 group relative policy optimization 展开。",
    "contributions": [
      "GRPO 核心出处之一",
      "可验证数学推理后训练路线"
    ],
    "limitations": [
      "需要结合 DeepSeek-R1 和后续 GRPO 改进理解其演进"
    ],
    "interviewQuestions": [
      "DeepSeekMath 为什么适合作为 GRPO 的背景论文？"
    ]
  },
  {
    "id": "deepseek-r1",
    "group": "DeepSeek",
    "title": "DeepSeek-R1",
    "status": "已下载",
    "source": "papers/model-series/deepseek-r1.pdf",
    "linkedTopics": [
      "grpo",
      "sampling-evaluation-rft",
      "moe-architecture",
      "chinese-model-series"
    ],
    "problem": "理解 reasoning RL / RLVR 如何推动大模型推理能力提升。",
    "method": "围绕冷启动、RL、蒸馏、可验证奖励和推理能力评估展开。",
    "contributions": [
      "reasoning RL 代表路线",
      "GRPO/RLVR 面试高频背景"
    ],
    "limitations": [
      "需要区分论文结论、复现经验和社区二次解读"
    ],
    "interviewQuestions": [
      "DeepSeek-R1 的 RL 路线和传统 RLHF 有什么不同？"
    ]
  },
  {
    "id": "deepseek-v3",
    "group": "DeepSeek",
    "title": "DeepSeek-V3",
    "status": "已下载",
    "source": "papers/model-series/deepseek-v3.pdf",
    "linkedTopics": [
      "mtp",
      "moe-architecture",
      "chinese-model-series"
    ],
    "problem": "理解 DeepSeek-V3 中 MoE、MTP、训练效率和推理效率相关设计。",
    "method": "从架构、训练目标、负载均衡、MTP 和系统优化角度精读。",
    "contributions": [
      "MTP 训练目标参考",
      "MoE 架构与训练系统参考"
    ],
    "limitations": [
      "后续 DeepSeek-V4 / DSpark 需要单独作为推理侧演进看待"
    ],
    "interviewQuestions": [
      "DeepSeek-V3 中 MTP 和 MoE 分别解决什么问题？"
    ]
  },
  {
    "id": "flashmemory-deepseek-v4",
    "group": "DeepSeek",
    "title": "FlashMemory / DeepSeek-V4 note",
    "status": "已下载",
    "source": "papers/model-series/latest-2026/flashmemory-deepseek-v4.pdf",
    "linkedTopics": [
      "chinese-model-series"
    ],
    "problem": "跟踪 DeepSeek-V4 相关最新推理与记忆机制讨论。",
    "method": "作为 2026 最新资料先进入论文库，后续再精读沉淀到专题页。",
    "contributions": [
      "DeepSeek 最新资料索引",
      "国产模型系列更新入口"
    ],
    "limitations": [
      "需要持续核验官方论文、技术报告和 model card 的证据等级"
    ],
    "interviewQuestions": [
      "最新模型资料没有完整论文时，如何判断哪些结论可以用于面试？"
    ]
  },
  {
    "id": "instructgpt-reading-card",
    "group": "RLHF",
    "title": "InstructGPT / RLHF reading card",
    "status": "待读",
    "source": "papers/post-training/ppo.pdf",
    "linkedTopics": [
      "sft",
      "dpo",
      "grpo"
    ],
    "problem": "作为 SFT、RM、PPO-RLHF 三阶段流程的背景索引。",
    "method": "暂以 PPO 与 RLHF 基础流程作为阅读入口，后续可补 InstructGPT 原论文。",
    "contributions": [
      "SFT/RLHF 面试背景",
      "偏好优化方法对比基线"
    ],
    "limitations": [
      "当前 source 暂指向 PPO，本地后续应补齐 InstructGPT PDF"
    ],
    "interviewQuestions": [
      "SFT、RM、PPO 三阶段分别解决什么问题？"
    ]
  },
  {
    "id": "glm-130b",
    "group": "GLM",
    "title": "GLM-130B",
    "status": "已下载",
    "source": "papers/model-series/glm-130b.pdf",
    "linkedTopics": [
      "chinese-model-series"
    ],
    "problem": "理解 GLM 系列早期大规模双语预训练模型和 GLM 架构传统。",
    "method": "待精读：blank infilling / autoregressive pretraining、双语数据、训练稳定性。",
    "contributions": [
      "GLM 系列基础路线",
      "国产大规模预训练模型代表"
    ],
    "limitations": [
      "与最新 agentic / reasoning 模型存在代际差距"
    ],
    "interviewQuestions": [
      "GLM 架构和标准 decoder-only LLM 有什么差异？"
    ]
  },
  {
    "id": "glm-4-5",
    "group": "GLM",
    "title": "GLM-4.5",
    "status": "已下载",
    "source": "papers/model-series/glm-4-5.pdf",
    "linkedTopics": [
      "chinese-model-series",
      "agent"
    ],
    "problem": "理解 GLM-4.5 在 agentic reasoning 和 deep research 方向的定位。",
    "method": "待精读：Agent 任务、推理训练、工具/研究能力评估。",
    "contributions": [
      "Agentic reasoning 路线",
      "Deep research 能力评估入口"
    ],
    "limitations": [
      "GLM-5.x 最新信息需结合模型卡继续跟踪"
    ],
    "interviewQuestions": [
      "GLM-4.5 的 agentic reasoning 和普通推理能力有什么区别？"
    ]
  },
  {
    "id": "glm-5-model-cards",
    "group": "GLM",
    "title": "GLM-5.1 / GLM-5.2 model cards",
    "status": "模型卡已下载",
    "source": "papers/model-series/model-cards/",
    "linkedTopics": [
      "chinese-model-series"
    ],
    "problem": "跟踪 GLM-5.1 / GLM-5.2 最新开源信息。",
    "method": "以 Hugging Face model card 为来源；待补技术报告或论文。",
    "contributions": [
      "最新开源状态跟踪",
      "模型卡证据"
    ],
    "limitations": [
      "不是论文证据，面试中需要谨慎表述"
    ],
    "interviewQuestions": [
      "最新模型只有模型卡没有论文时，如何进行可信技术分析？"
    ]
  },
  {
    "id": "kimi-k1-5",
    "group": "Kimi",
    "title": "Kimi k1.5",
    "status": "已下载",
    "source": "papers/model-series/kimi-k1-5.pdf",
    "linkedTopics": [
      "chinese-model-series",
      "grpo"
    ],
    "problem": "理解 Kimi 系列 reinforcement learning scaling 与长链路推理能力。",
    "method": "待精读：RL scaling、长上下文推理、测试时计算。",
    "contributions": [
      "RL scaling 线索",
      "推理模型训练路线"
    ],
    "limitations": [
      "需要和 DeepSeek-R1、MiniMax-M1 横向比较"
    ],
    "interviewQuestions": [
      "Kimi k1.5 的 RL scaling 和 DeepSeek-R1 的 RL 路线有什么可比较点？"
    ]
  },
  {
    "id": "kimi-k2",
    "group": "Kimi",
    "title": "Kimi K2",
    "status": "已下载",
    "source": "papers/model-series/kimi-k2.pdf",
    "linkedTopics": [
      "chinese-model-series",
      "agent"
    ],
    "problem": "理解 Kimi K2 的 agentic intelligence、MoE 和工具调用能力。",
    "method": "待精读：MoE 架构、Agent 评测、工具调用训练。",
    "contributions": [
      "Agentic intelligence 路线",
      "国产 MoE Agent 模型代表"
    ],
    "limitations": [
      "需要拆分模型结构贡献和训练数据贡献"
    ],
    "interviewQuestions": [
      "Kimi K2 的 Agent 能力应该如何评估？"
    ]
  },
  {
    "id": "kimi-k2-5",
    "group": "Kimi",
    "title": "Kimi K2.5 visual agentic intelligence",
    "status": "已下载",
    "source": "papers/model-series/latest-2026/kimi-k2-5-visual-agentic-intelligence.pdf",
    "linkedTopics": [
      "chinese-model-series",
      "agent"
    ],
    "problem": "理解 Kimi K2.5 在视觉 Agent / 多模态任务上的扩展。",
    "method": "待精读：视觉推理、GUI/工具交互、多模态 Agent 评估。",
    "contributions": [
      "多模态 Agent 路线",
      "Kimi 2026 最新线索"
    ],
    "limitations": [
      "需区分官方论文和第三方评测论文"
    ],
    "interviewQuestions": [
      "视觉 Agent 相比文本 Agent 多了哪些评估难点？"
    ]
  },
  {
    "id": "minimax-01",
    "group": "MiniMax",
    "title": "MiniMax-01",
    "status": "已下载",
    "source": "papers/model-series/minimax-01.pdf",
    "linkedTopics": [
      "chinese-model-series"
    ],
    "problem": "理解 Lightning Attention 和超长上下文模型扩展。",
    "method": "待精读：Lightning Attention、上下文长度扩展、训练效率。",
    "contributions": [
      "长上下文效率路线",
      "MiniMax 基础模型代表"
    ],
    "limitations": [
      "需要与后续 Sparse Attention 区分"
    ],
    "interviewQuestions": [
      "Lightning Attention 解决了长上下文中的什么瓶颈？"
    ]
  },
  {
    "id": "minimax-m1",
    "group": "MiniMax",
    "title": "MiniMax-M1",
    "status": "已下载",
    "source": "papers/model-series/minimax-m1.pdf",
    "linkedTopics": [
      "chinese-model-series",
      "grpo"
    ],
    "problem": "理解 test-time compute 和高效 RL 推理扩展。",
    "method": "待精读：test-time scaling、RL、推理预算与训练稳定性。",
    "contributions": [
      "Test-time compute 路线",
      "高效 reasoning RL 线索"
    ],
    "limitations": [
      "需要和 Kimi k1.5、DeepSeek-R1 比较"
    ],
    "interviewQuestions": [
      "为什么 reasoning 模型要关注 test-time compute？"
    ]
  },
  {
    "id": "minimax-sparse-attention",
    "group": "MiniMax",
    "title": "MiniMax Sparse Attention",
    "status": "已精读",
    "source": "papers/model-series/latest-2026/minimax-sparse-attention.pdf",
    "linkedTopics": [
      "chinese-model-series"
    ],
    "problem": "理解 2026 MiniMax 开源模型相关稀疏注意力机制。",
    "method": "Index Branch 为 KV blocks 打分，每个 GQA group 选择 top-k blocks；Main Branch 做精确 block-sparse attention，并配合 GPU kernel co-design。",
    "contributions": [
      "1M context per-token attention compute 降低 28.4x",
      "H800 上 prefill 14.2x、decoding 7.6x speedup"
    ],
    "limitations": [
      "需要继续精读 ablation，拆分 index branch、top-k、block size、kernel 各自贡献"
    ],
    "interviewQuestions": [
      "Sparse Attention 和 Lightning Attention 的优化目标有什么不同？"
    ]
  },
  {
    "id": "qwen2-5",
    "group": "Qwen",
    "title": "Qwen2.5",
    "status": "已下载",
    "source": "papers/model-series/qwen2-5.pdf",
    "linkedTopics": [
      "chinese-model-series"
    ],
    "problem": "理解 Qwen2.5 模型族、代码、多语言和工程生态。",
    "method": "待精读：模型族、数据、后训练、评估。",
    "contributions": [
      "Qwen 模型族路线",
      "开源生态代表"
    ],
    "limitations": [
      "和 Qwen3 thinking 模式需要分开讲"
    ],
    "interviewQuestions": [
      "Qwen2.5 的模型族设计对工程落地有什么价值？"
    ]
  },
  {
    "id": "qwen3",
    "group": "Qwen",
    "title": "Qwen3",
    "status": "已下载",
    "source": "papers/model-series/qwen3.pdf",
    "linkedTopics": [
      "chinese-model-series",
      "mtp"
    ],
    "problem": "理解 Qwen3 的 dense/MoE、thinking/non-thinking、Agent 和 coding 能力。",
    "method": "待精读：thinking 模式、MoE、训练与评测。",
    "contributions": [
      "Thinking 模式控制",
      "Qwen 新一代模型族"
    ],
    "limitations": [
      "Qwen3.6 最新模型卡需单独跟踪"
    ],
    "interviewQuestions": [
      "thinking/non-thinking 模式在系统侧如何控制？"
    ]
  },
  {
    "id": "qwen3-6-inference",
    "group": "Qwen",
    "title": "Qwen3.6-35B-A3B local inference",
    "status": "已下载",
    "source": "papers/model-series/latest-2026/qwen3-6-35b-a3b-local-inference.pdf",
    "linkedTopics": [
      "chinese-model-series"
    ],
    "problem": "理解 Qwen3.6 本地推理工程侧问题。",
    "method": "待精读：本地部署、推理性能、显存和吞吐。",
    "contributions": [
      "Qwen3.6 工程参考",
      "本地推理关注点"
    ],
    "limitations": [
      "不是 Qwen3.6 官方技术报告，需要结合模型卡"
    ],
    "interviewQuestions": [
      "模型技术报告和本地推理论文分别能支持什么结论？"
    ]
  },
  {
    "id": "react",
    "group": "Agent",
    "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "status": "已下载",
    "source": "papers/agent/react.pdf",
    "linkedTopics": [
      "agent"
    ],
    "problem": "理解 Agent 中 reasoning 与 acting 如何通过 Thought-Action-Observation 闭环结合。",
    "method": "让模型交替生成推理、工具动作和环境观察，提升多步任务可解释性和工具使用能力。",
    "contributions": [
      "ReAct 范式核心论文",
      "Reason + Act + Observe 控制循环"
    ],
    "limitations": [
      "推理轨迹可能冗长，工具错误会向后传播"
    ],
    "interviewQuestions": [
      "ReAct 和 Function Calling 的区别是什么？"
    ]
  },
  {
    "id": "rag",
    "group": "Agent",
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "status": "已下载",
    "source": "papers/agent/rag.pdf",
    "linkedTopics": [
      "agent"
    ],
    "problem": "理解 RAG 如何通过检索外部知识缓解模型知识过期和不可追溯问题。",
    "method": "先检索相关文档，再把检索结果作为上下文条件进行生成。",
    "contributions": [
      "RAG 基础论文",
      "检索增强生成范式"
    ],
    "limitations": [
      "最终质量强依赖召回、rerank、chunk 和上下文组织"
    ],
    "interviewQuestions": [
      "RAG 的瓶颈为什么通常不在生成，而在检索？"
    ]
  },
  {
    "id": "reflexion",
    "group": "Agent",
    "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
    "status": "已下载",
    "source": "papers/agent/reflexion.pdf",
    "linkedTopics": [
      "agent"
    ],
    "problem": "理解 Reflection 如何让 Agent 从失败尝试中生成语言反馈并改进行为。",
    "method": "将执行结果转化为 verbal reflection，再作为记忆/反馈影响下一次尝试。",
    "contributions": [
      "Reflection 范式代表",
      "语言形式的经验反馈"
    ],
    "limitations": [
      "缺少外部验证时，自我反思可能不可靠"
    ],
    "interviewQuestions": [
      "Reflection 为什么最好和测试、规则或检索证据结合？"
    ]
  },
  {
    "id": "tree-of-thoughts",
    "group": "Agent",
    "title": "Tree of Thoughts",
    "status": "已下载",
    "source": "papers/agent/tree-of-thoughts.pdf",
    "linkedTopics": [
      "agent"
    ],
    "problem": "理解 Planning 中如何对多个推理分支进行搜索和评估。",
    "method": "把中间思路作为树节点，通过生成、评估和搜索选择更优解题路径。",
    "contributions": [
      "Planning / search 范式",
      "多分支推理与评估"
    ],
    "limitations": [
      "搜索成本高，评估器质量决定上限"
    ],
    "interviewQuestions": [
      "Tree of Thoughts 和普通 CoT 的区别是什么？"
    ]
  },
  {
    "id": "generative-agents",
    "group": "Agent",
    "title": "Generative Agents",
    "status": "已下载",
    "source": "papers/agent/generative-agents.pdf",
    "linkedTopics": [
      "agent"
    ],
    "problem": "理解 Memory 在长期交互、角色行为和社会模拟中的作用。",
    "method": "通过 observation、reflection、planning 和 memory stream 组织长期行为。",
    "contributions": [
      "Agent Memory 经典案例",
      "长期情景记忆与反思机制"
    ],
    "limitations": [
      "工程系统中要额外处理隐私、记忆污染和过期机制"
    ],
    "interviewQuestions": [
      "Agent 的长期记忆为什么不是简单保存全部历史？"
    ]
  },
  {
    "id": "autogen",
    "group": "Agent",
    "title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
    "status": "已下载",
    "source": "papers/agent/autogen.pdf",
    "linkedTopics": [
      "agent"
    ],
    "problem": "理解 Multi-Agent 如何通过多角色对话组织复杂任务。",
    "method": "用多个可配置 Agent 进行协作、代码执行、审查和任务完成。",
    "contributions": [
      "Multi-Agent 系统代表",
      "角色分工与对话式协作"
    ],
    "limitations": [
      "协调成本和错误共振风险较高"
    ],
    "interviewQuestions": [
      "Multi-Agent 系统如何设计角色、共享状态和停止条件？"
    ]
  },
  {
    "id": "adam",
    "group": "Optimization",
    "title": "Adam: A Method for Stochastic Optimization",
    "status": "已下载 / 已初读",
    "source": "papers/optimization/adam.pdf",
    "linkedTopics": [
      "optimization-training"
    ],
    "problem": "理解 Adam 为什么成为深度学习默认优化器，以及它如何结合一阶动量和二阶矩估计。",
    "method": "使用梯度一阶矩和平方梯度二阶矩的指数滑动平均，并通过 bias correction 修正训练初期偏差。",
    "contributions": [
      "Adam/AdamW 的基础",
      "自适应学习率优化器核心起点"
    ],
    "limitations": [
      "Adam 中 L2 penalty 会被 adaptive scaling 影响，这也是 AdamW 解耦 weight decay 的动机"
    ],
    "interviewQuestions": [
      "Adam 为什么需要 bias correction？"
    ]
  },
  {
    "id": "adamw",
    "group": "Optimization",
    "title": "Decoupled Weight Decay Regularization",
    "status": "已下载 / 已初读",
    "source": "papers/optimization/adamw-decoupled-weight-decay.pdf",
    "linkedTopics": [
      "optimization-training"
    ],
    "problem": "解释 Adam 中 L2 regularization 和真正 weight decay 的差异。",
    "method": "将 weight decay 从 adaptive gradient update 中解耦出来，避免正则项被二阶矩缩放。",
    "contributions": [
      "AdamW 经典论文",
      "大模型训练默认优化器的重要基础"
    ],
    "limitations": [
      "weight decay 系数仍需要结合学习率、batch size 和训练阶段调参"
    ],
    "interviewQuestions": [
      "AdamW 的 decoupled weight decay 和 Adam + L2 有什么区别？"
    ]
  },
  {
    "id": "adafactor",
    "group": "Optimization",
    "title": "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost",
    "status": "已下载 / 已初读",
    "source": "papers/optimization/adafactor.pdf",
    "linkedTopics": [
      "optimization-training"
    ],
    "problem": "减少 Adam 二阶矩状态带来的显存开销。",
    "method": "用行/列因子分解近似二阶矩估计，降低 optimizer state 内存。",
    "contributions": [
      "低内存 adaptive optimizer",
      "大模型优化器状态压缩思路"
    ],
    "limitations": [
      "实现和超参更复杂，不一定在所有任务上优于 AdamW"
    ],
    "interviewQuestions": [
      "Adafactor 如何降低 optimizer state 内存？"
    ]
  },
  {
    "id": "lion",
    "group": "Optimization",
    "title": "Symbolic Discovery of Optimization Algorithms",
    "status": "已下载 / 已初读",
    "source": "papers/optimization/lion.pdf",
    "linkedTopics": [
      "optimization-training"
    ],
    "problem": "探索更简单、更低状态开销的优化器更新形式。",
    "method": "使用 sign momentum update，主要依赖动量方向而不是完整二阶矩状态。",
    "contributions": [
      "sign momentum 路线",
      "减少优化器状态的代表方法"
    ],
    "limitations": [
      "对 learning rate、batch size 和任务分布较敏感"
    ],
    "interviewQuestions": [
      "Lion 相比 AdamW 少了哪些 optimizer state？"
    ]
  },
  {
    "id": "sophia",
    "group": "Optimization",
    "title": "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training",
    "status": "已下载 / 已初读",
    "source": "papers/optimization/sophia.pdf",
    "linkedTopics": [
      "optimization-training"
    ],
    "problem": "在 LLM pretraining 中引入可扩展的曲率信息，提高训练效率。",
    "method": "用可计算的 Hessian 对角近似配合动量和 clipping，形成二阶优化近似。",
    "contributions": [
      "语言模型二阶优化代表",
      "训练效率优化思路"
    ],
    "limitations": [
      "Hessian 估计和 clipping 细节会带来额外实现复杂度"
    ],
    "interviewQuestions": [
      "Sophia 如何近似二阶信息？"
    ]
  },
  {
    "id": "shampoo",
    "group": "Optimization",
    "title": "Shampoo: Preconditioned Stochastic Tensor Optimization",
    "status": "已下载 / 已初读",
    "source": "papers/optimization/shampoo.pdf",
    "linkedTopics": [
      "optimization-training"
    ],
    "problem": "用矩阵预条件改善高维参数优化中的病态曲率问题。",
    "method": "对张量不同维度维护预条件矩阵，并进行矩阵逆根近似。",
    "contributions": [
      "张量/矩阵预条件优化",
      "二阶优化器重要路线"
    ],
    "limitations": [
      "计算和内存开销较高，工程实现复杂"
    ],
    "interviewQuestions": [
      "Shampoo 为什么比 AdamW 更像二阶优化器？"
    ]
  },
  {
    "id": "muon-scalable-llm-training",
    "group": "Optimization",
    "title": "Muon is Scalable for LLM Training",
    "status": "已下载 / 已初读",
    "source": "papers/optimization/muon-scalable-llm-training.pdf",
    "linkedTopics": [
      "optimization-training",
      "chinese-model-series"
    ],
    "problem": "分析 Muon 能否扩展到 LLM / MoE 训练，并与 AdamW 形成互补。",
    "method": "对部分矩阵参数使用 momentum + Newton-Schulz 正交化更新，对 embedding/head 等参数仍配合 AdamW。",
    "contributions": [
      "Muon 扩展到 LLM 训练",
      "提供 AdamW 之外的更新几何视角"
    ],
    "limitations": [
      "并非所有参数都适合 Muon，仍需处理 weight decay、scale 和实现兼容性"
    ],
    "interviewQuestions": [
      "Muon 和 AdamW 的更新几何有什么不同？",
      "为什么 Muon 常和 AdamW 混合使用？",
      "Newton-Schulz 正交化解决什么问题？"
    ]
  },
  {
    "id": "dpo",
    "group": "Preference Optimization",
    "title": "Direct Preference Optimization",
    "status": "已下载",
    "source": "papers/post-training/dpo.pdf",
    "linkedTopics": [
      "dpo"
    ],
    "problem": "理解 DPO 如何把 RLHF 中的 reward model + PPO 改写成直接偏好优化目标。",
    "method": "基于 Bradley-Terry 偏好模型和 KL-regularized RL 最优解关系，把 reward 写成 policy/reference log-ratio。",
    "contributions": [
      "DPO 核心论文",
      "偏好优化面试主线",
      "reference model 与隐式 reward 关系"
    ],
    "limitations": [
      "依赖高质量离线偏好对，不具备在线探索能力"
    ],
    "interviewQuestions": [
      "DPO 为什么可以不用显式 reward model？reference model 在其中起什么作用？"
    ]
  },
  {
    "id": "ipo",
    "group": "Preference Optimization",
    "title": "IPO: A General Theoretical Paradigm for Learning from Human Preferences",
    "status": "已下载",
    "source": "papers/post-training/ipo.pdf",
    "linkedTopics": [
      "dpo"
    ],
    "problem": "理解 DPO 类偏好优化目标的理论边界和过优化问题。",
    "method": "从 preference learning 理论视角重新分析 DPO，并提出 identity preference optimization。",
    "contributions": [
      "DPO 理论补充",
      "偏好过优化问题入口"
    ],
    "limitations": [
      "偏理论，需要和具体训练数据、beta、reference 设置结合理解"
    ],
    "interviewQuestions": [
      "IPO 为什么认为原始 DPO 在某些情况下可能过度优化？"
    ]
  },
  {
    "id": "kto",
    "group": "Preference Optimization",
    "title": "KTO: Model Alignment as Prospect Theoretic Optimization",
    "status": "已下载",
    "source": "papers/post-training/kto.pdf",
    "linkedTopics": [
      "dpo"
    ],
    "problem": "理解不依赖成对 chosen/rejected 数据的偏好优化路线。",
    "method": "用 desirable / undesirable 单样本反馈构造 prospect theory 风格的对齐目标。",
    "contributions": [
      "非成对偏好优化",
      "DPO 数据形态变体"
    ],
    "limitations": [
      "目标函数和数据假设不同，不能简单等同于 DPO"
    ],
    "interviewQuestions": [
      "KTO 和 DPO 最大的数据形态差异是什么？"
    ]
  },
  {
    "id": "orpo",
    "group": "Preference Optimization",
    "title": "ORPO: Monolithic Preference Optimization without Reference Model",
    "status": "已下载",
    "source": "papers/post-training/orpo.pdf",
    "linkedTopics": [
      "dpo"
    ],
    "problem": "理解去掉 reference model、合并 SFT 和偏好优化的单阶段路线。",
    "method": "在 supervised fine-tuning 目标上加入 odds-ratio preference penalty。",
    "contributions": [
      "无 reference 偏好优化",
      "SFT + preference 单阶段训练入口"
    ],
    "limitations": [
      "去掉 reference 后约束方式不同，需要关注分布漂移和稳定性"
    ],
    "interviewQuestions": [
      "ORPO 去掉 reference model 后靠什么约束 preferred / dispreferred 的差异？"
    ]
  },
  {
    "id": "ppo",
    "group": "GRPO / RL",
    "title": "Proximal Policy Optimization Algorithms",
    "status": "已下载",
    "source": "papers/post-training/ppo.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "理解 GRPO 所继承的 clipped policy optimization 背景。",
    "method": "PPO 用新旧策略概率比和 clipping 约束策略更新幅度，是 RLHF/GRPO 目标函数的重要基础。",
    "contributions": [
      "PPO clip objective 基础",
      "理解 policy ratio、advantage、KL 约束的入口"
    ],
    "limitations": [
      "原论文不是 LLM 专用方法，需要结合 RLHF/GRPO 场景理解"
    ],
    "interviewQuestions": [
      "PPO 中 clipping 的作用是什么？GRPO 为什么仍然保留类似思想？"
    ]
  },
  {
    "id": "dapo",
    "group": "GRPO / RL",
    "title": "DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
    "status": "已下载",
    "source": "papers/post-training/dapo.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "理解 GRPO 后续在大规模 LLM RL 系统中的稳定性改进。",
    "method": "从采样、过滤、动态裁剪、token-level policy gradient 等角度改进 GRPO/RLVR 训练。",
    "contributions": [
      "GRPO 改进谱系",
      "大规模 RL 系统工程参考",
      "on-policy 训练稳定性入口"
    ],
    "limitations": [
      "需要和 DeepSeekMath 原始 GRPO 区分，不要把 DAPO 机制倒灌回 GRPO 定义"
    ],
    "interviewQuestions": [
      "DAPO 相比原始 GRPO 主要想解决哪些训练稳定性问题？"
    ]
  },
  {
    "id": "high-entropy-minority-tokens",
    "group": "GRPO / RLVR Advanced",
    "title": "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/high-entropy-minority-tokens.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "解释 RLVR 中少数高熵 token 为什么可能主导有效学习信号。",
    "method": "基于 token entropy pattern 分析 reasoning fork，并讨论 entropy-aware token weighting。",
    "contributions": [
      "提出 high-entropy minority tokens 视角",
      "提醒 RLVR 不能等权看待所有 token"
    ],
    "limitations": [
      "高熵 token 的定义和阈值仍需结合任务验证"
    ],
    "interviewQuestions": [
      "为什么 token entropy 可以帮助定位关键推理 token？"
    ]
  },
  {
    "id": "cispo",
    "group": "GRPO / RLVR Advanced",
    "title": "Clipped Importance Sampling Policy Optimization (CISPO)",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/minimax-m1-cispo.pdf",
    "linkedTopics": [
      "grpo",
      "chinese-model-series"
    ],
    "problem": "缓解 GRPO 中 clipped importance sampling 的 token 级偏差。",
    "method": "围绕 importance sampling weight 与 detach 设计更稳定的 clip 估计。",
    "contributions": [
      "改进 token 级 ratio 处理",
      "MiniMax-M1 相关 RL 稳定化线索"
    ],
    "limitations": [
      "CISPO 的适用性需要结合具体 RL recipe 和任务分布验证"
    ],
    "interviewQuestions": [
      "CISPO 主要修正 GRPO 的哪个 ratio 问题？"
    ]
  },
  {
    "id": "gspo",
    "group": "GRPO / RLVR Advanced",
    "title": "Group Sequence Policy Optimization",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/gspo.pdf",
    "linkedTopics": [
      "grpo",
      "moe-architecture"
    ],
    "problem": "token-level importance ratios 在长序列和 MoE RL 中可能不稳定。",
    "method": "使用 sequence-level likelihood ratio 进行 sequence 级 clipping 与优化。",
    "contributions": [
      "sequence-level RLVR 路线",
      "Qwen3 相关 RL 稳定化方法",
      "适合讨论 MoE RL 稳定性"
    ],
    "limitations": [
      "sequence-level 稳定性增强后，token credit assignment 仍需额外处理"
    ],
    "interviewQuestions": [
      "GSPO 为什么从 token-level GRPO 转向 sequence-level ratio？"
    ]
  },
  {
    "id": "sapo",
    "group": "GRPO / RLVR Advanced",
    "title": "Soft Adaptive Policy Optimization",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/sapo.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "hard clipping 对 off-policy token 和边界样本处理过于粗糙。",
    "method": "用 temperature-controlled sigmoid soft gate 替代硬截断。",
    "contributions": [
      "soft gate 稳定化路线",
      "增强 near-on-policy token 利用",
      "兼顾 sequence coherence 与 token adaptivity"
    ],
    "limitations": [
      "soft gate 仍需要调温度和阈值，不能完全消除 off-policy 风险"
    ],
    "interviewQuestions": [
      "SAPO 相比 GSPO 和 GRPO 的核心差异是什么？"
    ]
  },
  {
    "id": "rloo",
    "group": "GRPO / RLVR Advanced",
    "title": "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/rloo.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "在不训练 critic 的情况下为 RLHF/RLVR 构造低方差 baseline。",
    "method": "使用 REINFORCE leave-one-out baseline 构造相对 advantage。",
    "contributions": [
      "RLOO 基线方法",
      "连接 REINFORCE-style 优化和 RLHF"
    ],
    "limitations": [
      "仍依赖 reward 质量和组内样本多样性"
    ],
    "interviewQuestions": [
      "RLOO 的 leave-one-out baseline 如何计算？"
    ]
  },
  {
    "id": "reinforce-plus-plus",
    "group": "GRPO / RLVR Advanced",
    "title": "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/reinforce-plus-plus.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "让 REINFORCE 在 LLM 对齐中更稳定，并吸收 PPO/GRPO 的工程技巧。",
    "method": "结合 advantage normalization、KL-in-reward、mini-batch update 等稳定化技巧。",
    "contributions": [
      "critic-free RLHF 路线",
      "与 GRPO/RLOO 形成对比"
    ],
    "limitations": [
      "不同实现中的 normalization、KL 和 batch 细节会显著影响结果"
    ],
    "interviewQuestions": [
      "REINFORCE++ 与 GRPO 的 reward normalization 有什么关系？"
    ]
  },
  {
    "id": "fipo",
    "group": "GRPO / RLVR Advanced",
    "title": "FIPO: Future-KL Influenced Policy Optimization",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/fipo.pdf",
    "linkedTopics": [
      "grpo",
      "multi-step-rl"
    ],
    "problem": "outcome reward 太粗，sequence-level advantage 难以分配到关键 token。",
    "method": "使用 discounted Future-KL 估计 token-level influence weight。",
    "contributions": [
      "Future-KL dense credit assignment",
      "缓解 reasoning length stagnation 的思路"
    ],
    "limitations": [
      "Future-KL 估计本身增加计算和实现复杂度"
    ],
    "interviewQuestions": [
      "FIPO 如何给 token 分配更细的 credit？"
    ]
  },
  {
    "id": "real-rlvr",
    "group": "GRPO / RLVR Advanced",
    "title": "Rewards as Labels: Revisiting RLVR from a Classification Perspective",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/real.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "重新解释 RLVR 中 reward 对 positive/negative rollout 的监督方式。",
    "method": "把 verifiable rewards 视作 categorical labels，将 policy optimization 改写成 classification-style objective。",
    "contributions": [
      "RLVR 分类视角",
      "bounded / monotonic gradient weighting"
    ],
    "limitations": [
      "classification-style objective 和 DPO/REAL 的边界需要结合公式说明"
    ],
    "interviewQuestions": [
      "为什么 REAL 可以把 reward 看成 label？"
    ]
  },
  {
    "id": "chord",
    "group": "GRPO / RLVR Advanced",
    "title": "On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/chord.pdf",
    "linkedTopics": [
      "grpo",
      "sft"
    ],
    "problem": "协调 RL exploration 和 SFT expert data，避免 RL 破坏基础能力。",
    "method": "用动态权重混合 GRPO loss 与 SFT loss，并按 token 调整 expert 信号。",
    "contributions": [
      "SFT/RL 动态混合",
      "off-policy expert data 与 on-policy rollout 协调"
    ],
    "limitations": [
      "SFT 权重过大可能抑制 RL 探索，过小又可能遗忘基础能力"
    ],
    "interviewQuestions": [
      "CHORD 如何平衡 SFT 和 RL 信号？"
    ]
  },
  {
    "id": "deepeyes",
    "group": "GRPO / RLVR Advanced",
    "title": "DeepEyes: Incentivizing Thinking with Images via Reinforcement Learning",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/deepeyes.pdf",
    "linkedTopics": [
      "grpo",
      "agent",
      "agent-interview-practice"
    ],
    "problem": "把 RLVR 扩展到多模态视觉推理和视觉工具调用。",
    "method": "激励模型在视觉任务中进行 image zoom-in 等工具辅助思考。",
    "contributions": [
      "多模态 RLVR + tool use",
      "thinking with images 路线"
    ],
    "limitations": [
      "视觉工具调用的 reward、grounding 和成本控制更复杂"
    ],
    "interviewQuestions": [
      "DeepEyes 如何让模型学会 thinking with images？"
    ]
  },
  {
    "id": "router-replay",
    "group": "GRPO / RLVR Advanced",
    "title": "Router Replay / Stabilizing MoE RL by Aligning Training and Inference Routers",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/router-replay.pdf",
    "linkedTopics": [
      "grpo",
      "moe-architecture",
      "training-inference-frameworks"
    ],
    "problem": "MoE RL 中 router 分布变化会造成 logprob 与 policy ratio mismatch。",
    "method": "使用 router replay / router alignment 缓解训练与推理路由不一致。",
    "contributions": [
      "MoE RL 稳定化",
      "连接 GSPO、router alignment 和 training-inference mismatch"
    ],
    "limitations": [
      "R2/R3 具体机制仍需结合论文细节精读"
    ],
    "interviewQuestions": [
      "MoE router mismatch 为什么会影响 RL logprob 和 policy ratio？"
    ]
  },
  {
    "id": "training-inference-mismatch-rlvr",
    "group": "GRPO / RLVR Advanced",
    "title": "Training-Inference-Mismatch / DeepSeek-V3.2 RL Notes",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/training-inference-mismatch-deepseek-v3-2.pdf",
    "linkedTopics": [
      "grpo",
      "training-inference-frameworks"
    ],
    "problem": "训练 rollout 与真实推理 serving 行为不一致会损害 RL 效果。",
    "method": "围绕 DeepSeek-V3.2 等场景分析训练-推理不一致、off-policy sequence masking 和 serving-aligned RL。",
    "contributions": [
      "serving-aligned RL 视角",
      "将 RL 训练和 inference workload 对齐"
    ],
    "limitations": [
      "具体结论需要结合 DeepSeek-V3.2 技术环境与开源实现验证"
    ],
    "interviewQuestions": [
      "为什么 RL 训练需要关注 training-inference mismatch？"
    ]
  },
  {
    "id": "treepo",
    "group": "GRPO / RLVR Advanced",
    "title": "TreePO: Bridging Policy Optimization and Inference Efficiency with Tree-based Modeling",
    "status": "已下载 / SWIFT 精读",
    "source": "papers/post-training/grpo-advanced/treepo.pdf",
    "linkedTopics": [
      "grpo",
      "multi-step-rl"
    ],
    "problem": "线性 rollout 对复杂推理搜索空间覆盖不足。",
    "method": "使用 tree-structured search、segment decoding、dynamic branching 和 tree-based advantage。",
    "contributions": [
      "结构化 rollout",
      "推理搜索与策略优化结合"
    ],
    "limitations": [
      "树搜索成本、segment advantage 和剪枝策略都更难调"
    ],
    "interviewQuestions": [
      "TreePO 和 Tree-GRPO 的关系是什么？"
    ]
  },
  {
    "id": "understanding-r1-zero",
    "group": "GRPO / RL",
    "title": "Understanding R1-Zero-Like Training: A Critical Perspective",
    "status": "已下载",
    "source": "papers/post-training/understanding-r1-zero.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "理解 R1-Zero 类 RL 训练现象、局限和常见误读。",
    "method": "分析可验证 reward、训练曲线、推理行为涌现与评估偏差。",
    "contributions": [
      "R1-Zero 训练现象分析",
      "GRPO/RLVR 面试反直觉问题来源"
    ],
    "limitations": [
      "属于分析视角论文，结论需要结合具体实现和实验设置"
    ],
    "interviewQuestions": [
      "为什么不能简单把 R1-Zero 的成功归因于某一个单独算法技巧？"
    ]
  },
  {
    "id": "grpo-u-statistic",
    "group": "GRPO / RL",
    "title": "A U-Statistic Perspective on Variance Reduction in GRPO",
    "status": "已下载",
    "source": "papers/post-training/grpo-u-statistic.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "从统计估计角度理解 group-relative advantage 的方差来源。",
    "method": "用 U-statistic 视角分析 GRPO 组内相对估计与方差降低问题。",
    "contributions": [
      "GRPO 理论分析入口",
      "解释 group size、baseline、variance 的面试材料"
    ],
    "limitations": [
      "偏理论，需要和工程实现中的 reward noise / sampling policy 结合理解"
    ],
    "interviewQuestions": [
      "组内相对优势为什么可能降低方差？组大小太小时会有什么问题？"
    ]
  },
  {
    "id": "revisiting-grpo-on-off-policy",
    "group": "GRPO / RL",
    "title": "Revisiting GRPO: On-Policy and Off-Policy Training",
    "status": "已下载",
    "source": "papers/post-training/revisiting-grpo-on-off-policy.pdf",
    "linkedTopics": [
      "grpo"
    ],
    "problem": "理解 GRPO 中 rollout 数据复用、on-policy / off-policy 偏差和训练稳定性。",
    "method": "围绕 policy ratio、采样分布和 off-policy 更新讨论 GRPO 训练细节。",
    "contributions": [
      "GRPO 稳定性分析",
      "理解 rollout 复用风险"
    ],
    "limitations": [
      "需要继续精读实验设置，避免泛化到所有 RLVR 系统"
    ],
    "interviewQuestions": [
      "为什么 GRPO 不能无限复用旧 rollout 数据？"
    ]
  },
  {
    "id": "tropd",
    "group": "OPD",
    "title": "TrOPD",
    "status": "已下载",
    "source": "papers/post-training/tropd.pdf",
    "linkedTopics": [
      "opd"
    ],
    "problem": "理解 trust-region on-policy distillation 如何修复 student rollout 与 teacher supervision 的不稳定区域。",
    "method": "待精读：trust region、local support、teacher reliability mask。",
    "contributions": [
      "OPD 变体谱系",
      "on-policy distillation 面试入口"
    ],
    "limitations": [
      "需要精读后确认目标函数细节"
    ],
    "interviewQuestions": [
      "为什么 OPD 需要 trust region？"
    ]
  },
  {
    "id": "oprd",
    "group": "OPD",
    "title": "OPRD",
    "status": "已下载",
    "source": "papers/post-training/oprd.pdf",
    "linkedTopics": [
      "opd"
    ],
    "problem": "理解 representation distillation 如何扩展 OPD。",
    "method": "待精读：hidden representation alignment、on-policy rollout。",
    "contributions": [
      "OPD 表征层扩展",
      "distillation 信号从 output 到 hidden"
    ],
    "limitations": [
      "需要确认和普通 feature distillation 的差异"
    ],
    "interviewQuestions": [
      "OPRD 为什么不只蒸馏 logits？"
    ]
  },
  {
    "id": "mad-opd",
    "group": "OPD",
    "title": "MAD-OPD",
    "status": "已下载",
    "source": "papers/post-training/mad-opd.pdf",
    "linkedTopics": [
      "opd",
      "agent"
    ],
    "problem": "理解 OPD 在 multi-agent debate / agentic trajectory 中的扩展。",
    "method": "待精读：multi-agent debate、on-policy trajectories、distillation signal。",
    "contributions": [
      "OPD 到 Agent 的扩展",
      "multi-agent debate 训练信号"
    ],
    "limitations": [
      "需要验证任务设置和泛化性"
    ],
    "interviewQuestions": [
      "OPD 用在 multi-agent debate 中会遇到什么新问题？"
    ]
  },
  {
    "id": "poweropd",
    "group": "OPD",
    "title": "PowerOPD: Stabilizing On-Policy Distillation with Bounded Power Transformation",
    "status": "已下载 / 已初读",
    "source": "papers/post-training/poweropd.pdf",
    "linkedTopics": [
      "opd"
    ],
    "problem": "解释 sampled-token OPD 为什么会出现训练不稳定、样本效率低和 response length 震荡。",
    "method": "把 OPD 看成 dense-reward policy gradient，诊断 log-ratio reward 无界、高方差且极端值集中在 rollout early tokens；用 Box-Cox power transformation 替代 log，得到 bounded 且 sign-consistent 的 token reward。",
    "contributions": [
      "把 vanilla OPD 的不稳定归因到 sampled-token log-ratio reward 本身",
      "提出 PowerOPD，用有界幂变换替代无界 log-ratio",
      "在 Qwen3 teacher-student 数学推理设置中提升 Avg@8 / Pass@8，同时节省 full-vocab OPD 的训练时间和显存"
    ],
    "limitations": [
      "实验主要集中在数学推理 benchmark 与 Qwen3 teacher-student pair",
      "Power transform 的 alpha 选择和不同任务/模型上的泛化还需要继续跟踪",
      "公众号解读是二级资料，定量结论应以论文为准"
    ],
    "interviewQuestions": [
      "为什么 sampled-token OPD 的 log-ratio reward 会导致高方差梯度？",
      "clip / tanh / z-score 为什么属于 post-hoc 修补而不是根因修复？",
      "PowerOPD 的 boundedness 和 sign consistency 分别解决什么问题？"
    ]
  },
  {
    "id": "multi-token-prediction",
    "group": "MTP",
    "title": "Better & Faster Large Language Models via Multi-token Prediction",
    "status": "已下载",
    "source": "papers/post-training/multi-token-prediction.pdf",
    "linkedTopics": [
      "mtp"
    ],
    "problem": "理解 Multi-Token Prediction 相比 next-token prediction 的训练信号差异。",
    "method": "待精读：multi-token heads、future shifted labels、auxiliary loss。",
    "contributions": [
      "MTP 基础论文",
      "sample efficiency 与推理能力入口"
    ],
    "limitations": [
      "需要和 speculative decoding 分开讲"
    ],
    "interviewQuestions": [
      "MTP 和 speculative decoding 是同一件事吗？"
    ]
  },
  {
    "id": "dspark",
    "group": "Inference / Speculative Decoding",
    "title": "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation",
    "status": "已导入",
    "source": "papers/inference/dspark.pdf",
    "linkedTopics": [
      "mtp",
      "training-inference-frameworks"
    ],
    "problem": "理解 DeepSeek-V4 serving 中如何从 MTP-1 baseline 进一步提升推理吞吐和交互速度。",
    "method": "使用 semi-autoregressive drafter 缓解 parallel drafter 的 suffix acceptance decay，并用 confidence-scheduled verification 根据 prefix survival probability 和系统吞吐 profile 动态裁剪验证长度。",
    "contributions": [
      "把 MTP-1 作为生产 baseline，报告 DSpark 在 DeepSeek-V4 serving 中提升 per-user generation speed",
      "把 speculative decoding 的问题拆成 draft quality 与 verification waste 两部分",
      "提出面向高并发 serving 的 hardware-aware prefix scheduler"
    ],
    "limitations": [
      "核心是推理加速系统，不是 MTP 训练目标本身",
      "线上收益依赖 serving engine、并发负载、target/drafter 配置和 SLA"
    ],
    "interviewQuestions": [
      "DSpark 和 MTP 的关系是什么，为什么不能把二者等同？",
      "semi-autoregressive drafter 解决了 parallel drafter 的什么问题？",
      "confidence-scheduled verification 为什么能改善高并发 serving 吞吐？"
    ]
  },
  {
    "id": "wechat-speculative-decoding-mtp",
    "group": "Community Notes / Inference",
    "title": "白得 2 到 3 倍加速的投机采样机制解析",
    "status": "社区解读 / 已归档",
    "source": "papers/deep-notes/wechat-speculative-decoding-mtp.txt",
    "linkedTopics": [
      "mtp",
      "training-inference-frameworks"
    ],
    "problem": "用直观语言解释 draft model 与 target model 如何配合完成 lossless speculative decoding。",
    "method": "围绕 draft-verify、rejection sampling、接受率、树状 speculative decoding、Medusa heads 和 EAGLE 做社区化讲解。",
    "contributions": [
      "适合帮助区分 MTP 训练目标和 speculative decoding 推理算法",
      "补充树状候选验证、Medusa、EAGLE 等 MTP 推理侧延伸",
      "把 KV cache 的空间优化和 speculative decoding 的时间优化做了直观区分"
    ],
    "limitations": [
      "这是二级社区资料，不是正式论文",
      "定量加速收益需要以 Speculative Decoding、SpecInfer、Medusa、EAGLE、DSpark 等论文或系统报告为准"
    ],
    "interviewQuestions": [
      "投机采样为什么能做到分布意义上的无损？",
      "MTP、Medusa、EAGLE、DSpark 分别处在训练目标、候选生成和 serving 系统的哪一层？",
      "为什么 draft 长度变长不一定带来更高吞吐？"
    ]
  },
  {
    "id": "pipo-latent-mtp-opd",
    "group": "MTP / OPD",
    "title": "PIPO: Latent Multi-Token Prediction and Online Policy Distillation",
    "status": "已下载",
    "source": "papers/post-training/pipo-latent-mtp-opd.pdf",
    "linkedTopics": [
      "mtp",
      "opd"
    ],
    "problem": "理解 MTP 和 OPD 在新后训练方法中的结合。",
    "method": "待精读：latent MTP、online policy distillation、training objective。",
    "contributions": [
      "MTP 与 OPD 交叉入口",
      "后训练前沿跟踪"
    ],
    "limitations": [
      "需要精读后判断是否适合面试主线"
    ],
    "interviewQuestions": [
      "为什么 MTP 可以和 online policy distillation 放在一起？"
    ]
  },
  {
    "id": "vla-survey-2505-04769",
    "group": "Embodied AI / VLA",
    "title": "Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges",
    "status": "综述精读 / 已索引",
    "source": "https://arxiv.org/abs/2505.04769",
    "linkedTopics": [
      "embodied-ai-robotics-vla",
      "vla-embodied-driving",
      "multimodal-vlm",
      "agent"
    ],
    "problem": "为 VLA 模型建立统一综述框架，梳理视觉、语言、动作统一建模在机器人、自动驾驶和具身智能中的概念基础、模型进展、应用与挑战。",
    "method": "系统回顾 2022-2025 年间 80 余个 VLA 相关模型，从概念基础、架构演进、高效训练、实时推理、应用落地和开放挑战等角度做结构化总结。",
    "contributions": [
      "把 VLA 从视觉-语言模型、动作规划器和分层控制器的融合角度统一起来",
      "整理 CLIPort、Gato、RT-1、VIMA、ACT、Diffusion Policy、RT-2、OpenVLA、Octo、GROOT 等模型谱系",
      "突出实时推理、动作表征、安全、泛化、跨具身迁移和 System 1/2 集成是 VLA 部署核心瓶颈",
      "适合作为具身智能和机器人 VLA 面试的总览型参考论文"
    ],
    "limitations": [
      "综述论文覆盖面广，但对每个模型的训练细节、数据配比和工程实现不会像原论文一样深入",
      "部分 2025 后续模型和工业实践需要继续结合官方论文、代码和真实评测跟踪",
      "博客园文章是中文转述，定量结论和模型细节应以 arXiv 原文为准"
    ],
    "interviewQuestions": [
      "VLA 和 VLM 的本质区别是什么？",
      "为什么机器人 VLA 不能只看离线准确率，还要看实时性和闭环安全？",
      "动作 token、continuous action head、Diffusion Policy 和 Flow Matching action expert 怎么比较？",
      "System 1/2 架构在具身智能里分别承担什么，为什么会有时间尺度冲突？"
    ]
  },
].concat(BJ_DEEPSEEK_PAPERS),
  readingQueue: [
  {
    "title": "优先精读 OPD / MTP 代表论文",
    "rule": "先读 TrOPD、Multi-token Prediction，再读 PIPO 这类交叉方法。"
  },
  {
    "title": "继续跟踪非 DeepSeek 国产模型论文",
    "rule": "GLM、Kimi、MiniMax、Qwen 的最新论文先进入通用论文库，精读后再决定是否沉淀到知识库。"
  }
]
};


