window.BJ_CONTENT = window.BJ_CONTENT || {};

window.BJ_CONTENT.deepseekPapers = {
  id: "deepseek-papers",
  title: "DeepSeek 论文线",
  eyebrow: "DeepSeek Paper Track",
  description:
    "单独维护 DeepSeek 相关论文、模型卡和精读笔记。这里的结论不直接塞进知识库，先作为资料线沉淀。",
  papers: [
    {
      id: "deepseek-v3",
      group: "DeepSeek",
      title: "DeepSeek-V3",
      status: "已下载",
      source: "papers/model-series/deepseek-v3.pdf",
      linkedTopics: ["chinese-model-series", "mtp"],
      problem: "理解 DeepSeek-V3 的 MoE、MLA、MTP 和训练效率。",
      method: "待精读：MoE routing、MLA、MTP auxiliary objective、训练系统和数据。",
      contributions: ["MoE + MLA 工程路线", "MTP 与训练效率入口"],
      limitations: ["需要和 DeepSeek-R1 的 RL 后训练区分"],
      interviewQuestions: ["MLA、MoE、MTP 分别解决 DeepSeek-V3 的什么问题？"]
    },
    {
      id: "deepseek-r1",
      group: "DeepSeek",
      title: "DeepSeek-R1",
      status: "已下载",
      source: "papers/model-series/deepseek-r1.pdf",
      linkedTopics: ["grpo", "chinese-model-series"],
      problem: "理解强化学习如何激发推理能力。",
      method: "待精读：cold-start、RL、蒸馏、可验证奖励。",
      contributions: ["Reasoning RL 代表", "GRPO/RLVR 面试核心"],
      limitations: ["需要理解 reward、采样和训练稳定性"],
      interviewQuestions: ["DeepSeek-R1 的 RL 路线和 DPO/OPD 有什么本质差异？"]
    },
    {
      id: "deepseekmath",
      group: "DeepSeek",
      title: "DeepSeekMath",
      status: "已下载",
      source: "papers/model-series/deepseekmath.pdf",
      linkedTopics: ["grpo", "chinese-model-series"],
      problem: "理解数学推理模型和强化学习早期路线。",
      method: "待精读：数学数据、RL、推理评估。",
      contributions: ["DeepSeek reasoning 前史", "数学推理 RL 线索"],
      limitations: ["和 R1 的大规模 reasoning RL 需要分代对比"],
      interviewQuestions: ["数学推理数据和通用 reasoning 数据有什么差异？"]
    },
    {
      id: "flashmemory-deepseek-v4",
      group: "DeepSeek-V4",
      title: "FlashMemory-DeepSeek-V4",
      status: "已精读",
      source: "papers/model-series/latest-2026/flashmemory-deepseek-v4.pdf",
      linkedTopics: ["chinese-model-series"],
      problem: "理解 DeepSeek-V4 相关长上下文推理中 KV cache 显存瓶颈和 Lookahead Sparse Attention。",
      method: "Memory Indexer 周期性预测 query-critical historical chunks，把冷历史留在 CPU，需要时拉回 GPU。",
      contributions: ["KV cache footprint 降至 13.5%", "500K context 约 90% memory reduction", "DeepSeek-V4 相关长上下文推理论文"],
      limitations: ["不是官方 DeepSeek-V4 technical report", "项目暂停说明需要谨慎看待工程持续性"],
      interviewQuestions: ["FlashMemory 和普通 sparse attention 解决的是同一个瓶颈吗？", "为什么 KV cache 驻留会成为超长上下文推理瓶颈？"]
    },
    {
      id: "deepseek-v4-pro-model-card",
      group: "DeepSeek-V4",
      title: "DeepSeek-V4-Pro model card",
      status: "模型卡已下载",
      source: "papers/model-series/model-cards/deepseek-v4-pro-model-card.md",
      linkedTopics: ["chinese-model-series"],
      problem: "跟踪 DeepSeek-V4-Pro 最新模型卡信息。",
      method: "以 Hugging Face model card 为来源；等待官方 technical report。",
      contributions: ["最新模型卡线索", "DeepSeek-V4 资料入口"],
      limitations: ["模型卡不是论文证据"],
      interviewQuestions: ["模型卡信息和论文结论应该如何区分？"]
    }
  ],
  readingQueue: [
    {
      title: "优先查找官方 DeepSeek-V4 Technical Report",
      rule: "目前只确认 DeepSeek-V4-Pro 模型卡与 FlashMemory-DeepSeek-V4 相关论文，不能把后者当成官方 V4 报告。"
    },
    {
      title: "精读 DeepSeek-V3 的 MTP",
      rule: "MTP 是知识库新增模块，需要从 DeepSeek-V3 和 Multi-token Prediction 论文中分别抽出训练目标。"
    }
  ]
};
