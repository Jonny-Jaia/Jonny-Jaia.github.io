window.BJ_CONTENT = window.BJ_CONTENT || {};

window.BJ_CONTENT.foundations = {
  id: "foundations",
  title: "基础知识",
  eyebrow: "Foundations",
  description:
    "单独维护计算机基础、机器学习基础、深度学习基础、强化学习基础、LLM 工程基础和 CV 基础模型，作为知识库专题的长期地基。",
  topics: [
    {
      id: "llm-engineering-foundations",
      group: "大模型工程基础",
      title: "LLM 工程基础",
      docId: "llm-engineering-foundations",
      docCollection: "foundations",
      status: "工程基础",
      summary: "沉淀归一化、PEFT、分布式训练、显存估算和推理优化等长期稳定基础。",
      tags: ["LLM", "Engineering", "PEFT", "Distributed Training", "Inference"],
      coreQuestions: ["Norm 类方法如何比较？", "LoRA/QLoRA 有什么差异？", "FlashAttention 和 PagedAttention 优化对象有何不同？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "computer-foundations",
      group: "计算机基础",
      title: "计算机基础",
      docId: "computer-foundations",
      docCollection: "foundations",
      status: "ReadyBlog 整理",
      summary: "围绕网络、Python 底层、数据结构、排序和图算法构建面试基础框架。",
      tags: ["CS", "Network", "Python", "Data Structure"],
      coreQuestions: ["TCP 可靠性如何保证？", "Python GIL 怎么解释？", "哈希表、堆、树分别适合什么场景？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "ml-foundations",
      group: "机器学习基础",
      title: "机器学习基础",
      docId: "ml-foundations",
      docCollection: "foundations",
      status: "ReadyBlog 整理",
      summary: "围绕优化、线性模型、正则化、损失函数和集成学习建立可复述框架。",
      tags: ["ML", "Optimization", "Regularization", "Ensemble"],
      coreQuestions: ["逻辑回归为什么用交叉熵？", "L1/L2 正则分别带来什么偏置？", "Bagging 和 Boosting 怎么从偏差-方差解释？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "dl-foundations",
      group: "深度学习基础",
      title: "深度学习基础",
      docId: "dl-foundations",
      docCollection: "foundations",
      status: "ReadyBlog 整理",
      summary: "围绕 Attention、Transformer、位置编码、Norm、激活函数、优化器基础、学习率调度、PLM 和 LLM 架构构建深度学习基础。",
      tags: ["DL", "Transformer", "Attention", "PLM"],
      coreQuestions: ["Attention 为什么除以 sqrt(d_k)？", "Pre-Norm 和 Post-Norm 有何影响？", "BERT 和 GPT 怎么比较？", "激活函数和学习率调度为什么影响训练稳定性？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], principleCodeRefs: ["sparse-block-attention"], paperRefs: []
    },
    {
      id: "cv-foundation-models",
      group: "视觉基础模型",
      title: "CV 基础模型：ViT / CLIP / MAE / SAM / Diffusion",
      docId: "cv-foundation-models",
      docCollection: "foundations",
      status: "本轮大幅补全",
      summary: "覆盖 ViT、CLIP、MAE、DINO、DETR、Grounding DINO、SAM、DDPM、Latent Diffusion 与多模态 LLM 的连接。",
      tags: ["CV", "ViT", "CLIP", "MAE", "SAM", "Diffusion", "VLM"],
      coreQuestions: ["ViT 如何把图像变成 token？", "CLIP 为什么能 zero-shot？", "SAM 为什么是 promptable segmentation？", "Latent Diffusion 为什么高效？"],
      answerRecords: [], followUps: ["请用 2 分钟讲清楚 CLIP、ViT、Diffusion 与 VLM 的关系。"], weaknesses: [], finalNotes: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "rl-foundations",
      group: "强化学习基础",
      title: "强化学习基础",
      docId: "rl-foundations",
      docCollection: "foundations",
      status: "ReadyBlog 整理",
      summary: "围绕 MDP、价值函数、策略梯度、Actor-Critic、PPO 和 LLM RL 的连接构建基础框架。",
      tags: ["RL", "PPO", "Policy Gradient", "Actor-Critic"],
      coreQuestions: ["Value-based 和 Policy-based 有何区别？", "Actor-Critic 为什么降方差？", "PPO clip 解决什么问题？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], principleCodeRefs: ["grpo-advantage"], paperRefs: ["ppo"]
    }
  ]
};
