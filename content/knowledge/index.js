window.BJ_CONTENT = window.BJ_CONTENT || {};

window.BJ_CONTENT.knowledge = {
  id: "knowledge",
  title: "知识库",
  eyebrow: "Interactive Knowledge Base",
  description: "面向大模型算法工程师面试的可扩展知识库。详细内容维护在 Markdown 章节中，前端只渲染生成后的文档缓存。",
  topics: [
    {
      id: "llm-engineering-interview-supplement",
      group: "工程基础",
      title: "LLM 工程基础面试补充",
      docId: "llm-engineering-interview-supplement",
      status: "持续补全",
      summary: "整理归一化、PEFT、分布式训练、显存估算和推理优化等高频工程基础。",
      tags: ["LLM", "工程基础"],
      coreQuestions: ["Norm 类方法如何比较？", "LoRA/QLoRA 怎么讲？", "DP/TP/PP/ZeRO 分别切什么？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "activation-functions",
      group: "训练优化",
      title: "激活函数与门控 FFN",
      docId: "activation-functions",
      status: "专题章节",
      summary: "整理 ReLU、GELU、SiLU、GLU/GEGLU/SwiGLU 与 Transformer FFN 的关系。",
      tags: ["Activation", "FFN"],
      coreQuestions: ["ReLU、GELU、SiLU 有什么区别？", "SwiGLU 改进在哪里？", "为什么 LLM 常用门控 FFN？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "optimization-training",
      group: "训练优化",
      title: "优化器与训练稳定性",
      docId: "optimization-training",
      status: "完整框架",
      summary: "整理 SGD、Momentum、AdamW、Adafactor、Lion、Sophia、Shampoo、Muon 与学习率调度。",
      tags: ["Optimizer", "AdamW", "Muon"],
      coreQuestions: ["AdamW 为什么常用？", "Muon 改变了哪类参数的更新几何？", "学习率调度如何影响训练稳定性？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "sft",
      group: "后训练核心",
      title: "SFT",
      docId: "sft",
      status: "持续补全",
      summary: "Instruction tuning、数据质量、chat template、loss mask 与后续对齐链路。",
      tags: ["SFT", "Data"],
      coreQuestions: ["为什么通常只对 assistant response 算 loss？", "SFT 数据质量如何评估？", "chat template 错位有什么后果？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["sft-loss-mask"], paperRefs: []
    },
    {
      id: "dpo",
      group: "后训练核心",
      title: "DPO / 偏好优化",
      docId: "dpo",
      status: "持续补全",
      summary: "Chosen/rejected、reference model、隐式 reward、beta 与 PPO/GRPO 对比。",
      tags: ["DPO", "Preference"],
      coreQuestions: ["DPO 为什么不显式训练 reward model？", "reference model 起什么作用？", "DPO/PPO/GRPO 怎么区分？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["dpo-loss"], paperRefs: []
    },
    {
      id: "grpo",
      group: "后训练核心",
      title: "GRPO",
      docId: "grpo",
      status: "持续补全",
      summary: "Group relative advantage、PPO 对比、RLVR、reward 设计与高级改进谱系。",
      tags: ["GRPO", "RLVR"],
      coreQuestions: ["GRPO 解决 PPO 哪些成本？", "组内相对优势如何替代 value model？", "DAPO/GSPO/CISPO 分别改进什么？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["grpo-advantage", "dapo-loss"], paperRefs: []
    },
    {
      id: "distillation",
      group: "后训练核心",
      title: "知识蒸馏：KD / GKD / OPD-RL / OPSD",
      docId: "distillation",
      status: "持续补全",
      summary: "作为 OPD 的前置知识，整理 KL/JSD、top-k、sampled-token teacher signal 等蒸馏基础。",
      tags: ["Distillation", "GKD"],
      coreQuestions: ["Forward KL 和 Reverse KL 如何区分？", "GKD 和 OPD-RL 的梯度路径有什么不同？", "教师信号粒度如何取舍？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["opd-loss"], paperRefs: []
    },
    {
      id: "opd",
      group: "后训练核心",
      title: "OPD / On-Policy Distillation",
      docId: "opd",
      status: "持续补全",
      summary: "Student on-policy rollout、teacher dense supervision、OPSD/SDPO/SDFT 与 PowerOPD。",
      tags: ["OPD", "Distillation"],
      coreQuestions: ["OPD 和离线蒸馏有什么区别？", "Forward KL、Reverse KL、JSD 如何影响训练？", "PowerOPD 改进了什么？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["opd-loss"], paperRefs: []
    },
    {
      id: "multi-step-rl",
      group: "后训练核心",
      title: "多步强化学习：ARPO / AEPO / RAPO / Tree-GRPO / GiGPO",
      docId: "multi-step-rl",
      status: "专题章节",
      summary: "作为 Agentic RL 的前置方法论，整理长链路推理、树/图轨迹和 step/branch credit assignment。",
      tags: ["RL", "Multi-step"],
      coreQuestions: ["多步 RL 和普通 GRPO 有何区别？", "为什么需要 step-level credit assignment？", "Tree-GRPO 怎么理解？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["grpo-advantage"], paperRefs: []
    },
    {
      id: "agentic-rl-system",
      group: "后训练核心",
      title: "Agentic RL 系统链路",
      docId: "agentic-rl-system",
      status: "专题章节",
      summary: "把多步 RL 落到工具/环境 trace，拆解 VeRL Agent Loop / Reward Loop、response_mask 与 credit assignment。",
      tags: ["Agentic RL", "VeRL"],
      coreQuestions: ["Agentic RL 和普通 RLVR 样本形态有什么不同？", "为什么 response_mask 是多轮工具训练核心？", "Reward Loop 为什么要单独抽象？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["agentic-rl-loop", "verl-dataflow-loop"], paperRefs: []
    },
    {
      id: "sampling-evaluation-rft",
      group: "工程闭环",
      title: "采样、评估与强化微调闭环",
      docId: "sampling-evaluation-rft",
      status: "持续补全",
      summary: "围绕 Sample、ORM/PRM、RFT/self-improvement、训练中评测和 Reward Loop 构建能力闭环。",
      tags: ["Sampling", "Evaluation"],
      coreQuestions: ["RFT 和 SFT 的区别是什么？", "ORM/PRM 分别解决什么？", "如何避免 reward hacking？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "mtp",
      group: "模型训练目标",
      title: "MTP / Multi-Token Prediction",
      docId: "mtp",
      status: "持续补全",
      summary: "多 token 预测目标、DeepSeek-V3 MTP、sample efficiency 与 speculative decoding 关系。",
      tags: ["MTP", "Pretraining"],
      coreQuestions: ["MTP 相比 NTP 多了什么信号？", "MTP 和投机解码是什么关系？", "MTP rollout 加速为什么不一定提速？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["mtp-loss"], paperRefs: []
    },
    {
      id: "moe-architecture",
      group: "模型架构",
      title: "MoE 模型架构演进",
      docId: "moe-architecture",
      status: "专题章节",
      summary: "整理 sparse routing、shared experts、fine-grained experts、load balancing 和 MoE 工程难点。",
      tags: ["MoE", "Routing"],
      coreQuestions: ["MoE 为什么能扩大总参数但控制激活计算？", "top-k routing 和 load balancing 解决什么？", "shared experts 有什么价值？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["moe-topk-router"], paperRefs: []
    },
    {
      id: "positional-encoding",
      group: "模型架构",
      title: "位置编码与长度外推",
      docId: "positional-encoding",
      status: "专题章节",
      summary: "整理 absolute/relative position、RoPE、ALiBi、PI、NTK/RTK、YaRN 等长上下文外推方法。",
      tags: ["RoPE", "Long Context"],
      coreQuestions: ["RoPE 如何注入相对位置信息？", "PI/NTK/YaRN 如何扩展上下文？", "长上下文为什么不只改位置编码？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "training-inference-frameworks",
      group: "工程框架",
      title: "训练推理框架",
      docId: "training-inference-frameworks",
      status: "持续补全",
      summary: "横向比较 DeepSpeed、Megatron-LM、VeRL、Slime、vLLM、SGLang、SWIFT 的定位。",
      tags: ["Training", "Inference", "RLHF"],
      coreQuestions: ["DeepSpeed 和 Megatron-LM 边界是什么？", "vLLM 和 SGLang 分别解决什么？", "RL 后训练系统如何组织 rollout/reward/update？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["verl-dataflow-loop"], paperRefs: []
    },
    {
      id: "verl-rl-framework",
      group: "工程框架",
      title: "VeRL：RL 后训练系统框架",
      docId: "verl-rl-framework",
      status: "框架精读",
      summary: "整理 HybridFlow、controller/WorkerGroup/DataProto/ResourcePool、PPO/GRPO 数据流、vLLM/SGLang rollout。",
      tags: ["VeRL", "HybridFlow"],
      coreQuestions: ["HybridFlow 为什么解耦控制流和计算流？", "DataProto/WorkerGroup 解决什么工程问题？", "VeRL 如何扩展算法？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["verl-dataflow-loop"], paperRefs: []
    },
    {
      id: "agent",
      group: "Agent 前沿",
      title: "Agent 系统",
      docId: "agent",
      status: "总览地图",
      summary: "系统梳理 ReAct、Function Calling、RAG、Memory、Planning、Multi-Agent、Agentic RL 和 Agent as Verifier。",
      tags: ["Agent", "ReAct", "RAG"],
      coreQuestions: ["Agent 和 chatbot 的区别是什么？", "Memory 和 RAG 有什么区别？", "多轮 Agent rollout 如何训练？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["agentic-rl-loop"], paperRefs: []
    },
    {
      id: "agent-rag-systems",
      group: "Agent 前沿",
      title: "Agent RAG 系统",
      docId: "agent-rag-systems",
      status: "新拆分专题",
      summary: "单独整理 RAG 离线建设、在线召回、chunk、embedding/reranker、hybrid search、引用与生产评估。",
      tags: ["Agent", "RAG"],
      coreQuestions: ["RAG 相比微调解决什么问题？", "完整 RAG pipeline 怎么讲？", "chunk size、embedding、reranker 怎么选？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["agent-rag-systems"], paperRefs: []
    },
    {
      id: "agent-memory-tooling",
      group: "Agent 前沿",
      title: "Agent Memory 与工具调用",
      docId: "agent-memory-tooling",
      status: "新拆分专题",
      summary: "单独整理 memory schema、写入/检索/过期策略、Tool Registry、schema 校验、权限控制、错误恢复和 response_mask 边界。",
      tags: ["Agent", "Memory", "Tool Calling"],
      coreQuestions: ["Memory 和 RAG 有什么区别？", "Tool Registry 需要维护哪些字段？", "为什么工具 observation 不能进 policy loss？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["agent-rag-systems", "llm-react-tool-loop", "agentic-rl-loop"], paperRefs: []
    },
    {
      id: "agent-planning-multi-agent",
      group: "Agent 前沿",
      title: "Agent 规划执行与 Multi-Agent",
      docId: "agent-planning-multi-agent",
      status: "新拆分专题",
      summary: "单独整理 Plan-and-Execute、ReAct、Reflection、Tree of Thoughts、Multi-Agent 角色协作、预算控制和 Agentic RL 轨迹连接。",
      tags: ["Agent", "Planning", "Multi-Agent"],
      coreQuestions: ["Planning 和 ReAct 的关系是什么？", "Reflection 为什么需要外部反馈？", "Multi-Agent 什么时候值得用？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["llm-react-tool-loop", "agentic-rl-loop"], paperRefs: []
    },
    {
      id: "agent-interview-practice",
      group: "Agent 前沿",
      title: "Agent 面试与实战补充",
      docId: "agent-interview-practice",
      status: "实战补充",
      summary: "补充 Context Engineering、Skills/MCP、GUI/Web Agent、工具设计、自进化与后训练实践。",
      tags: ["Agent", "Interview"],
      coreQuestions: ["Context Engineering 和 Prompt Engineering 有何区别？", "MCP 和 Skills 边界是什么？", "Web Agent 为什么不能只靠视觉？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "multimodal-vlm",
      group: "多模态大模型",
      title: "多模态大模型与 VLM",
      docId: "multimodal-vlm",
      status: "持续补全",
      summary: "整理 CLIP、BLIP-2、LLaVA、Flamingo、Qwen-VL 类模型的视觉 encoder、connector、指令微调和评估。",
      tags: ["VLM", "CLIP"],
      coreQuestions: ["CLIP 和 LLaVA 的区别是什么？", "Q-Former、MLP projector、cross-attention 有什么差异？", "VLM 幻觉如何缓解？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "vla-embodied-driving",
      group: "多模态大模型",
      title: "VLA：视觉-语言-动作与自动驾驶具身智能",
      docId: "vla-embodied-driving",
      status: "持续补全",
      summary: "整理 VLA 的动作输出、动态感知、3D 空间、多模态 CoT 与开闭环评估。",
      tags: ["VLA", "Embodied AI"],
      coreQuestions: ["VLA 和 VLM 最大区别是什么？", "自动驾驶 VLA 为什么需要 3D/时序/动态感知？", "动作轨迹应该用 token、文本还是 diffusion head？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "embodied-ai-robotics-vla",
      group: "多模态大模型",
      title: "具身智能与机器人 VLA",
      docId: "embodied-ai-robotics-vla",
      status: "持续补全",
      summary: "整理机器人 VLA 架构、动作表示、经典模型、数据训练、Sim-to-Real、生成式策略和机器人 RL。",
      tags: ["Embodied AI", "Robotics"],
      coreQuestions: ["具身智能和普通 VLM/Agent 有何区别？", "RT-2、OpenVLA、ACT、Diffusion Policy 怎么区分？", "Sim-to-Real 怎么讲？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "chinese-model-series",
      group: "模型家族",
      title: "国产模型系列：GLM / Kimi / MiniMax / Qwen / DeepSeek",
      docId: "chinese-model-series",
      status: "索引框架",
      summary: "保留模型家族路线和横向比较框架，具体论文总结放入论文库精读卡片。",
      tags: ["模型家族", "MoE"],
      coreQuestions: ["不同国产模型系列分别强调什么技术路线？", "MoE、Attention、后训练、Agent 如何横向比较？", "模型卡和论文证据边界怎么区分？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["moe-topk-router", "sparse-block-attention", "mtp-loss"], paperRefs: []
    },
    {
      id: "interview-question-bank",
      group: "面试专项",
      title: "大模型算法工程师面试八股总览",
      docId: "interview-question-bank",
      status: "框架整理",
      summary: "整理 LLM、VLM、RLHF、Agent、RAG、模型评估六类高频八股题回答主线。",
      tags: ["Interview", "LLM"],
      coreQuestions: ["LLM 基础如何组织？", "VLM 面试怎么答？", "RLHF/DPO/GRPO 如何串起来？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: [], paperRefs: []
    },
    {
      id: "agent-guide-interview-qa",
      group: "面试专项",
      title: "AgentGuide 面试问答补全",
      docId: "agent-guide-interview-qa",
      status: "问答补全",
      summary: "基于 AgentGuide 面试目录补齐 Agent、RAG、评估和系统手撕代码问题的中文回答模板。",
      tags: ["Agent", "RAG", "Interview"],
      coreQuestions: ["RAG pipeline 如何系统回答？", "Agent Memory、工具调用、冲突处理怎么讲？", "LLM-as-a-Judge 有哪些偏差？"],
      answerRecords: [], followUps: [], weaknesses: [], finalNotes: [], interviewQA: [], principleCodeRefs: ["agent-rag-systems", "llm-react-tool-loop"], paperRefs: []
    }
  ]
};
