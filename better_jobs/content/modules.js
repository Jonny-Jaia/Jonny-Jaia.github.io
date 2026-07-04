window.BJ_CONTENT = window.BJ_CONTENT || {};

window.BJ_MODULES = [
  {
    id: "knowledge",
    title: "知识库",
    description: "沉淀后训练、Agent、模型架构、模型系列、训练推理框架等大模型算法工程师核心专题。",
    contentKey: "knowledge",
    itemKey: "topics",
    renderer: "knowledge"
  },
  {
    id: "foundations",
    title: "基础知识",
    description: "单独维护计算机基础、机器学习、深度学习、强化学习、CV 基础模型和 LLM 工程基础。",
    contentKey: "foundations",
    itemKey: "topics",
    renderer: "knowledge"
  },
  {
    id: "paper-library",
    title: "论文库",
    description: "集中管理模型系列、后训练、OPD/MTP、优化器和框架论文卡片，保留来源索引和阅读状态。",
    contentKey: "paperLibrary",
    itemKey: "papers",
    renderer: "paperLibrary"
  },
  {
    id: "principle-code",
    title: "原理代码",
    description: "展示 SFT、DPO、GRPO、MoE、MTP、Agent/RAG 等原理的最小可解释实现代码。",
    contentKey: "principleCode",
    itemKey: "items",
    renderer: "codeSkill"
  },
  {
    id: "code-training",
    title: "代码训练",
    description: "独立训练 LeetCode 与手撕算法题，记录状态、薄弱点、复盘和下一题安排。",
    contentKey: "codeTraining",
    itemKey: "problemSets",
    renderer: "codeTraining"
  },
  {
    id: "learning-sessions",
    title: "学习会话",
    description: "记录每次互动后的薄弱点、追问队列、产出物和下一轮训练计划。",
    contentKey: "learningSessions",
    itemKey: "entries",
    renderer: "learningSessions"
  }
];
