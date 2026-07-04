window.BJ_CONTENT = window.BJ_CONTENT || {};

window.BJ_CONTENT.learningSessions = {
  "id": "learning-sessions",
  "title": "学习会话",
  "eyebrow": "Interactive Learning Loop",
  "description": "记录每次和 Codex 共创知识库、论文精读、面试追问和代码实现后的掌握状态、薄弱点与下一轮训练计划。",
  "dashboard": {
    "summary": "建议以后每次只选一个专题，通过“提问 - 回答 - 纠偏 - 追问 - 沉淀”的闭环推进，避免知识库变成一次性资料堆。",
    "activeLoop": [
      "选题",
      "追问",
      "回答",
      "纠偏",
      "沉淀",
      "复盘"
    ],
    "currentFocus": [
      "MTP / DSpark",
      "GRPO 及多步 RL",
      "Agent 工程机制",
      "MoE / RoPE 架构基础"
    ],
    "defaultCadence": "每天 1 个主专题 + 1 次代码训练 + 1 张论文卡片"
  },
  "protocols": [
    {
      "id": "paper-reading",
      "group": "交互协议",
      "title": "论文精读模式",
      "mode": "Paper Reading",
      "status": "推荐常用",
      "goal": "把新论文先沉淀到论文库，再判断是否进入知识库正文。",
      "steps": [
        "你提供 PDF、arXiv 或标题。",
        "我先判断它关联哪些知识章节和面试主题。",
        "再抽取问题、方法、贡献、局限、关键公式和可追问点。",
        "最后生成论文卡片，并只把稳定结论同步到知识库。"
      ],
      "output": [
        "论文库卡片",
        "证据边界",
        "关联章节",
        "面试追问题"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "这篇论文解决的是训练问题、推理问题、数据问题还是系统问题？",
        "它和已有知识库中哪个方法最容易混淆？",
        "哪些结论可以进知识库，哪些只能留在论文库？"
      ]
    },
    {
      "id": "interview-grill",
      "group": "交互协议",
      "title": "面试拷打模式",
      "mode": "Interview Drill",
      "status": "用于查漏补缺",
      "goal": "通过连续追问暴露概念混淆、公式解释薄弱和工程表达短板。",
      "steps": [
        "你指定专题，例如“拷打我 GRPO”。",
        "我给 5-10 个递进问题。",
        "你用自己的话回答，不要求一次完美。",
        "我标记已掌握、表达不清、概念混淆和需要代码实现的点。"
      ],
      "output": [
        "薄弱点记录",
        "下一轮追问",
        "知识库补丁",
        "面试表达模板"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "你能不用公式先讲清楚这个方法的直觉吗？",
        "如果面试官继续追问工程实现，你会怎么答？",
        "这个方法和相邻方法的边界在哪里？"
      ]
    },
    {
      "id": "chapter-coauthoring",
      "group": "交互协议",
      "title": "章节共创模式",
      "mode": "Chapter Coauthoring",
      "status": "用于系统化建设",
      "goal": "用对话逐步完善一个知识章节，而不是一次性塞满内容。",
      "steps": [
        "我先给章节大纲和面试地图。",
        "你确认哪些方向是当前重点。",
        "我围绕重点追问并补公式、图示和代码骨架。",
        "最终把稳定内容写入对应 md。"
      ],
      "output": [
        "章节 md",
        "核心问题清单",
        "后续扩展计划",
        "参考资料"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "这个章节应该服务哪类面试问题？",
        "哪些内容必须会讲，哪些只需要知道方向？",
        "这个章节需要配套什么代码实现？"
      ]
    },
    {
      "id": "code-implementation",
      "group": "交互协议",
      "title": "代码实现模式",
      "mode": "Code Implementation",
      "status": "用于手撕和原理代码",
      "goal": "把算法题训练和原理代码实现分开记录，提升手写代码能力。",
      "steps": [
        "我给可编辑 Python 脚本或原理代码骨架。",
        "你实现并运行样例。",
        "我 review 正确性、复杂度、边界条件和面试表达。",
        "完成后同步到代码训练或原理代码模块。"
      ],
      "output": [
        "代码文件",
        "review 记录",
        "边界用例",
        "能力评估更新"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "这段实现的不变量是什么？",
        "最容易漏掉的边界条件是什么？",
        "如果要在面试中 10 分钟写完，应该先写哪部分？"
      ]
    }
  ],
  "sessions": [
    {
      "id": "session-grpo-advanced-rlvr-2026-07-04",
      "group": "论文精读",
      "title": "GRPO / RLVR 高级方法谱系精读",
      "mode": "Paper Reading",
      "status": "已完成",
      "date": "2026-07-04",
      "topicRefs": [
        "grpo",
        "multi-step-rl",
        "training-inference-frameworks"
      ],
      "paperRefs": [
        "high-entropy-minority-tokens",
        "cispo",
        "dapo",
        "gspo",
        "sapo",
        "rloo",
        "reinforce-plus-plus",
        "fipo",
        "real-rlvr",
        "chord",
        "deepeyes",
        "router-replay",
        "training-inference-mismatch-rlvr",
        "treepo"
      ],
      "summary": "基于 SWIFT GRPO Advanced Research 的 14 篇材料，整理 RLVR 高级方法谱系，覆盖 token 熵、ratio/clip 稳定性、advantage 估计、token credit assignment、SFT/RL 混合、结构化 rollout 和训练-推理一致性。",
      "strengths": [
        "已经形成方法谱系",
        "能把 GRPO 扩展问题拆成多条主线"
      ],
      "weaknesses": [
        "具体公式仍需逐篇精读",
        "CISPO / GSPO / SAPO 的 loss 差异需要继续写代码验证"
      ],
      "nextQuestions": [
        "CISPO、GSPO、SAPO 分别如何处理 policy ratio？",
        "FIPO 与 REAL 如何从不同角度解决 credit assignment？",
        "TreePO、DeepEyes、Router Replay 分别对应哪类系统问题？"
      ],
      "nextActions": [
        "精读 DAPO / GSPO / FIPO",
        "整理 advanced RLVR 对照表",
        "补充 GRPO family loss 最小实现"
      ],
      "notes": [
        "基于 SWIFT GRPO Advanced Research 的 14 篇材料，整理 RLVR 改进谱系：token 熵、ratio/clip 稳定性、advantage 估计、token credit assignment、SFT/RL 混合、结构化 rollout 和训练-推理一致性。"
      ]
    },
    {
      "id": "session-poweropd-opd-2026-06-29",
      "group": "最近会话",
      "title": "PowerOPD 与 sampled-token OPD 病理",
      "mode": "Paper Reading",
      "status": "已沉淀",
      "date": "2026-06-29",
      "topicRefs": [
        "opd"
      ],
      "paperRefs": [
        "poweropd"
      ],
      "summary": "通过公众号解读和 PowerOPD 论文确认：vanilla sampled-token OPD 的核心不稳定来自无界 log-ratio reward，而不是简单的 RL 训练噪声。",
      "strengths": [
        "能把社区解读映射到 OPD 知识章节",
        "开始关注 reward design 对后训练稳定性的影响"
      ],
      "weaknesses": [
        "需要继续推导 Box-Cox reward 与 log-ratio 的极限关系",
        "需要练习 full-vocab OPD 与 sampled-token OPD 的成本/稳定性对比"
      ],
      "nextQuestions": [
        "为什么 full-vocab OPD 可以平均掉极端 token，而 sampled-token OPD 会直接吃到极端 reward？",
        "PowerOPD 的 sign consistency 为什么重要？",
        "如果 alpha 变大，reward 分布、response length 和训练稳定性可能怎么变化？"
      ],
      "nextActions": [
        "补 PowerOPD reward 最小代码实现",
        "精读论文 Section 2-4",
        "把 OPD 与 GRPO 的 reward shaping 差异做一次面试追问"
      ]
    },
    {
      "id": "session-dspark-mtp-2026-06-29",
      "group": "最近会话",
      "title": "DSpark 与 MTP 关系澄清",
      "mode": "Paper Reading",
      "status": "已沉淀",
      "date": "2026-06-29",
      "topicRefs": [
        "mtp",
        "training-inference-frameworks"
      ],
      "paperRefs": [
        "dspark"
      ],
      "summary": "确认 DSpark 与 MTP 关系很大，但 DSpark 更偏 speculative decoding / serving 系统，不等同于 MTP 训练目标。",
      "strengths": [
        "能主动发现新论文与已有 MTP 知识的关联",
        "开始关注推理系统和训练目标的边界"
      ],
      "weaknesses": [
        "需要继续区分 MTP 训练目标、multi-token drafter、speculative decoding 三者边界"
      ],
      "nextQuestions": [
        "DSpark 为什么用 semi-autoregressive drafter，而不是纯 parallel drafter？",
        "confidence-scheduled verification 如何减少高并发下的 verification waste？",
        "MTP-1 baseline 在生产 serving 中为什么会成为一个保守选择？"
      ],
      "nextActions": [
        "精读 DSpark Section 3",
        "补 speculative decoding 最小代码骨架",
        "横向比较 MTP / Eagle / DFlash / DSpark"
      ]
    },
    {
      "id": "session-speculative-decoding-mtp-2026-07-04",
      "group": "最近会话",
      "title": "投机采样与 MTP 推理侧边界",
      "mode": "Paper Reading",
      "status": "已沉淀",
      "date": "2026-07-04",
      "topicRefs": [
        "mtp",
        "training-inference-frameworks"
      ],
      "paperRefs": [
        "wechat-speculative-decoding-mtp",
        "dspark"
      ],
      "summary": "通过社区文章补齐 speculative decoding 的 draft-verify 直觉，并把 Medusa、EAGLE、SpecInfer、DSpark 与 MTP 训练目标区分开。",
      "strengths": [
        "能主动追问 MTP 和推理加速路线之间的关系",
        "开始把训练目标、候选生成、serving 调度拆层理解"
      ],
      "weaknesses": [
        "需要继续练习 rejection sampling 接受/拒绝公式推导",
        "需要补一个最小 speculative decoding 代码 demo"
      ],
      "nextQuestions": [
        "为什么 speculative decoding 的输出分布可以等价于 target model？",
        "Medusa 的多预测头和 MTP 的多 token head 在训练目标上有什么差异？",
        "树状 speculative decoding 为什么能缓解单路径 draft 的不确定性？"
      ],
      "nextActions": [
        "补 speculative decoding 最小实现",
        "横向精读 SpecInfer / Medusa / EAGLE",
        "把 MTP 章节中的推理侧谱系画成 SVG 流程图"
      ]
    },
    {
      "id": "session-architecture-map-2026-06-29",
      "group": "最近会话",
      "title": "模型架构专题扩展",
      "mode": "Chapter Coauthoring",
      "status": "已沉淀",
      "date": "2026-06-29",
      "topicRefs": [
        "moe-architecture",
        "positional-encoding"
      ],
      "paperRefs": [],
      "summary": "新增 MoE 架构演进、位置编码与长度外推章节，形成模型结构方向的独立面试地图。",
      "strengths": [
        "能快速识别 MoE、RoPE、PI、NTK-aware 等架构高频面试点"
      ],
      "weaknesses": [
        "需要通过追问把公式、工程代价和具体模型案例连接得更稳"
      ],
      "nextQuestions": [
        "RoPE 的相对位置性质如何从旋转矩阵点积推出来？",
        "MoE 的 load balancing loss 为什么可能和语言建模目标冲突？",
        "PI 和 NTK-aware scaling 分别牺牲了什么？"
      ],
      "nextActions": [
        "画 RoPE 旋转图",
        "补 MoE expert dispatch demo",
        "整理 LLaMA/Qwen/DeepSeek 的 RoPE 配置对比"
      ]
    }
  ],
  "entries": [
    {
      "id": "paper-reading",
      "group": "交互协议",
      "title": "论文精读模式",
      "mode": "Paper Reading",
      "status": "推荐常用",
      "goal": "把新论文先沉淀到论文库，再判断是否进入知识库正文。",
      "steps": [
        "你提供 PDF、arXiv 或标题。",
        "我先判断它关联哪些知识章节和面试主题。",
        "再抽取问题、方法、贡献、局限、关键公式和可追问点。",
        "最后生成论文卡片，并只把稳定结论同步到知识库。"
      ],
      "output": [
        "论文库卡片",
        "证据边界",
        "关联章节",
        "面试追问题"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "这篇论文解决的是训练问题、推理问题、数据问题还是系统问题？",
        "它和已有知识库中哪个方法最容易混淆？",
        "哪些结论可以进知识库，哪些只能留在论文库？"
      ]
    },
    {
      "id": "interview-grill",
      "group": "交互协议",
      "title": "面试拷打模式",
      "mode": "Interview Drill",
      "status": "用于查漏补缺",
      "goal": "通过连续追问暴露概念混淆、公式解释薄弱和工程表达短板。",
      "steps": [
        "你指定专题，例如“拷打我 GRPO”。",
        "我给 5-10 个递进问题。",
        "你用自己的话回答，不要求一次完美。",
        "我标记已掌握、表达不清、概念混淆和需要代码实现的点。"
      ],
      "output": [
        "薄弱点记录",
        "下一轮追问",
        "知识库补丁",
        "面试表达模板"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "你能不用公式先讲清楚这个方法的直觉吗？",
        "如果面试官继续追问工程实现，你会怎么答？",
        "这个方法和相邻方法的边界在哪里？"
      ]
    },
    {
      "id": "chapter-coauthoring",
      "group": "交互协议",
      "title": "章节共创模式",
      "mode": "Chapter Coauthoring",
      "status": "用于系统化建设",
      "goal": "用对话逐步完善一个知识章节，而不是一次性塞满内容。",
      "steps": [
        "我先给章节大纲和面试地图。",
        "你确认哪些方向是当前重点。",
        "我围绕重点追问并补公式、图示和代码骨架。",
        "最终把稳定内容写入对应 md。"
      ],
      "output": [
        "章节 md",
        "核心问题清单",
        "后续扩展计划",
        "参考资料"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "这个章节应该服务哪类面试问题？",
        "哪些内容必须会讲，哪些只需要知道方向？",
        "这个章节需要配套什么代码实现？"
      ]
    },
    {
      "id": "code-implementation",
      "group": "交互协议",
      "title": "代码实现模式",
      "mode": "Code Implementation",
      "status": "用于手撕和原理代码",
      "goal": "把算法题训练和原理代码实现分开记录，提升手写代码能力。",
      "steps": [
        "我给可编辑 Python 脚本或原理代码骨架。",
        "你实现并运行样例。",
        "我 review 正确性、复杂度、边界条件和面试表达。",
        "完成后同步到代码训练或原理代码模块。"
      ],
      "output": [
        "代码文件",
        "review 记录",
        "边界用例",
        "能力评估更新"
      ],
      "weaknesses": [],
      "nextQuestions": [
        "这段实现的不变量是什么？",
        "最容易漏掉的边界条件是什么？",
        "如果要在面试中 10 分钟写完，应该先写哪部分？"
      ]
    },
    {
      "id": "session-grpo-advanced-rlvr-2026-07-04",
      "group": "论文精读",
      "title": "GRPO / RLVR 高级方法谱系精读",
      "mode": "Paper Reading",
      "status": "已完成",
      "date": "2026-07-04",
      "topicRefs": [
        "grpo",
        "multi-step-rl",
        "training-inference-frameworks"
      ],
      "paperRefs": [
        "high-entropy-minority-tokens",
        "cispo",
        "dapo",
        "gspo",
        "sapo",
        "rloo",
        "reinforce-plus-plus",
        "fipo",
        "real-rlvr",
        "chord",
        "deepeyes",
        "router-replay",
        "training-inference-mismatch-rlvr",
        "treepo"
      ],
      "summary": "基于 SWIFT GRPO Advanced Research 的 14 篇材料，整理 RLVR 高级方法谱系，覆盖 token 熵、ratio/clip 稳定性、advantage 估计、token credit assignment、SFT/RL 混合、结构化 rollout 和训练-推理一致性。",
      "strengths": [
        "已经形成方法谱系",
        "能把 GRPO 扩展问题拆成多条主线"
      ],
      "weaknesses": [
        "具体公式仍需逐篇精读",
        "CISPO / GSPO / SAPO 的 loss 差异需要继续写代码验证"
      ],
      "nextQuestions": [
        "CISPO、GSPO、SAPO 分别如何处理 policy ratio？",
        "FIPO 与 REAL 如何从不同角度解决 credit assignment？",
        "TreePO、DeepEyes、Router Replay 分别对应哪类系统问题？"
      ],
      "nextActions": [
        "精读 DAPO / GSPO / FIPO",
        "整理 advanced RLVR 对照表",
        "补充 GRPO family loss 最小实现"
      ],
      "notes": [
        "基于 SWIFT GRPO Advanced Research 的 14 篇材料，整理 RLVR 改进谱系：token 熵、ratio/clip 稳定性、advantage 估计、token credit assignment、SFT/RL 混合、结构化 rollout 和训练-推理一致性。"
      ]
    },
    {
      "id": "session-poweropd-opd-2026-06-29",
      "group": "最近会话",
      "title": "PowerOPD 与 sampled-token OPD 病理",
      "mode": "Paper Reading",
      "status": "已沉淀",
      "date": "2026-06-29",
      "topicRefs": [
        "opd"
      ],
      "paperRefs": [
        "poweropd"
      ],
      "summary": "通过公众号解读和 PowerOPD 论文确认：vanilla sampled-token OPD 的核心不稳定来自无界 log-ratio reward，而不是简单的 RL 训练噪声。",
      "strengths": [
        "能把社区解读映射到 OPD 知识章节",
        "开始关注 reward design 对后训练稳定性的影响"
      ],
      "weaknesses": [
        "需要继续推导 Box-Cox reward 与 log-ratio 的极限关系",
        "需要练习 full-vocab OPD 与 sampled-token OPD 的成本/稳定性对比"
      ],
      "nextQuestions": [
        "为什么 full-vocab OPD 可以平均掉极端 token，而 sampled-token OPD 会直接吃到极端 reward？",
        "PowerOPD 的 sign consistency 为什么重要？",
        "如果 alpha 变大，reward 分布、response length 和训练稳定性可能怎么变化？"
      ],
      "nextActions": [
        "补 PowerOPD reward 最小代码实现",
        "精读论文 Section 2-4",
        "把 OPD 与 GRPO 的 reward shaping 差异做一次面试追问"
      ]
    },
    {
      "id": "session-dspark-mtp-2026-06-29",
      "group": "最近会话",
      "title": "DSpark 与 MTP 关系澄清",
      "mode": "Paper Reading",
      "status": "已沉淀",
      "date": "2026-06-29",
      "topicRefs": [
        "mtp",
        "training-inference-frameworks"
      ],
      "paperRefs": [
        "dspark"
      ],
      "summary": "确认 DSpark 与 MTP 关系很大，但 DSpark 更偏 speculative decoding / serving 系统，不等同于 MTP 训练目标。",
      "strengths": [
        "能主动发现新论文与已有 MTP 知识的关联",
        "开始关注推理系统和训练目标的边界"
      ],
      "weaknesses": [
        "需要继续区分 MTP 训练目标、multi-token drafter、speculative decoding 三者边界"
      ],
      "nextQuestions": [
        "DSpark 为什么用 semi-autoregressive drafter，而不是纯 parallel drafter？",
        "confidence-scheduled verification 如何减少高并发下的 verification waste？",
        "MTP-1 baseline 在生产 serving 中为什么会成为一个保守选择？"
      ],
      "nextActions": [
        "精读 DSpark Section 3",
        "补 speculative decoding 最小代码骨架",
        "横向比较 MTP / Eagle / DFlash / DSpark"
      ]
    },
    {
      "id": "session-speculative-decoding-mtp-2026-07-04",
      "group": "最近会话",
      "title": "投机采样与 MTP 推理侧边界",
      "mode": "Paper Reading",
      "status": "已沉淀",
      "date": "2026-07-04",
      "topicRefs": [
        "mtp",
        "training-inference-frameworks"
      ],
      "paperRefs": [
        "wechat-speculative-decoding-mtp",
        "dspark"
      ],
      "summary": "通过社区文章补齐 speculative decoding 的 draft-verify 直觉，并把 Medusa、EAGLE、SpecInfer、DSpark 与 MTP 训练目标区分开。",
      "strengths": [
        "能主动追问 MTP 和推理加速路线之间的关系",
        "开始把训练目标、候选生成、serving 调度拆层理解"
      ],
      "weaknesses": [
        "需要继续练习 rejection sampling 接受/拒绝公式推导",
        "需要补一个最小 speculative decoding 代码 demo"
      ],
      "nextQuestions": [
        "为什么 speculative decoding 的输出分布可以等价于 target model？",
        "Medusa 的多预测头和 MTP 的多 token head 在训练目标上有什么差异？",
        "树状 speculative decoding 为什么能缓解单路径 draft 的不确定性？"
      ],
      "nextActions": [
        "补 speculative decoding 最小实现",
        "横向精读 SpecInfer / Medusa / EAGLE",
        "把 MTP 章节中的推理侧谱系画成 SVG 流程图"
      ]
    },
    {
      "id": "session-architecture-map-2026-06-29",
      "group": "最近会话",
      "title": "模型架构专题扩展",
      "mode": "Chapter Coauthoring",
      "status": "已沉淀",
      "date": "2026-06-29",
      "topicRefs": [
        "moe-architecture",
        "positional-encoding"
      ],
      "paperRefs": [],
      "summary": "新增 MoE 架构演进、位置编码与长度外推章节，形成模型结构方向的独立面试地图。",
      "strengths": [
        "能快速识别 MoE、RoPE、PI、NTK-aware 等架构高频面试点"
      ],
      "weaknesses": [
        "需要通过追问把公式、工程代价和具体模型案例连接得更稳"
      ],
      "nextQuestions": [
        "RoPE 的相对位置性质如何从旋转矩阵点积推出来？",
        "MoE 的 load balancing loss 为什么可能和语言建模目标冲突？",
        "PI 和 NTK-aware scaling 分别牺牲了什么？"
      ],
      "nextActions": [
        "画 RoPE 旋转图",
        "补 MoE expert dispatch demo",
        "整理 LLaMA/Qwen/DeepSeek 的 RoPE 配置对比"
      ]
    }
  ]
};
