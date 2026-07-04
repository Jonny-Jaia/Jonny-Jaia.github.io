window.BJ_CONTENT = window.BJ_CONTENT || {};

window.BJ_CONTENT.principleCode = {
  id: "principle-code",
  title: "原理代码",
  eyebrow: "Principle Code",
  description:
    "这里的代码服务于理解大模型训练和架构原理，不等同于手撕算法训练。每段代码都对应一个知识点和一组面试追问。",
  items: [
    {
      id: "agent-rag-systems",
      group: "Agent/RAG 手撕代码",
      title: "Tool Registry / Memory / BM25 / Hybrid Search",
      status: "可运行实现",
      linkedTopics: ["agent-guide-interview-qa", "agent", "agent-interview-practice"],
      scenario: "覆盖 AgentGuide 系统类手撕题：工具注册、ReAct、记忆系统、文档切块、BM25、混合检索和语义缓存。",
      code: [
        "代码文件：content/principle-code/code/agent_rag_systems.py",
        "核心类/函数：ToolRegistry、simple_react_agent、ShortTermMemory、VectorMemory、BM25Index、hybrid_search、SemanticCache",
        "面试重点：工具 schema 校验、错误 fallback、短期/长期记忆分层、BM25 稀疏检索、dense+sparse 分数融合、语义缓存阈值。"
      ].join("\n"),
      interviewPrompts: [
        "Tool Registry 为什么要做 schema 校验和 unknown tool fallback？",
        "短期记忆和长期记忆分别存什么，如何避免旧记忆干扰？",
        "BM25 和向量检索为什么互补，融合分数怎么做？",
        "Semantic Cache 的阈值过高或过低分别有什么风险？"
      ]
    },
    {
      id: "llm-decoding-sampling",
      group: "LLM 手撕代码",
      title: "Top-k / Top-p sampling",
      status: "可运行实现",
      linkedTopics: ["llm-engineering-interview-supplement", "llm-engineering-foundations", "mtp"],
      scenario: "理解 temperature、top-k、top-p 在推理采样阶段如何过滤 logits。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：top_k_filtering、top_p_filtering",
        "面试重点：先 temperature，再过滤 logits；top-p 要保留第一个越过阈值的 token；过滤后再 softmax / multinomial。"
      ].join("\n"),
      interviewPrompts: ["top-k 和 top-p 的候选集合有什么区别？", "为什么 top-p 更能适应模型自信和犹豫两种状态？", "为什么过滤通常作用在 logits 而不是已经采样后的 token？"]
    },
    {
      id: "llm-norm-sft-ce",
      group: "LLM 手撕代码",
      title: "LayerNorm / RMSNorm / SFT loss / CE",
      status: "可运行实现",
      linkedTopics: ["llm-engineering-foundations", "sft"],
      scenario: "覆盖 Norm、数值稳定 Softmax、CrossEntropy、SFT shift-right loss 这些最常考基础算子。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数/类：ManualLayerNorm、ManualRMSNorm、stable_softmax、cross_entropy_from_logits、sft_cross_entropy_loss",
        "面试重点：LayerNorm 有 gamma/beta；RMSNorm 不减均值；CE 要用 logsumexp；SFT loss 用 logits[:, :-1] 对 labels[:, 1:]。"
      ].join("\n"),
      interviewPrompts: ["SFT 的 shift-right 为什么要去掉 logits 最后一位和 labels 第一位？", "CrossEntropy 为什么不要先显式 softmax 再 log？", "RMSNorm 比 LayerNorm 少了什么参数和操作？"]
    },
    {
      id: "llm-attention-rope",
      group: "LLM 手撕代码",
      title: "QKV / MHA / RoPE",
      status: "可运行实现",
      linkedTopics: ["llm-engineering-foundations", "positional-encoding"],
      scenario: "覆盖 Self-Attention、MHA 维度变换、causal mask 和 RoPE 应用位置。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数/类：scaled_dot_product_attention、MultiHeadSelfAttention、apply_rope",
        "面试重点：MHA 维度从 [B,S,H] 变为 [B,num_heads,S,head_dim]；RoPE 作用在 Q/K 上；Decoder-only 必须使用 causal mask。"
      ].join("\n"),
      interviewPrompts: ["Attention 为什么除以 sqrt(d_k)？", "RoPE 为什么只加在 Q/K 上？", "如果去掉 K 变成 QQV，会破坏什么匹配关系？"]
    },
    {
      id: "llm-kv-cache-serving",
      group: "推理系统手撕代码",
      title: "Paged KV Cache / Radix Prefix Cache / Continuous Batching",
      status: "可运行实现",
      linkedTopics: ["training-inference-frameworks", "mtp", "agentic-rl-system"],
      scenario: "覆盖 vLLM / SGLang 面试里最容易被追问的推理系统机制：KV cache block table、prefix reuse、decode step 动态合批。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数/类：KVBlock、PagedKVCache、RadixPrefixCache、DecodeRequest、continuous_batching_step",
        "面试重点：PagedAttention 把 KV cache 拆成 block table；RadixAttention 用前缀树复用共享 prompt；continuous batching 每个 decode step 后移除完成请求并补入 waiting queue。"
      ].join("\n"),
      interviewPrompts: [
        "PagedAttention 为什么能缓解 KV cache 碎片？",
        "RadixAttention 和普通 prefix cache 的区别是什么？",
        "continuous batching 相比静态 batching 提升在哪里？",
        "为什么 RL rollout 场景尤其依赖高吞吐推理引擎？"
      ]
    },    {
      id: "cv-foundation-kernels",
      group: "CV / 多模态手撕代码",
      title: "ViT / CLIP / MAE / DDPM kernels",
      status: "可运行实现",
      linkedTopics: ["multimodal-vlm", "vla-embodied-driving"],
      scenario: "把 CV 基础模型中最容易在面试里追问的四个核心操作放在同一个代码入口：图像 patch 化、图文对比学习、masked image modeling、扩散前向加噪。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：vit_patchify、clip_contrastive_loss、mae_random_mask、diffusion_q_sample",
        "面试重点：ViT 把图像切成 token；CLIP 用 batch 内双向 InfoNCE 对齐图文；MAE 只把可见 patch 送入 encoder；DDPM 训练时通常预测加入的噪声 epsilon。"
      ].join("\n"),
      interviewPrompts: [
        "ViT patchify 后 token 数量和 patch size 的关系是什么？",
        "CLIP 为什么要同时做 image-to-text 和 text-to-image loss？",
        "MAE 为什么高 mask ratio 仍然有效？",
        "DDPM 的 q_sample 为什么可以直接由 x0 和 epsilon 闭式采样？"
      ]
    },    {
      id: "llm-rl-losses-gae",
      group: "LLM 手撕代码",
      title: "PPO / DPO / GRPO loss and GAE",
      status: "可运行实现",
      linkedTopics: ["grpo", "dpo", "rl-foundations"],
      scenario: "覆盖 PPO clip、GAE、DPO log-ratio、GRPO group relative advantage 和 k3 KL 近似。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：ppo_clipped_policy_loss、compute_gae、dpo_loss、grpo_group_advantages、grpo_loss",
        "面试重点：PPO 使用 importance ratio + clip；DPO 比较 policy/ref 的 chosen-rejected log-ratio；GRPO 按同 prompt 组内 reward 标准化。"
      ].join("\n"),
      interviewPrompts: ["PPO 的 ratio 为什么用 exp(new_logprob - old_logprob)？", "DPO 的 reference model 起什么作用？", "GRPO 为什么要按 prompt 分组计算 advantage？"]
    },
    {
      id: "llm-react-tool-loop",
      group: "LLM 手撕代码",
      title: "ReAct tool loop",
      status: "可运行骨架",
      linkedTopics: ["agent", "agent-interview-practice", "agentic-rl-system"],
      scenario: "覆盖 Thought -> Action -> Observation 的 Agent 工具调用循环和异常 fallback。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：react_loop",
        "面试重点：工具调用不是输出函数名这么简单，还要处理 schema、unknown tool、tool error、max_steps 和 final answer。"
      ].join("\n"),
      interviewPrompts: ["ReAct 和普通 CoT 的差异是什么？", "工具调用失败时如何 fallback？", "Agent loop 为什么必须有 max_steps 和权限边界？"]
    },
    {
      id: "agentic-rl-loop",
      group: "Agent/RL 手撕代码",
      title: "Agentic RL trace / mask / reward loop",
      status: "可运行实现",
      linkedTopics: ["agentic-rl-system", "agent", "verl-rl-framework", "sampling-evaluation-rft", "multi-step-rl"],
      scenario: "把 VeRL Agent Loop / Reward Loop 中最适合面试手写的部分拆成 trace step、response mask、step credit assignment、reward 聚合和可复盘摘要。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：AgentTraceStep、build_agent_response_mask、assign_step_credit、build_token_advantages_from_steps、score_agent_trace、replayable_trace_summary",
        "面试重点：assistant token 才是 policy action；tool observation 进上下文但不进 loss；credit assignment 要把 final reward、step reward、成本和错误惩罚映射回 step/token。"
      ].join("\n"),
      interviewPrompts: ["为什么工具返回 token 不能参与 policy loss？", "只用 final reward 做广播有什么问题？", "Agentic RL 里 step-level advantage 怎么构造？", "如何排查 response_mask 错位导致的训练异常？"]
    },
    {
      id: "opd-loss",
      group: "后训练蒸馏",
      title: "OPD forward KL / PG OPD / PowerOPD reward",
      status: "可运行实现",
      linkedTopics: ["opd", "distillation", "verl-rl-framework"],
      scenario: "理解 OPD 为什么既可以作为分布级 KL loss，也可以作为 sampled-token policy-gradient reward，并用 PowerOPD 缓解 log-ratio 无界问题。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：token_level_forward_kl、topk_forward_kl、sampled_reverse_kl_reward、pg_opd_loss、power_opd_reward",
        "面试重点：GKD OPD 直接最小化 teacher/student KL；PG OPD 把 teacher-student log-ratio 作为 stop-gradient reward；PowerOPD 用有界幂变换降低 sampled-token 高方差。"
      ].join("\n"),
      interviewPrompts: [
        "GKD OPD 和 PG OPD 的梯度路径有什么不同？",
        "为什么 PG OPD 的 teacher reward 必须 stop-gradient？",
        "top-k teacher logits 相比 full-vocab KL 省了什么、损失了什么？",
        "PowerOPD 为什么说修的是 log-ratio reward 的根因？"
      ]
    },
    {
      id: "dapo-loss",
      group: "后训练 RL",
      title: "DAPO clip higher / dynamic sampling / token loss",
      status: "可运行实现",
      linkedTopics: ["grpo", "training-inference-frameworks", "verl-rl-framework"],
      scenario: "把 SWIFT / VeRL DAPO 文档中的 clip higher、dynamic sampling、token-level loss、soft overlong punishment 拆成可手写的最小函数。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：dapo_clipped_policy_loss、filter_zero_variance_reward_groups、token_mean_loss、soft_overlong_penalty",
        "面试重点：DAPO 不是单个公式，而是一套修 GRPO 稳定性的 recipe：非对称 clip 鼓励探索，动态采样过滤零方差 group，token-mean 缓解长度偏差，soft overlong 平滑控制超长输出。"
      ].join("\n"),
      interviewPrompts: [
        "DAPO 的 clip higher 为什么只放宽上界？",
        "为什么 group reward 全一样时要动态采样或过滤？",
        "token-mean 和 seq-mean-token-sum 的长度偏差有什么不同？",
        "soft overlong punishment 和硬截断相比有什么好处？"
      ]
    },
    {
      id: "mtp-loss",
      group: "模型训练目标",
      title: "MTP shifted labels / multi-head loss / speculative verify",
      status: "可运行实现",
      linkedTopics: ["mtp", "training-inference-frameworks", "verl-rl-framework"],
      scenario: "理解 Multi-Token Prediction 如何把同一 hidden state 扩展为多个 future-token 监督信号，并区分 MTP 训练目标与 speculative decoding 的验证逻辑。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：build_mtp_shifted_labels、mtp_multi_head_loss、speculative_accept_mask、accepted_prefix_lengths",
        "面试重点：MTP 的第 k 个 head 预测 t+k token；多头 loss 需要权重；投机解码的 draft token 必须由 target 模型验证，真实无损版本还要做概率校正。"
      ].join("\n"),
      interviewPrompts: [
        "MTP 的 shifted labels 和普通 SFT shift-right 有什么关系？",
        "为什么 MTP loss 权重不能无脑设很大？",
        "MTP、Medusa、EAGLE、DSpark 分别属于训练目标、草稿生成还是 serving 系统？",
        "为什么 speculative decoding 接受前缀越长不一定吞吐越高？"
      ]
    },
    {
      id: "sft-loss-mask",
      group: "后训练 loss",
      title: "SFT loss mask",
      status: "骨架",
      linkedTopics: ["sft"],
      scenario: "理解为什么只对 assistant response 计算监督信号。",
      code: [
        "def masked_cross_entropy(logits, labels, loss_mask):",
        "    # logits: [batch, seq, vocab]",
        "    # labels: [batch, seq]",
        "    # loss_mask: 1 means this token contributes to loss",
        "    pass"
      ].join("\n"),
      interviewPrompts: ["为什么 prompt token 通常不参与 loss？", "多轮对话中 mask 应该如何构造？"]
    },
    {
      id: "dpo-loss",
      group: "后训练 loss",
      title: "DPO loss",
      status: "骨架",
      linkedTopics: ["dpo"],
      scenario: "理解 chosen/rejected 和 reference model logprob ratio。",
      code: [
        "def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):",
        "    # Fill in log-ratio based preference objective.",
        "    pass"
      ].join("\n"),
      interviewPrompts: ["reference model 的 logprob 为什么要进入公式？", "beta 控制了什么？"]
    },
    {
      id: "grpo-advantage",
      group: "后训练 RL",
      title: "GRPO / RLOO / GSPO advantage and ratio",
      status: "可运行实现",
      linkedTopics: ["grpo", "multi-step-rl", "agentic-rl-system"],
      scenario: "把 GRPO 的 group-relative advantage、RLOO 的 leave-one-out baseline、GSPO 的 sequence-level importance ratio 放在同一个手写代码入口中，方便面试时解释 RLVR 高级谱系的共同问题。",
      code: [
        "代码文件：content/principle-code/code/llm_handwritten_kernels.py",
        "核心函数：grpo_group_advantages、rloo_advantages、sequence_level_importance_ratio、grpo_loss",
        "面试重点：GRPO 用组内均值/方差替代 critic；RLOO 用其他样本均值做 baseline；GSPO 把 token ratio 聚合到序列级，缓解长 CoT / MoE 场景下 token 级 ratio 抖动。"
      ].join("\n"),
      interviewPrompts: [
        "GRPO 的 group baseline 和 RLOO 的 leave-one-out baseline 有什么区别？",
        "为什么 group_size=1 时 GRPO/RLOO 都退化？",
        "GSPO 为什么要从 token-level ratio 转向 sequence-level ratio？",
        "如果 response 长度差异很大，sequence ratio 是否需要 length normalization？"
      ]
    },
    {
      id: "moe-topk-router",
      group: "模型架构",
      title: "MoE top-k routing",
      status: "骨架",
      linkedTopics: ["chinese-model-series"],
      scenario: "理解 DeepSeek、Qwen、Kimi 等 MoE 模型中的专家选择和负载均衡直觉。",
      code: [
        "def moe_topk_router(hidden_states, router_weight, k=2):",
        "    # hidden_states: [tokens, hidden]",
        "    # router_weight: [hidden, num_experts]",
        "    # return top-k expert ids and routing probabilities",
        "    logits = hidden_states @ router_weight",
        "    probs = softmax(logits, dim=-1)",
        "    topk_probs, topk_ids = topk(probs, k=k, dim=-1)",
        "    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)",
        "    return topk_ids, topk_probs"
      ].join("\n"),
      interviewPrompts: ["MoE 为什么能扩大参数量但控制激活计算？", "top-k routing 会带来什么负载均衡问题？"]
    },
    {
      id: "sparse-block-attention",
      group: "模型架构",
      title: "Block sparse attention mask",
      status: "骨架",
      linkedTopics: ["chinese-model-series"],
      scenario: "理解 MiniMax Sparse Attention / 长上下文模型如何通过稀疏模式降低注意力成本。",
      code: [
        "def build_block_sparse_mask(seq_len, block_size, local_blocks, global_tokens):",
        "    # Return a boolean attention mask: True means this query can attend to key.",
        "    mask = zeros((seq_len, seq_len), dtype=bool)",
        "    for q in range(seq_len):",
        "        q_block = q // block_size",
        "        start_block = max(0, q_block - local_blocks)",
        "        for b in range(start_block, q_block + 1):",
        "            left = b * block_size",
        "            right = min(seq_len, (b + 1) * block_size)",
        "            mask[q, left:right] = True",
        "        mask[q, :global_tokens] = True",
        "    return causal(mask)"
      ].join("\n"),
      interviewPrompts: ["稀疏注意力减少的是什么复杂度？", "local block 和 global token 分别解决什么问题？"]
    },
    {
      id: "thinking-budget-gate",
      group: "推理控制",
      title: "Thinking budget gate",
      status: "骨架",
      linkedTopics: ["chinese-model-series"],
      scenario: "理解 Qwen3 thinking/non-thinking、MiniMax test-time compute 等推理预算控制思路。",
      code: [
        "def choose_reasoning_budget(query_features, max_budget):",
        "    # query_features may include difficulty score, tool need, and risk level.",
        "    difficulty = query_features['difficulty']",
        "    tool_need = query_features.get('tool_need', 0.0)",
        "    risk = query_features.get('risk', 0.0)",
        "    score = 0.6 * difficulty + 0.3 * tool_need + 0.1 * risk",
        "    if score < 0.3:",
        "        return 0",
        "    if score < 0.7:",
        "        return max_budget // 2",
        "    return max_budget"
      ].join("\n"),
      interviewPrompts: ["为什么 reasoning 模型需要控制 test-time compute？", "thinking 模式打开过多会有什么代价？"]
    }
  ]
};




