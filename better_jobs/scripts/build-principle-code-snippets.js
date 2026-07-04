const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const codeDir = path.join(root, "content", "principle-code", "code");
const outputFile = path.join(root, "content", "principle-code", "generated-code.js");

const itemConfigs = {
  "agent-rag-systems": {
    file: "agent_rag_systems.py",
    symbols: [
      "ToolRegistry",
      "AgentStep",
      "simple_react_agent",
      "ShortTermMemory",
      "MemoryItem",
      "VectorMemory",
      "fixed_size_chunks",
      "overlap_chunks",
      "BM25Index",
      "hybrid_search",
      "SemanticCache",
      "hashing_embedder"
    ],
    note: "Agent/RAG 手撕代码：工具注册、ReAct 循环、记忆、切块、BM25、混合检索和语义缓存。"
  },
  "llm-decoding-sampling": {
    file: "llm_handwritten_kernels.py",
    symbols: ["top_k_filtering", "top_p_filtering"],
    note: "解码采样手撕代码：先做 temperature 缩放，再用 top-k 或 top-p 过滤 logits。"
  },
  "llm-norm-sft-ce": {
    file: "llm_handwritten_kernels.py",
    symbols: ["stable_softmax", "cross_entropy_from_logits", "ManualLayerNorm", "ManualRMSNorm", "sft_cross_entropy_loss"],
    note: "基础算子手撕代码：稳定 softmax、交叉熵、LayerNorm/RMSNorm 和 SFT shift-right loss。"
  },
  "llm-attention-rope": {
    file: "llm_handwritten_kernels.py",
    symbols: ["scaled_dot_product_attention", "MultiHeadSelfAttention", "apply_rope"],
    note: "模型架构手撕代码：QKV、causal mask、多头 reshape 和 RoPE。"
  },
  "llm-kv-cache-serving": {
    file: "llm_handwritten_kernels.py",
    symbols: ["KVBlock", "PagedKVCache", "longest_common_prefix", "RadixNode", "RadixPrefixCache", "DecodeRequest", "continuous_batching_step"],
    note: "推理系统手写代码：Paged KV Cache 的 block table、SGLang/RadixAttention 前缀复用，以及 continuous batching 调度骨架。"
  },  "cv-foundation-kernels": {
    file: "llm_handwritten_kernels.py",
    symbols: ["vit_patchify", "clip_contrastive_loss", "mae_random_mask", "diffusion_q_sample"],
    note: "CV 基础模型手撕代码：ViT patchify、CLIP 双向对比损失、MAE 随机 mask 和 DDPM 前向加噪。"
  },
  "llm-rl-losses-gae": {
    file: "llm_handwritten_kernels.py",
    symbols: ["dpo_loss", "ppo_clipped_policy_loss", "compute_gae", "grpo_group_advantages", "grpo_loss"],
    note: "后训练 loss 手撕代码：DPO、PPO clip、GAE、GRPO 组内 advantage 和 KL 约束。"
  },
  "llm-react-tool-loop": {
    file: "llm_handwritten_kernels.py",
    symbols: ["ToolResult", "react_loop"],
    note: "Agent 循环手撕代码：Thought -> Action -> Observation，并处理未知工具、工具错误和最大步数。"
  },
  "agentic-rl-loop": {
    file: "llm_handwritten_kernels.py",
    symbols: ["AgentTraceStep", "build_agent_response_mask", "assign_step_credit", "build_token_advantages_from_steps", "score_agent_trace", "replayable_trace_summary"],
    note: "Agentic RL 手撕代码：构造多轮 trace、response_mask、reward loop 和可复盘摘要。"
  },
  "opd-loss": {
    file: "llm_handwritten_kernels.py",
    symbols: ["token_level_forward_kl", "topk_forward_kl", "sampled_reverse_kl_reward", "pg_opd_loss", "power_opd_reward"],
    note: "OPD 手撕代码：token-level forward KL、teacher top-k KL、PG OPD sampled-token reward 和 PowerOPD 有界 reward。"
  },
  "dapo-loss": {
    file: "llm_handwritten_kernels.py",
    symbols: ["dapo_clipped_policy_loss", "filter_zero_variance_reward_groups", "token_mean_loss", "soft_overlong_penalty"],
    note: "DAPO 手撕代码：非对称 clip、动态采样过滤、token-level loss 和 soft overlong punishment。"
  },
  "mtp-loss": {
    file: "llm_handwritten_kernels.py",
    symbols: ["build_mtp_shifted_labels", "mtp_multi_head_loss", "speculative_accept_mask", "accepted_prefix_lengths"],
    note: "MTP 手撕代码：多步 shifted labels、多头 future-token loss，以及 speculative decoding 的连续接受前缀验证。"
  },
  "verl-dataflow-loop": {
    file: "llm_handwritten_kernels.py",
    symbols: ["MiniDataProto", "mock_verl_rl_dataflow", "validate_verl_token_alignment", "verl_masked_policy_loss"],
    note: "VeRL dataflow 手撕代码：用 DataProto 风格容器串起 rollout、logprob、reward、advantage、mask 对齐和 actor update。"
  },
  "sft-loss-mask": {
    file: "llm_handwritten_kernels.py",
    symbols: ["sft_cross_entropy_loss"],
    note: "SFT loss 手撕代码：shift-right labels，并用 ignore_index/loss_mask 只训练 assistant token。"
  },
  "dpo-loss": {
    file: "llm_handwritten_kernels.py",
    symbols: ["dpo_loss"],
    note: "DPO loss 手撕代码：比较 policy 相对 reference 的 chosen/rejected log-ratio。"
  },
  "grpo-advantage": {
    file: "llm_handwritten_kernels.py",
    symbols: ["grpo_group_advantages", "rloo_advantages", "sequence_level_importance_ratio", "grpo_loss"],
    note: "GRPO/RLOO/GSPO 手撕代码：组内 reward 标准化、leave-one-out baseline、序列级 importance ratio 和最小 GRPO loss。"
  }
};

function readLines(file) {
  return fs.readFileSync(path.join(codeDir, file), "utf8").split(/\r?\n/);
}

function symbolName(line) {
  const match = line.match(/^(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b/);
  return match?.[1] || null;
}

function topLevelDefinition(line) {
  return /^(?:@|def\s+|class\s+|async\s+def\s+)/.test(line);
}

function extractSymbol(lines, name) {
  let defIndex = -1;
  for (let i = 0; i < lines.length; i += 1) {
    if (symbolName(lines[i]) === name) {
      defIndex = i;
      break;
    }
  }
  if (defIndex === -1) return "";

  let start = defIndex;
  while (start > 0 && lines[start - 1].startsWith("@")) start -= 1;

  let end = lines.length;
  for (let i = defIndex + 1; i < lines.length; i += 1) {
    if (topLevelDefinition(lines[i])) {
      end = i;
      break;
    }
  }
  return lines.slice(start, end).join("\n").trimEnd();
}

function makeSnippet(config) {
  const lines = readLines(config.file);
  const parts = config.symbols
    .map((symbol) => {
      const code = extractSymbol(lines, symbol);
      if (!code) return "";
      return `# 代码片段：${symbol}。面试讲解时先说输入输出、核心不变量、边界条件和复杂度。\n${code}`;
    })
    .filter(Boolean);
  return [
    `# ${config.note}`,
    `# 源文件：content/principle-code/code/${config.file}`,
    "",
    ...parts
  ].join("\n\n");
}

const snippets = {};
for (const [id, config] of Object.entries(itemConfigs)) {
  snippets[id] = {
    file: config.file,
    symbols: config.symbols,
    note: config.note,
    code: makeSnippet(config)
  };
}

fs.writeFileSync(
  outputFile,
  `window.BJ_PRINCIPLE_CODE_SNIPPETS = ${JSON.stringify(snippets, null, 2)};\n`,
  "utf8"
);

console.log(`Generated ${Object.keys(snippets).length} principle-code snippets.`);





