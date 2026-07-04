window.BJ_RENDERERS = window.BJ_RENDERERS || {};

function principleSnippet(item) {
  return window.BJ_PRINCIPLE_CODE_SNIPPETS?.[item.id] || null;
}

function readableCodeGroup(item) {
  const groups = {
    "agent-rag-systems": "Agent/RAG 手撕代码",
    "llm-decoding-sampling": "LLM 手撕代码",
    "llm-norm-sft-ce": "LLM 手撕代码",
    "llm-attention-rope": "LLM 手撕代码",
    "llm-rl-losses-gae": "LLM 手撕代码",
    "llm-react-tool-loop": "LLM 手撕代码",
    "sft-loss-mask": "后训练 loss",
    "dpo-loss": "后训练 loss",
    "grpo-advantage": "后训练 RL",
    "moe-topk-router": "模型架构",
    "sparse-block-attention": "模型架构",
    "thinking-budget-gate": "推理控制"
  };
  return groups[item.id] || item.group || "未分组";
}

function codeNarration(item) {
  const snippet = principleSnippet(item);
  return snippet?.note || item.scenario || "该条目提供一个面试中可手写、可解释的最小实现骨架。";
}

function codeForDisplay(item) {
  const snippet = principleSnippet(item);
  return snippet?.code || item.code || "# 暂无代码片段";
}

window.BJ_RENDERERS.codeSkill = {
  title(item) {
    return item.title;
  },
  subtitle(item) {
    return `${readableCodeGroup(item)} / ${item.status || "未标记"}`;
  },
  group(item) {
    return readableCodeGroup(item);
  },
  searchText(item) {
    const snippet = principleSnippet(item);
    return [
      item.title,
      readableCodeGroup(item),
      item.status,
      item.scenario,
      codeForDisplay(item),
      ...(item.linkedTopics || []),
      ...(item.interviewPrompts || []),
      ...(snippet?.symbols || [])
    ].join(" ");
  },
  renderOverview(item) {
    return `
      <h3>${item.title}</h3>
      <p>${codeNarration(item)}</p>
      <div class="meta">
        <span class="pill">${readableCodeGroup(item)}</span>
        ${principleSnippet(item) ? '<span class="pill subtle">内联代码</span>' : '<span class="pill warning">骨架</span>'}
      </div>
    `;
  },
  renderMain(item) {
    const snippet = principleSnippet(item);
    return `
      <header class="article-header">
        <div>
          <h2>${item.title}</h2>
          <p>${codeNarration(item)}</p>
        </div>
        <div class="meta">
          <span class="pill warning">${item.status || "未标记"}</span>
          <span class="pill">${readableCodeGroup(item)}</span>
        </div>
      </header>
      <section class="section-block grid-two">
        <div class="info-box">
          <strong>面试讲解重点</strong>
          <p>${codeNarration(item)}</p>
        </div>
        <div class="info-box">
          <strong>代码来源</strong>
          <p>${snippet ? `已从 content/principle-code/code/${snippet.file} 抽取：${snippet.symbols.join(", ")}` : "当前条目使用索引内的最小骨架代码。"}</p>
        </div>
      </section>
      <section class="section-block">
        <h3>可直接阅读的代码实现</h3>
        <div class="code-box code-box-large"><pre><code class="language-python">${escapeHtml(codeForDisplay(item))}</code></pre></div>
      </section>
      <section class="section-block grid-two">
        <div class="info-box">
          <strong>关联知识点</strong>
          <p>${(item.linkedTopics || []).join(", ") || "暂无"}</p>
        </div>
        <div class="info-box">
          <strong>手写时要说清楚</strong>
          <ul>${(item.interviewPrompts || []).map((prompt) => `<li>${prompt}</li>`).join("") || "<li>先讲输入输出、核心不变量和复杂度。</li>"}</ul>
        </div>
      </section>
    `;
  },
  focus(item) {
    const snippet = principleSnippet(item);
    return [
      codeNarration(item),
      `关联知识点：${(item.linkedTopics || []).join(", ") || "暂无"}`,
      snippet ? `代码片段：${snippet.symbols.join(", ")}` : "代码片段：索引内骨架"
    ];
  },
  prompts(item) {
    return (item.interviewPrompts || []).map((prompt) => ({ q: "代码追问", a: prompt }));
  }
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

