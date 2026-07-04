window.BJ_RENDERERS = window.BJ_RENDERERS || {};

function knowledgeDocFor(item) {
  const docs = item.docCollection === "foundations" ? window.BJ_FOUNDATION_DOCS : window.BJ_MARKDOWN_DOCS;
  return docs?.[item.docId] || "";
}

function readableTitle(item) {
  const doc = knowledgeDocFor(item);
  const match = doc.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : item.title;
}

function readableGroup(item) {
  const groups = {
    "agent-guide-interview-qa": "面试专项",
    "interview-question-bank": "面试专项",
    "llm-engineering-interview-supplement": "工程基础",
    grpo: "后训练核心",
    sft: "后训练核心",
    dpo: "后训练核心",
    distillation: "后训练核心",
    opd: "后训练核心",
    "multi-step-rl": "后训练核心",
    "optimization-training": "训练优化",
    "sampling-evaluation-rft": "工程闭环",
    mtp: "模型训练目标",
    "moe-architecture": "模型架构",
    "positional-encoding": "模型架构",
    agent: "Agent 前沿",
    "agent-interview-practice": "Agent 前沿",
    "training-inference-frameworks": "工程框架",
    "verl-rl-framework": "工程框架",
    "chinese-model-series": "模型家族",
    "llm-engineering-foundations": "大模型工程基础",
    "computer-foundations": "计算机基础",
    "ml-foundations": "机器学习基础",
    "dl-foundations": "深度学习基础",
    "rl-foundations": "强化学习基础",
    "cv-foundation-models": "视觉基础模型"
  };
  return groups[item.id] || item.group || "未分组";
}

function docIntro(item) {
  const doc = knowledgeDocFor(item);
  const line = doc
    .split(/\r?\n/)
    .map((value) => value.trim())
    .find((value) => value && !value.startsWith("#") && !value.startsWith("|") && !value.startsWith(">") && !value.startsWith("```"));
  return line || item.summary || "暂无摘要，优先阅读章节正文。";
}

function knowledgeDocStats(item) {
  const doc = knowledgeDocFor(item);
  const headings = (doc.match(/^##\s+/gm) || []).length;
  const tables = (doc.match(/^\|.+\|$/gm) || []).length;
  const formulas = (doc.match(/\$\$/g) || []).length / 2;
  return {
    chars: doc.length,
    headings,
    tables,
    formulas,
    hasSourceIndex: doc.includes("知识索引引用") || doc.includes("参考资料") || doc.includes("参考论文") || doc.includes("参考资料与检索入口"),
    hasInterviewQA: doc.includes("面试 QA") || doc.includes("面试八股") || doc.includes("面试回答模板") || doc.includes("高频问答")
  };
}

function knowledgeDocDensity(stats) {
  if (stats.chars > 16000 || stats.headings > 12 || stats.tables > 18) {
    return { label: "深读", className: "dense", hint: "长文档：优先看目录、表格和 QA，再回看公式细节。" };
  }
  if (stats.chars > 8000 || stats.headings > 7) {
    return { label: "标准", className: "standard", hint: "中等密度：适合按章节顺序完整复习。" };
  }
  return { label: "速览", className: "light", hint: "短文档：适合快速建立知识定位。" };
}

function learningBrief(item, stats) {
  const questions = (item.coreQuestions || []).slice(0, 3);
  const tags = (item.tags || []).slice(0, 5);
  const codeRefs = (item.principleCodeRefs || []).slice(0, 4);
  const paperRefs = (item.paperRefs || []).slice(0, 4);
  return `
    <section class="learning-brief" aria-label="学习入口">
      <div class="brief-main">
        <span class="brief-kicker">Learning Entry</span>
        <strong>${stats.hasInterviewQA ? "先看 QA，再回到公式和代码。" : "先建立概念骨架，再补面试问答。"}</strong>
        <p>${questions[0] || docIntro(item)}</p>
      </div>
      <div class="brief-grid">
        <div><span>关键词</span><p>${tags.join(" / ") || "待补充"}</p></div>
        <div><span>核心问题</span><p>${questions.join(" / ") || "待补充"}</p></div>
        <div><span>代码入口</span><p>${codeRefs.join(" / ") || "暂无"}</p></div>
        <div><span>论文索引</span><p>${paperRefs.join(" / ") || "暂无"}</p></div>
      </div>
    </section>
  `;
}

function contentAdvice(item, stats) {
  const advice = [];
  if (stats.chars < 6000) advice.push("内容仍偏提纲型，建议后续补充公式推导、边界条件、面试追问和最小代码实现。");
  if (!stats.hasSourceIndex) advice.push("缺少来源索引，建议补充论文、官方文档或可靠博客链接，避免结论无出处。");
  if (!stats.hasInterviewQA) advice.push("缺少面试问答区，建议至少补 3 个高频问题：直觉、公式/流程、工程落地。");
  if ((item.principleCodeRefs || []).length === 0 && /GRPO|DPO|SFT|MoE|RoPE|Agent|RAG|MTP|优化器|训练|推理/.test(readableTitle(item))) {
    advice.push("可以考虑关联原理代码区，补一个能手写的最小实现或伪代码片段。");
  }
  if (readableGroup(item).includes("基础") && stats.chars < 5000) {
    advice.push("基础知识章节建议继续从定义、核心公式、常见误区、手撕代码四层展开。");
  }
  if (!advice.length) advice.push("当前结构完整。后续扩展时优先补充新论文证据、真实面试追问和代码化理解。 ");
  return advice;
}

window.BJ_RENDERERS.knowledge = {
  title(item) {
    return readableTitle(item);
  },
  subtitle(item) {
    return `${readableGroup(item)} / ${item.status || "未标记"}`;
  },
  group(item) {
    return readableGroup(item);
  },
  searchText(item) {
    return [
      readableTitle(item),
      item.title,
      readableGroup(item),
      item.group,
      item.summary,
      item.status,
      knowledgeDocFor(item),
      ...(item.tags || []),
      ...(item.coreQuestions || []),
      ...(item.followUps || []),
      ...(item.weaknesses || [])
    ].join(" ");
  },
  renderOverview(item) {
    const stats = knowledgeDocStats(item);
    const density = knowledgeDocDensity(stats);
    return `
      <h3>${readableTitle(item)}</h3>
      <p>${docIntro(item)}</p>
      <div class="meta">
        ${(item.tags || []).slice(0, 4).map((tag) => `<span class="pill">${tag}</span>`).join("")}
        <span class="pill subtle">${density.label}</span>
      </div>
    `;
  },
  renderMain(item) {
    const stats = knowledgeDocStats(item);
    const density = knowledgeDocDensity(stats);
    const advice = contentAdvice(item, stats);
    return `
      <header class="article-header">
        <div>
          <h2>${readableTitle(item)}</h2>
          <p>${docIntro(item)}</p>
        </div>
        <div class="meta">
          <span class="pill warning">${item.status || "未标记"}</span>
          <span class="pill">${readableGroup(item)}</span>
        </div>
      </header>
      <section class="doc-status-bar" aria-label="章节状态">
        <span class="density-pill ${density.className}"><strong>${density.label}</strong> ${density.hint}</span>
        <span><strong>${Math.round(stats.chars / 100) / 10}k</strong> 字符</span>
        <span><strong>${stats.headings}</strong> 小节</span>
        <span><strong>${stats.tables}</strong> 表格行</span>
        <span><strong>${Math.round(stats.formulas)}</strong> 公式块</span>
        <span><strong>${(item.coreQuestions || []).length}</strong> 核心问题</span>
        <span><strong>${(item.paperRefs || []).length}</strong> 论文索引</span>
        <span class="${stats.hasSourceIndex ? "ok" : "warn"}">${stats.hasSourceIndex ? "已标注来源" : "待补来源"}</span>
        <span class="${stats.hasInterviewQA ? "ok" : "warn"}">${stats.hasInterviewQA ? "含面试问答" : "待补 QA"}</span>
      </section>
      ${learningBrief(item, stats)}
      <section class="section-block grid-two">
        <div class="info-box">
          <strong>内容体检建议</strong>
          <ul>${advice.map((value) => `<li>${value}</li>`).join("")}</ul>
        </div>
        <div class="info-box">
          <strong>关联维护动作</strong>
          <ul>
            <li>格式问题：运行 <code>node scripts\\audit-workbench-content.js</code>。</li>
            <li>知识正文变更后：运行 <code>node scripts\\build-knowledge-docs.js</code> 或 foundations 构建脚本。</li>
            <li>原理代码变更后：运行 <code>node scripts\\build-principle-code-snippets.js</code>。</li>
          </ul>
        </div>
      </section>
      <section class="section-block">
        <h3>章节文档</h3>
        <div class="doc-card density-${density.className}">${window.BJ_MARKDOWN.render(knowledgeDocFor(item) || "暂无 Markdown 文档。")}</div>
      </section>
      <section class="section-block">
        <h3>核心问题</h3>
        <ul>${(item.coreQuestions || []).map((question) => `<li>${question}</li>`).join("") || "<li>待通过章节共创补充。</li>"}</ul>
      </section>
      <section class="section-block grid-two">
        <div class="info-box">
          <strong>我的回答记录</strong>
          <p>${(item.answerRecords || []).length ? item.answerRecords.join("<br>") : "还没有沉淀回答。先在右侧记录你的回答，后续再追问并沉淀。"}</p>
        </div>
        <div class="info-box">
          <strong>薄弱点</strong>
          <p>${(item.weaknesses || []).length ? item.weaknesses.join("<br>") : "待通过问答诊断。"}</p>
        </div>
      </section>
      <section class="section-block">
        <h3>最终沉淀</h3>
        <p>${(item.finalNotes || []).length ? item.finalNotes.join("<br>") : "待交互完成后沉淀为可复习笔记。"}</p>
      </section>
      <section class="section-block">
        <h3>关联资源</h3>
        <p>原理代码：${(item.principleCodeRefs || []).join(", ") || "暂无"}；论文卡片：${(item.paperRefs || []).join(", ") || "暂无"}。</p>
      </section>
    `;
  },
  focus(item) {
    return (item.coreQuestions || []).slice(0, 4);
  },
  prompts(item) {
    return (item.followUps || []).map((prompt) => ({ q: "追问", a: prompt }));
  }
};


