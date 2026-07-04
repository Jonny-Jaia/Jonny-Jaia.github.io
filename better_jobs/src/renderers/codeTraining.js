window.BJ_RENDERERS = window.BJ_RENDERERS || {};

window.BJ_RENDERERS.codeTraining = {
  title(item) {
    return item.title;
  },
  subtitle(item) {
    return `${item.group || "未分组"} / ${item.difficulty || "未标难度"} / ${item.status || "未标记"}`;
  },
  group(item) {
    return item.group || "未分组";
  },
  searchText(item) {
    return [
      item.title,
      item.group,
      item.difficulty,
      item.rewrittenPrompt,
      item.status,
      item.weakness,
      item.reviewPlan,
      item.codeRef,
      item.guideRef,
      window.BJ_TRAINING_GUIDES?.[item.guideRef] || "",
      window.BJ_TRAINING_GUIDES?.[item.reviewRef] || "",
      ...(item.tags || [])
    ].join(" ");
  },
  renderOverview(item) {
    return `
      <h3>${item.title}</h3>
      <p>${item.rewrittenPrompt || "待补充题目重述。"}</p>
      <div class="meta">
        <span class="pill">${item.group || "未分组"}</span>
        <span class="pill warning">${item.status || "未标记"}</span>
      </div>
    `;
  },
  renderDashboard(moduleContent) {
    const assessment = moduleContent.assessment;
    if (!assessment) return "";
    return `
      <section class="overview-card overview-wide dashboard-card">
        <h3>代码能力评估</h3>
        <p>${assessment.summary}</p>
        <div class="grid-two section-block">
          <div class="info-box">
            <strong>当前水平</strong>
            <p>${assessment.level}</p>
          </div>
          <div class="info-box">
            <strong>下一步重点</strong>
            <p>${(assessment.nextFocus || []).join(" / ")}</p>
          </div>
          <div class="info-box">
            <strong>优势</strong>
            <ul>${renderList(assessment.strengths, "继续沉淀优势。")}</ul>
          </div>
          <div class="info-box">
            <strong>薄弱点</strong>
            <ul>${renderList(assessment.weaknesses, "继续诊断薄弱点。")}</ul>
          </div>
        </div>
      </section>
    `;
  },
  renderMain(item) {
    return `
      <header class="article-header">
        <div>
          <h2>${item.title}</h2>
          <p>${item.rewrittenPrompt || "待补充题目重述。"}</p>
        </div>
        <div class="meta">
          <span class="pill">${item.difficulty || "未标难度"}</span>
          <span class="pill warning">${item.status || "未标记"}</span>
        </div>
      </header>
      <section class="section-block grid-two">
        <div class="info-box">
          <strong>算法标签</strong>
          <p>${(item.tags || []).join(" / ") || "暂无标签"}</p>
        </div>
        <div class="info-box">
          <strong>薄弱状态</strong>
          <p>${item.weakness || "待复盘。"}</p>
        </div>
      </section>
      <section class="section-block">
        <h3>训练计划</h3>
        <p>${item.reviewPlan || "待安排下一步训练。"}</p>
      </section>
      <section class="section-block">
        <h3>算法模板</h3>
        <div class="doc-card">${window.BJ_MARKDOWN.render(window.BJ_TRAINING_GUIDES?.[item.guideRef] || "暂无算法模板。")}</div>
      </section>
      ${
        item.reviewRef
          ? `<section class="section-block">
              <h3>训练复盘</h3>
              <div class="doc-card">${window.BJ_MARKDOWN.render(window.BJ_TRAINING_GUIDES?.[item.reviewRef] || "暂无复盘文档。")}</div>
            </section>`
          : ""
      }
      <section class="section-block">
        <h3>代码实现</h3>
        <p class="muted">来源：content/code-training/code/${item.codeRef}.py</p>
        <div class="code-box code-box-large"><pre><code class="language-python">${escapeHtml(window.BJ_CODE_SNIPPETS?.[item.codeRef] || "暂无代码文件。")}</code></pre></div>
      </section>
      <section class="section-block">
        <h3>题目链接</h3>
        <p>${item.source ? `<a href="${item.source}" target="_blank" rel="noreferrer">${item.source}</a>` : "暂无链接。"}</p>
      </section>
    `;
  },
  focus(item) {
    return [
      `题型：${item.group || "未分组"}`,
      `状态：${item.status || "未标记"}`,
      `薄弱点：${item.weakness || "待复盘"}`,
      item.reviewPlan || "先完成一轮可运行代码。"
    ];
  },
  prompts() {
    return [
      { q: "训练追问", a: "你能先说出这道题对应模板的不变量吗？" },
      { q: "复盘追问", a: "上次卡住的是边界、状态定义、复杂度，还是代码实现？" }
    ];
  }
};

function renderList(values, fallback) {
  const items = values && values.length ? values : [fallback];
  return items.map((item) => `<li>${item}</li>`).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
