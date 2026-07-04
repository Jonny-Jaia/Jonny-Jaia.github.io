window.BJ_RENDERERS = window.BJ_RENDERERS || {};

window.BJ_RENDERERS.learningSessions = {
  title(item) {
    return item.title;
  },
  subtitle(item) {
    return `${item.mode || "未标模式"} / ${item.status || "未标记"}`;
  },
  group(item) {
    return item.group || item.mode || "未分组";
  },
  searchText(item) {
    return [
      item.title,
      item.group,
      item.mode,
      item.status,
      item.goal,
      item.summary,
      ...(item.steps || []),
      ...(item.output || []),
      ...(item.strengths || []),
      ...(item.weaknesses || []),
      ...(item.nextQuestions || []),
      ...(item.nextActions || []),
      ...(item.topicRefs || []),
      ...(item.paperRefs || [])
    ].join(" ");
  },
  renderOverview(item) {
    return `
      <h3>${item.title}</h3>
      <p>${item.goal || item.summary || "待补充学习目标。"}</p>
      <div class="meta">
        <span class="pill">${item.mode || "未标模式"}</span>
        <span class="pill warning">${item.status || "未标记"}</span>
      </div>
    `;
  },
  renderDashboard(moduleContent) {
    const dashboard = moduleContent.dashboard;
    if (!dashboard) return "";
    return `
      <section class="overview-card overview-wide dashboard-card">
        <h3>学习闭环</h3>
        <p>${dashboard.summary}</p>
        <div class="grid-two section-block">
          <div class="info-box">
            <strong>默认节奏</strong>
            <p>${dashboard.defaultCadence}</p>
          </div>
          <div class="info-box">
            <strong>当前重点</strong>
            <p>${(dashboard.currentFocus || []).join(" / ")}</p>
          </div>
        </div>
        <div class="meta">${(dashboard.activeLoop || []).map((item) => `<span class="pill">${item}</span>`).join("")}</div>
      </section>
    `;
  },
  renderMain(item) {
    const isProtocol = Boolean(item.goal);
    return `
      <header class="article-header">
        <div>
          <h2>${item.title}</h2>
          <p>${item.goal || item.summary || "待补充学习目标。"}</p>
        </div>
        <div class="meta">
          <span class="pill">${item.mode || "未标模式"}</span>
          <span class="pill warning">${item.status || "未标记"}</span>
        </div>
      </header>
      ${
        isProtocol
          ? `<section class="section-block">
              <h3>交互步骤</h3>
              <ol>${renderList(item.steps, "待补充步骤。")}</ol>
            </section>
            <section class="section-block">
              <h3>输出物</h3>
              <div class="meta">${(item.output || []).map((output) => `<span class="pill">${output}</span>`).join("") || '<span class="pill subtle">待补充</span>'}</div>
            </section>`
          : `<section class="section-block grid-two">
              <div class="info-box">
                <strong>关联专题</strong>
                <p>${(item.topicRefs || []).join(" / ") || "暂无"}</p>
              </div>
              <div class="info-box">
                <strong>关联论文</strong>
                <p>${(item.paperRefs || []).join(" / ") || "暂无"}</p>
              </div>
            </section>
            <section class="section-block grid-two">
              <div class="info-box">
                <strong>优势</strong>
                <ul>${renderList(item.strengths, "待补充优势。")}</ul>
              </div>
              <div class="info-box">
                <strong>薄弱点</strong>
                <ul>${renderList(item.weaknesses, "待补充薄弱点。")}</ul>
              </div>
            </section>`
      }
      <section class="section-block">
        <h3>下一轮追问</h3>
        <ul>${renderList(item.nextQuestions, "待生成下一轮追问。")}</ul>
      </section>
      ${
        item.nextActions
          ? `<section class="section-block">
              <h3>下一步动作</h3>
              <ul>${renderList(item.nextActions, "待安排动作。")}</ul>
            </section>`
          : ""
      }
    `;
  },
  focus(item) {
    return [
      `模式：${item.mode || "未标模式"}`,
      `状态：${item.status || "未标记"}`,
      ...(item.weaknesses || []).slice(0, 2),
      ...(item.nextActions || []).slice(0, 1)
    ];
  },
  prompts(item) {
    return (item.nextQuestions || []).map((question) => ({ q: "下一轮追问", a: question }));
  }
};

function renderList(values, fallback) {
  const items = values && values.length ? values : [fallback];
  return items.map((value) => `<li>${value}</li>`).join("");
}
