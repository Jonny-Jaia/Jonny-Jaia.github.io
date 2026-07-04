window.BJ_RENDERERS = window.BJ_RENDERERS || {};

window.BJ_RENDERERS.paperLibrary = {
  title(item) {
    return item.title;
  },
  subtitle(item) {
    return `${item.group || "未分组"} / ${item.status || "未标记"}`;
  },
  group(item) {
    return item.group || "未分组";
  },
  searchText(item) {
    return [
      item.title,
      item.group,
      item.status,
      item.source,
      item.problem,
      item.method,
      ...(item.linkedTopics || []),
      ...(item.contributions || []),
      ...(item.limitations || []),
      ...(item.interviewQuestions || [])
    ].join(" ");
  },
  renderOverview(item) {
    return `
      <h3>${item.title}</h3>
      <p>${item.problem || "待补充论文问题定位。"}</p>
      <div class="meta">
        <span class="pill">${item.group || "未分组"}</span>
        <span class="pill warning">${item.status || "未标记"}</span>
      </div>
    `;
  },
  renderMain(item) {
    return `
      <header class="article-header">
        <div>
          <h2>${item.title}</h2>
          <p>${item.problem || "待补充论文问题定位。"}</p>
        </div>
        <div class="meta">
          <span class="pill warning">${item.status || "未标记"}</span>
          <span class="pill">${item.group || "未分组"}</span>
        </div>
      </header>
      <section class="section-block">
        <h3>来源索引</h3>
        ${renderSource(item.source)}
      </section>
      <section class="section-block">
        <h3>方法摘要</h3>
        <p>${item.method || "待精读后补充。"}</p>
      </section>
      <section class="section-block grid-two">
        <div class="info-box">
          <strong>主要贡献</strong>
          <ul>${renderList(item.contributions, "待补充贡献点。")}</ul>
        </div>
        <div class="info-box">
          <strong>局限与风险</strong>
          <ul>${renderList(item.limitations, "待补充局限分析。")}</ul>
        </div>
      </section>
      <section class="section-block">
        <h3>关联专题</h3>
        <p>${(item.linkedTopics || []).join(" / ") || "暂无关联专题。"}</p>
      </section>
    `;
  },
  focus(item) {
    return [
      item.problem || "先确认论文解决的问题。",
      `关联专题：${(item.linkedTopics || []).join(" / ") || "暂无"}`,
      `阅读状态：${item.status || "未标记"}`
    ];
  },
  prompts(item) {
    return (item.interviewQuestions || []).map((question) => ({ q: "论文转面试题", a: question }));
  }
};

function renderSource(source) {
  if (!source) return "<p>待补充来源链接或本地论文路径。</p>";
  if (/^https?:\/\//.test(source)) {
    return `<p><a href="${source}" target="_blank" rel="noreferrer">${source}</a></p>`;
  }
  return `<p>${source}</p>`;
}

function renderList(values, fallback) {
  const items = values && values.length ? values : [fallback];
  return items.map((point) => `<li>${point}</li>`).join("");
}
