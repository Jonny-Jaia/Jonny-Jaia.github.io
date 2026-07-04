const fallbackRenderer = {
  title(item) {
    return item.title || item.id || "未命名条目";
  },
  subtitle(item) {
    return item.group || "未分组";
  },
  group(item) {
    return item.group || "未分组";
  },
  searchText(item) {
    return JSON.stringify(item);
  },
  renderOverview(item) {
    return `<h3>${this.title(item)}</h3><p>${this.subtitle(item)}</p>`;
  },
  renderMain(item) {
    return `<pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre>`;
  },
  focus() {
    return [];
  },
  prompts() {
    return [];
  }
};

(function () {
  const modules = window.BJ_MODULES || [];
  const content = window.BJ_CONTENT || {};
  const renderers = window.BJ_RENDERERS || {};

  const moduleNav = document.querySelector("#moduleNav");
  const itemNav = document.querySelector("#itemNav");
  const overview = document.querySelector("#overview");
  const article = document.querySelector("#mainArticle");
  const focusList = document.querySelector("#focusList");
  const promptList = document.querySelector("#promptList");
  const noteInput = document.querySelector("#noteInput");
  const noteStatus = document.querySelector("#noteStatus");
  const searchInput = document.querySelector("#searchInput");
  const readingProgress = document.querySelector("#readingProgress span");
  const moduleEyebrow = document.querySelector("#moduleEyebrow");
  const moduleTitle = document.querySelector("#moduleTitle");
  const moduleDescription = document.querySelector("#moduleDescription");
  const moduleStats = document.querySelector("#moduleStats");

  let activeModuleId = modules[0]?.id;
  let activeItemId = getItems(getActiveModule())[0]?.id;
  const activeReadingMode = "study";

  function getActiveModule() {
    return modules.find((module) => module.id === activeModuleId) || modules[0] || {};
  }

  function getModuleContent(module) {
    return content[module.contentKey] || {};
  }

  function getItems(module) {
    const moduleContent = getModuleContent(module);
    return moduleContent[module.itemKey] || [];
  }

  function getRenderer(module) {
    return renderers[module.renderer] || fallbackRenderer;
  }

  function itemGroup(renderer, item) {
    return renderer.group ? renderer.group(item) : item.group || "未分组";
  }

  function groupItems(items, renderer) {
    return items.reduce((groups, item) => {
      const group = itemGroup(renderer, item);
      if (!groups[group]) groups[group] = [];
      groups[group].push(item);
      return groups;
    }, {});
  }

  function filteredItems() {
    const module = getActiveModule();
    const renderer = getRenderer(module);
    const keyword = searchInput.value.trim().toLowerCase();
    const items = getItems(module);
    if (!keyword) return items;
    return items.filter((item) => renderer.searchText(item).toLowerCase().includes(keyword));
  }

  function renderModuleNav() {
    moduleNav.innerHTML = modules
      .map(
        (module) => `
          <button class="module-button${module.id === activeModuleId ? " active" : ""}" type="button" data-module="${module.id}">
            <strong>${module.title}</strong>
            <span>${module.description}</span>
          </button>
        `
      )
      .join("");

    moduleNav.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        activeModuleId = button.dataset.module;
        searchInput.value = "";
        activeItemId = getItems(getActiveModule())[0]?.id;
        render();
      });
    });
  }

  function renderItemNav(items) {
    const module = getActiveModule();
    const renderer = getRenderer(module);
    const groups = groupItems(items, renderer);
    itemNav.innerHTML = "";

    Object.entries(groups).forEach(([group, groupItems]) => {
      const groupTitle = document.createElement("div");
      groupTitle.className = "nav-title";
      groupTitle.textContent = group;
      itemNav.appendChild(groupTitle);

      groupItems.forEach((item) => {
        const button = document.createElement("button");
        button.className = `item-button${item.id === activeItemId ? " active" : ""}`;
        button.type = "button";
        button.dataset.item = item.id;
        button.innerHTML = `<strong>${renderer.title(item)}</strong><span>${renderer.subtitle(item)}</span>`;
        button.addEventListener("click", () => {
          activeItemId = item.id;
          render();
        });
        itemNav.appendChild(button);
      });
    });
  }

  function renderOverview(items) {
    const module = getActiveModule();
    const moduleContent = getModuleContent(module);
    const renderer = getRenderer(module);
    const dashboard = renderer.renderDashboard ? renderer.renderDashboard(moduleContent) : "";
    if (!items.length) {
      overview.innerHTML = `${dashboard}<div class="empty-state">没有匹配条目。换一个关键词试试。</div>`;
      return;
    }
    overview.innerHTML =
      dashboard +
      items
        .slice(0, 6)
        .map(
          (item) => `
            <button class="overview-card" type="button" data-item="${item.id}">
              ${renderer.renderOverview(item)}
            </button>
          `
        )
        .join("");
    overview.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        activeItemId = button.dataset.item;
        render();
      });
    });
  }

  function updateReadingProgress() {
    if (!readingProgress) return;
    const doc = document.documentElement;
    const scrollTop = doc.scrollTop || document.body.scrollTop;
    const maxScroll = Math.max(1, doc.scrollHeight - doc.clientHeight);
    const progress = Math.min(1, Math.max(0, scrollTop / maxScroll));
    readingProgress.style.transform = `scaleX(${progress})`;
  }

  function updateActiveToc() {
    const links = Array.from(article.querySelectorAll(".doc-toc a"));
    if (!links.length) return;
    const headings = links
      .map((link) => ({ link, target: document.getElementById(link.getAttribute("href").slice(1)) }))
      .filter((item) => item.target);
    let active = headings[0];
    for (const item of headings) {
      if (item.target.getBoundingClientRect().top <= 120) active = item;
      else break;
    }
    links.forEach((link) => link.classList.toggle("active", active && link === active.link));
  }

  function updateReadingAids() {
    updateReadingProgress();
    updateActiveToc();
  }

  function renderHero(items) {
    const module = getActiveModule();
    const moduleContent = getModuleContent(module);
    moduleEyebrow.textContent = moduleContent.eyebrow || "Interview Workspace";
    moduleTitle.textContent = module.title || moduleContent.title || "LLM 面试准备工作台";
    moduleDescription.textContent = module.description || moduleContent.description || "持续沉淀知识、论文、代码与学习记录。";

    moduleStats.innerHTML = [
      { label: "全部条目", value: getItems(module).length },
      { label: "当前筛选", value: items.length },
      { label: "主模块", value: modules.length },
      { label: "记录方式", value: "本地" }
    ]
      .map((stat) => `<div class="stat"><strong>${stat.value}</strong><span>${stat.label}</span></div>`)
      .join("");
  }

  function renderArticle(item) {
    const renderer = getRenderer(getActiveModule());
    article.innerHTML = renderer.renderMain(item);
  }

  function renderInsight(item) {
    const renderer = getRenderer(getActiveModule());
    const focus = renderer.focus(item).filter(Boolean);
    const prompts = renderer.prompts(item).filter(Boolean);
    focusList.innerHTML = focus.length ? focus.map((value) => `<li>${value}</li>`).join("") : "<li>暂无重点，先从正文目录开始。</li>";
    promptList.innerHTML = prompts.length
      ? prompts
          .map(
            (prompt) => `
              <div class="qa-item">
                <strong>${prompt.q}</strong>
                <p>${prompt.a}</p>
              </div>
            `
          )
          .join("")
      : '<div class="empty-state">暂无追问。完成一轮回答后再沉淀。</div>';
  }

  function noteKey() {
    return `better-jobs-note:${activeModuleId}:${activeItemId}`;
  }

  function loadNote() {
    noteInput.value = localStorage.getItem(noteKey()) || "";
    noteStatus.textContent = "本地自动保存";
  }

  function render() {
    if (!modules.length) return;
    const items = filteredItems();
    const activeItem = items.find((item) => item.id === activeItemId) || items[0] || getItems(getActiveModule())[0];
    activeItemId = activeItem?.id;
    document.body.dataset.module = activeModuleId;
    document.body.dataset.readingMode = activeReadingMode;

    renderModuleNav();
    renderItemNav(items);
    renderOverview(items);
    renderHero(items);

    if (!activeItem) {
      article.innerHTML = '<div class="empty-state">当前模块还没有条目。</div>';
      focusList.innerHTML = "";
      promptList.innerHTML = "";
      noteInput.value = "";
      return;
    }

    renderArticle(activeItem);
    renderInsight(activeItem);
    loadNote();
    requestAnimationFrame(updateReadingAids);
  }

  article.addEventListener("click", (event) => {
    const link = event.target.closest("[data-module-link][data-item-link]");
    if (!link) return;
    const targetModule = modules.find((module) => module.id === link.dataset.moduleLink);
    if (!targetModule) return;
    const targetItem = getItems(targetModule).find((item) => item.id === link.dataset.itemLink);
    if (!targetItem) return;
    event.preventDefault();
    activeModuleId = targetModule.id;
    activeItemId = targetItem.id;
    searchInput.value = "";
    render();
  });

  searchInput.addEventListener("input", () => {
    const items = filteredItems();
    if (!items.some((item) => item.id === activeItemId)) {
      activeItemId = items[0]?.id;
    }
    render();
  });

  window.addEventListener("scroll", updateReadingAids, { passive: true });
  window.addEventListener("resize", updateReadingAids);

  noteInput.addEventListener("input", () => {
    localStorage.setItem(noteKey(), noteInput.value);
    noteStatus.textContent = "已保存到本地浏览器";
  });

  render();
})();

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}




