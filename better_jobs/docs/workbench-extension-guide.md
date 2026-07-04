# 工作台扩展约定

这个工作台的前端只负责通用导航、搜索、状态记录和渲染调度。知识正文、论文摘要、代码训练记录都应优先维护在 `content/**` 下，再由脚本或索引文件注册到页面。

## 新增一个主模块

1. 在 `content/<module>/index.js` 中挂载数据到 `window.BJ_CONTENT.<contentKey>`。
2. 在 `content/modules.js` 中新增一项：`id`、`title`、`description`、`contentKey`、`itemKey`、`renderer`。
3. 如果已有 renderer 能复用，直接填写对应 renderer；否则在 `src/renderers/<name>.js` 新增 renderer 并在 `index.html` 引入。

## Renderer 契约

每个 renderer 推荐提供以下方法：

```js
{
  title(item) {},
  subtitle(item) {},
  group(item) {},
  searchText(item) {},
  renderOverview(item) {},
  renderMain(item) {},
  renderDashboard(moduleContent) {},
  focus(item) {},
  prompts(item) {}
}
```

其中 `renderDashboard` 可选。主应用会使用 `group(item)` 控制左侧条目分组，使用 `searchText(item)` 做当前模块搜索。

## Markdown 内容扩展

知识库和基础知识正文应继续采用“一章一个 md”的方式维护：

- 知识库：`content/knowledge/chapters/*.md`
- 基础知识：`content/foundations/chapters/*.md`

修改后运行对应构建脚本：

```bash
node scripts/build-knowledge-docs.js
node scripts/build-foundation-docs.js
```

## 代码展示扩展

原理代码优先维护在 `content/principle-code/code/*.py`，再通过抽取脚本生成页面内联代码块：

```bash
node scripts/build-principle-code-snippets.js
```

代码训练题目与模板保持独立，避免与知识库正文混在一起。

## 上线前检查

```bash
node scripts/audit-workbench-content.js
node scripts/verify-structure.js
node --check src/app.js
```
## 内容展示偏好

这些偏好是工作台的长期设计约定，后续补强知识库时默认遵循：

- **知识正文仍然维护在 md 中**，前端只负责渲染；不要把长知识内容直接写死在 `index.html` 或 renderer 里。
- **关联资源、知识索引引用、参考资料、资料索引必须写成可点击链接卡片**。推荐写法是资源标题下的 Markdown 表格，至少包含“资源/知识点 + 链接 + 使用方式/为什么重要”。
- **原理代码不要只放本地文件路径**，需要通过 `content/principle-code/code/*.py` + `build-principle-code-snippets.js` 渲染成页面内联代码块，并使用中文注释。
- **章节补强不要追加在资料索引之后**。正文、面试 QA、原理代码说明在前，知识索引/参考资料/后续计划放在末尾。
- **不要一股脑加粗**。只加粗结论、风险、反直觉点和面试模板中的关键词，普通解释保持正常字重。

资源卡片推荐格式：

```md
## 关联资源

| 资源 | 链接 | 为什么重要 |
|---|---|---|
| VeRL HybridFlow | https://verl.readthedocs.io/en/latest/hybrid_flow.html | 用于解释 control flow / computation flow 解耦 |
```

