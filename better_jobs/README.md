# LLM 面试准备工作台

这是一个本地静态面试准备工作台，用来通过交互问答逐步构建大模型算法工程师知识库，并独立维护论文、原理代码和手撕代码训练。

## 使用方式

直接用浏览器打开 `index.html`。

如果修改了 `content/knowledge/chapters/*.md`，运行：

```bash
node scripts/build-knowledge-docs.js
```

如果修改了 `content/code-training/code/**/*.py`，运行：

```bash
node scripts/build-code-training-code.js
```

然后刷新页面。

## 文件说明

- `index.html`：通用页面壳。
- `assets/styles.css`：视觉样式和响应式布局。
- `src/app.js`：模块切换、搜索、记录保存。
- `content/knowledge/chapters/*.md`：知识章节源文档。
- `content/knowledge/generated-docs.js`：供静态页面直接渲染的 Markdown 文档镜像。
- `content/paper-library/index.js`：论文库和阅读队列。
- `content/principle-code/index.js`：原理代码模块。
- `content/code-training/index.js`：手撕代码训练题单和状态。
- `content/code-training/code/**/*.py`：代码训练的独立代码文件。
- `content/code-training/generated-code.js`：供静态页面渲染的代码文件镜像。

## 维护原则

1. 知识库通过互动沉淀，每个章节维护一个 Markdown。
2. 论文库维护原始来源和精读卡片，再把结论沉淀到知识库。
3. 原理代码只服务大模型训练理解。
4. 代码训练独立服务 LeetCode / 手撕代码题，不和知识库混在一起。
