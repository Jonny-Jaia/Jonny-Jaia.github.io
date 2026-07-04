const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.resolve(__dirname, "..");
const markdownDirs = [
  path.join(root, "content", "knowledge", "chapters"),
  path.join(root, "content", "foundations", "chapters"),
  path.join(root, "content", "code-training", "guides"),
  path.join(root, "content", "code-training", "reviews")
];

const jsFiles = [
  "index.html",
  "content/modules.js",
  "src/markdown.js",
  "src/app.js",
  "src/renderers/knowledge.js",
  "src/renderers/codeSkill.js",
  "src/renderers/codeTraining.js",
  "src/renderers/paperLibrary.js",
  "src/renderers/learningSessions.js"
].map((file) => path.join(root, file));

const interviewSectionPatterns = [/\u9762\u8bd5 QA/, /\u9ad8\u9891\u95ee\u7b54/, /\u9762\u8bd5\u56de\u7b54\u6a21\u677f/, /\u9762\u8bd5\u516b\u80a1/];
const sourceSectionPatterns = [/\u77e5\u8bc6\u7d22\u5f15\u5f15\u7528/, /\u53c2\u8003\u8d44\u6599/, /\u53c2\u8003\u8bba\u6587/, /\u53c2\u8003\u8d44\u6599\u4e0e\u68c0\u7d22\u5165\u53e3/, /\u5173\u8054\u8d44\u6e90/, /\u8d44\u6599\u7d22\u5f15/, /\u8d44\u6e90\u7d22\u5f15/];
const mojibakePattern = /[鐭绯闈㈠閲鍚钖]|�/;

function walkMarkdownFiles() {
  const files = [];
  for (const dir of markdownDirs) {
    if (!fs.existsSync(dir)) continue;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.endsWith(".md")) {
        files.push(path.join(dir, entry.name));
      }
    }
  }
  return files;
}


function hasResourceUrlSection(text) {
  const lines = text.split(/\r?\n/);
  let inResourceSection = false;
  for (const line of lines) {
    if (/^#{2,4}\s+/.test(line)) {
      inResourceSection = sourceSectionPatterns.some((pattern) => pattern.test(line));
      continue;
    }
    if (inResourceSection && /https?:\/\//.test(line)) return true;
  }
  return false;
}
function parseFenceIssues(file, text) {
  const lines = text.split(/\r?\n/);
  const issues = [];
  let open = null;
  let content = [];

  lines.forEach((line, index) => {
    if (!line.startsWith("```")) {
      if (open) content.push(line);
      return;
    }
    if (!open) {
      open = { lang: line.slice(3).trim().toLowerCase(), line: index + 1 };
      content = [];
      return;
    }

    if (open.lang === "flow") {
      const source = content.join("\n").trim();
      if (!source.includes("->")) {
        issues.push({
          level: "error",
          file,
          line: open.line,
          message: "flow block must contain at least one `->` edge"
        });
      }
      if (source.split(/\s*->\s*/).filter(Boolean).length < 2) {
        issues.push({
          level: "error",
          file,
          line: open.line,
          message: "flow block must contain at least two nodes"
        });
      }
    }

    open = null;
    content = [];
  });

  if (open) {
    issues.push({
      level: "error",
      file,
      line: open.line,
      message: `unclosed code fence: ${open.lang || "plain"}`
    });
  }

  return issues;
}

function auditMarkdown(file) {
  const text = fs.readFileSync(file, "utf8");
  const rel = path.relative(root, file).replaceAll("\\", "/");
  const issues = [];
  const isKnowledge = rel.startsWith("content/knowledge/chapters/");
  const isFoundation = rel.startsWith("content/foundations/chapters/");

  if (!/^#\s+.+/m.test(text)) {
    issues.push({ level: "error", file: rel, line: 1, message: "missing H1 title" });
  }
  if (mojibakePattern.test(text)) {
    issues.push({ level: "warn", file: rel, line: 1, message: "possible mojibake characters detected" });
  }
  if ((isKnowledge || isFoundation) && !interviewSectionPatterns.some((pattern) => pattern.test(text))) {
    issues.push({ level: "warn", file: rel, line: 1, message: "missing interview QA section" });
  }
  if (isKnowledge && !sourceSectionPatterns.some((pattern) => pattern.test(text))) {
    issues.push({ level: "warn", file: rel, line: 1, message: "missing source/provenance section" });
  }

  issues.push(...parseFenceIssues(rel, text));
  return issues;
}

function loadMarkdownRenderer() {
  const context = { window: {}, console };
  vm.createContext(context);
  context.window.katex = {
    renderToString(value, options) {
      return `${options.displayMode ? "DISPLAY" : "INLINE"}:${value.trim()}`;
    }
  };
  vm.runInContext(fs.readFileSync(path.join(root, "src", "markdown.js"), "utf8"), context);
  return context.window.BJ_MARKDOWN;
}

function auditRenderedMarkdown(file, renderer) {
  const text = fs.readFileSync(file, "utf8");
  const html = renderer.render(text);
  const rel = path.relative(root, file).replaceAll("\\", "/");
  const issues = [];
  if (html.includes("```flow") || html.includes("```")) {
    issues.push({ level: "error", file: rel, line: 1, message: "rendered html leaks markdown fences" });
  }
  if (text.includes("```flow") && !html.includes("flow-diagram")) {
    issues.push({ level: "error", file: rel, line: 1, message: "flow block did not render as diagram" });
  }
  if (hasResourceUrlSection(text) && !html.includes("resource-card")) {
    issues.push({ level: "error", file: rel, line: 1, message: "resource/source URL section should render as link cards" });
  }
  return issues;
}

function auditJs(file) {
  const text = fs.readFileSync(file, "utf8");
  const rel = path.relative(root, file).replaceAll("\\", "/");
  const issues = [];
  if (mojibakePattern.test(text)) {
    issues.push({ level: "warn", file: rel, line: 1, message: "possible mojibake characters detected" });
  }
  return issues;
}

const markdownFiles = walkMarkdownFiles();
const renderer = loadMarkdownRenderer();
const issues = [
  ...markdownFiles.flatMap(auditMarkdown),
  ...markdownFiles.flatMap((file) => auditRenderedMarkdown(file, renderer)),
  ...jsFiles.flatMap(auditJs)
];

const errors = issues.filter((issue) => issue.level === "error");
const warnings = issues.filter((issue) => issue.level === "warn");

for (const issue of issues) {
  console.log(`${issue.level.toUpperCase()} ${issue.file}:${issue.line} ${issue.message}`);
}

console.log(`Workbench content audit: ${errors.length} errors, ${warnings.length} warnings.`);
if (errors.length) process.exit(1);

