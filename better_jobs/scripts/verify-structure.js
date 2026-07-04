const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const root = path.resolve(__dirname, "..");
const requiredFiles = [
  "index.html",
  "assets/styles.css",
  "src/markdown.js",
  "src/app.js",
  "content/modules.js",
  "content/knowledge/index.js",
  "content/knowledge/generated-docs.js",
  "content/foundations/index.js",
  "content/foundations/generated-docs.js",
  "content/paper-library/index.js",
  "content/principle-code/index.js",
  "content/principle-code/generated-code.js",
  "content/code-training/index.js",
  "content/code-training/generated-code.js",
  "content/code-training/generated-guides.js",
  "assets/diagrams/manifest.js"
];

const knowledgeDocs = [
  "llm-engineering-interview-supplement",
  "activation-functions",
  "optimization-training",
  "sft",
  "dpo",
  "grpo",
  "distillation",
  "opd",
  "multi-step-rl",
  "agentic-rl-system",
  "sampling-evaluation-rft",
  "mtp",
  "moe-architecture",
  "positional-encoding",
  "training-inference-frameworks",
  "verl-rl-framework",
  "agent",
  "agent-rag-systems",
  "agent-memory-tooling",
  "agent-planning-multi-agent",
  "agent-interview-practice",
  "multimodal-vlm",
  "vla-embodied-driving",
  "embodied-ai-robotics-vla",
  "chinese-model-series",
  "interview-question-bank",
  "agent-guide-interview-qa"
];

for (const id of knowledgeDocs) requiredFiles.push(`content/knowledge/chapters/${id}.md`);
for (const file of requiredFiles) {
  assert.ok(fs.existsSync(path.join(root, file)), `Missing required file: ${file}`);
}

const context = { window: {}, console };
vm.createContext(context);
context.window.katex = {
  renderToString(value, options) {
    return `${options.displayMode ? "DISPLAY" : "INLINE"}:${value.trim()}`;
  }
};

function runFile(file) {
  vm.runInContext(fs.readFileSync(path.join(root, file), "utf8"), context, { filename: file });
}

for (const file of [
  "src/markdown.js",
  "content/knowledge/generated-docs.js",
  "content/knowledge/index.js",
  "content/foundations/generated-docs.js",
  "content/foundations/index.js",
  "content/paper-library/index.js",
  "content/principle-code/generated-code.js",
  "content/principle-code/index.js",
  "content/code-training/generated-code.js",
  "content/code-training/generated-guides.js",
  "content/code-training/index.js",
  "content/learning-sessions/index.js",
  "content/modules.js",
  "assets/diagrams/manifest.js"
]) {
  if (fs.existsSync(path.join(root, file))) runFile(file);
}

assert.deepStrictEqual(
  Array.from(context.window.BJ_CONTENT.knowledge.topics, (topic) => topic.id),
  knowledgeDocs,
  "Knowledge topics should keep the intended learning order"
);

const closedGroups = new Set();
let currentGroup = null;
for (const topic of context.window.BJ_CONTENT.knowledge.topics) {
  if (topic.group === currentGroup) continue;
  if (currentGroup) closedGroups.add(currentGroup);
  assert.ok(!closedGroups.has(topic.group), `Knowledge group is split in navigation order: ${topic.group}`);
  currentGroup = topic.group;
}

for (const topic of context.window.BJ_CONTENT.knowledge.topics) {
  assert.ok(topic.docId, `Topic needs docId: ${topic.id}`);
  assert.ok(Array.isArray(topic.coreQuestions), `Topic needs coreQuestions: ${topic.id}`);
  const doc = context.window.BJ_MARKDOWN_DOCS[topic.docId];
  assert.ok(doc, `Generated doc missing: ${topic.docId}`);
  assert.ok(!/\?\?\?+/.test(doc), `Doc contains placeholder question marks: ${topic.id}`);
  assert.ok(
    doc.includes("知识索引引用") || doc.includes("参考资料") || doc.includes("参考论文") || doc.includes("关联资源"),
    `Doc should expose source/provenance index: ${topic.id}`
  );
  assert.ok(
    doc.includes("面试 QA") || doc.includes("面试八股") || doc.includes("面试回答模板") || doc.includes("高频问答"),
    `Doc should include interview practice section: ${topic.id}`
  );
}

const docs = context.window.BJ_MARKDOWN_DOCS;
assert.ok(docs["agent-rag-systems"].includes("hybrid_score") && docs["agent-rag-systems"].includes("Recall@k"), "Agent RAG doc should include retrieval details");
assert.ok(docs["agent-memory-tooling"].includes("ToolRegistry") && docs["agent-memory-tooling"].includes("response_mask"), "Agent memory/tooling doc should include tool and mask details");
assert.ok(docs["agent-planning-multi-agent"].includes("Plan-and-Execute") && docs["agent-planning-multi-agent"].includes("Multi-Agent"), "Agent planning doc should include planning and multi-agent details");
assert.ok(docs["training-inference-frameworks"].includes("framework-landscape.html"), "Training framework doc should embed framework landscape diagram");
assert.ok(docs["training-inference-frameworks"].includes("deepspeed-zero-stages.html"), "Training framework doc should embed DeepSpeed ZeRO diagram");
assert.ok(docs["training-inference-frameworks"].includes("megatron-parallelism.html"), "Training framework doc should embed Megatron parallelism diagram");
assert.ok(docs["training-inference-frameworks"].includes("vllm-pagedattention.html"), "Training framework doc should embed vLLM PagedAttention diagram");
assert.ok(docs["training-inference-frameworks"].includes("sglang-radixattention.html"), "Training framework doc should embed SGLang RadixAttention diagram");
assert.ok(docs["agent-rag-systems"].includes("rag-production-pipeline.html"), "RAG doc should embed RAG diagram");
assert.ok(docs["agent-memory-tooling"].includes("memory-lifecycle.html"), "Memory doc should embed memory diagram");
assert.ok(docs["agent-memory-tooling"].includes("tool-calling-safety.html"), "Tooling doc should embed tool diagram");
assert.ok(docs["agentic-rl-system"].includes("agentic-rl-system.html"), "Agentic RL doc should embed system diagram");
assert.ok(docs.opd.includes("opd-distillation-loop.html"), "OPD doc should embed OPD diagram");
assert.ok(docs["verl-rl-framework"].includes("verl-hybridflow-dataflow.html"), "VeRL doc should embed HybridFlow diagram");

const manifest = context.window.BJ_DIAGRAM_MANIFEST;
assert.ok(manifest && Array.isArray(manifest.diagrams), "Diagram manifest should load");
for (const diagram of manifest.diagrams) {
  assert.ok(fs.existsSync(path.join(root, diagram.source)), `Missing diagram source: ${diagram.source}`);
  assert.ok(fs.existsSync(path.join(root, diagram.output)), `Missing diagram output: ${diagram.output}`);
}

const rendered = context.window.BJ_MARKDOWN.render("# Demo\n\nTeacher gives $p_{\\theta_t}(\\cdot \\mid x, y_{<t})$.\n\n```archify\nDemo|assets/diagrams/html/framework-landscape.html\n```");
assert.ok(rendered.includes("INLINE:p_{\\theta_t}(\\cdot \\mid x, y_{<t})"), "Markdown renderer should preserve inline LaTeX");
assert.ok(rendered.includes("archify-diagram"), "Markdown renderer should render Archify fences");

console.log("Structure verification passed.");
