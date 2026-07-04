const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const codeDir = path.join(root, "content", "code-training", "code");
const outputFile = path.join(root, "content", "code-training", "generated-code.js");

const snippets = {};

function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath);
      continue;
    }
    if (!entry.name.endsWith(".py")) continue;

    const relative = path.relative(codeDir, fullPath).replaceAll("\\", "/");
    snippets[relative.replace(/\.py$/, "")] = fs.readFileSync(fullPath, "utf8");
  }
}

walk(codeDir);

fs.writeFileSync(
  outputFile,
  `window.BJ_CODE_SNIPPETS = ${JSON.stringify(snippets, null, 2)};\n`,
  "utf8"
);

console.log(`Generated ${Object.keys(snippets).length} code snippets.`);
