const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const chapterDir = path.join(root, "content", "foundations", "chapters");
const outputFile = path.join(root, "content", "foundations", "generated-docs.js");

const docs = {};

for (const file of fs.readdirSync(chapterDir)) {
  if (!file.endsWith(".md")) continue;
  const id = path.basename(file, ".md");
  docs[id] = fs.readFileSync(path.join(chapterDir, file), "utf8");
}

const body = `window.BJ_FOUNDATION_DOCS = ${JSON.stringify(docs, null, 2)};\n`;
fs.writeFileSync(outputFile, body, "utf8");

console.log(`Generated ${Object.keys(docs).length} foundation docs.`);
