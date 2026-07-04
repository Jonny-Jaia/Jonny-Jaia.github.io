const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const folders = [
  path.join(root, "content", "code-training", "guides"),
  path.join(root, "content", "code-training", "reviews")
];
const outputFile = path.join(root, "content", "code-training", "generated-guides.js");

const docs = {};

for (const folder of folders) {
  for (const file of fs.readdirSync(folder)) {
    if (!file.endsWith(".md")) continue;
    const id = path.basename(file, ".md");
    docs[id] = fs.readFileSync(path.join(folder, file), "utf8");
  }
}

fs.writeFileSync(
  outputFile,
  `window.BJ_TRAINING_GUIDES = ${JSON.stringify(docs, null, 2)};\n`,
  "utf8"
);

console.log(`Generated ${Object.keys(docs).length} training guide docs.`);
