window.BJ_MARKDOWN = {
  render(markdown) {
    const lines = markdown.trim().split(/\r?\n/);
    const html = [];
    const toc = [];
    let listOpen = false;
    let tableRows = [];
    let codeOpen = false;
    let codeLang = "";
    let codeMeta = "";
    let codeLines = [];
    let mathOpen = false;
    let mathLines = [];
    let resourceBlock = false;
    let resourceListItems = [];

    function closeList() {
      if (listOpen) {
        html.push("</ul>");
        listOpen = false;
      }
    }

    function closeResourceList() {
      if (!resourceListItems.length) return;
      html.push(renderResourceCardsFromList(resourceListItems));
      resourceListItems = [];
    }

    function closeTable() {
      if (!tableRows.length) return;
      const rows = tableRows.map(parseTableRow);
      const separatorIndex = rows.findIndex((row) => row.every((cell) => /^:?-{3,}:?$/.test(cell)));
      if (separatorIndex > 0) {
        const header = rows[0];
        const body = rows.slice(separatorIndex + 1);
        if (resourceBlock && rowsContainResourceLinks(rows)) {
          html.push(renderResourceCardsFromTable(header, body));
        } else {
          html.push(
            `<div class="table-wrap"><table><thead><tr>${header
              .map((cell) => `<th>${inlineMarkdown(cell)}</th>`)
              .join("")}</tr></thead><tbody>${body
              .map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`)
              .join("")}</tbody></table></div>`
          );
        }
      } else {
        tableRows.forEach((row) => html.push(`<p>${inlineMarkdown(row)}</p>`));
      }
      tableRows = [];
    }

    function closeCode() {
      if (!codeOpen) return;
      if (codeLang === "archify") {
        html.push(renderArchifyDiagram(codeLines.join("\n"), codeMeta));
      } else if (codeLang === "flow" || codeLang === "mermaid") {
        html.push(codeLang === "mermaid" ? renderMermaidFlowDiagram(codeLines.join("\n")) : renderFlowDiagram(codeLines.join("\n")));
      } else {
        html.push(`<div class="code-box"><pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre></div>`);
      }
      codeOpen = false;
      codeLang = "";
      codeMeta = "";
      codeLines = [];
    }

    function closeMath() {
      if (!mathOpen) return;
      html.push(`<div class="math-block">${renderMath(mathLines.join("\n"), true)}</div>`);
      mathOpen = false;
      mathLines = [];
    }

    lines.forEach((line) => {
      if (line.startsWith("```")) {
        closeList();
        closeResourceList();
        closeTable();
        closeMath();
        if (codeOpen) closeCode();
        else {
          codeOpen = true;
          const fenceInfo = line.slice(3).trim();
          codeLang = fenceInfo.toLowerCase().split(/\s+/)[0] || "";
          codeMeta = fenceInfo.slice(codeLang.length).trim();
        }
        return;
      }

      if (codeOpen) {
        codeLines.push(line);
        return;
      }

      if (line.trim() === "$$") {
        closeList();
        closeResourceList();
        closeTable();
        if (mathOpen) closeMath();
        else mathOpen = true;
        return;
      }

      if (mathOpen) {
        mathLines.push(line);
        return;
      }

      if (line.startsWith("$$") && line.endsWith("$$") && line.length > 4) {
        closeList();
        closeResourceList();
        closeTable();
        html.push(`<div class="math-block">${renderMath(line.slice(2, -2), true)}</div>`);
        return;
      }

      if (isTableLine(line)) {
        closeList();
        closeResourceList();
        tableRows.push(line);
        return;
      }
      closeTable();

      if (!line.trim()) {
        closeList();
        closeResourceList();
        return;
      }

      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        closeList();
        closeResourceList();
        const level = heading[1].length;
        const text = heading[2].trim();
        const id = slugify(text);
        resourceBlock = isResourceHeading(text);
        if (level <= 2) toc.push({ id, level, text });
        html.push(`<h${Math.min(level + 2, 6)} id="${id}">${inlineMarkdown(text)}</h${Math.min(level + 2, 6)}>`);
        return;
      }

      if (line.startsWith("- ")) {
        if (resourceBlock && lineHasUrl(line)) {
          closeList();
          resourceListItems.push(line.slice(2));
          return;
        }
        closeResourceList();
        if (!listOpen) {
          html.push("<ul>");
          listOpen = true;
        }
        html.push(`<li>${inlineMarkdown(line.slice(2))}</li>`);
        return;
      }

      if (line.startsWith("> ")) {
        closeList();
        closeResourceList();
        html.push(`<blockquote class="doc-callout">${inlineMarkdown(line.slice(2))}</blockquote>`);
        return;
      }

      if (/^\d+\.\s/.test(line)) {
        closeList();
        closeResourceList();
        html.push(`<p>${inlineMarkdown(line)}</p>`);
        return;
      }

      closeList();
      closeResourceList();
      html.push(`<p>${inlineMarkdown(line)}</p>`);
    });

    closeList();
    closeResourceList();
    closeTable();
    closeCode();
    closeMath();

    const tocHtml =
      toc.length > 2
        ? `<nav class="doc-toc" aria-label="文档目录"><strong>目录</strong>${toc
            .map((item) => `<a class="toc-level-${item.level}" href="#${item.id}">${escapeHtml(item.text)}</a>`)
            .join("")}</nav>`
        : "";

    return `<div class="doc-layout">${tocHtml}<div class="markdown-body">${html.join("")}</div></div>`;
  }
};

function inlineMarkdown(value) {
  const parts = [];
  const mathPattern = /(\$\$[^$\n]+\$\$|\$[^$\n]+\$)/g;
  let lastIndex = 0;
  let match;

  while ((match = mathPattern.exec(value)) !== null) {
    if (match.index > lastIndex) {
      parts.push(renderInlineText(value.slice(lastIndex, match.index)));
    }

    const token = match[0];
    const displayMode = token.startsWith("$$");
    const expr = displayMode ? token.slice(2, -2) : token.slice(1, -1);
    parts.push(`<span class="math-inline">${renderMath(expr, false)}</span>`);
    lastIndex = mathPattern.lastIndex;
  }

  if (lastIndex < value.length) {
    parts.push(renderInlineText(value.slice(lastIndex)));
  }

  return parts.join("");
}

function renderInlineText(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/==([^=]+)==/g, "<mark>$1</mark>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\(#([a-z0-9-]+)\/([a-z0-9-]+)\)/g, '<a href="#$2/$3" data-module-link="$2" data-item-link="$3">$1</a>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/(^|[\s（(])((https?:\/\/)[^\s`，。；、)）]+)(?=$|[\s，。；、)）])/g, '$1<a href="$2" target="_blank" rel="noreferrer">$2</a>');
}

function renderResourceCardsFromTable(header, body) {
  const cards = body
    .map((row) => buildResourceCardFromRow(header, row))
    .filter(Boolean)
    .join("");
  return cards ? `<div class="resource-grid">${cards}</div>` : "";
}

function renderResourceCardsFromList(items) {
  const cards = items
    .map((item) => buildResourceCardFromText(item))
    .filter(Boolean)
    .join("");
  return cards ? `<div class="resource-grid">${cards}</div>` : "";
}

function buildResourceCardFromRow(header, row) {
  const joined = row.join(" ");
  const url = extractUrl(joined);
  if (!url) return null;

  const linkCellIndex = row.findIndex((cell) => cell.includes(url));
  const titleIndex = chooseTitleIndex(header, row, linkCellIndex);
  const title = cleanupResourceTitle(row[titleIndex] || row[0] || url, url);
  const description = row
    .filter((cell, index) => index !== titleIndex && index !== linkCellIndex && cell.trim())
    .join("；");
  const sourceLabel = getSourceLabel(url, row[linkCellIndex]);

  return renderResourceCard({ title, description, url, sourceLabel });
}

function buildResourceCardFromText(text) {
  const url = extractUrl(text);
  if (!url) return null;
  const markdownLink = text.match(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/);
  const beforeUrl = text.slice(0, text.indexOf(url)).replace(/[：:：`\[]+$/g, "").trim();
  const title = cleanupResourceTitle(markdownLink ? markdownLink[1] : beforeUrl || url, url);
  const description = text
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, "")
    .replace(url, "")
    .replace(/^[-：:：\s`]+|[`\s]+$/g, "")
    .trim();

  return renderResourceCard({ title, description, url, sourceLabel: getSourceLabel(url, text) });
}

function renderResourceCard({ title, description, url, sourceLabel }) {
  const safeUrl = escapeHtml(url);
  return `<a class="resource-card" href="${safeUrl}" target="_blank" rel="noreferrer">
    <span class="resource-card-kicker">${escapeHtml(sourceLabel)}</span>
    <strong>${inlineMarkdown(title)}</strong>
    ${description ? `<span class="resource-card-desc">${inlineMarkdown(description)}</span>` : ""}
    <span class="resource-card-action">打开资源</span>
  </a>`;
}

function chooseTitleIndex(header, row, linkCellIndex) {
  const preferred = header.findIndex((cell) => /知识点|主题|标题|论文|资源|文档|来源/.test(cell));
  if (preferred >= 0 && preferred !== linkCellIndex && row[preferred]) return preferred;
  const firstText = row.findIndex((cell, index) => index !== linkCellIndex && cell.trim() && !extractUrl(cell));
  return firstText >= 0 ? firstText : 0;
}

function cleanupResourceTitle(value, url) {
  return value
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, "$1")
    .replace(url, "")
    .replace(/`/g, "")
    .replace(/^[-：:：\s]+|[-：:：\s]+$/g, "")
    .trim() || url;
}

function getSourceLabel(url, raw) {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    if (/arxiv/.test(host)) return "Paper";
    if (/github|raw\.githubusercontent/.test(host)) return "GitHub";
    if (/readthedocs/.test(host)) return "Docs";
    if (/zhihu|cnblogs|mp\.weixin/.test(host)) return "Blog";
    return host;
  } catch (_error) {
    return raw && /论文|paper/i.test(raw) ? "Paper" : "Resource";
  }
}

function isResourceHeading(text) {
  return /知识索引引用|参考资源|参考资料|参考论文|关联资源|资料来源|资料索引|资源索引|来源索引|References/i.test(text);
}

function rowsContainResourceLinks(rows) {
  return rows.some((row) => row.some((cell) => lineHasUrl(cell)));
}

function lineHasUrl(value) {
  return /https?:\/\//.test(value);
}

function extractUrl(value) {
  const match = String(value).match(/https?:\/\/[^\s`)，。；、)）]+/);
  return match ? match[0] : "";
}

function renderMath(expr, displayMode) {
  const raw = expr.trim();
  if (window.katex) {
    try {
      return window.katex.renderToString(raw, { displayMode, throwOnError: false });
    } catch (_error) {
      return escapeHtml(raw);
    }
  }
  return escapeHtml(displayMode ? `$$${raw}$$` : `$${raw}$`);
}


function renderArchifyDiagram(source, meta) {
  const raw = [meta, source].filter(Boolean).join("\n").trim();
  const [firstLine = ""] = raw.split(/\r?\n/);
  const parts = firstLine.split("|").map((part) => part.trim()).filter(Boolean);
  const title = parts.length > 1 ? parts[0] : "Archify Diagram";
  const src = parts.length > 1 ? parts[1] : parts[0];

  if (!src || !/^assets\/diagrams\/html\/[a-z0-9._/-]+\.html$/i.test(src)) {
    return `<div class="code-box"><pre><code>${escapeHtml(raw || source)}</code></pre></div>`;
  }

  const safeSrc = escapeHtml(src);
  const safeTitle = escapeHtml(title);
  return `
    <figure class="archify-diagram">
      <figcaption>
        <span>${safeTitle}</span>
        <a href="${safeSrc}" target="_blank" rel="noreferrer">Open diagram</a>
      </figcaption>
      <iframe title="${safeTitle}" src="${safeSrc}" loading="lazy"></iframe>
    </figure>
  `;
}

function renderMermaidFlowDiagram(source) {
  const lines = source
    .split(/\r?\n/g)
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !/^flowchart\s+/i.test(line));

  const labels = [];
  for (const line of lines) {
    const matches = [...line.matchAll(/\["([^"]+)"\]/g)].map((match) => match[1]);
    if (matches.length) {
      for (const label of matches) {
        if (!labels.length || labels[labels.length - 1] !== label) labels.push(label);
      }
      continue;
    }
    if (line.includes("-->")) {
      for (const part of line.split(/\s*-->\s*/g)) {
        const label = part.replace(/^[A-Za-z0-9_]+/, "").replace(/[\[\]"]/g, "").trim() || part.trim();
        if (label && (!labels.length || labels[labels.length - 1] !== label)) labels.push(label);
      }
    }
  }

  if (labels.length >= 2) return renderFlowDiagram(labels.join(" -> "));
  const compactSource = lines.join("\\n");
  if (compactSource.includes("->")) return renderFlowDiagram(compactSource);
  return `<div class="code-box"><pre><code>${escapeHtml(source)}</code></pre></div>`;
}
function renderFlowDiagram(source) {
  const steps = source
    .split(/\s*->\s*/g)
    .map((step) => step.trim())
    .filter(Boolean);

  if (steps.length < 2) {
    return `<div class="code-box"><pre><code>${escapeHtml(source)}</code></pre></div>`;
  }

  const boxWidth = 150;
  const gap = 48;
  const height = 118;
  const width = 32 + steps.length * boxWidth + (steps.length - 1) * gap;
  const ariaLabel = steps.join(" to ");

  const boxes = steps
    .map((step, index) => {
      const x = 16 + index * (boxWidth + gap);
      const lines = splitSvgLabel(step);
      const text = lines
        .map((line, lineIndex) => {
          const y = 54 + lineIndex * 18 - (lines.length - 1) * 9;
          return `<text x="${x + boxWidth / 2}" y="${y}" text-anchor="middle">${escapeHtml(line)}</text>`;
        })
        .join("");

      const arrow =
        index < steps.length - 1
          ? `<path d="M ${x + boxWidth + 8} 58 L ${x + boxWidth + gap - 10} 58" class="flow-arrow" marker-end="url(#flowArrow)" />`
          : "";

      return `<g><rect x="${x}" y="24" width="${boxWidth}" height="68" rx="8" />${text}</g>${arrow}`;
    })
    .join("");

  return `
    <figure class="flow-diagram" role="img" aria-label="${escapeHtml(ariaLabel)}">
      <svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="flowArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M 0 0 L 8 4 L 0 8 z" />
          </marker>
        </defs>
        ${boxes}
      </svg>
    </figure>
  `;
}

function splitSvgLabel(label) {
  const chars = Array.from(label);
  if (chars.length <= 14) return [label];
  const midpoint = Math.ceil(chars.length / 2);
  return [chars.slice(0, midpoint).join(""), chars.slice(midpoint).join("")];
}

function isTableLine(line) {
  return line.trim().startsWith("|") && line.trim().endsWith("|");
}

function parseTableRow(row) {
  const trimmed = row.trim().replace(/^\|/, "").replace(/\|$/, "");
  const cells = [];
  let cell = "";
  let escaped = false;

  for (const char of trimmed) {
    if (char === "\\" && !escaped) {
      escaped = true;
      cell += char;
      continue;
    }
    if (char === "|" && !escaped) {
      cells.push(cell.trim());
      cell = "";
      continue;
    }
    cell += char;
    escaped = false;
  }

  cells.push(cell.trim());
  return cells;
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}





