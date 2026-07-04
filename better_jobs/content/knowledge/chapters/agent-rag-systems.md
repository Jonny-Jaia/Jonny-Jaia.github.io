# Agent RAG 系统：检索增强、重排与可追溯生成

## 当前定位

RAG 在 Agent 系统里不是“外挂知识库”这么简单，而是把 **知识更新、权限控制、证据召回、上下文组织和答案可追溯性** 从模型参数里拆出来。面试中要把它讲成一条离线建设链路和一条在线查询链路，而不是只说 embedding + 向量库。

> **面试抓手**：微调更适合改变模型行为、格式和任务风格；RAG 更适合接入私有知识、实时知识和可引用证据。RAG 能缓解知识过时和私有知识缺失，但检索错、上下文噪声和生成不忠实仍然会导致幻觉。

```archify
RAG Production Pipeline|assets/diagrams/html/rag-production-pipeline.html
```

## 一、完整 Pipeline 怎么讲

| 阶段 | 核心步骤 | 面试要点 | 常见风险 |
|---|---|---|---|
| 离线建设 | 文档加载、清洗、切块、metadata、embedding、向量库/倒排索引、权限索引 | 数据质量决定上限，chunk 结构决定召回粒度 | 脏数据、语义切断、版本混乱、权限泄漏 |
| 在线查询 | query rewrite、意图识别、召回、rerank、过滤、上下文拼接、生成、引用、日志 | 召回和生成要分开评估，不要只看最终答案 | 召回不足、噪声过多、Lost in the Middle、延迟高 |
| 生产闭环 | 反馈采集、失败样例、索引更新、权限审计、监控告警 | RAG 是持续运营系统，不是一次性脚本 | 数据漂移、引用不可追溯、评估集老化 |

一句话模板：**Load -> Clean -> Chunk -> Embed -> Store -> Retrieve -> Rerank -> Compose Context -> Generate -> Cite -> Monitor**。

### 离线链路的关键设计

- **清洗**：去掉导航栏、页脚、重复模板、广告、乱码和过期版本；保留标题层级、表格、代码块和来源信息。
- **切块**：优先按标题、段落、表格、函数、类、章节边界切，而不是无脑固定 token 窗口。
- **metadata**：至少保留 `source_id`、`title`、`section`、`version`、`updated_at`、`permission_scope`、`chunk_index`。
- **索引**：高召回场景用 hybrid search，把 BM25 的关键词匹配和 dense embedding 的语义匹配结合起来。
- **权限**：权限必须进入召回前过滤，不能等生成后再删，因为敏感片段一旦进入上下文就已经泄漏。

### 在线链路的关键设计

- **query rewrite**：处理口语化、省略指代、多跳问题和领域缩写，但要避免改写改变原问题意图。
- **hybrid retrieval**：BM25 擅长术语、编号、函数名、错误码；embedding 擅长语义相似、同义表达和概念匹配。
- **rerank**：cross-encoder 或 LLM reranker 负责精排，通常比纯 embedding 更能判断证据是否真正回答问题。
- **context packing**：去重、按证据链排序、压缩长片段、保留引用编号，避免把相互冲突的片段直接塞给模型。
- **answer verification**：检查答案是否被证据支持，必要时拒答或返回“证据不足”。

## 二、Chunk Size 和 Overlap 怎么选

核心取舍是：chunk 越小，召回更精准但上下文可能不完整；chunk 越大，语义更完整但噪声更多、召回粒度更粗、上下文成本更高。overlap 能缓解边界切断，但会增加索引冗余和重复召回。

| 文档类型 | 推荐切块方式 | 原因 |
|---|---|---|
| FAQ / 短问答 | 小 chunk，低 overlap | 单个问题通常可以独立回答 |
| 法律 / 论文 / 技术文档 | 标题层级 + 段落 + 表格边界 | 需要保留论证结构和引用上下文 |
| 代码文档 | 函数、类、文件结构切分 | 函数签名和实现细节必须一起看 |
| 产品知识库 | 标题层级 + 版本 metadata | 需要处理过期版本和权限范围 |

最终不要凭感觉定参数，应该用 **Recall@k、MRR、nDCG、答案忠实度、引用准确率、上下文 token 成本和端到端延迟** 共同调参。

## 三、Embedding、Reranker 与 Hybrid Search

选择 embedding 模型时可以从四个维度回答：

| 维度 | 看什么 | 面试表达 |
|---|---|---|
| 语义匹配能力 | MTEB / C-MTEB、Recall@k、MRR、nDCG | 先用离线评估集验证召回，不要只看模型榜单 |
| 领域适配 | 金融、法律、代码、医疗术语是否稳定 | 领域词汇和缩写会显著影响召回 |
| 多语言能力 | 中英混合、专有名词、缩写 | 中文知识库尤其要关注 C-MTEB 和实际 query |
| 工程成本 | 向量维度、吞吐、延迟、部署方式、增量更新 | 高维向量更贵，索引更新和存储也更重 |

**embedding 负责粗召回，reranker 负责精排**。不要指望 embedding 单独解决所有相关性问题。生产 RAG 中常见组合是：BM25 召回一批、dense 召回一批、合并去重、rerank 排序、再做权限/版本/质量过滤。

### 分数融合的直觉

```python
def hybrid_score(bm25_score: float, dense_score: float, alpha: float = 0.55) -> float:
    """alpha 越大越偏关键词匹配，越小越偏语义匹配。"""
    return alpha * bm25_score + (1 - alpha) * dense_score
```

真实系统里要先做 score normalization，因为 BM25 分数和向量相似度不在同一尺度。常见做法包括 min-max、z-score、rank fusion 或 reciprocal rank fusion。

## 四、RAG 为什么会失败

| 失败点 | 表现 | 排查方式 |
|---|---|---|
| 召回失败 | 正确文档没有进入候选集 | 查 Recall@k、query rewrite、chunk 策略、embedding 模型 |
| 排序失败 | 正确文档在候选里但没进上下文 | 查 reranker、metadata boost、去重逻辑 |
| 上下文污染 | 错误、过期、冲突片段被塞进 prompt | 查版本过滤、权限过滤、source quality |
| Lost in the Middle | 证据在长上下文中间被忽略 | 调整 packing 顺序、压缩证据、分段回答 |
| 生成不忠实 | 模型没有按证据回答 | 加引用约束、answer verification、拒答策略 |

**结论**：RAG 的瓶颈通常不在 LLM，而在 **检索质量、rerank、上下文组织、权限控制和答案可追溯性**。

## 五、生产化检查清单

- 权限是否在检索前过滤？
- chunk 是否保留标题、来源、版本和更新时间？
- 是否同时评估召回、排序、生成和引用？
- 是否保留 query、retrieved chunks、reranked chunks、prompt、answer、citation 的日志？
- 是否有过期知识、冲突知识和敏感知识的处理策略？
- 是否有失败样例回流机制，用于更新 query rewrite、chunk、embedding 或 reranker？

## 面试 QA

**Q：RAG 相比直接微调解决什么问题？**

A：RAG 把知识更新和模型行为拆开。微调更适合改变模型输出风格、格式和任务行为；RAG 更适合接入私有知识、实时知识和可追溯证据。它通过外部知识库召回证据，再把证据放进上下文生成答案，缓解知识过时和私有知识缺失问题。但 RAG 不等于一定不幻觉，检索错、上下文噪声和生成不忠实都会出错。

**Q：chunk size 和 overlap 怎么选？**

A：chunk 小召回精准但上下文可能断裂；chunk 大语义完整但噪声多、成本高。FAQ 可以小 chunk，论文和技术文档应该按标题层级、段落和表格切，代码文档按函数、类和文件结构切。最终用 Recall@k、MRR、答案忠实度和 token 成本共同调参。

**Q：为什么 embedding 之后还要 rerank？**

A：embedding 更适合粗召回，目标是别漏掉可能相关的候选；reranker 更适合精排，判断候选是否真正回答 query。尤其在长文档、相似术语和多跳问题里，reranker 往往能显著提升上下文质量。

**Q：RAG 的线上监控看什么？**

A：至少看召回命中率、rerank 后证据质量、引用准确率、答案忠实度、拒答率、延迟、token 成本、权限过滤命中、用户反馈和失败样例分布。只看最终满意度会很难定位问题在召回、排序还是生成。

## 知识索引引用

| 知识点 | 主要来源 | 本页使用方式 |
|---|---|---|
| RAG 基本范式 | https://arxiv.org/abs/2005.11401 | 用于定义 retrieval-augmented generation 的基本目标和证据增强思想 |
| AgentGuide 面试问题 | https://github.com/adongwanai/AgentGuide/tree/main/docs/04-interview | 用于整理 RAG pipeline、chunk、embedding、评估等面试问法 |
| Hello-Agents Extra Chapter | https://github.com/datawhalechina/hello-agents/tree/main/Extra-Chapter | 用于补充 Agent/RAG 工程面试表达 |
| WDN LLM Interview Note | https://wdndev.github.io/llm_interview_note/#/ | 用于补充 RAG、Agent、LLM 工程面试高频问题 |

## 相关章节

| 章节 | 关系 |
|---|---|
| [Agent 系统](#knowledge/agent) | Agent 总览入口 |
| [Agent Memory 与工具调用](#knowledge/agent-memory-tooling) | RAG 与 memory、tool registry 的边界 |
| [Agent 规划与多智能体](#knowledge/agent-planning-multi-agent) | RAG 作为规划执行过程中的外部证据来源 |
| [AgentGuide 面试问答补全](#knowledge/agent-guide-interview-qa) | 高频面试题模板 |
