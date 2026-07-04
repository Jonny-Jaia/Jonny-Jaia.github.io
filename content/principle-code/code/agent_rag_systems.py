"""
Agent/RAG 面试手撕代码合集。

特点：
- 只依赖 Python 标准库，方便在面试或白板环境中讲解。
- 每个实现都保留“最小可运行骨架”，避免工程框架细节掩盖核心思想。
- 覆盖工具注册、ReAct 循环、短期记忆、向量记忆、文本切块、BM25、混合检索和语义缓存。

运行：
    python content/principle-code/code/agent_rag_systems.py
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def tokenize(text: str) -> List[str]:
    """
    极简分词器：英文/数字按词切，中文按单字切。

    面试里重点不是分词效果，而是说明 BM25/稀疏检索需要把文档转成 token 序列。
    """
    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度：向量检索里最常见的打分函数之一。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ToolRegistry:
    """
    工具注册表：把工具名称、函数、参数 schema 统一管理。

    面试讲解重点：
    1. LLM 只负责选择工具名和参数，真正执行由宿主程序完成。
    2. schema 校验可以减少模型幻觉参数导致的运行时错误。
    3. 真实系统还会补鉴权、超时、重试、审计日志和沙箱隔离。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tuple[Callable[..., Any], Dict[str, type], str]] = {}

    def register(self, name: str, func: Callable[..., Any], schema: Dict[str, type], description: str = "") -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = (func, schema, description)

    def describe(self) -> List[Dict[str, Any]]:
        # 返回给模型看的工具描述；真实 Function Calling 会转成 JSON Schema。
        return [
            {"name": name, "schema": schema, "description": description}
            for name, (_, schema, description) in self._tools.items()
        ]

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        func, schema, _ = self._tools[name]

        # 最小参数校验：缺参和类型错误要在工具执行前暴露。
        for key, typ in schema.items():
            if key not in kwargs:
                raise ValueError(f"missing required argument: {key}")
            if not isinstance(kwargs[key], typ):
                raise TypeError(f"{key} expects {typ.__name__}, got {type(kwargs[key]).__name__}")
        return func(**kwargs)


@dataclass
class AgentStep:
    """记录一次 ReAct 轨迹：思考、动作、动作参数、工具观察结果。"""

    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str


def simple_react_agent(
    question: str,
    planner: Callable[[str, List[AgentStep]], Tuple[str, str, Dict[str, Any]]],
    registry: ToolRegistry,
    max_steps: int = 4,
) -> Tuple[str, List[AgentStep]]:
    """
    极简 ReAct 循环：Thought -> Action -> Observation。

    不变量：每一轮都把 observation 追加到 trace，让下一轮 planner 基于历史继续决策。
    退出条件：
    - planner 返回 action == "final"，说明已经得到最终回答。
    - 达到 max_steps，避免 Agent 无限调用工具。
    """
    trace: List[AgentStep] = []
    for _ in range(max_steps):
        thought, action, action_input = planner(question, trace)
        if action == "final":
            return thought, trace
        try:
            observation = str(registry.call(action, **action_input))
        except Exception as exc:
            # 工具失败也写入轨迹，让模型有机会自我修正或降级回答。
            observation = f"tool_error: {exc}"
        trace.append(AgentStep(thought, action, action_input, observation))
    return "max_steps reached; ask for clarification or fallback", trace


class ShortTermMemory:
    """
    短期记忆：保留最近 N 条对话上下文。

    适用场景：多轮问答里维持短期上下文。
    局限：容量固定，不能解决长期偏好、事实记忆和跨会话检索问题。
    """

    def __init__(self, max_items: int = 8):
        self.items: deque[str] = deque(maxlen=max_items)

    def add(self, text: str) -> None:
        self.items.append(text)

    def context(self) -> str:
        return "\n".join(self.items)


@dataclass
class MemoryItem:
    """一条长期记忆：文本、向量、写入时间和重要性权重。"""

    text: str
    vector: List[float]
    timestamp: float
    importance: float = 1.0


class VectorMemory:
    """
    向量记忆：用 embedding 相似度召回历史信息。

    打分 = 语义相似度 * 时间衰减 * 重要性。
    - 语义相似度：query 和 memory 的 cosine。
    - 时间衰减：越旧的记忆默认越不重要。
    - 重要性：用户偏好、明确事实可以给更高权重。
    """

    def __init__(self, embedder: Callable[[str], List[float]], half_life_seconds: float = 86400.0):
        self.embedder = embedder
        self.half_life_seconds = half_life_seconds
        self.items: List[MemoryItem] = []

    def add(self, text: str, importance: float = 1.0) -> None:
        self.items.append(MemoryItem(text, self.embedder(text), time.time(), importance))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        now = time.time()
        q = self.embedder(query)
        scored = []
        for item in self.items:
            age = max(0.0, now - item.timestamp)
            decay = 0.5 ** (age / self.half_life_seconds)
            score = cosine(q, item.vector) * decay * item.importance
            scored.append((item.text, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


def fixed_size_chunks(text: str, chunk_size: int) -> List[str]:
    """固定长度切块：实现简单，但可能切断语义边界。"""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def overlap_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    滑窗重叠切块：通过 overlap 保留相邻 chunk 的上下文连续性。

    面试常问：overlap 不能大于等于 chunk_size，否则 step <= 0 会死循环。
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


class BM25Index:
    """
    BM25 稀疏检索最小实现。

    核心公式由三部分组成：
    - IDF：区分词越少见，权重越高。
    - TF 饱和：词频增加有收益，但收益递减。
    - 长度归一：长文档天然命中更多词，需要惩罚。
    """

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in self.docs]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: str, doc_id: int) -> float:
        score = 0.0
        tf = self.term_freqs[doc_id]
        dl = self.doc_lens[doc_id]
        for term in tokenize(query):
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


def minmax_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """把不同检索器的分数压到 0-1，方便线性融合。"""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def hybrid_search(
    query: str,
    docs: Sequence[str],
    bm25: BM25Index,
    embedder: Callable[[str], List[float]],
    alpha: float = 0.5,
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    """
    混合检索：融合 BM25 稀疏分数和向量 dense 分数。

    alpha 越大越偏语义检索，越小越偏关键词检索。
    工程里通常还会加 reranker，把 top-k 候选再精排。
    """
    q_vec = embedder(query)
    sparse = {i: bm25.score(query, i) for i in range(len(docs))}
    dense = {i: cosine(q_vec, embedder(doc)) for i, doc in enumerate(docs)}
    sparse_n = minmax_normalize(sparse)
    dense_n = minmax_normalize(dense)
    fused = {
        i: alpha * dense_n.get(i, 0.0) + (1 - alpha) * sparse_n.get(i, 0.0)
        for i in range(len(docs))
    }
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


class SemanticCache:
    """
    语义缓存：不是 query 字符串完全相同才命中，而是语义相似即可复用答案。

    关键边界：
    - threshold 太低会误命中，回答错问题。
    - ttl 太长会复用过期信息。
    - 真实系统还要考虑用户隔离和权限边界。
    """

    def __init__(self, embedder: Callable[[str], List[float]], threshold: float = 0.9, ttl_seconds: float = 3600.0):
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.entries: List[Tuple[str, List[float], str, float]] = []

    def get(self, query: str) -> Optional[str]:
        now = time.time()
        q = self.embedder(query)
        best_answer = None
        best_score = -1.0
        for cached_query, vec, answer, created_at in self.entries:
            if now - created_at > self.ttl_seconds:
                continue
            score = cosine(q, vec)
            if score > best_score:
                best_answer = answer
                best_score = score
        if best_score >= self.threshold:
            return best_answer
        return None

    def set(self, query: str, answer: str) -> None:
        self.entries.append((query, self.embedder(query), answer, time.time()))


def hashing_embedder(text: str, dim: int = 32) -> List[float]:
    """
    面试演示用 embedding：把 token 哈希到固定维度向量。

    它不是高质量语义模型，但足够说明“文本 -> 向量 -> 相似度检索”的流程。
    """
    vec = [0.0] * dim
    for token in tokenize(text):
        vec[hash(token) % dim] += 1.0
    return vec


def _quick_check() -> None:
    registry = ToolRegistry()
    registry.register("search", lambda query: f"result for {query}", {"query": str})

    def planner(question: str, trace: List[AgentStep]) -> Tuple[str, str, Dict[str, Any]]:
        if trace:
            return f"final based on {trace[-1].observation}", "final", {}
        return "need search", "search", {"query": question}

    answer, trace = simple_react_agent("what is rag", planner, registry)
    assert "final" in answer and len(trace) == 1

    docs = ["RAG uses retrieval augmented generation", "Agent uses tools and memory", "BM25 is sparse retrieval"]
    bm25 = BM25Index(docs)
    assert bm25.search("retrieval")[0][0] in {0, 2}
    assert hybrid_search("agent memory", docs, bm25, hashing_embedder)[0][0] == 1

    cache = SemanticCache(hashing_embedder, threshold=0.5)
    cache.set("what is agent memory", "memory answer")
    assert cache.get("agent memory") == "memory answer"

    print("quick check passed")


if __name__ == "__main__":
    _quick_check()
