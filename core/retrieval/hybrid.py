"""混合检索的项目级编排层。

边界：LlamaIndex 原生组件负责向量检索、BM25、RRF 和 CrossEncoder 重排；本服务只负责
多 query、多通道、索引刷新、异常降级、证据阈值和 trace，不重新实现检索算法。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from config.settings import Settings, settings
from core.llama_embedding import get_llama_embedding
from core.retrieval.native import build_native_bm25, native_rrf_fuse

logger = logging.getLogger(__name__)
VectorSearch = Callable[[str, int, dict | None], list[NodeWithScore]]
CorpusLoader = Callable[[], list[BaseNode]]
RetrievalProfile = Literal["vector_only", "vector_rerank", "hybrid_rerank"]


@dataclass(frozen=True)
class HybridRetrievalResult:
    """统一检索结果；trace 中的分数保留原始语义，不伪装成概率。"""

    nodes: list[NodeWithScore]
    trace: dict[str, Any]
    insufficient_evidence: bool


class HybridRetrievalService:
    """组合 LlamaIndex 原生 Retriever，并补充项目级控制逻辑。

    三个可注入参数用于单元测试绕开 Chroma 和真实模型；生产环境不传时走原生链路。
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        *,
        vector_search: VectorSearch | None = None,
        corpus_loader: CorpusLoader | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.settings = app_settings
        self._injected_vector_search = vector_search
        self._injected_corpus_loader = corpus_loader
        self._vector_store: ChromaVectorStore | None = None
        self._index: VectorStoreIndex | None = None

        # BM25 算法由官方组件提供。项目只缓存不同 top_k/过滤条件的 Retriever；
        # 文档变化时清空缓存，避免关键词索引仍引用旧 Node。
        self._corpus_nodes: list[BaseNode] = []
        self._corpus_fingerprint: str | None = None
        self._bm25_cache: dict[str, BM25Retriever] = {}

        # 官方重排器构造时会加载模型，因此延迟到第一次真实查询。
        self._reranker = reranker
        local_path = app_settings.MODELS_DIR / app_settings.RERANKER_MODEL_NAME.replace("/", "_")
        self._reranker_model = (
            str(local_path) if local_path.exists() else app_settings.RERANKER_MODEL_NAME
        )

    def _ensure_vector_runtime(self) -> None:
        """延迟连接 Chroma；向量相似度仍由 VectorStoreIndex 原生实现。"""
        if self._vector_store is not None and self._index is not None:
            return
        client = chromadb.PersistentClient(
            path=str(self.settings.VECTOR_DB_DIR),
            settings=ChromaSettings(
                anonymized_telemetry=self.settings.CHROMA_ANONYMIZED_TELEMETRY
            ),
        )
        collection = client.get_or_create_collection(self.settings.CHROMA_COLLECTION_NAME)
        self._vector_store = ChromaVectorStore(chroma_collection=collection)
        self._index = VectorStoreIndex.from_vector_store(
            self._vector_store,
            embed_model=get_llama_embedding(self.settings),
        )

    def _load_corpus(self) -> list[BaseNode]:
        """读取 BM25 建索引所需的完整 Node；测试时可以替换数据来源。"""
        if self._injected_corpus_loader is not None:
            return self._injected_corpus_loader()
        self._ensure_vector_runtime()
        assert self._vector_store is not None
        return self._vector_store.get_nodes(node_ids=None)

    @staticmethod
    def _fingerprint(nodes: Sequence[BaseNode]) -> str:
        """ID 和内容 hash 同时参与指纹，同数量内容更新也能触发 BM25 重建。"""
        identity = "\n".join(sorted(f"{node.node_id}:{node.hash}" for node in nodes))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _refresh_corpus_if_needed(self) -> None:
        """检测知识库变化，并使官方 BM25Retriever 缓存失效。"""
        nodes = self._load_corpus()
        fingerprint = self._fingerprint(nodes)
        if fingerprint != self._corpus_fingerprint:
            self._corpus_nodes = list(nodes)
            self._corpus_fingerprint = fingerprint
            self._bm25_cache.clear()
            logger.info("BM25 语料已刷新，Node 数量: %s", len(nodes))

    @staticmethod
    def _metadata_filters(filter_dict: dict | None) -> MetadataFilters | None:
        """把项目 API 的字典过滤条件转换为 LlamaIndex 原生过滤对象。"""
        if not filter_dict:
            return None
        return MetadataFilters(
            filters=[MetadataFilter(key=key, value=value) for key, value in filter_dict.items()]
        )

    def _get_bm25(self, top_k: int, filter_dict: dict | None) -> BM25Retriever:
        """按查询参数复用官方 BM25Retriever；不在项目中实现 BM25 公式。"""
        filter_key = json.dumps(filter_dict or {}, ensure_ascii=False, sort_keys=True, default=str)
        cache_key = f"{top_k}:{filter_key}"
        if cache_key not in self._bm25_cache:
            self._bm25_cache[cache_key] = build_native_bm25(
                self._corpus_nodes,
                top_k=top_k,
                filters=self._metadata_filters(filter_dict),
            )
        return self._bm25_cache[cache_key]

    def _get_reranker(self, top_n: int) -> Any | None:
        """延迟创建 LlamaIndex 原生重排器，并同步本次查询需要返回的数量。"""
        if not self.settings.USE_RERANKER:
            return None
        if self._reranker is None:
            self._reranker = SentenceTransformerRerank(
                model=self._reranker_model,
                top_n=top_n,
                device=self.settings.EMBEDDING_DEVICE,
                # 避免把 retrieval_score 写入 metadata 后改变 node.hash，导致 BM25 误刷新。
                keep_retrieval_score=False,
            )
        elif hasattr(self._reranker, "top_n"):
            self._reranker.top_n = top_n
        return self._reranker

    def _default_vector_search(
        self, query: str, top_k: int, filter_dict: dict | None
    ) -> list[NodeWithScore]:
        self._ensure_vector_runtime()
        assert self._index is not None
        # as_retriever 返回原生 VectorIndexRetriever；项目不计算向量相似度。
        retriever = self._index.as_retriever(
            similarity_top_k=top_k,
            filters=self._metadata_filters(filter_dict),
        )
        return list(retriever.retrieve(query))

    def _vector_search(
        self, query: str, top_k: int, filter_dict: dict | None
    ) -> list[NodeWithScore]:
        if self._injected_vector_search is not None:
            return self._injected_vector_search(query, top_k, filter_dict)
        return self._default_vector_search(query, top_k, filter_dict)

    @staticmethod
    def _trace_items(items: Sequence[NodeWithScore]) -> list[dict[str, Any]]:
        """把框架对象变成可序列化 trace；只做展示，不修改框架得分。"""
        return [
            {
                "rank": rank,
                "node_id": item.node.node_id,
                # SentenceTransformer 可能返回 numpy.float32，API 出口统一使用 Python float。
                "score": float(item.score) if item.score is not None else None,
                "file_name": item.node.metadata.get("file_name"),
                "page": item.node.metadata.get("page"),
                "preview": item.node.get_content()[:120],
            }
            for rank, item in enumerate(items, start=1)
        ]

    @staticmethod
    def _contributors(
        rankings: dict[str, list[NodeWithScore]],
    ) -> dict[str, dict[str, int]]:
        """从融合前结果派生来源排名，仅用于解释，不参与 RRF 分数计算。"""
        contributors: dict[str, dict[str, int]] = {}
        for channel, items in rankings.items():
            # 原生 QueryFusionRetriever 同样先按通道 score 降序，再基于 rank 做 RRF。
            ordered = sorted(items, key=lambda item: item.score or 0.0, reverse=True)
            for rank, item in enumerate(ordered, start=1):
                contributors.setdefault(item.node.hash, {})[channel] = rank
        return contributors

    @staticmethod
    def _fusion_trace(
        items: Sequence[NodeWithScore], contributors: dict[str, dict[str, int]]
    ) -> list[dict[str, Any]]:
        """展示原生 RRF 结果，并附上项目派生的通道来源。"""
        return [
            {
                "rank": rank,
                "node_id": item.node.node_id,
                "score": float(item.score) if item.score is not None else None,
                "contributors": contributors.get(item.node.hash, {}),
                "file_name": item.node.metadata.get("file_name"),
                "page": item.node.metadata.get("page"),
                "preview": item.node.get_content()[:120],
            }
            for rank, item in enumerate(items, start=1)
        ]

    def retrieve(
        self,
        query: str,
        *,
        alternate_queries: Sequence[str] | None = None,
        top_k: int | None = None,
        filter_dict: dict | None = None,
        profile: RetrievalProfile = "hybrid_rerank",
    ) -> HybridRetrievalResult:
        """按评测 Profile 执行检索，默认行为仍是完整混合链路。

        ``alternate_queries`` 为 query 改写预留。各 query 的两种召回都是独立通道；本层保留
        原始结果，方便排错，也避免融合组件再次执行昂贵检索。

        Profiles:
            vector_only: 只取原问题的向量结果，不执行 BM25、RRF 或 Rerank。
            vector_rerank: 向量候选经过 Reranker，不执行 BM25/RRF。
            hybrid_rerank: 生产默认链路，向量 + BM25 + RRF + 可选 Rerank。
        """
        started_at = time.perf_counter()
        if profile not in {"vector_only", "vector_rerank", "hybrid_rerank"}:
            raise ValueError(f"不支持的检索 Profile: {profile}")
        if not query.strip():
            return HybridRetrievalResult(
                nodes=[],
                trace={"query": query, "queries": [], "error": "query is empty"},
                insufficient_evidence=True,
            )

        final_top_k = top_k or self.settings.SEARCH_TOP_K
        # dict 保序去重：原问题永远在第一位，改写问题不会覆盖原问题。
        queries = list(
            dict.fromkeys(
                value.strip()
                for value in [query, *(alternate_queries or [])]
                if value.strip()
            )
        )
        if profile != "hybrid_rerank":
            # 消融基线只评估原始向量问题，不享受 Query Rewrite 或多 Query 带来的额外召回。
            queries = [query.strip()]
        self._refresh_corpus_if_needed()
        if not self._corpus_nodes:
            return HybridRetrievalResult(
                nodes=[],
                trace={
                    "query": query,
                    "queries": queries,
                    "profile": profile,
                    "channels": {},
                    "fusion": [],
                    "rerank": [],
                    "rerank_applied": False,
                    "score_type": (
                        "rrf" if profile == "hybrid_rerank" else "vector_similarity"
                    ),
                    "threshold": self.settings.RETRIEVAL_MIN_FUSION_SCORE,
                    "insufficient_evidence": True,
                    "errors": [],
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
                insufficient_evidence=True,
            )

        # 不向 Retriever 请求超过语料总量的结果。
        candidate_k = min(
            max(final_top_k, self.settings.RETRIEVAL_CANDIDATE_K), len(self._corpus_nodes)
        )
        rankings: dict[str, list[NodeWithScore]] = {}
        errors: list[str] = []

        bm25 = None
        if profile == "hybrid_rerank":
            # 当 metadata filter 过滤掉全部 Node 时，官方 BM25Retriever 会在构造阶段抛错。
            # 这只代表关键词通道没有候选，不应阻断仍然可用的向量通道。
            try:
                bm25 = self._get_bm25(candidate_k, filter_dict)
            except Exception as exc:
                logger.exception("BM25 初始化失败")
                errors.append(f"bm25:init: {type(exc).__name__}: {exc}")

        for index, current_query in enumerate(queries):
            try:
                rankings[f"vector:{index}"] = self._vector_search(
                    current_query, candidate_k, filter_dict
                )
            except Exception as exc:
                # 单通道失败时保留其他通道，trace 记录原因，避免整个问答不可用。
                logger.exception("向量召回失败: query=%s", current_query)
                rankings[f"vector:{index}"] = []
                errors.append(f"vector:{index}: {type(exc).__name__}: {exc}")

            if profile == "hybrid_rerank":
                try:
                    if bm25 is None:
                        rankings[f"bm25:{index}"] = []
                        continue
                    native_bm25_results = list(bm25.retrieve(current_query))
                    # 官方 BM25 会返回 0 分占位项；它们不是关键词命中，不能进入 RRF 获得排名分。
                    rankings[f"bm25:{index}"] = [
                        item for item in native_bm25_results if (item.score or 0.0) > 0.0
                    ]
                except Exception as exc:
                    logger.exception("BM25 召回失败: query=%s", current_query)
                    rankings[f"bm25:{index}"] = []
                    errors.append(f"bm25:{index}: {type(exc).__name__}: {exc}")

        contributors: dict[str, dict[str, int]] = {}
        if profile == "hybrid_rerank":
            contributors = self._contributors(rankings)
            # 真正的 RRF 公式由 QueryFusionRetriever 执行，项目只传入已获取的通道结果。
            fused = native_rrf_fuse(rankings, query=query, top_k=candidate_k)
            fusion_candidates = fused[:candidate_k]
        else:
            # 两个向量基线只使用原始问题，防止 Query Rewrite 给基线额外增益。
            # 复制列表避免 Reranker 原地改分污染 vector channel 的调试 trace。
            fused = []
            fusion_candidates = [
                NodeWithScore(node=item.node, score=item.score)
                for item in rankings.get("vector:0", [])[:candidate_k]
            ]

        rerank_applied = False
        rerank_error: str | None = None
        reranker = self._get_reranker(final_top_k) if profile != "vector_only" else None
        if reranker is not None and fusion_candidates:
            try:
                # SentenceTransformerRerank 会原地覆盖 NodeWithScore.score。传入浅复制对象，
                # 让 fused 中的原生 RRF 分数继续用于 fusion trace，避免两种分数语义混淆。
                rerank_candidates = [
                    NodeWithScore(node=item.node, score=item.score)
                    for item in fusion_candidates
                ]
                final_nodes = list(
                    reranker.postprocess_nodes(
                        rerank_candidates, query_bundle=QueryBundle(query_str=query)
                    )
                )
                # NodeWithScore 还会直接返回给脚本/API，不能只在 trace 中转换类型。
                for item in final_nodes:
                    if item.score is not None:
                        item.score = float(item.score)
                rerank_applied = True
            except Exception as exc:
                logger.exception("Rerank 失败，降级为 RRF 结果")
                rerank_error = f"{type(exc).__name__}: {exc}"
                final_nodes = fusion_candidates[:final_top_k]
        else:
            final_nodes = fusion_candidates[:final_top_k]

        threshold = (
            self.settings.RETRIEVAL_MIN_RERANK_SCORE
            if rerank_applied
            else self.settings.RETRIEVAL_MIN_FUSION_SCORE
        )
        top_score = (
            final_nodes[0].score
            if final_nodes and final_nodes[0].score is not None
            else float("-inf")
        )
        # 原生 reranker 的 score 可能是 numpy.float32，比较结果会成为 numpy.bool_；
        # 显式转成 Python bool，确保 FastAPI/JSON 调试脚本可以稳定序列化。
        insufficient = bool(not final_nodes or top_score < threshold)
        trace = {
            "query": query,
            "queries": queries,
            "profile": profile,
            "channels": {
                channel: self._trace_items(items) for channel, items in rankings.items()
            },
            "fusion": self._fusion_trace(fused[:candidate_k], contributors),
            "rerank": self._trace_items(final_nodes) if rerank_applied else [],
            "rerank_applied": rerank_applied,
            "score_type": (
                "rerank_raw"
                if rerank_applied
                else "rrf"
                if profile == "hybrid_rerank"
                else "vector_similarity"
            ),
            "threshold": threshold,
            "insufficient_evidence": insufficient,
            "errors": [*errors, *([rerank_error] if rerank_error else [])],
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        return HybridRetrievalResult(
            nodes=[] if insufficient else final_nodes,
            trace=trace,
            insufficient_evidence=insufficient,
        )


_hybrid_retrieval_service: HybridRetrievalService | None = None


def get_hybrid_retrieval_service() -> HybridRetrievalService:
    """返回进程级单例，复用 Embedding、Chroma、BM25 与 Reranker 模型。"""
    global _hybrid_retrieval_service
    if _hybrid_retrieval_service is None:
        _hybrid_retrieval_service = HybridRetrievalService()
    return _hybrid_retrieval_service
