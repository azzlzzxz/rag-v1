"""LlamaIndex 混合检索与旧脚本之间的兼容层。

阶段 3 的真实检索逻辑位于 ``core/retrieval``。本文件只负责把 LlamaIndex NodeWithScore
转换为 LangChain Document。阶段 4 的 LangGraph 已直接使用标准 Retrieval Tool；这里仅为
已有脚本和调用方保留旧方法，不再参与 Agent 主链。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config.settings import settings
from core.retrieval.hybrid import (
    HybridRetrievalResult,
    HybridRetrievalService,
    get_hybrid_retrieval_service,
)


def _to_langchain_document(node_with_score) -> Document:
    """转换 Node，并把分数类型写入 metadata，避免调用方误把分数当概率。"""
    metadata = dict(node_with_score.node.metadata)
    metadata["retrieval_score"] = node_with_score.score
    return Document(page_content=node_with_score.node.get_content(), metadata=metadata)


class HybridLangChainRetriever(BaseRetriever):
    """满足旧 LangChain Retriever 调用方的薄适配器。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    service: HybridRetrievalService
    top_k: int
    filter_dict: dict | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        result = self.service.retrieve(
            query,
            top_k=self.top_k,
            filter_dict=self.filter_dict,
        )
        return [_to_langchain_document(item) for item in result.nodes]


class RAGRetriever:
    """保留旧公开方法名，内部委托给新的 HybridRetrievalService。"""

    def __init__(self, service: HybridRetrievalService | None = None) -> None:
        self.service = service or get_hybrid_retrieval_service()
        self.last_trace: dict[str, Any] | None = None

    def retrieve_with_trace(
        self,
        query: str,
        *,
        alternate_queries: list[str] | None = None,
        top_k: int | None = None,
        filter_dict: dict | None = None,
    ) -> HybridRetrievalResult:
        """新接口：返回 LlamaIndex NodeWithScore、证据判断和完整检索轨迹。"""
        result = self.service.retrieve(
            query,
            alternate_queries=alternate_queries,
            top_k=top_k,
            filter_dict=filter_dict,
        )
        self.last_trace = result.trace
        return result

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_dict: dict | None = None,
        return_score: bool = False,
    ) -> list:
        """旧接口：返回 LangChain Document，供已有脚本平滑迁移。"""
        result = self.retrieve_with_trace(
            query,
            top_k=top_k,
            filter_dict=filter_dict,
        )
        documents = [_to_langchain_document(item) for item in result.nodes]
        if return_score:
            return [(document, item.score) for document, item in zip(documents, result.nodes)]
        return documents

    def get_compression_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        """返回 LangChain BaseRetriever；rerank 已在混合检索服务内部完成。"""
        search_kwargs = search_kwargs or {}
        return HybridLangChainRetriever(
            service=self.service,
            top_k=search_kwargs.get("k", settings.SEARCH_TOP_K),
            filter_dict=search_kwargs.get("filter"),
        )

    def get_compresstion_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        """兼容旧代码中的拼写错误；新代码应使用 get_compression_retriever。"""
        return self.get_compression_retriever(search_kwargs)


_rag_retriever_instance: RAGRetriever | None = None


def get_rag_retriever() -> RAGRetriever:
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    return _rag_retriever_instance
