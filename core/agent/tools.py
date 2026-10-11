"""可复用的 Agent Tool。

检索算法仍在 ``core.retrieval``，Tool 只把项目服务包装成 LangChain 标准工具，并将
LlamaIndex Node 转成 SQLite checkpoint 可安全持久化的普通字典。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config.settings import settings
from core.retrieval.hybrid import HybridRetrievalService


class RetrievalToolInput(BaseModel):
    original_query: str = Field(description="用户原始问题")
    alternate_queries: list[str] = Field(default_factory=list, description="保留约束后的改写问题")
    top_k: int = Field(default=settings.SEARCH_TOP_K, ge=1, le=100)
    filter_dict: dict[str, Any] | None = None


def _json_safe(value: Any) -> Any:
    """将 metadata/trace 中可能出现的 Path、NumPy 标量等转换为 SQLite 可序列化类型。"""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def create_retrieval_tool(service: HybridRetrievalService) -> StructuredTool:
    """把混合检索服务包装成标准 Tool，便于当前 Graph 和后续 create_agent 共同复用。"""

    def retrieve_knowledge(
        original_query: str,
        alternate_queries: list[str] | None = None,
        top_k: int = settings.SEARCH_TOP_K,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """从企业知识库检索与问题相关的证据，并返回候选内容和完整检索轨迹。"""
        result = service.retrieve(
            original_query,
            alternate_queries=alternate_queries or [],
            top_k=top_k,
            filter_dict=filter_dict,
        )
        nodes = [
            {
                "node_id": item.node.node_id,
                "content": item.node.get_content(),
                "metadata": _json_safe(dict(item.node.metadata)),
                "score": float(item.score) if item.score is not None else None,
            }
            for item in result.nodes
        ]
        return {
            "nodes": nodes,
            "trace": _json_safe(result.trace),
            "insufficient_evidence": bool(result.insufficient_evidence),
        }

    return StructuredTool.from_function(
        func=retrieve_knowledge,
        name="retrieve_knowledge",
        description="使用向量、中文 BM25、RRF 和可选 Reranker 检索企业知识库",
        args_schema=RetrievalToolInput,
    )
