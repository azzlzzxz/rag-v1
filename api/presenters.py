"""把内部 Agent 结果转换成稳定、可公开的 HTTP 响应。

这里是安全边界：内部 trace 可以包含 chunk 预览和异常细节，普通 API 不得直接透传。
"""

from __future__ import annotations

import re
from typing import Any

from api.schemas.qa import AnswerResponse, CitationResponse


def _safe_error(error: Any) -> str:
    """只保留通道/节点和异常类型，删除路径、密钥附近文本及第三方响应正文。"""
    text = str(error)
    match = re.match(
        r"^(.+?):\s*([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)", text
    )
    return f"{match.group(1)}: {match.group(2)}" if match else "internal_error"


def sanitize_retrieval_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    """保留排名和分数，删除 preview/content 与详细异常消息。"""
    trace = trace or {}

    def safe_items(items: list[dict[str, Any]] | None, *, contributors: bool = False):
        result = []
        for item in items or []:
            safe = {
                "rank": item.get("rank"),
                "node_id": item.get("node_id"),
                "score": item.get("score"),
                "file_name": item.get("file_name"),
                "page": item.get("page"),
            }
            if contributors:
                safe["contributors"] = item.get("contributors", {})
            result.append(safe)
        return result

    return {
        "queries": list(trace.get("queries") or []),
        "channels": {
            name: safe_items(items)
            for name, items in (trace.get("channels") or {}).items()
        },
        "fusion": safe_items(trace.get("fusion"), contributors=True),
        "rerank": safe_items(trace.get("rerank")),
        "rerank_applied": bool(trace.get("rerank_applied", False)),
        "score_type": trace.get("score_type"),
        "threshold": trace.get("threshold"),
        "insufficient_evidence": bool(trace.get("insufficient_evidence", False)),
        "elapsed_ms": trace.get("elapsed_ms"),
        "errors": [_safe_error(error) for error in trace.get("errors") or []],
    }


def present_answer(result: dict[str, Any], thread_id: str, *, debug: bool) -> AnswerResponse:
    """建立引用编号并按 debug 开关控制 trace 可见性。"""
    citations = [
        CitationResponse(
            citation_id=index,
            document_id=item.get("document_id"),
            chunk_id=item.get("chunk_id") or item.get("node_id") or "unknown",
            file_name=item.get("file_name") or item.get("source") or "未知",
            page=item.get("page"),
            content=item.get("content", ""),
            score=item.get("score"),
        )
        for index, item in enumerate(result.get("citations") or [], start=1)
    ]
    return AnswerResponse(
        answer=result["answer"],
        decision=result.get("decision", "refuse"),
        thread_id=thread_id,
        original_query=result.get("original_query", ""),
        rewritten_query=result.get("rewritten_query", result.get("original_query", "")),
        rewrite_applied=bool(result.get("rewrite_applied", False)),
        citations=citations,
        trace=sanitize_retrieval_trace(result.get("trace")) if debug else None,
    )
