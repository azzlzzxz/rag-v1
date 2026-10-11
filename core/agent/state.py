"""LangGraph 状态定义。

状态只保存可序列化数据，不能直接放入 Chroma client、Retriever 或 Node 对象。这样 SQLite
checkpoint 才能跨进程恢复，也避免把运行时连接意外持久化。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

AgentDecision = Literal["answer", "clarify", "refuse"]


class RAGAgentState(TypedDict, total=False):
    """RAG 工作流的共享状态。

    ``messages`` 使用 LangGraph 的 add_messages reducer：每次请求只传入新的 HumanMessage，
    checkpoint 会自动和历史消息合并，不需要项目再维护一份内存会话。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: str
    rewritten_query: str
    rewrite_applied: bool
    rewrite_reason: str
    protected_constraints: dict[str, list[str]]
    intent: str
    retrieved_nodes: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    insufficient_evidence: bool
    decision: AgentDecision
    answer: str
    citations: list[dict[str, Any]]
    errors: list[str]
