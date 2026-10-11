"""阶段 4 兼容入口：旧类名，内部完全委托给 LangGraph Agent。

API 和现有调用方暂时继续使用 ``RAGChain``，但本文件已经不再创建 LangChain Classic chain。
阶段 5 稳定新 API 契约后，可以把对外命名统一成 ``RAGAgent``。
"""

from __future__ import annotations

from typing import Any

from core.agent.graph import RAGAgent

DEFAULT_SESSION_ID = "default"


class RAGChain:
    """保持旧公开方法兼容的薄门面。"""

    def __init__(self, agent: RAGAgent | None = None) -> None:
        self.agent = agent or RAGAgent()

    def ask(self, question: str, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
        """使用 session_id 作为 LangGraph thread_id 执行一轮问答。"""
        return self.agent.ask(question, thread_id=session_id)

    def get_chat_history(self, session_id: str = DEFAULT_SESSION_ID) -> list[dict[str, str]]:
        """读取 SQLite checkpoint 中的持久化历史。"""
        return self.agent.get_chat_history(thread_id=session_id)

    def clear_session(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        """删除指定 thread 的 checkpoint。"""
        self.agent.clear_thread(thread_id=session_id)


_rag_chain_instance: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """返回进程级单例，复用模型、检索器、图和 SQLite 连接。"""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = RAGChain()
    return _rag_chain_instance
