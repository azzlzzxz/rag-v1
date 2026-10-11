"""基于 LangGraph thread_id 的会话 API。"""

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from api.dependencies import get_rag_chain_dep
from api.schemas.qa import HistoryResponse
from core.rag_chain import RAGChain

router = APIRouter(prefix="/sessions", tags=["会话"])


@router.get("/{thread_id}/history", response_model=HistoryResponse)
async def get_session_history(
    thread_id: str,
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
) -> HistoryResponse:
    """读取 SQLite checkpoint 中的 Human/AI 消息。"""
    history = await run_in_threadpool(rag_chain.get_chat_history, thread_id)
    return HistoryResponse(thread_id=thread_id, history=history)


@router.delete("/{thread_id}")
async def delete_session(
    thread_id: str,
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
) -> dict[str, str]:
    """只删除指定 thread，不影响其他会话。"""
    await run_in_threadpool(rag_chain.clear_session, thread_id)
    return {"status": "deleted", "thread_id": thread_id}
