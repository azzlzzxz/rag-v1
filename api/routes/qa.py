"""稳定问答 API：同步回答、SSE 生命周期事件和旧路径兼容。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.dependencies import get_rag_chain_dep
from api.presenters import present_answer
from api.schemas.qa import AnswerResponse, QuestionRequest
from core.rag_chain import RAGChain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/qa", tags=["智能问答"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
) -> AnswerResponse:
    """执行一次问答；阻塞的模型和检索调用放入线程池，避免卡住 FastAPI 事件循环。"""
    try:
        result = await run_in_threadpool(rag_chain.ask, request.question, request.thread_id)
    except Exception as exc:
        logger.exception("同步问答失败: thread_id=%s", request.thread_id)
        raise HTTPException(503, "问答服务暂时不可用") from exc
    return present_answer(result, request.thread_id, debug=request.debug)


def _sse(event: str, payload: dict) -> str:
    """生成标准 Server-Sent Events 帧；ensure_ascii=False 便于直接调试中文。"""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/stream")
async def stream_answer(
    request: QuestionRequest,
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
) -> StreamingResponse:
    """以 SSE 返回稳定生命周期事件。

    当前图在 ``completed`` 事件中发送完整回答；阶段 6 前端可先展示 started 状态，再渲染最终
    结果。未来增加模型 token 流时，可以新增 token 事件而不破坏现有消费者。
    """

    async def events():
        yield _sse("started", {"thread_id": request.thread_id})
        try:
            result = await run_in_threadpool(
                rag_chain.ask, request.question, request.thread_id
            )
            response = present_answer(result, request.thread_id, debug=request.debug)
            yield _sse("completed", response.model_dump(mode="json"))
            yield _sse("done", {"thread_id": request.thread_id})
        except Exception:
            # SSE 已经发出 200 后不能再改变 HTTP 状态码，只发送不含内部异常的公开错误事件。
            logger.exception("流式问答失败: thread_id=%s", request.thread_id)
            yield _sse(
                "error",
                {"thread_id": request.thread_id, "message": "问答处理失败，请稍后重试"},
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 以下两个路径保留给旧前端；新调用方使用 /sessions/{thread_id}。
@router.post("/clear_memory", deprecated=True)
async def clear_conversation_memory(
    session_id: str = Query("default", description="旧版会话 ID"),
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
):
    rag_chain.clear_session(session_id)
    return {"status": "success", "thread_id": session_id}


@router.get("/history", deprecated=True)
async def get_chat_history(
    session_id: str = Query("default", description="旧版会话 ID"),
    rag_chain: RAGChain = Depends(get_rag_chain_dep),
):
    return {"thread_id": session_id, "history": rag_chain.get_chat_history(session_id)}
