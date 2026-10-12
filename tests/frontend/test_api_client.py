import json

import httpx
import pytest

from frontend.api_client import APIClientError, RAGAPIClient


def test_api_client_calls_stable_contracts_and_parses_sse():
    """前端必须只依赖阶段 5/6 的公开 URL，并正确解析中文 SSE 数据。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/system/health":
            return httpx.Response(200, json={"status": "healthy", "version": "2.0.0"})
        if request.url.path == "/api/v1/documents":
            return httpx.Response(200, json={"total": 0, "documents": []})
        if request.url.path == "/api/v1/qa/stream":
            body = (
                'event: started\ndata: {"thread_id":"demo"}\n\n'
                'event: completed\ndata: {"answer":"制度答案","citations":[]}\n\n'
                'event: done\ndata: {"thread_id":"demo"}\n\n'
            )
            return httpx.Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, json={"detail": "not found"})

    client = RAGAPIClient("http://test", transport=httpx.MockTransport(handler))
    assert client.health()["status"] == "healthy"
    assert client.list_documents() == {"total": 0, "documents": []}
    events = list(client.stream_answer("请假制度？", "demo"))
    assert [event for event, _ in events] == ["started", "completed", "done"]
    assert events[1][1]["answer"] == "制度答案"


def test_api_client_exposes_safe_server_detail():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "问答服务暂时不可用"})

    client = RAGAPIClient("http://test", transport=httpx.MockTransport(handler))
    with pytest.raises(APIClientError, match="问答服务暂时不可用"):
        client.health()
