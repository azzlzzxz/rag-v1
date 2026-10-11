import httpx
import pytest

from api.dependencies import get_rag_chain_dep
from api.main import app


class FakeRAGChain:
    def __init__(self):
        self.cleared: list[str] = []

    def ask(self, question: str, session_id: str):
        return {
            "answer": "员工应先提交申请。[1]",
            "decision": "answer",
            "original_query": question,
            "rewritten_query": "员工加班应该怎么申请？",
            "rewrite_applied": True,
            "citations": [
                {
                    "node_id": "chunk-1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "file_name": "员工制度.pdf",
                    "source": "员工制度.pdf",
                    "page": 12,
                    "content": "员工应先在系统中提交加班申请。",
                    "score": 0.98,
                }
            ],
            "trace": {
                "queries": [question, "员工加班应该怎么申请？"],
                "channels": {
                    "vector:0": [
                        {
                            "rank": 1,
                            "node_id": "chunk-1",
                            "score": 0.8,
                            "file_name": "员工制度.pdf",
                            "page": 12,
                            "preview": "这段内部 chunk 正文不能出现在 debug trace",
                        }
                    ]
                },
                "fusion": [],
                "rerank": [],
                "rerank_applied": True,
                "score_type": "rerank_raw",
                "threshold": 0.2,
                "insufficient_evidence": False,
                "elapsed_ms": 31.2,
                "errors": ["vector:0: RuntimeError: /private/secret/path"],
            },
            "errors": [],
        }

    def get_chat_history(self, session_id: str):
        return [
            {"role": "human", "content": "加班制度是什么？"},
            {"role": "assistant", "content": "这是加班制度。"},
        ]

    def clear_session(self, session_id: str):
        self.cleared.append(session_id)


@pytest.fixture
def fake_chain():
    chain = FakeRAGChain()
    app.dependency_overrides[get_rag_chain_dep] = lambda: chain
    yield chain
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ask_contract_hides_trace_by_default_and_numbers_citations(fake_chain):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/qa/ask",
            json={"question": "那怎么申请？", "thread_id": "demo-001"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "demo-001"
    assert body["decision"] == "answer"
    assert body["trace"] is None
    assert body["citations"][0] == {
        "citation_id": 1,
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "file_name": "员工制度.pdf",
        "page": 12,
        "content": "员工应先在系统中提交加班申请。",
        "score": 0.98,
    }


@pytest.mark.asyncio
async def test_debug_trace_removes_preview_and_internal_error_detail(fake_chain):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/qa/ask",
            json={"question": "那怎么申请？", "thread_id": "debug", "debug": True},
        )

    trace = response.json()["trace"]
    assert "preview" not in trace["channels"]["vector:0"][0]
    assert "/private/secret/path" not in str(trace)
    assert trace["errors"] == ["vector:0: RuntimeError"]


@pytest.mark.asyncio
async def test_old_session_id_alias_and_input_validation(fake_chain):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        compatible = await client.post(
            "/api/v1/qa/ask",
            json={"question": "加班制度", "session_id": "old-client"},
        )
        invalid = await client.post(
            "/api/v1/qa/ask",
            json={"question": "   ", "thread_id": "demo"},
        )

    assert compatible.status_code == 200
    assert compatible.json()["thread_id"] == "old-client"
    assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_sse_stream_has_stable_lifecycle_events(fake_chain):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/qa/stream",
            json={"question": "那怎么申请？", "thread_id": "stream-001"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: started" in response.text
    assert "event: completed" in response.text
    assert "event: done" in response.text
    assert response.text.index("event: started") < response.text.index("event: completed")


@pytest.mark.asyncio
async def test_session_history_and_delete_contract(fake_chain):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        history = await client.get("/api/v1/sessions/demo-001/history")
        deleted = await client.delete("/api/v1/sessions/demo-001")

    assert history.status_code == 200
    assert history.json()["history"][0]["role"] == "human"
    assert deleted.json() == {"status": "deleted", "thread_id": "demo-001"}
    assert fake_chain.cleared == ["demo-001"]


@pytest.mark.asyncio
async def test_sync_failure_returns_public_503_without_internal_detail():
    class FailingChain(FakeRAGChain):
        def ask(self, question: str, session_id: str):
            raise RuntimeError("/private/secret/path provider payload")

    app.dependency_overrides[get_rag_chain_dep] = lambda: FailingChain()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/qa/ask",
                json={"question": "测试", "thread_id": "failure"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "问答服务暂时不可用"}
    assert "/private/secret/path" not in response.text
