"""阶段 5 稳定 HTTP 契约的轻量客户端。

把所有网络调用集中在这里有两个好处：页面代码只负责展示；后端返回错误时也能统一转成
适合最终用户阅读的中文消息，而不会把 httpx 异常栈显示在 Streamlit 页面上。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class APIClientError(RuntimeError):
    """可安全显示在前端的 API 调用异常。"""


class RAGAPIClient:
    """只依赖公开 API 的前端客户端，便于单测和未来替换 UI 技术。"""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """执行普通 JSON 请求，并把服务端 detail 提取为用户可读错误。"""
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise APIClientError(str(detail or f"API 返回 {exc.response.status_code}")) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise APIClientError(f"无法连接 API：{exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/system/health")

    def knowledge_stats(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/knowledge/stats")

    def list_documents(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/documents")

    def upload_document(self, name: str, content: bytes, category: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/documents/upload",
            files={"file": (name, content)},
            data={"category": category},
        )

    def document_status(self, file_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/documents/{file_id}/status")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/documents/{document_id}")

    def ask(self, question: str, thread_id: str, *, debug: bool = False) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/qa/ask",
            json={"question": question, "thread_id": thread_id, "debug": debug},
        )

    def history(self, thread_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/sessions/{thread_id}/history")

    def delete_session(self, thread_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/sessions/{thread_id}")

    def stream_answer(
        self, question: str, thread_id: str, *, debug: bool = False
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """逐帧解析 SSE。

        当前后端流的是 started/completed/done 生命周期；以后新增 token 事件，这个解析器也
        会原样向页面透传，不需要改网络层。
        """
        try:
            with self._client.stream(
                "POST",
                "/api/v1/qa/stream",
                json={"question": question, "thread_id": thread_id, "debug": debug},
            ) as response:
                response.raise_for_status()
                event_name = "message"
                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line:
                        event_name = "message"
                    elif line.startswith("event:"):
                        event_name = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        payload = json.loads(line.split(":", 1)[1].strip())
                        yield event_name, payload
        except httpx.HTTPStatusError as exc:
            raise APIClientError(f"流式接口返回 {exc.response.status_code}") from exc
        except (httpx.RequestError, json.JSONDecodeError) as exc:
            raise APIClientError(f"流式回答中断：{exc}") from exc
