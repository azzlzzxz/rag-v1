from pathlib import Path

import httpx
import pytest

from api.dependencies import get_ingestion_service_dep
from api.main import app
from core.ingestion import DeletionResult
from core.ingestion.jobs import IngestionJobRepository


class FakeIngestionService:
    def __init__(self, root: Path):
        self.jobs = IngestionJobRepository(root / "jobs.db")
        self.deleted_ids: list[str] = []

    def process_job(self, file_id, file_path, category, original_filename):
        self.jobs.update(file_id, "succeeded", node_count=2)

    def delete_document(self, document_id):
        self.deleted_ids.append(document_id)
        if document_id == "missing":
            return DeletionResult(document_id, False, 0)
        return DeletionResult(document_id, True, 2)

    def get_knowledge_stats(self):
        return {"document_count": 1, "node_count": 2, "categories": {"hr": 1}}


@pytest.mark.asyncio
async def test_document_upload_status_and_delete_contract(tmp_path: Path, monkeypatch):
    ingestion = FakeIngestionService(tmp_path)
    ingestion.jobs.create("failed-job", "broken.pdf")
    ingestion.jobs.update(
        "failed-job",
        "failed",
        error="RuntimeError: /private/secret/path provider response",
    )
    monkeypatch.setattr("api.routes.upload.settings.UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()
    app.dependency_overrides[get_ingestion_service_dep] = lambda: ingestion
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            uploaded = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("policy.txt", b"policy content", "text/plain")},
                data={"category": "hr"},
            )
            upload_body = uploaded.json()
            status = await client.get(
                f"/api/v1/documents/{upload_body['file_id']}/status"
            )
            failed_status = await client.get("/api/v1/documents/failed-job/status")
            deleted = await client.delete(
                f"/api/v1/documents/{upload_body['document_id']}"
            )
            missing = await client.delete("/api/v1/documents/missing")
            stats = await client.get("/api/v1/knowledge/stats")
    finally:
        app.dependency_overrides.clear()

    assert uploaded.status_code == 202
    assert upload_body["file_name"] == "policy.txt"
    assert upload_body["category"] == "hr"
    assert status.json()["status"] == "succeeded"
    assert status.json()["node_count"] == 2
    assert failed_status.json()["error"] == "文档处理失败，请检查文件内容或联系管理员"
    assert "/private/secret/path" not in failed_status.text
    assert deleted.json()["deleted_chunks"] == 2
    assert missing.status_code == 404
    assert stats.json() == {
        "document_count": 1,
        "node_count": 2,
        "categories": {"hr": 1},
    }
