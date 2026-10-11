from pathlib import Path

import httpx
import pytest

from api.dependencies import get_ingestion_service_dep
from api.main import app
from core.ingestion.jobs import IngestionJobRepository


class FakeIngestionService:
    def __init__(self, path: Path):
        self.jobs = IngestionJobRepository(path / "jobs.db")


@pytest.mark.asyncio
async def test_status_endpoint_returns_job_and_404(tmp_path: Path):
    fake = FakeIngestionService(tmp_path)
    fake.jobs.create("known", "policy.txt")
    app.dependency_overrides[get_ingestion_service_dep] = lambda: fake
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            known = await client.get("/api/v1/upload/status/known")
            missing = await client.get("/api/v1/upload/status/missing")
    finally:
        app.dependency_overrides.clear()

    assert known.status_code == 200
    assert known.json()["status"] == "pending"
    assert missing.status_code == 404
