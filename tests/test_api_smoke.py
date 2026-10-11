import httpx
import pytest

from api.main import app


@pytest.mark.asyncio
async def test_public_health_routes_are_reachable():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        root = await client.get("/")
        health = await client.get("/api/v1/system/health")
        ready = await client.get("/api/v1/system/ready")

    assert root.status_code == 200
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_app_metadata_comes_from_settings():
    assert app.title
    assert app.version
