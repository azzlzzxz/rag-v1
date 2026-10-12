import json
from pathlib import Path

import httpx
import pytest

from api.dependencies import get_evaluation_service_dep
from api.main import app
from config.settings import Settings
from evaluation.service import EvaluationService


def make_service(tmp_path: Path) -> EvaluationService:
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "unknown",
                    "type": "no_answer",
                    "query": "未知问题",
                    "relevant_files": [],
                    "expected_keywords": [],
                    "expected_decision": "refuse",
                }
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings(
        DATA_DIR=tmp_path / "data",
        VECTOR_DB_DIR=tmp_path / "vectors",
        MODELS_DIR=tmp_path / "models",
        UPLOAD_DIR=tmp_path / "uploads",
        INGESTION_STORAGE_DIR=tmp_path / "ingestion",
        CHECKPOINT_DB_PATH=tmp_path / "checkpoints.db",
        EVALUATION_DATASET_PATH=dataset,
        EVALUATION_REPORT_DIR=tmp_path / "reports",
        EVALUATION_JOB_DB_PATH=tmp_path / "evaluation-jobs.db",
        LOG_TO_FILE=False,
    )
    return EvaluationService(
        settings,
        executor=lambda **kwargs: {
            "schema_version": 1,
            "dataset": "cases.json",
            "cases": 1,
            "top_k": kwargs["top_k"],
            "mrr": 1.0,
        },
    )


@pytest.mark.asyncio
async def test_evaluation_api_run_status_report_and_metric_formulas(tmp_path: Path):
    service = make_service(tmp_path)
    app.dependency_overrides[get_evaluation_service_dep] = lambda: service
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            metadata = await client.get("/api/v1/evaluations/metadata")
            rejected = await client.post(
                "/api/v1/evaluations/runs",
                json={"mode": "all", "top_k": 5, "category": "demo"},
            )
            created = await client.post(
                "/api/v1/evaluations/runs",
                json={"mode": "retrieval", "top_k": 3, "category": "demo"},
            )
            run_id = created.json()["run_id"]
            status = await client.get(f"/api/v1/evaluations/runs/{run_id}")
            report = await client.get(f"/api/v1/evaluations/runs/{run_id}/report")
            latest = await client.get("/api/v1/evaluations/latest")
    finally:
        app.dependency_overrides.clear()

    assert metadata.status_code == 200
    assert metadata.json()["case_count"] == 1
    assert "active_run" in metadata.json()
    assert any(item["name"] == "MRR" and item["formula"] for item in metadata.json()["metrics"])
    assert rejected.status_code == 400
    assert created.status_code == 202
    assert status.json()["status"] == "succeeded"
    assert status.json()["report_ready"] is True
    assert report.json()["top_k"] == 3
    assert latest.json()["mrr"] == 1.0
