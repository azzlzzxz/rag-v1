from pathlib import Path

from core.ingestion.jobs import IngestionJobRepository


def test_job_lifecycle_and_error_sanitization(tmp_path: Path):
    repository = IngestionJobRepository(tmp_path / "jobs.db")
    assert repository.create("file-1", "policy.txt").status == "pending"
    assert repository.update("file-1", "running").status == "running"

    failed = repository.update("file-1", "failed", error="x" * 1000)

    assert failed.status == "failed"
    assert len(failed.error or "") == 500
