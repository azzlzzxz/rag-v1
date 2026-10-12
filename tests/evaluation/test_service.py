import json
from pathlib import Path

import pytest

from config.settings import Settings
from evaluation.service import EvaluationBusyError, EvaluationService


def make_settings(tmp_path: Path) -> Settings:
    dataset = tmp_path / "evaluation" / "cases.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "no-answer",
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
    return Settings(
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


def test_evaluation_service_persists_job_and_reports(tmp_path: Path):
    settings = make_settings(tmp_path)
    calls = []

    def executor(**kwargs):
        calls.append(kwargs)
        return {"schema_version": 1, "cases": 1, "mrr": 1.0}

    service = EvaluationService(settings, executor=executor)
    job = service.create_run("retrieval", 5, "demo")
    with pytest.raises(EvaluationBusyError):
        service.create_run("retrieval", 5, "demo")

    service.process_run(job.run_id)

    completed = service.jobs.get(job.run_id)
    assert completed is not None and completed.status == "succeeded"
    assert calls[0]["dataset"] == settings.EVALUATION_DATASET_PATH
    assert service.get_report(job.run_id)["mrr"] == 1.0
    assert service.get_latest_report()["cases"] == 1
    assert service.metadata()["case_count"] == 1


def test_service_recovers_interrupted_job_after_restart(tmp_path: Path):
    settings = make_settings(tmp_path)
    first = EvaluationService(settings, executor=lambda **kwargs: {})
    job = first.create_run("retrieval", 5, "demo")

    EvaluationService(settings, executor=lambda **kwargs: {})

    recovered = first.jobs.get(job.run_id)
    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.error == "ProcessInterrupted"
