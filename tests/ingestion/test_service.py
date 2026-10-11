from pathlib import Path
from types import SimpleNamespace

from config.settings import Settings
from core.ingestion.service import IngestionService


class FakeStore:
    def __init__(self):
        self.deleted = []
        self.persisted = 0

    def delete_ref_doc(self, source_id, raise_error=False):
        self.deleted.append(source_id)

    def delete(self, source_id):
        self.deleted.append(source_id)

    def persist(self, _path):
        self.persisted += 1


class FakePipeline:
    def __init__(self):
        self.docstore = FakeStore()
        self.vector_store = FakeStore()
        self.cache = FakeStore()
        self.calls = 0

    def run(self, *, documents, show_progress):
        self.calls += 1
        return [SimpleNamespace(node_id=f"node-{index}") for index, _ in enumerate(documents)]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        DATA_DIR=tmp_path / "data",
        VECTOR_DB_DIR=tmp_path / "vectors",
        MODELS_DIR=tmp_path / "models",
        UPLOAD_DIR=tmp_path / "uploads",
        INGESTION_STORAGE_DIR=tmp_path / "data" / "ingestion",
        LOG_TO_FILE=False,
    )


def test_identical_file_is_skipped_before_pipeline_runs_again(tmp_path: Path):
    pipeline = FakePipeline()
    service = IngestionService(make_settings(tmp_path), pipeline=pipeline)
    file_path = tmp_path / "policy.txt"
    file_path.write_text("第一版制度", encoding="utf-8")

    first = service.ingest_file(file_path, category="hr")
    second = service.ingest_file(file_path, category="hr")

    assert first.skipped is False
    assert second.skipped is True
    assert second.node_count == 0
    assert pipeline.calls == 1


def test_changed_file_is_ingested_as_update(tmp_path: Path):
    pipeline = FakePipeline()
    service = IngestionService(make_settings(tmp_path), pipeline=pipeline)
    file_path = tmp_path / "policy.txt"
    file_path.write_text("第一版制度", encoding="utf-8")
    service.ingest_file(file_path, category="hr")

    file_path.write_text("第二版制度", encoding="utf-8")
    updated = service.ingest_file(file_path, category="hr")

    assert updated.skipped is False
    assert pipeline.calls == 2
