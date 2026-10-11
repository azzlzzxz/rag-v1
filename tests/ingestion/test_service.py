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


def test_delete_document_removes_storage_manifest_and_only_safe_upload(tmp_path: Path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.UPLOAD_DIR.mkdir(parents=True)
    pipeline = FakePipeline()
    service = IngestionService(settings, pipeline=pipeline)
    uploaded_file = settings.UPLOAD_DIR / "abc_policy.txt"
    uploaded_file.write_text("制度", encoding="utf-8")
    service._save_manifest(
        {
            "doc-1": {
                "source_ids": ["source-1"],
                "upload_path": str(uploaded_file),
            }
        }
    )

    class FakeCollection:
        def __init__(self):
            self.deleted = []

        def get(self, **kwargs):
            assert kwargs["where"] == {"document_id": "doc-1"}
            return {"ids": ["chunk-1", "chunk-2"]}

        def delete(self, ids):
            self.deleted.extend(ids)

    collection = FakeCollection()
    fake_client = SimpleNamespace(get_or_create_collection=lambda _name: collection)
    monkeypatch.setattr(
        "core.ingestion.service.chromadb.PersistentClient",
        lambda **kwargs: fake_client,
    )

    result = service.delete_document("doc-1")

    assert result.deleted is True
    assert result.deleted_chunks == 2
    assert collection.deleted == ["chunk-1", "chunk-2"]
    assert pipeline.docstore.deleted == ["source-1"]
    assert service._load_manifest() == {}
    assert uploaded_file.exists() is False


def test_delete_document_never_unlinks_path_outside_upload_dir(tmp_path: Path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.UPLOAD_DIR.mkdir(parents=True)
    outside_file = tmp_path / "must-stay.txt"
    outside_file.write_text("不能删除", encoding="utf-8")
    service = IngestionService(settings, pipeline=FakePipeline())
    service._save_manifest(
        {"doc-outside": {"source_ids": [], "upload_path": str(outside_file)}}
    )

    collection = SimpleNamespace(get=lambda **kwargs: {"ids": []})
    fake_client = SimpleNamespace(get_or_create_collection=lambda _name: collection)
    monkeypatch.setattr(
        "core.ingestion.service.chromadb.PersistentClient",
        lambda **kwargs: fake_client,
    )

    result = service.delete_document("doc-outside")

    assert result.deleted is True
    assert outside_file.read_text(encoding="utf-8") == "不能删除"
