from pathlib import Path

from core.document_loader import DocumentLoader
from core.vector_store import VectorStoreManager


def test_document_loader_merges_upload_metadata(tmp_path: Path):
    source = tmp_path / "sample.txt"
    source.write_text("第一条制度。第二条制度。", encoding="utf-8")

    documents = DocumentLoader(chunk_size=10, chunk_overlap=0).load_file(
        str(source), {"category": "hr", "original_filename": "制度.txt"}
    )

    assert documents
    assert all(document.metadata["category"] == "hr" for document in documents)
    assert all(document.metadata["original_filename"] == "制度.txt" for document in documents)
    assert all(document.metadata["file_name"] == "sample.txt" for document in documents)


def test_vector_store_exposes_public_api_contracts():
    assert callable(VectorStoreManager.delete_by_source)
    assert callable(VectorStoreManager.get_collection_stats)
