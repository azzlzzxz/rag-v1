from pathlib import Path

from llama_index.core.schema import NodeWithScore, TextNode

from config.settings import Settings
from core.retrieval.hybrid import HybridRetrievalService
from core.retriever import RAGRetriever


def test_old_retriever_contract_returns_langchain_documents(tmp_path: Path):
    item = TextNode(id_="node-1", text="事假最多十五天", metadata={"file_name": "制度.txt"})
    settings = Settings(
        DATA_DIR=tmp_path / "data",
        VECTOR_DB_DIR=tmp_path / "vectors",
        MODELS_DIR=tmp_path / "models",
        UPLOAD_DIR=tmp_path / "uploads",
        INGESTION_STORAGE_DIR=tmp_path / "data" / "ingestion",
        USE_RERANKER=False,
        LOG_TO_FILE=False,
    )
    service = HybridRetrievalService(
        settings,
        vector_search=lambda query, top_k, filters: [NodeWithScore(node=item, score=0.9)],
        corpus_loader=lambda: [item],
    )
    retriever = RAGRetriever(service)

    documents = retriever.retrieve("事假", top_k=1)
    chain_retriever = retriever.get_compression_retriever({"k": 1})

    assert documents[0].page_content == "事假最多十五天"
    assert "retrieval_score" in documents[0].metadata
    assert chain_retriever.invoke("事假")[0].page_content == "事假最多十五天"
