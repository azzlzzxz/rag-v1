"""
FastAPI依赖注入模块
"""
from core.rag_chain import get_rag_chain
from core.ingestion import IngestionService, get_ingestion_service

def get_rag_chain_dep():
    """获取RAG链依赖"""
    return get_rag_chain()

def get_ingestion_service_dep() -> IngestionService:
    """获取 LlamaIndex 摄取服务；Embedding 会在后台任务中延迟初始化。"""
    return get_ingestion_service()
