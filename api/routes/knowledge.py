"""知识库只读统计 API。

文档删除已统一迁移到 ``DELETE /documents/{document_id}``，不再接受不安全的 source_path。
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_ingestion_service_dep
from core.ingestion import IngestionService

router = APIRouter(prefix="/knowledge", tags=["知识库"])


@router.get("/stats")
async def get_knowledge_stats(
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
) -> dict:
    """返回 manifest 文档数、Chroma chunk 数和分类分布。"""
    return ingestion.get_knowledge_stats()
