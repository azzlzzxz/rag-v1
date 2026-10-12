"""LlamaIndex 文档摄取子系统。"""

from core.ingestion.jobs import IngestionJob, IngestionJobRepository
from core.ingestion.service import (
    DeletionResult,
    DocumentRecord,
    IngestionResult,
    IngestionService,
    get_ingestion_service,
)

__all__ = [
    "IngestionJob",
    "IngestionJobRepository",
    "DeletionResult",
    "DocumentRecord",
    "IngestionResult",
    "IngestionService",
    "get_ingestion_service",
]
