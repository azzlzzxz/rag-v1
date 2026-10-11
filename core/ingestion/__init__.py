"""LlamaIndex 文档摄取子系统。"""

from core.ingestion.jobs import IngestionJob, IngestionJobRepository
from core.ingestion.service import (
    DeletionResult,
    IngestionResult,
    IngestionService,
    get_ingestion_service,
)

__all__ = [
    "IngestionJob",
    "IngestionJobRepository",
    "DeletionResult",
    "IngestionResult",
    "IngestionService",
    "get_ingestion_service",
]
