"""稳定 HTTP 契约使用的 Pydantic Schema。"""

from api.schemas.documents import (
    DocumentDeleteResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from api.schemas.qa import (
    AnswerResponse,
    CitationResponse,
    HistoryMessage,
    HistoryResponse,
    QuestionRequest,
)

__all__ = [
    "AnswerResponse",
    "CitationResponse",
    "DocumentDeleteResponse",
    "DocumentStatusResponse",
    "DocumentUploadResponse",
    "HistoryMessage",
    "HistoryResponse",
    "QuestionRequest",
]
