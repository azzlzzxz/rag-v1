from .upload import router as upload_router
from .qa import router as qa_router
from .knowledge import router as knowledge_router
from .system import router as system_router
from .documents import router as documents_router
from .sessions import router as sessions_router
from .evaluations import router as evaluations_router

__all__ = [
    "documents_router",
    "evaluations_router",
    "knowledge_router",
    "qa_router",
    "sessions_router",
    "system_router",
    "upload_router",
]
