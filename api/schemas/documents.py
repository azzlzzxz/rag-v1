"""文档上传、任务状态和删除 API 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    file_id: str
    document_id: str
    file_name: str
    category: str
    status: Literal["pending"]


class DocumentStatusResponse(BaseModel):
    file_id: str
    file_name: str
    status: Literal["pending", "running", "succeeded", "failed"]
    node_count: int
    skipped: bool
    error: str | None
    created_at: str
    updated_at: str


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    deleted_chunks: int
