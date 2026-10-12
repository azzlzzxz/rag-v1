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


class DocumentSummary(BaseModel):
    """知识库页面展示所需的文档摘要，不暴露服务器本地路径和文件 hash。"""

    document_id: str
    file_name: str
    category: str
    source_count: int
    ingested_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    """稳定的文档列表契约；未来增加分页字段时无需改变顶层类型。"""

    total: int
    documents: list[DocumentSummary]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    deleted_chunks: int
