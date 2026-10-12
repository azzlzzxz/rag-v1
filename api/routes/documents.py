"""阶段 5 稳定文档 API。

旧 ``/upload`` 路径继续兼容；新调用方统一使用 ``/documents``。
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_ingestion_service_dep
from api.routes.upload import _save_upload
from api.schemas.documents import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from core.ingestion import IngestionService
from core.ingestion.ids import build_document_id

router = APIRouter(prefix="/documents", tags=["文档"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
) -> DocumentListResponse:
    """返回知识库文档摘要，供 Streamlit 刷新后恢复列表。"""
    records = ingestion.list_documents()
    documents = [DocumentSummary(**record.__dict__) for record in records]
    return DocumentListResponse(total=len(documents), documents=documents)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general"),
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
) -> DocumentUploadResponse:
    """保存文件并提交后台摄取，立即返回任务 ID 和稳定 document_id。"""
    file_id, original_filename, save_path = _save_upload(file)
    document_id = build_document_id(category, original_filename)
    ingestion.jobs.create(file_id, original_filename)
    background_tasks.add_task(
        ingestion.process_job,
        file_id,
        str(save_path),
        category,
        original_filename,
    )
    return DocumentUploadResponse(
        file_id=file_id,
        document_id=document_id,
        file_name=original_filename,
        category=category,
        status="pending",
    )


@router.get("/{file_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    file_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
) -> DocumentStatusResponse:
    """查询后台摄取任务；路径 ID 是 upload 返回的 file_id。"""
    job = ingestion.jobs.get(file_id)
    if job is None:
        raise HTTPException(404, "文档任务不存在")
    # 详细异常只进入服务端日志/SQLite；公开接口避免泄漏本地路径或第三方响应正文。
    public_error = "文档处理失败，请检查文件内容或联系管理员" if job.error else None
    return DocumentStatusResponse(
        file_id=job.file_id,
        file_name=job.filename,
        status=job.status,
        node_count=job.node_count,
        skipped=job.skipped,
        error=public_error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
) -> DocumentDeleteResponse:
    """按 upload 返回的稳定 document_id 删除整篇文档。"""
    result = ingestion.delete_document(document_id)
    if not result.deleted:
        raise HTTPException(404, "文档不存在")
    return DocumentDeleteResponse(
        document_id=result.document_id,
        deleted=result.deleted,
        deleted_chunks=result.deleted_chunks,
    )
