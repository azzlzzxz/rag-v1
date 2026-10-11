"""文件上传 API：保存原文件并提交 LlamaIndex 后台摄取任务。"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.dependencies import get_ingestion_service_dep
from config.settings import settings
from core.ingestion import IngestionService
from core.ingestion.ids import build_document_id
from core.ingestion.readers import EnterpriseDocumentReader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["文件上传"])


def _save_upload(file: UploadFile) -> tuple[str, str, Path]:
    original_filename = Path(file.filename or "").name
    file_ext = Path(original_filename).suffix.lower()
    if file_ext not in EnterpriseDocumentReader.SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {file_ext}")
    file_id = uuid.uuid4().hex[:8]
    save_path = settings.UPLOAD_DIR / f"{file_id}_{original_filename}"
    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.exception("文件保存失败: %s", original_filename)
        raise HTTPException(500, "文件保存失败") from exc
    return file_id, original_filename, save_path


@router.post("/file", status_code=202)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general"),
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
):
    file_id, original_filename, save_path = _save_upload(file)
    # 先持久化 pending，再把任务交给后台；即使进程中途异常，客户端也能查询到任务记录。
    ingestion.jobs.create(file_id, original_filename)
    background_tasks.add_task(
        ingestion.process_job, file_id, str(save_path), category, original_filename
    )
    return JSONResponse(
        status_code=202,
        content={
            "status": "pending",
            "message": f"文件 {original_filename} 上传成功，正在后台处理",
            "file_id": file_id,
            "document_id": build_document_id(category, original_filename),
            "filename": save_path.name,
        },
    )


@router.get("/status/{file_id}")
async def get_upload_status(
    file_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
):
    job = ingestion.jobs.get(file_id)
    if job is None:
        raise HTTPException(404, "上传任务不存在")
    return job.to_dict()


@router.post("/batch", status_code=202)
async def upload_batch_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    category: str = Form("general"),
    ingestion: IngestionService = Depends(get_ingestion_service_dep),
):
    results = []
    for file in files:
        try:
            file_id, original_filename, save_path = _save_upload(file)
            ingestion.jobs.create(file_id, original_filename)
            background_tasks.add_task(
                ingestion.process_job, file_id, str(save_path), category, original_filename
            )
            results.append({"filename": original_filename, "status": "pending", "file_id": file_id})
        except HTTPException as exc:
            results.append({"filename": file.filename, "status": "skipped", "reason": exc.detail})
        except Exception:
            logger.exception("批量上传保存失败: %s", file.filename)
            results.append({"filename": file.filename, "status": "error", "reason": "文件保存失败"})
    return JSONResponse(status_code=202, content={"results": results})
