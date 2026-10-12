"""页面评测任务、报告和指标说明 API。"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.dependencies import get_evaluation_service_dep
from api.schemas.evaluations import (
    EvaluationMetadataResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationStatusResponse,
)
from evaluation.metrics import METRIC_DEFINITIONS
from evaluation.service import EvaluationBusyError, EvaluationService

router = APIRouter(prefix="/evaluations", tags=["评测"])


def _run_response(job) -> EvaluationRunResponse:
    return EvaluationRunResponse(
        run_id=job.run_id,
        mode=job.mode,
        top_k=job.top_k,
        category=job.category,
        status=job.status,
        created_at=job.created_at,
    )


@router.get("/metadata", response_model=EvaluationMetadataResponse)
async def evaluation_metadata(
    service: EvaluationService = Depends(get_evaluation_service_dep),
) -> EvaluationMetadataResponse:
    """返回数据集规模、实验 Profile 和页面展示所需的统一指标公式。"""
    return EvaluationMetadataResponse(**service.metadata(), metrics=METRIC_DEFINITIONS)


@router.post("/runs", response_model=EvaluationRunResponse, status_code=202)
async def create_evaluation_run(
    request: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    service: EvaluationService = Depends(get_evaluation_service_dep),
) -> EvaluationRunResponse:
    """创建后台评测；会调用 LLM 的模式必须由页面显式确认费用。"""
    if request.mode in {"qa", "all"} and not request.confirm_llm_cost:
        raise HTTPException(400, "完整评测会调用 LLM，请先确认可能产生费用")
    try:
        job = service.create_run(request.mode, request.top_k, request.category)
    except EvaluationBusyError as exc:
        raise HTTPException(409, f"已有评测任务正在运行: {exc}") from exc
    background_tasks.add_task(service.process_run, job.run_id)
    return _run_response(job)


@router.get("/runs/{run_id}", response_model=EvaluationStatusResponse)
async def get_evaluation_status(
    run_id: str,
    service: EvaluationService = Depends(get_evaluation_service_dep),
) -> EvaluationStatusResponse:
    job = service.jobs.get(run_id)
    if job is None:
        raise HTTPException(404, "评测任务不存在")
    # 内部异常类型保存在 SQLite/日志，页面只显示统一文案，避免泄漏模型或本地路径信息。
    public_error = "评测执行失败，请查看 API 日志" if job.status == "failed" else None
    return EvaluationStatusResponse(
        **_run_response(job).model_dump(),
        updated_at=job.updated_at,
        report_ready=job.status == "succeeded" and bool(job.report_name),
        error=public_error,
    )


@router.get("/runs/{run_id}/report")
async def get_evaluation_report(
    run_id: str,
    service: EvaluationService = Depends(get_evaluation_service_dep),
) -> dict:
    report = service.get_report(run_id)
    if report is None:
        raise HTTPException(404, "评测报告尚未生成")
    return report


@router.get("/latest")
async def get_latest_evaluation_report(
    service: EvaluationService = Depends(get_evaluation_service_dep),
) -> dict:
    report = service.get_latest_report()
    if report is None:
        raise HTTPException(404, "尚无评测报告")
    return report
