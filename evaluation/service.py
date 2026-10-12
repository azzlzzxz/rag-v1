"""FastAPI 页面评测的任务服务。

服务只负责任务生命周期、并发保护和报告文件；指标计算继续复用 evaluation.run，保证页面
和 CLI 没有两套实现。
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from config.settings import Settings, settings
from evaluation.jobs import EvaluationJob, EvaluationJobRepository
from evaluation.models import load_cases
from evaluation.run import execute_evaluation, write_report

logger = logging.getLogger(__name__)
EvaluationExecutor = Callable[..., dict[str, Any]]


class EvaluationBusyError(RuntimeError):
    """已有评测占用模型资源。"""


class EvaluationService:
    def __init__(
        self,
        app_settings: Settings = settings,
        *,
        executor: EvaluationExecutor = execute_evaluation,
    ) -> None:
        self.settings = app_settings
        self.executor = executor
        self.jobs = EvaluationJobRepository(app_settings.EVALUATION_JOB_DB_PATH)
        recovered = self.jobs.recover_interrupted()
        if recovered:
            logger.warning("已恢复 %s 个被进程重启中断的评测任务", recovered)
        self._create_lock = threading.Lock()

    def create_run(self, mode: str, top_k: int, category: str) -> EvaluationJob:
        """原子检查 active 任务并创建新任务，避免页面重复点击启动多套模型。"""
        with self._create_lock:
            active = self.jobs.active()
            if active is not None:
                raise EvaluationBusyError(active.run_id)
            return self.jobs.create(uuid.uuid4().hex, mode, top_k, category)

    def process_run(self, run_id: str) -> None:
        """在 FastAPI BackgroundTasks 中执行；详细异常只写服务端日志。"""
        job = self.jobs.get(run_id)
        if job is None:
            raise KeyError(run_id)
        self.jobs.update(run_id, "running")
        try:
            report = self.executor(
                mode=job.mode,
                dataset=self.settings.EVALUATION_DATASET_PATH,
                top_k=job.top_k,
                category=job.category or None,
            )
            report_name = f"{run_id}.json"
            run_path = self.settings.EVALUATION_REPORT_DIR / report_name
            latest_path = self.settings.EVALUATION_REPORT_DIR / "latest.json"
            write_report(report, run_path)
            write_report(report, latest_path)
            self.jobs.update(run_id, "succeeded", report_name=report_name)
        except Exception as exc:
            logger.exception("评测任务失败: run_id=%s", run_id)
            self.jobs.update(run_id, "failed", error=type(exc).__name__)

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(run_id)
        if job is None or job.status != "succeeded" or not job.report_name:
            return None
        # report_name 只能由服务端生成；仍取 name 进一步阻断数据库异常值带来的路径越界。
        path = self.settings.EVALUATION_REPORT_DIR / Path(job.report_name).name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def get_latest_report(self) -> dict[str, Any] | None:
        path = self.settings.EVALUATION_REPORT_DIR / "latest.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def metadata(self) -> dict[str, Any]:
        cases = load_cases(self.settings.EVALUATION_DATASET_PATH)
        active_job = self.jobs.active()
        return {
            "dataset": self.settings.EVALUATION_DATASET_PATH.name,
            "case_count": len(cases),
            "case_types": sorted({case.type for case in cases}),
            "profiles": ["vector_only", "vector_rerank", "hybrid_rerank"],
            # 仅暴露 pending/running 任务；已完成结果统一从 latest 接口读取。
            "active_run": active_job.to_dict() if active_job else None,
        }


_evaluation_service: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service
