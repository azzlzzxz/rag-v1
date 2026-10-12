"""页面评测任务的稳定 API 契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EvaluationRunRequest(BaseModel):
    mode: Literal["retrieval", "qa", "all"] = "retrieval"
    top_k: int = Field(default=5, ge=1, le=20)
    category: str = Field(default="demo", max_length=64)
    confirm_llm_cost: bool = False

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        return value.strip()


class EvaluationRunResponse(BaseModel):
    run_id: str
    mode: str
    top_k: int
    category: str
    status: Literal["pending", "running", "succeeded", "failed"]
    created_at: str


class EvaluationStatusResponse(EvaluationRunResponse):
    updated_at: str
    report_ready: bool
    error: str | None


class EvaluationMetadataResponse(BaseModel):
    dataset: str
    case_count: int
    case_types: list[str]
    profiles: list[str]
    metrics: list[dict[str, Any]]
    # 页面重新连接后可借此找回尚未完成的任务，不必只依赖 Streamlit 会话状态。
    active_run: dict[str, Any] | None = None
