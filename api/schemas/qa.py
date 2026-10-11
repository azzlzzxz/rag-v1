"""问答与会话 API 的稳定数据契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class QuestionRequest(BaseModel):
    """问答请求。

    HTTP 新契约使用 ``thread_id``；validation_alias 暂时接受旧 ``session_id``，使旧前端可以
    平滑升级。OpenAPI 和返回结果只展示 thread_id。
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("thread_id", "session_id"),
    )
    debug: bool = False

    @field_validator("question", "thread_id")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空")
        return value


class CitationResponse(BaseModel):
    """回答中可稳定定位到原文 chunk 的引用。"""

    citation_id: int = Field(ge=1)
    document_id: str | None = None
    chunk_id: str
    file_name: str
    page: int | None = None
    content: str
    score: float | None = None


class AnswerResponse(BaseModel):
    answer: str
    decision: Literal["answer", "clarify", "refuse"]
    thread_id: str
    original_query: str
    rewritten_query: str
    rewrite_applied: bool
    citations: list[CitationResponse]
    # 普通请求必须为 null；仅 debug=true 时返回经过脱敏的检索轨迹。
    trace: dict[str, Any] | None = None


class HistoryMessage(BaseModel):
    role: Literal["human", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    history: list[HistoryMessage]
