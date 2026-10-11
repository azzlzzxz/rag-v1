"""带约束保护的 Query Rewrite。

LLM 负责理解上下文并输出结构化结果；确定性的校验器负责阻止数字、时间、否定词和显式实体
在改写时丢失。两层职责分开，避免完全相信模型的自我声明。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class RewriteResult(BaseModel):
    """LLM 必须返回的结构化改写结果。"""

    rewritten_query: str = Field(description="可独立理解的完整问题")
    rewrite_applied: bool = Field(description="是否真的修改了原问题")
    rewrite_reason: str = Field(description="改写或保持原样的简短原因")


_CONTEXT_MARKERS = (
    "这个",
    "那个",
    "这些",
    "那些",
    "它",
    "他们",
    "上述",
    "前面",
    "刚才",
    "那",
    "那怎么",
    "那需要",
    "怎么办",
    "呢",
)
_NEGATIONS = ("不", "不能", "不得", "不要", "无需", "未", "没有", "禁止", "严禁", "除外")
_EXPLICIT_TOPICS = (
    "制度",
    "政策",
    "流程",
    "规定",
    "办法",
    "病假",
    "事假",
    "年假",
    "婚假",
    "产假",
    "调休",
    "加班",
    "报销",
    "发票",
)
_GENERIC_QUESTIONS = ("怎么", "如何", "多少", "多久", "谁", "需要什么", "可以吗")


def needs_rewrite(query: str, history: Sequence[AnyMessage]) -> bool:
    """仅对确实依赖上下文的问题调用 LLM，独立问题保持原样。"""
    if not history:
        return False
    stripped = query.strip()
    if any(marker in stripped for marker in _CONTEXT_MARKERS):
        return True
    # “怎么申请？”虽然没有代词，但缺少申请对象；“病假制度是什么？”已有明确主题，不改写。
    has_explicit_topic = any(topic in stripped for topic in _EXPLICIT_TOPICS)
    return (
        not has_explicit_topic
        and len(stripped) <= 10
        and any(marker in stripped for marker in _GENERIC_QUESTIONS)
    )


def extract_protected_constraints(query: str) -> dict[str, list[str]]:
    """提取改写时不允许丢失的显式约束。

    这里不是通用 NER，而是针对 RAG 查询漂移风险做的保守保护：数字与单位、时间表达、
    否定约束、制度编号，以及书名号/引号/常见业务后缀标记出的实体。
    """
    numbers = re.findall(r"\d+(?:\.\d+)?(?:天|小时|分钟|年|月|日|次|个|%|％)?", query)
    times = re.findall(
        r"(?:今天|明天|昨天|本周|下周|上周|本月|下月|今年|明年|去年|"
        r"\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?)",
        query,
    )
    negations = [token for token in _NEGATIONS if token in query]
    identifiers = re.findall(r"(?i)\b(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9_-]+\b", query)
    quoted = re.findall(r"《([^》]+)》|[“\"]([^”\"]+)[”\"]", query)
    quoted_entities = [left or right for left, right in quoted]
    suffixed_entities = re.findall(
        r"[\u4e00-\u9fff]{2,12}(?:制度|政策|流程|规定|办法|部门|产品|项目)", query
    )
    return {
        "numbers": list(dict.fromkeys(numbers)),
        "times": list(dict.fromkeys(times)),
        "negations": list(dict.fromkeys(negations)),
        "identifiers": list(dict.fromkeys(identifiers)),
        "entities": list(dict.fromkeys([*quoted_entities, *suffixed_entities])),
    }


def missing_constraints(candidate: str, constraints: dict[str, list[str]]) -> list[str]:
    """返回改写文本中丢失的约束，调用方据此回退到原问题。"""
    return [
        value
        for values in constraints.values()
        for value in values
        if value and value.lower() not in candidate.lower()
    ]


class QueryRewriter:
    """使用结构化输出改写 query，并在输出后执行确定性一致性校验。"""

    def __init__(self, llm, *, max_retries: int = 1) -> None:
        self._structured_llm = llm.with_structured_output(RewriteResult)
        self.max_retries = max_retries

    def rewrite(
        self,
        original_query: str,
        history: Sequence[AnyMessage],
    ) -> tuple[RewriteResult, dict[str, list[str]], list[str]]:
        constraints = extract_protected_constraints(original_query)
        if not needs_rewrite(original_query, history):
            return (
                RewriteResult(
                    rewritten_query=original_query,
                    rewrite_applied=False,
                    rewrite_reason="问题可以独立理解，无需改写",
                ),
                constraints,
                [],
            )

        prompt = [
            SystemMessage(
                content=(
                    "你是企业知识库的 query 改写器。根据历史对话把当前追问改写成可独立理解的"
                    "问题。不得新增事实，不得删除当前问题中的实体、数字、时间和否定约束。"
                    "如果无需改写，原样返回。"
                )
            ),
            *history,
            HumanMessage(content=original_query),
        ]
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            try:
                raw_result = self._structured_llm.invoke(prompt)
                result = (
                    raw_result
                    if isinstance(raw_result, RewriteResult)
                    else RewriteResult.model_validate(raw_result)
                )
                candidate = result.rewritten_query.strip()
                missing = missing_constraints(candidate, constraints)
                if not candidate:
                    raise ValueError("改写结果为空")
                if missing:
                    return (
                        RewriteResult(
                            rewritten_query=original_query,
                            rewrite_applied=False,
                            rewrite_reason=f"改写丢失约束，已回退原问题：{', '.join(missing)}",
                        ),
                        constraints,
                        errors,
                    )
                result.rewritten_query = candidate
                result.rewrite_applied = candidate != original_query
                return result, constraints, errors
            except Exception as exc:  # LLM/结构化解析错误按同一类别重试
                errors.append(f"rewrite attempt {attempt + 1}: {type(exc).__name__}: {exc}")

        return (
            RewriteResult(
                rewritten_query=original_query,
                rewrite_applied=False,
                rewrite_reason="改写失败，已安全回退原问题",
            ),
            constraints,
            errors,
        )
