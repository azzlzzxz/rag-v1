"""离线评测数据模型与输入校验。

评测集是代码之外的长期资产，因此在运行昂贵模型前先一次性校验所有字段。这样拼错
expected_decision 或重复 case id 时会立即失败，不会跑到一半才留下不可比较的报告。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CaseType = Literal[
    "direct",
    "semantic_rewrite",
    "numbered_term",
    "table",
    "multi_turn",
    "similar_policy",
    "no_answer",
    "ambiguous",
]
Decision = Literal["answer", "clarify", "refuse"]

ALLOWED_CASE_TYPES = {
    "direct",
    "semantic_rewrite",
    "numbered_term",
    "table",
    "multi_turn",
    "similar_policy",
    "no_answer",
    "ambiguous",
}
ALLOWED_DECISIONS = {"answer", "clarify", "refuse"}


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    type: CaseType
    query: str
    relevant_files: tuple[str, ...]
    expected_keywords: tuple[str, ...]
    expected_decision: Decision
    history: tuple[str, ...] = ()


def load_cases(path: Path) -> list[EvaluationCase]:
    """读取并严格校验评测集，错误信息包含具体用例位置。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("评测集必须是非空 JSON 数组")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条评测用例必须是 JSON 对象")
        case_id = str(item.get("id") or "").strip()
        case_type = str(item.get("type") or "").strip()
        query = str(item.get("query") or "").strip()
        decision = str(item.get("expected_decision") or "").strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"第 {index} 条用例 id 为空或重复: {case_id!r}")
        if case_type not in ALLOWED_CASE_TYPES:
            raise ValueError(f"用例 {case_id} 的 type 不受支持: {case_type!r}")
        if not query:
            raise ValueError(f"用例 {case_id} 的 query 不能为空")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"用例 {case_id} 的 expected_decision 不合法: {decision!r}")

        relevant_files = tuple(str(value).strip() for value in item.get("relevant_files", []))
        expected_keywords = tuple(
            str(value).strip() for value in item.get("expected_keywords", [])
        )
        history = tuple(str(value).strip() for value in item.get("history", []))
        if decision == "answer" and (not relevant_files or not expected_keywords):
            raise ValueError(
                f"回答型用例 {case_id} 必须同时提供 relevant_files 和 expected_keywords"
            )
        seen_ids.add(case_id)
        cases.append(
            EvaluationCase(
                id=case_id,
                type=case_type,  # type: ignore[arg-type]
                query=query,
                relevant_files=relevant_files,
                expected_keywords=expected_keywords,
                expected_decision=decision,  # type: ignore[arg-type]
                history=history,
            )
        )
    return cases
