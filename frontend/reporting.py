"""评测页的只读报告加载器。

评测计算始终在 ``evaluation`` 包中；本模块只校验并读取 JSON，避免 Streamlit rerun 时
重复调用 Retriever/LLM 或复制指标公式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any] | None:
    """报告不存在时返回 None；存在但格式错误时抛出明确异常。"""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("评测报告顶层必须是 JSON 对象")
    return data


METRIC_LABELS = {
    "recall@5": "Recall@5",
    "mrr": "MRR",
    "source_hit_rate": "来源命中率",
    "decision_accuracy": "决策准确率",
    "citation_accuracy": "引用正确率",
    "keyword_coverage": "关键词覆盖率",
    "refusal_accuracy": "拒答准确率",
    "clarify_accuracy": "澄清准确率",
    "avg_retrieval_ms": "平均检索耗时",
    "p95_retrieval_ms": "P95 检索耗时",
    "avg_generation_ms": "平均生成耗时",
    "p95_generation_ms": "P95 生成耗时",
}


def metric_items(report: dict[str, Any]) -> list[tuple[str, float]]:
    """按固定白名单提取指标，避免把 schema_version/top_k 误显示成性能指标。"""
    items: list[tuple[str, float]] = []
    for key, label in METRIC_LABELS.items():
        if key in report and isinstance(report[key], (int, float)):
            items.append((label, float(report[key])))
    # Top-K 可以调整，因此动态识别 recall@N，同时避免重复默认 recall@5。
    for key, value in report.items():
        if key.startswith("recall@") and key != "recall@5" and isinstance(value, (int, float)):
            items.insert(0, (key.upper(), float(value)))
    return items


def experiment_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """把嵌套消融实验变成 Streamlit 表格行。"""
    top_k = int(report.get("top_k") or 5)
    return [
        {
            "方案": profile,
            f"Recall@{top_k}": metrics.get(f"recall@{top_k}"),
            "MRR": metrics.get("mrr"),
            "来源命中率": metrics.get("source_hit_rate"),
            "平均检索(ms)": metrics.get("avg_retrieval_ms"),
            "P95检索(ms)": metrics.get("p95_retrieval_ms"),
            "降级用例": metrics.get("degraded_cases", 0),
        }
        for profile, metrics in (report.get("experiments") or {}).items()
    ]
