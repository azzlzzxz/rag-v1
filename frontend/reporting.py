"""阶段 6 评测页的只读报告加载器。

评测计算属于阶段 7；本模块只校验并读取阶段 7 将生成的 JSON，避免 UI 为了展示而复制
评测逻辑或伪造指标。
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


def metric_items(report: dict[str, Any]) -> list[tuple[str, float]]:
    """提取数字型顶层指标，排除明细和布尔值。"""
    return [
        (key, float(value))
        for key, value in report.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "cases"
    ]
