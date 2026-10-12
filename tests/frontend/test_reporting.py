import json

import pytest

from frontend.reporting import load_report, metric_items


def test_report_loader_and_numeric_metrics(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps({"cases": 3, "recall@5": 0.8, "mrr": 0.65, "details": []}),
        encoding="utf-8",
    )
    report = load_report(path)
    assert report is not None
    assert metric_items(report) == [("recall@5", 0.8), ("mrr", 0.65)]


def test_report_loader_rejects_non_object(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层"):
        load_report(path)
