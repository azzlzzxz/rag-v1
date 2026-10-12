import json

import pytest

from frontend.reporting import experiment_rows, load_report, metric_items


def test_report_loader_and_numeric_metrics(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps({"cases": 3, "recall@5": 0.8, "mrr": 0.65, "details": []}),
        encoding="utf-8",
    )
    report = load_report(path)
    assert report is not None
    assert metric_items(report) == [("Recall@5", 0.8), ("MRR", 0.65)]


def test_report_loader_rejects_non_object(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层"):
        load_report(path)


def test_experiment_rows_flattens_comparison_table():
    report = {
        "top_k": 3,
        "experiments": {
            "vector_only": {
                "recall@3": 0.5,
                "mrr": 0.4,
                "source_hit_rate": 0.5,
                "avg_retrieval_ms": 10,
                "p95_retrieval_ms": 20,
            }
        },
    }
    assert experiment_rows(report)[0]["方案"] == "vector_only"
    assert experiment_rows(report)[0]["Recall@3"] == 0.5
