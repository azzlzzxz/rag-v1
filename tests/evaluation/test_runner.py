from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from evaluation.models import EvaluationCase, load_cases
from evaluation.run import print_summary, write_report
from evaluation.runner import build_report, evaluate_qa, evaluate_retrieval


def case(
    case_id="answer",
    *,
    decision="answer",
    files=("policy.txt",),
    keywords=("15天",),
):
    return EvaluationCase(
        id=case_id,
        type="direct" if decision == "answer" else "no_answer",
        query="问题",
        relevant_files=files,
        expected_keywords=keywords,
        expected_decision=decision,
    )


def retrieval_result(*files, rerank_applied=False):
    nodes = [
        NodeWithScore(
            node=TextNode(id_=f"n-{index}", text="内容", metadata={"file_name": name}),
            score=1.0 / index,
        )
        for index, name in enumerate(files, start=1)
    ]
    return SimpleNamespace(
        nodes=nodes,
        insufficient_evidence=not nodes,
        trace={"rerank_applied": rerank_applied},
    )


def test_retrieval_metrics_and_bad_case_are_deterministic():
    cases = [case("hit"), case("miss", files=("missing.txt",))]

    def retrieve(current, profile, top_k):
        return retrieval_result("other.txt", "policy.txt")

    result = evaluate_retrieval(cases, "vector_only", 2, retrieve)
    assert result["recall@2"] == 0.5
    assert result["mrr"] == 0.25
    assert result["source_hit_rate"] == 0.5
    assert result["details"][1]["failure_reasons"] == ["expected_source_not_retrieved"]


def test_qa_metrics_cover_citation_keyword_refusal_and_clarify():
    cases = [
        case(),
        case("refuse", decision="refuse", files=(), keywords=()),
        EvaluationCase("clarify", "ambiguous", "这个呢", (), (), "clarify"),
    ]

    def ask(current):
        if current.id == "answer":
            return {
                "decision": "answer",
                "answer": "上限是15天",
                "citations": [{"file_name": "policy.txt"}],
                "trace": {"elapsed_ms": 1},
            }
        return {"decision": current.expected_decision, "answer": "", "citations": [], "trace": {}}

    result = evaluate_qa(cases, ask)
    assert result["decision_accuracy"] == 1.0
    assert result["citation_accuracy"] == 1.0
    assert result["keyword_coverage"] == 1.0
    assert result["refusal_accuracy"] == 1.0
    assert result["clarify_accuracy"] == 1.0
    assert all(item["passed"] for item in result["details"])


def test_build_report_exposes_primary_metrics_and_bad_cases():
    cases = [case()]
    report = build_report(
        cases,
        top_k=1,
        profiles=["vector_only", "hybrid_rerank"],
        retrieve=lambda current, profile, top_k: retrieval_result(
            rerank_applied=profile != "vector_only"
        ),
        ask=lambda current: {"decision": "refuse", "answer": "", "citations": [], "trace": {}},
        dataset_name="cases.json",
    )
    assert report["primary_profile"] == "hybrid_rerank"
    assert report["recall@1"] == 0.0
    assert {item["stage"] for item in report["bad_cases"]} == {"retrieval", "qa"}


def test_dataset_validation_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"id":"x","type":"no_answer","query":"a","expected_decision":"refuse"},'
        '{"id":"x","type":"no_answer","query":"b","expected_decision":"refuse"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        load_cases(path)


def test_report_write_is_atomic_and_terminal_summary_stays_compact(tmp_path, capsys):
    report = {
        "dataset": "cases.json",
        "cases": 1,
        "top_k": 5,
        "experiments": {
            "vector_only": {"recall@5": 1.0, "mrr": 1.0, "p95_retrieval_ms": 8.0}
        },
        "qa": None,
        "bad_cases": [],
    }
    output = tmp_path / "reports" / "latest.json"
    write_report(report, output)
    print_summary(report, output)

    terminal = capsys.readouterr().out
    assert output.exists()
    assert not output.with_suffix(".json.tmp").exists()
    assert "vector_only" in terminal
    assert "完整报告" in terminal
    assert "Degraded=0" in terminal
    assert len(terminal.splitlines()) == 4
