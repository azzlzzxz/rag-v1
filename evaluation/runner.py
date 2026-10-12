"""可注入、可单测的 RAG 离线评测核心。

这里不直接创建模型或数据库连接：CLI 负责注入真实 Retriever/Agent，单元测试注入 Fake。
指标计算因此可以在无网络环境稳定复现，也不会把“模型是否可下载”和“公式是否正确”混在一起。
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from evaluation.models import EvaluationCase

RetrievalCall = Callable[[EvaluationCase, str, int], Any]
QACall = Callable[[EvaluationCase], dict[str, Any]]


def _round(value: float) -> float:
    return round(float(value), 4)


def _mean(values: Sequence[float]) -> float:
    return _round(statistics.fmean(values)) if values else 0.0


def _p95(values: Sequence[float]) -> float:
    """使用 nearest-rank P95，小样本也返回真实观测值而不是插值出的虚构延迟。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return _round(ordered[rank - 1])


def _normalize(text: str) -> str:
    """关键词覆盖使用确定性规范化，不引入另一个 LLM Judge。"""
    return "".join(text.lower().split())


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    profile: str,
    top_k: int,
    retrieve: RetrievalCall,
) -> dict[str, Any]:
    """计算 Recall@K、MRR、来源覆盖率和检索延迟。

    无答案/澄清用例没有 relevant source，不参与检索排名指标；它们由 Agent 决策指标负责。
    """
    positive_cases = [case for case in cases if case.relevant_files]
    hits = 0
    reciprocal_rank = 0.0
    expected_sources = 0
    hit_sources = 0
    latencies: list[float] = []
    degraded_cases = 0
    details: list[dict[str, Any]] = []

    for case in positive_cases:
        started = time.perf_counter()
        result = retrieve(case, profile, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)

        files = [str(item.node.metadata.get("file_name") or "") for item in result.nodes]
        trace = getattr(result, "trace", {}) or {}
        expects_rerank = profile in {"vector_rerank", "hybrid_rerank"}
        degraded = expects_rerank and not bool(trace.get("rerank_applied", False))
        degraded_cases += int(degraded)
        rank = next(
            (
                index
                for index, file_name in enumerate(files, start=1)
                if file_name in case.relevant_files
            ),
            None,
        )
        matched_sources = sorted(set(files).intersection(case.relevant_files))
        hits += int(rank is not None)
        reciprocal_rank += 1.0 / rank if rank else 0.0
        expected_sources += len(case.relevant_files)
        hit_sources += len(matched_sources)
        reasons = [] if rank else ["expected_source_not_retrieved"]
        if degraded:
            # 不保存 trace.errors 的内部正文，只记录组件未实际生效这一事实。
            reasons.append("reranker_degraded")
        details.append(
            {
                "id": case.id,
                "type": case.type,
                "query": case.query,
                "profile": profile,
                "expected_files": list(case.relevant_files),
                "retrieved_files": files,
                "rank": rank,
                "matched_sources": matched_sources,
                "insufficient_evidence": bool(result.insufficient_evidence),
                "rerank_applied": bool(trace.get("rerank_applied", False)),
                "degraded": degraded,
                "latency_ms": _round(elapsed_ms),
                "passed": not reasons,
                "failure_reasons": reasons,
            }
        )

    total = len(positive_cases)
    return {
        "cases": total,
        f"recall@{top_k}": _round(hits / total) if total else 0.0,
        "mrr": _round(reciprocal_rank / total) if total else 0.0,
        "source_hit_rate": _round(hit_sources / expected_sources) if expected_sources else 0.0,
        "avg_retrieval_ms": _mean(latencies),
        "p95_retrieval_ms": _p95(latencies),
        "degraded_cases": degraded_cases,
        "details": details,
    }


def evaluate_qa(cases: Sequence[EvaluationCase], ask: QACall) -> dict[str, Any]:
    """评估决策、引用、关键词覆盖以及端到端/生成近似延迟。

    ``generation_ms`` 用端到端耗时减去 trace 中检索耗时估算，包含 rewrite、路由、证据门控
    和回答节点，因此应理解为“非检索阶段耗时”，不是底层模型首 token 延迟。
    """
    details: list[dict[str, Any]] = []
    decision_hits = 0
    refusal_hits = 0
    refusal_total = 0
    clarify_hits = 0
    clarify_total = 0
    citation_scores: list[float] = []
    keyword_scores: list[float] = []
    source_hits: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        result = ask(case)
        total_ms = (time.perf_counter() - started) * 1000
        trace = result.get("trace") or {}
        retrieval_ms = float(trace.get("elapsed_ms") or 0.0)
        generation_ms = max(0.0, total_ms - retrieval_ms)
        retrieval_latencies.append(retrieval_ms)
        generation_latencies.append(generation_ms)

        actual_decision = str(result.get("decision") or "")
        decision_ok = actual_decision == case.expected_decision
        decision_hits += int(decision_ok)
        if case.expected_decision == "refuse":
            refusal_total += 1
            refusal_hits += int(decision_ok)
        if case.expected_decision == "clarify":
            clarify_total += 1
            clarify_hits += int(decision_ok)

        answer = str(result.get("answer") or "")
        normalized_answer = _normalize(answer)
        keyword_hits = [
            keyword for keyword in case.expected_keywords if _normalize(keyword) in normalized_answer
        ]
        keyword_score = (
            len(keyword_hits) / len(case.expected_keywords) if case.expected_keywords else 1.0
        )

        citations = list(result.get("citations") or [])
        citation_files = [
            str(item.get("file_name") or item.get("source") or "") for item in citations
        ]
        if case.expected_decision == "answer":
            citation_score = (
                sum(file_name in case.relevant_files for file_name in citation_files)
                / len(citation_files)
                if citation_files
                else 0.0
            )
            source_hit = float(bool(set(citation_files).intersection(case.relevant_files)))
            citation_scores.append(citation_score)
            keyword_scores.append(keyword_score)
            source_hits.append(source_hit)
        else:
            citation_score = 1.0 if not citations else 0.0
            source_hit = 1.0 if not citations else 0.0

        reasons: list[str] = []
        if not decision_ok:
            reasons.append("decision_mismatch")
        if case.expected_decision == "answer" and keyword_score < 1.0:
            reasons.append("keyword_missing")
        if case.expected_decision == "answer" and source_hit < 1.0:
            reasons.append("citation_source_mismatch")
        if case.expected_decision != "answer" and citations:
            reasons.append("unexpected_citation")
        details.append(
            {
                "id": case.id,
                "type": case.type,
                "query": case.query,
                "expected_decision": case.expected_decision,
                "actual_decision": actual_decision,
                "expected_keywords": list(case.expected_keywords),
                "matched_keywords": keyword_hits,
                "keyword_coverage": _round(keyword_score),
                "expected_files": list(case.relevant_files),
                "citation_files": citation_files,
                "citation_accuracy": _round(citation_score),
                "total_ms": _round(total_ms),
                "retrieval_ms": _round(retrieval_ms),
                "generation_ms": _round(generation_ms),
                "passed": not reasons,
                "failure_reasons": reasons,
            }
        )

    total = len(cases)
    return {
        "cases": total,
        "decision_accuracy": _round(decision_hits / total) if total else 0.0,
        "citation_accuracy": _mean(citation_scores),
        "keyword_coverage": _mean(keyword_scores),
        "source_hit_rate": _mean(source_hits),
        "refusal_accuracy": _round(refusal_hits / refusal_total) if refusal_total else 0.0,
        "clarify_accuracy": _round(clarify_hits / clarify_total) if clarify_total else 0.0,
        "avg_retrieval_ms": _mean(retrieval_latencies),
        "p95_retrieval_ms": _p95(retrieval_latencies),
        "avg_generation_ms": _mean(generation_latencies),
        "p95_generation_ms": _p95(generation_latencies),
        "details": details,
    }


def build_report(
    cases: Sequence[EvaluationCase],
    *,
    top_k: int,
    profiles: Sequence[str],
    retrieve: RetrievalCall | None = None,
    ask: QACall | None = None,
    dataset_name: str,
) -> dict[str, Any]:
    """组装稳定报告；顶层保留主实验指标，便于 Streamlit 直接展示。"""
    experiments = (
        {profile: evaluate_retrieval(cases, profile, top_k, retrieve) for profile in profiles}
        if retrieve
        else {}
    )
    primary_profile = "hybrid_rerank" if "hybrid_rerank" in experiments else next(
        iter(experiments), None
    )
    primary = experiments.get(primary_profile, {}) if primary_profile else {}
    qa = evaluate_qa(cases, ask) if ask else None

    bad_cases: list[dict[str, Any]] = []
    for item in primary.get("details", []):
        if not item["passed"]:
            bad_cases.append({"stage": "retrieval", **item})
    for item in (qa or {}).get("details", []):
        if not item["passed"]:
            bad_cases.append({"stage": "qa", **item})

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "cases": len(cases),
        "top_k": top_k,
        "primary_profile": primary_profile,
        "experiments": experiments,
        "qa": qa,
        "bad_cases": bad_cases,
        "details": (qa or primary).get("details", []),
    }
    # 常用指标复制到顶层，保持 Streamlit 与后续 CI 阈值读取简单稳定。
    for key in (f"recall@{top_k}", "mrr", "source_hit_rate", "avg_retrieval_ms", "p95_retrieval_ms"):
        if key in primary:
            report[key] = primary[key]
    if qa:
        for key in (
            "decision_accuracy",
            "citation_accuracy",
            "keyword_coverage",
            "refusal_accuracy",
            "clarify_accuracy",
            "avg_generation_ms",
            "p95_generation_ms",
        ):
            report[key] = qa[key]
    return report
