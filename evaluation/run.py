"""阶段 7 离线评测命令。

准备知识库：
    python scripts/ingest_demo.py tests/sample_docs evaluation/sample_docs --category demo

运行完整评测：
    python -m evaluation.run --all --output reports/latest.json

默认只在终端打印紧凑摘要，完整逐条结果写入 JSON，避免大量 trace 淹没人工判断。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.rag_chain import get_rag_chain
from core.retrieval.hybrid import RetrievalProfile, get_hybrid_retrieval_service
from evaluation.models import EvaluationCase, load_cases
from evaluation.runner import build_report

PROFILES: tuple[RetrievalProfile, ...] = (
    "vector_only",
    "vector_rerank",
    "hybrid_rerank",
)


def write_report(report: dict[str, Any], output: Path) -> None:
    """原子写入报告，评测中断时不覆盖上一次可用结果。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)


def print_summary(report: dict[str, Any], output: Path | None) -> None:
    """只打印人能一眼判断的摘要；Bad Case 细节请看报告或 Streamlit。"""
    top_k = report["top_k"]
    print(f"评测集: {report['dataset']} | 用例: {report['cases']} | Top-K: {top_k}")
    experiments = report.get("experiments") or {}
    for profile, metrics in experiments.items():
        print(
            f"  {profile:<16} "
            f"Recall={metrics.get(f'recall@{top_k}', 0):.4f} "
            f"MRR={metrics.get('mrr', 0):.4f} "
            f"P95={metrics.get('p95_retrieval_ms', 0):.2f}ms "
            f"Degraded={metrics.get('degraded_cases', 0)}"
        )
    qa = report.get("qa")
    if qa:
        print(
            "  qa               "
            f"Decision={qa['decision_accuracy']:.4f} "
            f"Citation={qa['citation_accuracy']:.4f} "
            f"Keyword={qa['keyword_coverage']:.4f} "
            f"Refusal={qa['refusal_accuracy']:.4f}"
        )
    print(f"Bad Case: {len(report.get('bad_cases') or [])}")
    if output:
        print(f"完整报告: {output.resolve()}")


def make_retrieval_call(category: str | None):
    service = get_hybrid_retrieval_service()

    def retrieve(case: EvaluationCase, profile: str, top_k: int):
        return service.retrieve(
            case.query,
            top_k=top_k,
            filter_dict={"category": category} if category else None,
            profile=profile,  # type: ignore[arg-type]
        )

    return retrieve


def make_qa_call():
    """为每条用例建立临时 thread，支持多轮评测并在结束后清理 checkpoint。"""
    chain = get_rag_chain()
    sequence = 0

    def ask(case: EvaluationCase) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        thread_id = f"eval-{case.id}-{sequence}"
        try:
            for history_query in case.history:
                chain.ask(history_query, thread_id)
            return chain.ask(case.query, thread_id)
        finally:
            # 评测 thread 不属于业务会话，完成后立即删除，避免 checkpoint 数据库持续膨胀。
            chain.clear_session(thread_id)

    return ask


def execute_evaluation(
    *,
    mode: str,
    dataset: Path,
    top_k: int,
    category: str | None,
) -> dict[str, Any]:
    """执行一次真实评测，供 CLI 和 FastAPI 后台任务共同复用。

    运行入口统一后，页面评测与命令行评测使用完全相同的数据、Profile 和指标公式，避免
    出现“页面结果”和“脚本结果”对不上的双实现。
    """
    if mode not in {"retrieval", "qa", "all"}:
        raise ValueError(f"不支持的评测模式: {mode}")
    if top_k < 1:
        raise ValueError("top_k 必须大于等于 1")
    cases = load_cases(dataset)
    return build_report(
        cases,
        top_k=top_k,
        profiles=PROFILES if mode in {"retrieval", "all"} else (),
        retrieve=(
            make_retrieval_call(category)
            if mode in {"retrieval", "all"}
            else None
        ),
        ask=make_qa_call() if mode in {"qa", "all"} else None,
        dataset_name=dataset.name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="企业 RAG 离线评测与 Bad Case 报告")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "qa", "all"],
        default="retrieval",
        help="retrieval 不调用 LLM；all 同时运行三组检索实验和 Agent 评测",
    )
    parser.add_argument("--all", action="store_true", help="等价于 --mode all")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("rag_cases.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default="demo", help="空字符串表示不过滤分类")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-json", action="store_true", help="额外在终端打印完整 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k 必须大于等于 1")
    mode = "all" if args.all else args.mode
    report = execute_evaluation(
        mode=mode,
        dataset=args.dataset,
        top_k=args.top_k,
        category=args.category or None,
    )
    if args.output:
        write_report(report, args.output)
    print_summary(report, args.output)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
