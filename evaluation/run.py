"""最小离线检索评测。

先摄取测试文档，再执行：
    .venv-v2/bin/python scripts/ingest_demo.py tests/sample_docs --category demo
    .venv-v2/bin/python -m evaluation.run --mode retrieval

数据集每项包含 query 和 relevant_file。本阶段计算 Recall@K 与 MRR；阶段 6 会在此基础
上增加更完整的数据集、生成质量和忠实度评测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.retrieval.hybrid import get_hybrid_retrieval_service


def run_retrieval(dataset_path: Path, top_k: int, category: str | None) -> dict:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    service = get_hybrid_retrieval_service()
    hits = 0
    reciprocal_rank_sum = 0.0
    details = []
    for case in cases:
        result = service.retrieve(
            case["query"],
            top_k=top_k,
            filter_dict={"category": category} if category else None,
        )
        files = [item.node.metadata.get("file_name") for item in result.nodes]
        rank = next(
            (index for index, file_name in enumerate(files, start=1) if file_name == case["relevant_file"]),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / rank
        details.append(
            {
                "query": case["query"],
                "expected": case["relevant_file"],
                "rank": rank,
                "retrieved": files,
                "insufficient_evidence": result.insufficient_evidence,
            }
        )
    total = len(cases)
    return {
        "cases": total,
        f"recall@{top_k}": hits / total if total else 0.0,
        "mrr": reciprocal_rank_sum / total if total else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["retrieval"], default="retrieval")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("retrieval_cases.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default="demo")
    args = parser.parse_args()
    result = run_retrieval(args.dataset, args.top_k, args.category)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
