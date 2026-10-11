"""打印混合检索结果与完整轨迹。

用法：
    .venv-v2/bin/python scripts/retrieve_demo.py "员工全年事假最多多少天？"
    .venv-v2/bin/python scripts/retrieve_demo.py "那需要什么证明？" \
        --alternate-query "员工申请病假需要提供什么证明？" --category demo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retrieval.hybrid import get_hybrid_retrieval_service


def main() -> None:
    parser = argparse.ArgumentParser(description="查看向量、BM25、RRF 和 Rerank 检索轨迹")
    parser.add_argument("query")
    parser.add_argument("--alternate-query", action="append", default=[])
    parser.add_argument("--category")
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    result = get_hybrid_retrieval_service().retrieve(
        args.query,
        alternate_queries=args.alternate_query,
        top_k=args.top_k,
        filter_dict={"category": args.category} if args.category else None,
    )
    payload = {
        "insufficient_evidence": result.insufficient_evidence,
        "results": [
            {
                "node_id": item.node.node_id,
                "score": item.score,
                "file_name": item.node.metadata.get("file_name"),
                "page": item.node.metadata.get("page"),
                "content": item.node.get_content(),
            }
            for item in result.nodes
        ],
        "trace": result.trace,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
