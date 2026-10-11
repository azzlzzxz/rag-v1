from llama_index.core.schema import NodeWithScore, TextNode

from core.retrieval.native import native_rrf_fuse


def hit(node_id: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=node_id), score=score)


def test_native_rrf_uses_rank_instead_of_mixing_raw_scores():
    results = native_rrf_fuse(
        {
            "vector": [hit("shared", 0.01), hit("vector-only", 999.0)],
            "bm25": [hit("shared", 5000.0), hit("bm25-only", 0.001)],
        },
        query="shared",
        top_k=3,
    )

    # shared 同时出现在两个通道，使用原生 QueryFusionRetriever 的 RRF 后应排第一。
    assert results[0].node.node_id == "shared"
    # 极大的原始分数不会直接累加；两个单通道结果都只获得一次基于 rank 的小分值。
    assert all(item.score < results[0].score for item in results[1:])
    assert all(item.score < 0.02 for item in results[1:])
