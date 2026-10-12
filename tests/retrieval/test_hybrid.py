from pathlib import Path

from llama_index.core.schema import NodeWithScore, TextNode

from config.settings import Settings
from core.retrieval.hybrid import HybridRetrievalService


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "DATA_DIR": tmp_path / "data",
        "VECTOR_DB_DIR": tmp_path / "vectors",
        "MODELS_DIR": tmp_path / "models",
        "UPLOAD_DIR": tmp_path / "uploads",
        "INGESTION_STORAGE_DIR": tmp_path / "data" / "ingestion",
        "USE_RERANKER": False,
        "SEARCH_TOP_K": 2,
        "RETRIEVAL_CANDIDATE_K": 4,
        "LOG_TO_FILE": False,
    }
    values.update(overrides)
    return Settings(**values)


def node(node_id: str, text: str, category: str = "hr") -> TextNode:
    return TextNode(
        id_=node_id,
        text=text,
        metadata={"category": category, "file_name": "policy.txt"},
    )


def test_hybrid_retrieval_has_channels_fusion_and_alternate_query(tmp_path: Path):
    leave = node("leave", "员工事假全年最多十五天")
    sick = node("sick", "员工病假需要医院证明")
    corpus = [leave, sick]

    def vector_search(query, top_k, filter_dict):
        selected = sick if "证明" in query else leave
        return [NodeWithScore(node=selected, score=0.8)]

    service = HybridRetrievalService(
        make_settings(tmp_path),
        vector_search=vector_search,
        corpus_loader=lambda: corpus,
    )
    result = service.retrieve("事假多久", alternate_queries=["请假期限证明"])

    assert result.insufficient_evidence is False
    assert result.nodes[0].node.node_id == "leave"
    assert set(result.trace["channels"]) == {"vector:0", "bm25:0", "vector:1", "bm25:1"}
    assert result.trace["score_type"] == "rrf"


def test_minimum_evidence_threshold_returns_no_nodes(tmp_path: Path):
    weak = node("weak", "无关内容")
    service = HybridRetrievalService(
        make_settings(tmp_path, RETRIEVAL_MIN_FUSION_SCORE=1.0),
        vector_search=lambda query, top_k, filters: [NodeWithScore(node=weak, score=999.0)],
        corpus_loader=lambda: [weak],
    )

    result = service.retrieve("问题")

    assert result.insufficient_evidence is True
    assert result.nodes == []
    assert result.trace["fusion"]


def test_bm25_filter_without_matching_nodes_degrades_to_vector(tmp_path: Path):
    """官方 BM25 无法为全空过滤结果建索引时，向量通道仍应正常工作。"""
    leave = node("leave", "员工事假全年最多十五天", "hr")
    service = HybridRetrievalService(
        make_settings(tmp_path),
        vector_search=lambda query, top_k, filters: [NodeWithScore(node=leave, score=0.9)],
        corpus_loader=lambda: [leave],
    )

    result = service.retrieve("事假多久", filter_dict={"category": "finance"})

    assert result.nodes[0].node.node_id == "leave"
    assert result.trace["channels"]["bm25:0"] == []
    assert result.trace["errors"][0].startswith("bm25:init: ValueError")


def test_bm25_refreshes_when_content_changes_but_count_is_same(tmp_path: Path):
    corpus = [node("same-id", "旧的考勤制度")]
    service = HybridRetrievalService(
        make_settings(tmp_path),
        vector_search=lambda query, top_k, filters: [],
        corpus_loader=lambda: corpus,
    )
    first = service.retrieve("考勤制度")
    corpus[:] = [node("same-id", "新的发票制度")]
    second = service.retrieve("发票制度")

    assert first.nodes[0].node.get_content() == "旧的考勤制度"
    assert second.nodes[0].node.get_content() == "新的发票制度"


class FakeReranker:
    def postprocess_nodes(self, nodes, query_bundle):
        selected = next(item for item in nodes if item.node.node_id == "second")
        # 模拟官方组件的原地改分行为，验证融合 trace 不会被 rerank score 污染。
        selected.score = 3.5
        return [selected]


def test_reranker_score_and_trace_are_labeled_as_raw(tmp_path: Path):
    first = node("first", "第一条")
    second = node("second", "第二条")
    service = HybridRetrievalService(
        make_settings(tmp_path, USE_RERANKER=True, RETRIEVAL_MIN_RERANK_SCORE=3.0),
        vector_search=lambda query, top_k, filters: [
            NodeWithScore(node=first, score=0.9),
            NodeWithScore(node=second, score=0.8),
        ],
        corpus_loader=lambda: [first, second],
        reranker=FakeReranker(),
    )

    result = service.retrieve("第二条")

    assert result.nodes[0].node.node_id == "second"
    assert result.trace["score_type"] == "rerank_raw"
    assert result.trace["rerank_applied"] is True
    assert result.trace["fusion"][0]["score"] != 3.5
    assert type(result.insufficient_evidence) is bool


def test_evaluation_profiles_keep_ablation_boundaries(tmp_path: Path):
    """消融实验必须真的关闭组件，不能只给同一结果换一个名字。"""
    first = node("first", "普通内容")
    second = node("second", "需要命中的第二条")
    service = HybridRetrievalService(
        make_settings(tmp_path, USE_RERANKER=True),
        vector_search=lambda query, top_k, filters: [
            NodeWithScore(node=first, score=0.9),
            NodeWithScore(node=second, score=0.8),
        ],
        corpus_loader=lambda: [first, second],
        reranker=FakeReranker(),
    )

    vector = service.retrieve("第二条", profile="vector_only")
    vector_rerank = service.retrieve("第二条", profile="vector_rerank")
    hybrid = service.retrieve("第二条", profile="hybrid_rerank")

    assert vector.nodes[0].node.node_id == "first"
    assert vector.trace["rerank_applied"] is False
    assert set(vector.trace["channels"]) == {"vector:0"}
    assert vector_rerank.nodes[0].node.node_id == "second"
    assert vector_rerank.trace["fusion"] == []
    assert any(name.startswith("bm25:") for name in hybrid.trace["channels"])
