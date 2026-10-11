import re

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

from core.retrieval.native import CHINESE_TOKEN_PATTERN, build_native_bm25


def make_node(node_id: str, text: str, category: str = "policy") -> TextNode:
    return TextNode(id_=node_id, text=text, metadata={"category": category})


def test_chinese_token_pattern_keeps_numbers_and_chinese_characters():
    # 只定制官方 BM25 的 token 边界，不在项目里维护另一套分词/打分实现。
    tokens = re.findall(CHINESE_TOKEN_PATTERN, "HR-2024-01 第十五条规定")
    assert "HR" in tokens
    assert "2024" in tokens
    assert all(value in tokens for value in ["第", "十", "五", "条"])


def test_native_bm25_exact_term_and_metadata_filter():
    nodes = [
        make_node("leave", "员工事假全年最多十五天", "hr"),
        make_node("invoice", "财务发票报销流程", "finance"),
        make_node("security", "信息安全事件处理流程", "it"),
    ]
    retriever = build_native_bm25(nodes, top_k=3, filters=None)

    # 类型断言明确保证这里使用官方集成，而不是项目自研 BM25。
    assert isinstance(retriever, BM25Retriever)
    assert retriever.retrieve("发票")[0].node.node_id == "invoice"

    filtered = build_native_bm25(
        nodes,
        top_k=3,
        filters=MetadataFilters(filters=[MetadataFilter(key="category", value="hr")]),
    )
    # 官方实现会用 0 分结果补齐 top_k；业务编排层会丢弃这些占位项。
    actual_hits = [item for item in filtered.retrieve("事假") if (item.score or 0.0) > 0.0]
    assert actual_hits
    assert all(item.node.metadata["category"] == "hr" for item in actual_hits)
