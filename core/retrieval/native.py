"""LlamaIndex 原生检索组件的项目级配置入口。

边界说明：

- BM25 计算由官方 ``llama-index-retrievers-bm25`` 完成；
- RRF 计算由 Core 的 ``QueryFusionRetriever`` 完成；
- CrossEncoder 重排由 Core 的 ``SentenceTransformerRerank`` 完成；
- 本模块只配置中文 token 规则，并把已经执行过的多通道结果交给原生 RRF。

这里没有重新实现任何打分公式。把构造代码集中起来，是为了避免业务编排层散落框架参数，
并让“原生框架能力”和“项目业务规则”的边界一眼可见。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

# 官方 BM25 使用正则切 token。默认 ``\w\w+`` 会把连续中文当成一个超长 token，
# 因此项目只定制 token 边界：中文按单字、英文/数字编号按连续串。BM25 公式仍完全由官方实现。
CHINESE_TOKEN_PATTERN = r"(?u)[\u4e00-\u9fff]|[A-Za-z0-9_]+"


class StaticResultsRetriever(BaseRetriever):
    """把已得到的通道结果适配成 LlamaIndex Retriever。

    多 query 是项目上游生成的，不让 QueryFusionRetriever 再调用 LLM 改写。各通道先执行一次，
    再用这个只读适配器交给原生 RRF，既避免重复检索，也能保留融合前 trace。
    """

    def __init__(self, results: Sequence[NodeWithScore]) -> None:
        super().__init__()
        self._results = list(results)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # QueryFusionRetriever 会改写融合后的 score，因此返回副本，保护原始通道分数用于 trace。
        return [NodeWithScore(node=item.node, score=item.score) for item in self._results]


def build_native_bm25(
    nodes: Sequence[BaseNode],
    *,
    top_k: int,
    filters: MetadataFilters | None,
) -> BM25Retriever:
    """构造官方 BM25Retriever，只覆盖中文 token 边界。"""
    return BM25Retriever.from_defaults(
        nodes=list(nodes),
        similarity_top_k=top_k,
        language="chinese",
        skip_stemming=True,
        token_pattern=CHINESE_TOKEN_PATTERN,
        filters=filters,
    )


def native_rrf_fuse(
    rankings: Mapping[str, Sequence[NodeWithScore]],
    *,
    query: str,
    top_k: int,
) -> list[NodeWithScore]:
    """使用 LlamaIndex QueryFusionRetriever 的原生 RRF 合并已有通道结果。

    ``num_queries=1`` 表示禁止内部生成额外 query；``MockLLM`` 只是避免组件解析默认 OpenAI
    LLM，本路径不会调用它。RRF 常数由 LlamaIndex 按论文实现固定为 60。
    """
    non_empty = [StaticResultsRetriever(items) for items in rankings.values() if items]
    if not non_empty:
        return []
    fusion = QueryFusionRetriever(
        retrievers=non_empty,
        llm=MockLLM(),
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=top_k,
        num_queries=1,
        use_async=False,
    )
    return list(fusion.retrieve(query))
