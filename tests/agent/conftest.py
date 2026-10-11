from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from llama_index.core.schema import NodeWithScore, TextNode

from config.settings import Settings
from core.agent.rewrite import RewriteResult
from core.retrieval.hybrid import HybridRetrievalResult


class FakeStructuredLLM:
    """模拟结构化改写；特定问题故意丢约束，用于验证安全回退。"""

    def __init__(self, owner) -> None:
        self.owner = owner

    def invoke(self, messages):
        self.owner.rewrite_calls += 1
        query = messages[-1].content
        if "15天" in query:
            return RewriteResult(
                rewritten_query="事假可以补交吗？",
                rewrite_applied=True,
                rewrite_reason="结合历史补全事假",
            )
        return RewriteResult(
            rewritten_query="员工加班应该怎么申请？",
            rewrite_applied=True,
            rewrite_reason="把指代补全为员工加班",
        )


class FakeLLM:
    def __init__(self) -> None:
        self.rewrite_calls = 0
        self.answer_calls = 0

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self)

    def invoke(self, messages):
        self.answer_calls += 1
        return AIMessage(content="根据制度，员工应先提交申请并等待审批。[1]")


class FakeRetrievalService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(
        self,
        query,
        *,
        alternate_queries=None,
        top_k=None,
        filter_dict=None,
    ):
        self.calls.append(
            {
                "query": query,
                "alternate_queries": alternate_queries or [],
                "top_k": top_k,
            }
        )
        if "检索异常" in query:
            raise RuntimeError("vector database unavailable")
        if "这个呢" in query or "量子纠错" in query:
            return HybridRetrievalResult(
                nodes=[],
                trace={"query": query, "errors": []},
                insufficient_evidence=True,
            )
        node = TextNode(
            id_="policy-1",
            text="员工加班或请假应先提交申请，经负责人审批通过后执行。",
            metadata={"file_name": "员工制度.txt", "page": 2},
        )
        return HybridRetrievalResult(
            nodes=[NodeWithScore(node=node, score=0.95)],
            trace={"query": query, "errors": [], "rerank_applied": True},
            insufficient_evidence=False,
        )


@pytest.fixture
def app_settings(tmp_path: Path) -> Settings:
    return Settings(
        DATA_DIR=tmp_path / "data",
        VECTOR_DB_DIR=tmp_path / "vectors",
        MODELS_DIR=tmp_path / "models",
        UPLOAD_DIR=tmp_path / "uploads",
        INGESTION_STORAGE_DIR=tmp_path / "ingestion",
        CHECKPOINT_DB_PATH=tmp_path / "checkpoints.db",
        LOG_TO_FILE=False,
        USE_RERANKER=False,
        AGENT_LLM_MAX_RETRIES=0,
    )
