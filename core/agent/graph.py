"""LangGraph RAG Agent 主流程。

图只做有状态业务编排；检索由 LlamaIndex 服务完成，模型调用继续使用 LangChain 1.x。
每个节点返回增量状态，SQLite checkpointer 以 ``thread_id`` 隔离并恢复多轮会话。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config.settings import Settings, settings
from core.agent.rewrite import QueryRewriter
from core.agent.state import AgentDecision, RAGAgentState
from core.agent.tools import create_retrieval_tool
from core.intent_recognizer import IntentRecognizer
from core.llm_client import get_llm
from core.retrieval.hybrid import HybridRetrievalService, get_hybrid_retrieval_service

logger = logging.getLogger(__name__)

_AMBIGUOUS_MARKERS = ("这个", "那个", "它", "怎么办", "怎么弄", "呢", "如何处理")


def _message_text(message: AnyMessage) -> str:
    """兼容字符串和多模态 content；当前问答只取可读文本。"""
    if isinstance(message.content, str):
        return message.content
    return str(message.content)


class RAGAgent:
    """可恢复会话的 LangGraph RAG Agent。

    生产环境默认创建 SQLite checkpointer。测试可注入 MemorySaver、Fake LLM 和 Fake Retriever，
    因而图节点无需连接外部模型或真实向量库也能验证。
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        *,
        llm: Any | None = None,
        retrieval_service: HybridRetrievalService | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = app_settings
        self.llm = llm or get_llm()
        self.retrieval_service = retrieval_service or get_hybrid_retrieval_service()
        self.intent_recognizer = IntentRecognizer()
        self.query_rewriter = QueryRewriter(
            self.llm,
            max_retries=app_settings.AGENT_LLM_MAX_RETRIES,
        )
        self.retrieval_tool = create_retrieval_tool(self.retrieval_service)
        self._checkpoint_connection: sqlite3.Connection | None = None
        self.checkpointer = checkpointer or self._create_sqlite_checkpointer(
            app_settings.CHECKPOINT_DB_PATH
        )
        self.graph = self._build_graph()

    def _create_sqlite_checkpointer(self, path: Path) -> SqliteSaver:
        """创建线程安全的 SQLite 连接，并初始化 LangGraph checkpoint 表。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI 的同步依赖可能在线程池中执行，所以关闭 sqlite 的同线程限制。
        self._checkpoint_connection = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(self._checkpoint_connection)
        saver.setup()
        return saver

    def _build_graph(self):
        """显式声明流程节点，方便调试、测试和后续增加人工审批节点。"""
        builder = StateGraph(RAGAgentState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("rewrite_query", self._rewrite_query)
        builder.add_node("route_intent", self._route_intent)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade_evidence", self._grade_evidence)
        builder.add_node("answer", self._answer)
        builder.add_node("clarify", self._clarify)
        builder.add_node("refuse", self._refuse)
        builder.add_node("persist_result", self._persist_result)

        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "rewrite_query")
        builder.add_edge("rewrite_query", "route_intent")
        builder.add_edge("route_intent", "retrieve")
        builder.add_edge("retrieve", "grade_evidence")
        builder.add_conditional_edges(
            "grade_evidence",
            self._select_decision,
            {"answer": "answer", "clarify": "clarify", "refuse": "refuse"},
        )
        builder.add_edge("answer", "persist_result")
        builder.add_edge("clarify", "persist_result")
        builder.add_edge("refuse", "persist_result")
        builder.add_edge("persist_result", END)
        return builder.compile(checkpointer=self.checkpointer, name="enterprise-rag-agent")

    @staticmethod
    def _prior_messages(state: RAGAgentState) -> list[AnyMessage]:
        """当前请求的最后一条 HumanMessage 不算历史。"""
        messages = list(state.get("messages", []))
        return messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages

    def _load_context(self, state: RAGAgentState) -> dict[str, Any]:
        """初始化本轮临时字段，防止 checkpoint 中上一轮结果泄漏到本轮。"""
        messages = list(state.get("messages", []))
        original_query = state.get("original_query", "").strip()
        if not original_query and messages and isinstance(messages[-1], HumanMessage):
            original_query = _message_text(messages[-1]).strip()
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
            "rewrite_applied": False,
            "rewrite_reason": "",
            "protected_constraints": {},
            "intent": "通用问答",
            "retrieved_nodes": [],
            "retrieval_trace": {},
            "insufficient_evidence": True,
            "decision": "refuse",
            "answer": "",
            "citations": [],
            "errors": [],
        }

    def _rewrite_query(self, state: RAGAgentState) -> dict[str, Any]:
        """结构化改写追问；不确定或约束丢失时自动使用原问题。"""
        history = self._prior_messages(state)[-self.settings.AGENT_HISTORY_WINDOW * 2 :]
        result, constraints, errors = self.query_rewriter.rewrite(
            state["original_query"], history
        )
        return {
            "rewritten_query": result.rewritten_query,
            "rewrite_applied": result.rewrite_applied,
            "rewrite_reason": result.rewrite_reason,
            "protected_constraints": constraints,
            "errors": [*state.get("errors", []), *errors],
        }

    def _route_intent(self, state: RAGAgentState) -> dict[str, Any]:
        """意图当前用于 trace 和回答风格；后续可扩展为不同工具路由。"""
        return {"intent": self.intent_recognizer.recognize(state["rewritten_query"])}

    def _retrieve(self, state: RAGAgentState) -> dict[str, Any]:
        """通过标准 LangChain Tool 调用阶段 3 的 LlamaIndex 混合检索。"""
        alternate_queries = (
            [state["rewritten_query"]] if state.get("rewrite_applied") else []
        )
        try:
            payload = self.retrieval_tool.invoke(
                {
                    "original_query": state["original_query"],
                    "alternate_queries": alternate_queries,
                    "top_k": self.settings.SEARCH_TOP_K,
                }
            )
            return {
                "retrieved_nodes": payload["nodes"],
                "retrieval_trace": payload["trace"],
                "insufficient_evidence": payload["insufficient_evidence"],
            }
        except Exception as exc:
            logger.exception("知识库检索失败")
            return {
                "retrieved_nodes": [],
                "retrieval_trace": {
                    "query": state["original_query"],
                    "errors": [f"agent retrieve: {type(exc).__name__}: {exc}"],
                },
                "insufficient_evidence": True,
                "errors": [
                    *state.get("errors", []),
                    f"retrieve: {type(exc).__name__}: {exc}",
                ],
            }

    def _is_ambiguous_without_context(self, state: RAGAgentState) -> bool:
        query = state["original_query"]
        return not self._prior_messages(state) and (
            len(query) <= 8 or any(marker in query for marker in _AMBIGUOUS_MARKERS)
        )

    def _grade_evidence(self, state: RAGAgentState) -> dict[str, AgentDecision]:
        """把证据判断变成显式决策，禁止无证据时继续调用回答模型。"""
        if state.get("retrieved_nodes") and not state.get("insufficient_evidence", True):
            decision: AgentDecision = "answer"
        elif self._is_ambiguous_without_context(state):
            decision = "clarify"
        else:
            decision = "refuse"
        return {"decision": decision}

    @staticmethod
    def _select_decision(state: RAGAgentState) -> AgentDecision:
        return state["decision"]

    def _invoke_answer_llm(self, messages: Sequence[AnyMessage]) -> str:
        """回答模型失败时按配置重试；超过次数后由 answer 节点生成安全降级文案。"""
        last_error: Exception | None = None
        for attempt in range(self.settings.AGENT_LLM_MAX_RETRIES + 1):
            try:
                return _message_text(self.llm.invoke(list(messages))).strip()
            except Exception as exc:
                last_error = exc
                logger.warning("回答模型第 %s 次调用失败: %s", attempt + 1, exc)
        assert last_error is not None
        raise last_error

    def _answer(self, state: RAGAgentState) -> dict[str, Any]:
        """只把已通过证据门控的内容交给 LLM，并生成结构化引用。"""
        nodes = state.get("retrieved_nodes", [])
        context = "\n\n".join(
            f"[{index}] 来源={node['metadata'].get('file_name', node['metadata'].get('source', '未知'))}"
            f" 页码={node['metadata'].get('page', '无')}\n{node['content']}"
            for index, node in enumerate(nodes, start=1)
        )
        citations = [
            {
                "node_id": node["node_id"],
                "source": node["metadata"].get(
                    "file_name", node["metadata"].get("source", "未知")
                ),
                "page": node["metadata"].get("page"),
                "content": node["content"][:200] + ("..." if len(node["content"]) > 200 else ""),
                "score": node.get("score"),
            }
            for node in nodes
        ]
        history = self._prior_messages(state)[-self.settings.AGENT_HISTORY_WINDOW * 2 :]
        prompt = [
            SystemMessage(
                content=(
                    "你是企业知识库助手。只能根据给定证据回答，不得使用证据外事实。"
                    "回答简洁、准确；引用证据时使用 [1]、[2] 标记。\n\n"
                    f"证据：\n{context}"
                )
            ),
            *history,
            HumanMessage(
                content=(
                    f"原问题：{state['original_query']}\n"
                    f"用于检索的完整问题：{state['rewritten_query']}"
                )
            ),
        ]
        try:
            answer = self._invoke_answer_llm(prompt)
            return {"answer": answer, "citations": citations}
        except Exception as exc:
            logger.exception("生成回答失败")
            return {
                "answer": "已找到相关资料，但生成回答时出现临时错误，请稍后重试。",
                "citations": citations,
                "errors": [
                    *state.get("errors", []),
                    f"answer: {type(exc).__name__}: {exc}",
                ],
            }

    @staticmethod
    def _clarify(state: RAGAgentState) -> dict[str, Any]:
        """问题缺少指代对象且没有历史时，不猜测用户意图。"""
        return {
            "answer": "你的问题里缺少明确的对象，请补充要咨询的制度、流程或事项名称。",
            "citations": [],
        }

    @staticmethod
    def _refuse(state: RAGAgentState) -> dict[str, Any]:
        """知识库证据不足时明确拒答，避免模型凭常识编造。"""
        return {
            "answer": "当前知识库中没有找到足够证据，暂时无法可靠回答这个问题。",
            "citations": [],
        }

    @staticmethod
    def _persist_result(state: RAGAgentState) -> dict[str, Any]:
        """把最终回答写入 messages；checkpointer 会随节点完成自动持久化。"""
        return {"messages": [AIMessage(content=state["answer"])]}

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def ask(self, question: str, thread_id: str = "default") -> dict[str, Any]:
        """执行一轮图，并返回阶段 4 的完整结果。"""
        question = question.strip()
        if not question:
            return {
                "answer": "问题不能为空",
                "sources": [],
                "citations": [],
                "intent": "invalid",
                "decision": "refuse",
                "trace": {},
            }
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "original_query": question,
            },
            config=self._config(thread_id),
        )
        citations = result.get("citations", [])
        return {
            "answer": result["answer"],
            # sources 保留阶段 3/旧 API 字段；citations 是阶段 4 的完整结构。
            "sources": [
                {"source": item["source"], "page": item["page"], "content": item["content"]}
                for item in citations
            ],
            "citations": citations,
            "intent": result.get("intent", "通用问答"),
            "decision": result.get("decision", "refuse"),
            "original_query": result.get("original_query", question),
            "rewritten_query": result.get("rewritten_query", question),
            "rewrite_applied": result.get("rewrite_applied", False),
            "rewrite_reason": result.get("rewrite_reason", ""),
            "trace": result.get("retrieval_trace", {}),
            "errors": result.get("errors", []),
        }

    def get_chat_history(self, thread_id: str = "default") -> list[dict[str, str]]:
        """直接读取 checkpoint 中的消息，进程重启后仍可恢复。"""
        snapshot = self.graph.get_state(self._config(thread_id))
        messages = snapshot.values.get("messages", []) if snapshot.values else []
        return [
            {
                "role": "human" if isinstance(message, HumanMessage) else "assistant",
                "content": _message_text(message),
            }
            for message in messages
            if isinstance(message, (HumanMessage, AIMessage))
        ]

    def clear_thread(self, thread_id: str = "default") -> None:
        """删除该 thread 的所有 checkpoint，不影响其他会话。"""
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        """测试或进程退出时可显式关闭本实例持有的 SQLite 连接。"""
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None
