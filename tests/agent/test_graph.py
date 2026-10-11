from langgraph.checkpoint.memory import MemorySaver

from core.agent.graph import RAGAgent
from scripts.chat_cli import build_debug_summary, save_full_trace

from .conftest import FakeLLM, FakeRetrievalService


def make_agent(app_settings):
    llm = FakeLLM()
    retrieval = FakeRetrievalService()
    agent = RAGAgent(
        app_settings,
        llm=llm,
        retrieval_service=retrieval,
        checkpointer=MemorySaver(),
    )
    return agent, llm, retrieval


def test_independent_question_is_not_rewritten_and_returns_citation(app_settings):
    agent, llm, retrieval = make_agent(app_settings)

    result = agent.ask("加班制度是什么？", thread_id="independent")

    assert result["decision"] == "answer"
    assert result["rewrite_applied"] is False
    assert result["rewritten_query"] == "加班制度是什么？"
    assert llm.rewrite_calls == 0
    assert result["citations"][0]["source"] == "员工制度.txt"
    assert retrieval.calls[0]["alternate_queries"] == []


def test_follow_up_is_rewritten_but_original_and_rewrite_both_retrieved(app_settings):
    agent, llm, retrieval = make_agent(app_settings)
    agent.ask("加班制度是什么？", thread_id="follow-up")

    result = agent.ask("那怎么申请？", thread_id="follow-up")

    assert result["rewrite_applied"] is True
    assert result["rewritten_query"] == "员工加班应该怎么申请？"
    assert retrieval.calls[-1]["query"] == "那怎么申请？"
    assert retrieval.calls[-1]["alternate_queries"] == ["员工加班应该怎么申请？"]
    assert len(agent.get_chat_history("follow-up")) == 4


def test_independent_question_stays_unchanged_even_when_history_exists(app_settings):
    agent, llm, retrieval = make_agent(app_settings)
    agent.ask("加班制度是什么？", thread_id="topic-switch")

    result = agent.ask("病假制度是什么？", thread_id="topic-switch")

    assert result["rewrite_applied"] is False
    assert result["rewritten_query"] == "病假制度是什么？"
    assert llm.rewrite_calls == 0
    assert retrieval.calls[-1]["alternate_queries"] == []


def test_rewrite_losing_number_and_negation_falls_back_to_original(app_settings):
    agent, llm, retrieval = make_agent(app_settings)
    agent.ask("事假制度是什么？", thread_id="constraint")

    result = agent.ask("那15天内不能补交吗？", thread_id="constraint")

    assert result["rewrite_applied"] is False
    assert result["rewritten_query"] == "那15天内不能补交吗？"
    assert "丢失约束" in result["rewrite_reason"]
    assert retrieval.calls[-1]["alternate_queries"] == []


def test_ambiguous_question_without_history_asks_for_clarification(app_settings):
    agent, llm, _ = make_agent(app_settings)

    result = agent.ask("这个呢？", thread_id="ambiguous")

    assert result["decision"] == "clarify"
    assert "补充" in result["answer"]
    assert llm.answer_calls == 0


def test_no_evidence_refuses_without_calling_answer_llm(app_settings):
    agent, llm, _ = make_agent(app_settings)

    result = agent.ask("量子纠错的完整技术规范是什么？", thread_id="no-evidence")

    assert result["decision"] == "refuse"
    assert result["citations"] == []
    assert llm.answer_calls == 0


def test_retrieval_exception_is_recorded_and_degrades_to_refusal(app_settings):
    agent, _, _ = make_agent(app_settings)

    result = agent.ask("请查询检索异常的制度内容", thread_id="failure")

    assert result["decision"] == "refuse"
    assert result["errors"][0].startswith("retrieve: RuntimeError")
    assert result["trace"]["errors"][0].startswith("agent retrieve: RuntimeError")


def test_debug_summary_hides_chunk_content_and_full_trace_is_saved(app_settings, tmp_path):
    agent, _, _ = make_agent(app_settings)
    result = agent.ask("加班制度是什么？", thread_id="debug-output")

    summary = build_debug_summary(result)
    output_path = save_full_trace(result, "debug/output", tmp_path / "traces")

    assert "决策：answer" in summary
    assert "vector" not in summary or "检索通道" in summary
    assert "员工加班或请假应先提交申请" not in summary
    assert output_path.exists()
    assert "员工加班或请假应先提交申请" in output_path.read_text(encoding="utf-8")
