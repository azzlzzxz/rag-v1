from core.agent.graph import RAGAgent

from .conftest import FakeLLM, FakeRetrievalService


def test_sqlite_checkpoint_restores_history_after_new_agent_instance(app_settings):
    first_llm = FakeLLM()
    first = RAGAgent(
        app_settings,
        llm=first_llm,
        retrieval_service=FakeRetrievalService(),
    )
    first.ask("加班制度是什么？", thread_id="persistent-thread")
    first.close()

    second_llm = FakeLLM()
    second_retrieval = FakeRetrievalService()
    second = RAGAgent(
        app_settings,
        llm=second_llm,
        retrieval_service=second_retrieval,
    )
    result = second.ask("那怎么申请？", thread_id="persistent-thread")

    assert result["rewrite_applied"] is True
    assert second_llm.rewrite_calls == 1
    assert len(second.get_chat_history("persistent-thread")) == 4
    second.close()


def test_clear_thread_only_removes_target_checkpoint(app_settings):
    agent = RAGAgent(
        app_settings,
        llm=FakeLLM(),
        retrieval_service=FakeRetrievalService(),
    )
    agent.ask("加班制度是什么？", thread_id="to-clear")
    agent.ask("请假制度是什么？", thread_id="to-keep")

    agent.clear_thread("to-clear")

    assert agent.get_chat_history("to-clear") == []
    assert len(agent.get_chat_history("to-keep")) == 2
    agent.close()
