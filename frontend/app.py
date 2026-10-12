"""企业知识库 RAG 的 Streamlit 面试演示界面。

页面层只调用 ``RAGAPIClient``，不直接 import IngestionService、RAGChain 或 LangGraph。
这个边界保证 UI 重启不会破坏后端单例，也便于将来把 Streamlit 替换成独立 Web 前端。
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

# Streamlit 用文件路径启动时，先把项目根目录放入 import path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import APIClientError, RAGAPIClient  # noqa: E402
from frontend.reporting import load_report, metric_items  # noqa: E402


st.set_page_config(
    page_title="Enterprise RAG Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.18);}
      .hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(120,120,120,.2);
             border-radius: 18px; margin-bottom: 1.4rem;
             background: linear-gradient(135deg, rgba(34,139,230,.12), rgba(99,102,241,.07));}
      .hero h1 {font-size: 2rem; margin: 0 0 .35rem 0;}
      .hero p {margin: 0; opacity: .75;}
      .status-ok {color: #16a34a; font-weight: 650;}
      .status-off {color: #dc2626; font-weight: 650;}
      .citation {padding: .75rem 1rem; border-left: 3px solid #4f46e5;
                 background: rgba(99,102,241,.06); border-radius: 8px; margin: .4rem 0;}
      .muted {opacity: .68; font-size: .92rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(120,120,120,.18);
                                   border-radius: 14px; padding: .8rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_client(base_url: str) -> RAGAPIClient:
    """同一 API 地址复用连接池；地址变化时 Streamlit 会创建新客户端。"""
    return RAGAPIClient(base_url)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_error(message: str) -> None:
    """统一错误样式，让演示时不会出现难读的 Python 异常栈。"""
    st.error(message, icon="⚠️")


def display_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.caption("本次回答没有可展示的引用。")
        return
    with st.expander(f"查看引用来源（{len(citations)}）", expanded=True):
        for item in citations:
            location = f" · 第 {item['page']} 页" if item.get("page") is not None else ""
            score = f" · score {item['score']:.4f}" if item.get("score") is not None else ""
            st.markdown(
                f"**[{item.get('citation_id', '-')}] {item.get('file_name', '未知')}**"
                f"{location}{score}"
            )
            st.markdown(
                f'<div class="citation">{item.get("content") or "（无引用摘要）"}</div>',
                unsafe_allow_html=True,
            )


def display_trace(answer: dict[str, Any]) -> None:
    """把脱敏 trace 拆成面试时容易讲解的 Query → 召回 → 融合 → 重排结构。"""
    trace = answer.get("trace") or {}
    left, right = st.columns(2)
    with left:
        st.markdown("#### Query Rewrite")
        st.code(answer.get("original_query") or "", language=None)
        st.code(answer.get("rewritten_query") or "", language=None)
        st.caption("已改写" if answer.get("rewrite_applied") else "本次无需改写")
    with right:
        st.markdown("#### 检索决策")
        st.metric("总耗时", f"{trace.get('elapsed_ms') or 0} ms")
        st.write(
            {
                "decision": answer.get("decision"),
                "score_type": trace.get("score_type"),
                "threshold": trace.get("threshold"),
                "insufficient_evidence": trace.get("insufficient_evidence"),
            }
        )

    st.markdown("#### 多路召回")
    channels = trace.get("channels") or {}
    if channels:
        tabs = st.tabs(list(channels))
        for tab, (name, items) in zip(tabs, channels.items()):
            with tab:
                st.dataframe(items, width="stretch", hide_index=True)
    else:
        st.info("没有召回通道数据。")

    fusion_col, rerank_col = st.columns(2)
    with fusion_col:
        st.markdown("#### RRF Fusion")
        st.dataframe(trace.get("fusion") or [], width="stretch", hide_index=True)
    with rerank_col:
        st.markdown("#### Rerank")
        st.caption("已启用" if trace.get("rerank_applied") else "本次未启用")
        st.dataframe(trace.get("rerank") or [], width="stretch", hide_index=True)

    if trace.get("errors"):
        st.warning("降级信息：" + "；".join(trace["errors"]))


def init_thread_state(client: RAGAPIClient) -> str:
    """从 URL 恢复 thread_id，并在首次进入该 thread 时从后端加载 checkpoint 历史。"""
    query_thread = st.query_params.get("thread")
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = query_thread or f"demo-{uuid.uuid4().hex[:8]}"

    entered = st.sidebar.text_input(
        "Thread ID",
        value=st.session_state.thread_id,
        help="复制这个 ID 可在刷新或另一浏览器中恢复同一会话。",
    ).strip()
    if entered and entered != st.session_state.thread_id:
        st.session_state.thread_id = entered
        st.session_state.pop("history_loaded_for", None)
    thread_id = st.session_state.thread_id
    st.query_params["thread"] = thread_id

    if st.session_state.get("history_loaded_for") != thread_id:
        try:
            history = client.history(thread_id).get("history", [])
            st.session_state.messages = [
                {"role": item["role"], "content": item["content"]} for item in history
            ]
            st.session_state.history_loaded_for = thread_id
        except APIClientError:
            # API 离线时仍允许打开页面；联网后点击“重新加载历史”即可恢复。
            st.session_state.setdefault("messages", [])
    return thread_id


def sidebar(client: RAGAPIClient, thread_id: str) -> str:
    page = st.sidebar.radio(
        "演示导航",
        ["智能问答", "知识库", "检索调试", "评测看板"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    if st.sidebar.button("新建会话", width="stretch"):
        st.session_state.thread_id = f"demo-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.pop("history_loaded_for", None)
        st.query_params["thread"] = st.session_state.thread_id
        st.rerun()
    if st.sidebar.button("重新加载历史", width="stretch"):
        st.session_state.pop("history_loaded_for", None)
        st.rerun()
    if st.sidebar.button("清空当前会话", width="stretch"):
        try:
            client.delete_session(thread_id)
            st.session_state.messages = []
            st.session_state.history_loaded_for = thread_id
            st.sidebar.success("会话已清空")
        except APIClientError as exc:
            st.sidebar.error(str(exc))
    st.sidebar.caption("LlamaIndex 检索 · LangGraph 编排 · FastAPI 契约")
    return page


def render_chat(client: RAGAPIClient, thread_id: str) -> None:
    page_header("智能问答", "展示多轮记忆、生命周期流式状态和可追溯引用")
    st.caption(f"当前 thread：`{thread_id}` · URL 已保存此 ID，刷新后可恢复")

    for message in st.session_state.get("messages", []):
        role = "user" if message["role"] in {"user", "human"} else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])
            if role == "assistant" and message.get("citations"):
                display_citations(message["citations"])

    question = st.chat_input("例如：员工申请病假需要哪些材料？")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            completed: dict[str, Any] | None = None
            with st.status("正在改写问题并检索知识库…", expanded=True) as status:
                for event, payload in client.stream_answer(question, thread_id):
                    if event == "started":
                        st.write("已进入 LangGraph 工作流")
                    elif event == "completed":
                        completed = payload
                        st.write("证据门控与回答生成完成")
                    elif event == "done":
                        status.update(label="回答完成", state="complete", expanded=False)
                    elif event == "error":
                        raise APIClientError(payload.get("message", "问答失败"))
            if completed is None:
                raise APIClientError("流式响应未返回完成事件")
            st.markdown(completed["answer"])
            display_citations(completed.get("citations", []))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": completed["answer"],
                    "citations": completed.get("citations", []),
                }
            )
            st.session_state.last_answer = completed
        except APIClientError as exc:
            show_error(str(exc))


def render_knowledge(client: RAGAPIClient) -> None:
    page_header("知识库", "上传、任务状态、文档清单、删除和容量统计")
    try:
        stats = client.knowledge_stats()
        cols = st.columns(3)
        cols[0].metric("文档数", stats.get("document_count", 0))
        cols[1].metric("Chunk 数", stats.get("node_count", 0))
        cols[2].metric("分类数", len(stats.get("categories") or {}))
    except APIClientError as exc:
        show_error(str(exc))

    st.markdown("### 上传文档")
    upload_col, category_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader("支持 PDF、DOCX、XLSX、TXT、Markdown")
    with category_col:
        category = st.text_input("分类", value="general")
    if st.button("提交摄取", type="primary", disabled=uploaded is None):
        try:
            result = client.upload_document(uploaded.name, uploaded.getvalue(), category or "general")
            st.session_state.last_upload = result
            st.success(f"已提交：{result['file_name']}（file_id={result['file_id']}）")
        except APIClientError as exc:
            show_error(str(exc))

    task = st.session_state.get("last_upload")
    if task:
        task_col, button_col = st.columns([4, 1])
        task_col.info(f"最近任务：{task['file_name']} · `{task['file_id']}`")
        if button_col.button("刷新任务状态", width="stretch"):
            try:
                state = client.document_status(task["file_id"])
                if state["status"] == "succeeded":
                    st.success(f"处理完成：{state['node_count']} 个新 Chunk")
                elif state["status"] == "failed":
                    st.error(state.get("error") or "处理失败")
                else:
                    st.info(f"当前状态：{state['status']}")
            except APIClientError as exc:
                show_error(str(exc))

    st.markdown("### 已摄取文档")
    try:
        result = client.list_documents()
        documents = result.get("documents", [])
        if not documents:
            st.info("知识库尚无文档。上传成功后在这里查看。")
        for document in documents:
            info_col, delete_col = st.columns([6, 1])
            with info_col:
                st.markdown(f"**{document['file_name']}** · `{document['category']}`")
                st.caption(
                    f"{document['document_id']} · {document['source_count']} 个来源单元 · "
                    f"更新于 {document['updated_at']}"
                )
            with delete_col:
                if st.button("删除", key=f"delete-{document['document_id']}", width="stretch"):
                    client.delete_document(document["document_id"])
                    st.success("文档已删除")
                    st.rerun()
            st.divider()
    except APIClientError as exc:
        show_error(str(exc))


def render_debug(client: RAGAPIClient, thread_id: str) -> None:
    page_header("检索调试", "观察 Query Rewrite、双路召回、RRF、Rerank 与证据门控")
    with st.form("debug-query"):
        question = st.text_input("调试问题", placeholder="例如：那报销标准呢？")
        submitted = st.form_submit_button("运行完整 Trace", type="primary")
    if submitted and question.strip():
        try:
            with st.spinner("正在执行带 debug trace 的问答…"):
                st.session_state.last_debug = client.ask(question, thread_id, debug=True)
        except APIClientError as exc:
            show_error(str(exc))

    answer = st.session_state.get("last_debug")
    if answer:
        display_trace(answer)
        st.markdown("#### 最终回答与引用")
        st.markdown(answer["answer"])
        display_citations(answer.get("citations", []))
    else:
        st.info("输入一个问题后，这里会展示后端已脱敏的完整检索轨迹。")


def render_evaluation() -> None:
    page_header("评测看板", "读取离线评测报告，定位没有命中的 Bad Case")
    report_path = PROJECT_ROOT / "reports" / "latest.json"
    dataset_path = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
    try:
        report = load_report(report_path)
    except (ValueError, OSError) as exc:
        show_error(f"评测文件读取失败：{exc}")
        return

    # retrieval_cases.json 是数组，不符合 report 格式，因此只在下方用标准 json 读取数量。
    import json

    cases = json.loads(dataset_path.read_text(encoding="utf-8")) if dataset_path.exists() else []
    st.metric("已准备评测问题", len(cases))
    if report is None:
        st.info(
            "尚未生成 `reports/latest.json`。本页已完成展示契约；Recall、MRR、Bad Case "
            "计算将在阶段 7 实现，避免阶段 6 为了 UI 重复评测逻辑。"
        )
        if cases:
            st.dataframe(cases, width="stretch", hide_index=True)
        return

    metrics = metric_items(report)
    if metrics:
        columns = st.columns(min(4, len(metrics)))
        for index, (name, value) in enumerate(metrics):
            columns[index % len(columns)].metric(name, f"{value:.3f}")

    details = report.get("details") or []
    bad_cases = [item for item in details if item.get("rank") is None or item.get("passed") is False]
    st.markdown(f"### Bad Case（{len(bad_cases)} / {len(details)}）")
    if bad_cases:
        st.dataframe(bad_cases, width="stretch", hide_index=True)
    else:
        st.success("当前报告中没有 Bad Case。")
    with st.expander("查看原始评测报告"):
        st.json(report)


def main() -> None:
    api_url = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")
    client = get_client(api_url)
    st.sidebar.markdown("## ◈ RAG Console")
    try:
        health = client.health()
        st.sidebar.markdown(
            f'<span class="status-ok">● API 在线</span> · v{health.get("version", "-")}',
            unsafe_allow_html=True,
        )
    except APIClientError:
        st.sidebar.markdown('<span class="status-off">● API 离线</span>', unsafe_allow_html=True)
    thread_id = init_thread_state(client)
    page = sidebar(client, thread_id)
    if page == "智能问答":
        render_chat(client, thread_id)
    elif page == "知识库":
        render_knowledge(client)
    elif page == "检索调试":
        render_debug(client, thread_id)
    else:
        render_evaluation()


if __name__ == "__main__":
    main()
