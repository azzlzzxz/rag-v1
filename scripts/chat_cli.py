"""阶段 4 LangGraph Agent 命令行验收工具。

用法：
    .venv-v2/bin/python scripts/chat_cli.py --thread demo-001 --debug
    .venv-v2/bin/python scripts/chat_cli.py --thread demo-001 --question "加班制度是什么？"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent.graph import RAGAgent


def _score_text(score) -> str:
    return "-" if score is None else f"{float(score):.4f}"


def build_debug_summary(result: dict) -> str:
    """生成适合终端阅读的摘要，不展开 chunk 正文和完整 trace。"""
    lines = [
        f"决策：{result.get('decision', '-')}",
        f"意图：{result.get('intent', '-')}",
        f"原问题：{result.get('original_query', '-')}",
        f"改写问题：{result.get('rewritten_query', '-')}",
        f"是否改写：{result.get('rewrite_applied', False)}",
        f"改写原因：{result.get('rewrite_reason') or '-'}",
    ]

    trace = result.get("trace") or {}
    queries = trace.get("queries") or []
    if queries:
        lines.append(f"实际检索问题：{' | '.join(queries)}")

    lines.append("检索通道：")
    channels = trace.get("channels") or {}
    if not channels:
        lines.append("  - 无")
    for channel, items in channels.items():
        top = items[0] if items else {}
        lines.append(
            f"  - {channel}: {len(items)} 条，top={top.get('node_id', '-')}，"
            f"score={_score_text(top.get('score'))}"
        )

    fusion = trace.get("fusion") or []
    rerank = trace.get("rerank") or []
    lines.extend(
        [
            f"融合结果：{len(fusion)} 条，top={fusion[0].get('node_id', '-') if fusion else '-'}，"
            f"score={_score_text(fusion[0].get('score') if fusion else None)}",
            f"重排结果：{len(rerank)} 条，top={rerank[0].get('node_id', '-') if rerank else '-'}，"
            f"score={_score_text(rerank[0].get('score') if rerank else None)}",
            f"证据判断：{'不足' if trace.get('insufficient_evidence') else '通过'}",
            f"耗时：{trace.get('elapsed_ms', '-')} ms",
        ]
    )

    citations = result.get("citations") or []
    lines.append("引用：")
    if not citations:
        lines.append("  - 无")
    for index, item in enumerate(citations, start=1):
        lines.append(
            f"  - [{index}] {item.get('source', '未知')}#"
            f"{item.get('page') or '-'}，score={_score_text(item.get('score'))}"
        )

    errors = [*(result.get("errors") or []), *(trace.get("errors") or [])]
    if errors:
        lines.append("错误/降级：")
        lines.extend(f"  - {error}" for error in dict.fromkeys(errors))
    return "\n".join(lines)


def save_full_trace(result: dict, thread_id: str, output_dir: Path) -> Path:
    """完整结果写入被 Git 忽略的 data 目录，终端只返回文件路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_thread = re.sub(r"[^A-Za-z0-9_.-]+", "-", thread_id).strip("-") or "default"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = output_dir / f"{safe_thread}-{timestamp}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_result(
    result: dict,
    *,
    debug: bool,
    full_trace: bool,
    thread_id: str,
    trace_dir: Path,
) -> None:
    if full_trace:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"助手：{result['answer']}")
    if debug:
        print("\n--- 调试摘要 ---")
        print(build_debug_summary(result))
        trace_path = save_full_trace(result, thread_id, trace_dir)
        print(f"完整 trace：{trace_path}")
    elif result.get("citations"):
        sources = ", ".join(
            f"{item['source']}#{item.get('page') or '-'}" for item in result["citations"]
        )
        print(f"引用：{sources}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph RAG 多轮对话验收")
    parser.add_argument("--thread", default="demo-001", help="SQLite checkpoint thread_id")
    parser.add_argument("--question", help="只执行一个问题；不传则进入交互模式")
    parser.add_argument("--debug", action="store_true", help="显示紧凑调试摘要，完整 trace 保存到文件")
    parser.add_argument("--full-trace", action="store_true", help="把完整 JSON 打印到终端（通常不建议）")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("data/debug_traces"),
        help="--debug 模式保存完整 trace 的目录",
    )
    parser.add_argument("--clear", action="store_true", help="开始前清空该 thread")
    args = parser.parse_args()

    agent = RAGAgent()
    if args.clear:
        agent.clear_thread(args.thread)
        print(f"已清空 thread：{args.thread}")

    if args.question:
        print_result(
            agent.ask(args.question, args.thread),
            debug=args.debug,
            full_trace=args.full_trace,
            thread_id=args.thread,
            trace_dir=args.trace_dir,
        )
        return

    print(f"当前 thread：{args.thread}；输入 exit 退出，clear 清空当前会话。")
    while True:
        question = input("你：").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "clear":
            agent.clear_thread(args.thread)
            print("当前会话已清空。")
            continue
        print_result(
            agent.ask(question, args.thread),
            debug=args.debug,
            full_trace=args.full_trace,
            thread_id=args.thread,
            trace_dir=args.trace_dir,
        )


if __name__ == "__main__":
    main()
