"""评测指标的页面说明元数据。

公式只在这里维护，API 原样返回给 Streamlit，避免文档、页面和实际实现各写一套定义。
"""

from __future__ import annotations

from typing import Any


METRIC_DEFINITIONS: list[dict[str, Any]] = [
    {
        "key": "recall_at_k",
        "name": "Recall@K",
        "formula": "Top-K 至少命中一个正确来源的正例数 ÷ 正例总数",
        "meaning": "检索阶段有没有把正确资料找回来；越高越好。",
        "range": "0～1",
        "scope": "仅有正确来源的用例",
    },
    {
        "key": "mrr",
        "name": "MRR",
        "formula": "Σ(1 ÷ 第一个正确来源的排名) ÷ 正例总数",
        "meaning": "正确资料排得是否靠前；未命中贡献 0，越高越好。",
        "range": "0～1",
        "scope": "仅有正确来源的用例",
    },
    {
        "key": "source_hit_rate",
        "name": "来源命中率",
        "formula": "实际命中的期望来源数 ÷ 全部期望来源数",
        "meaning": "需要联合多份资料时，是否遗漏某个来源。",
        "range": "0～1",
        "scope": "检索/回答型用例",
    },
    {
        "key": "decision_accuracy",
        "name": "决策准确率",
        "formula": "answer / clarify / refuse 决策正确数 ÷ 全部用例数",
        "meaning": "Agent 是否进入了正确业务分支。",
        "range": "0～1",
        "scope": "全部 Agent 用例",
    },
    {
        "key": "citation_accuracy",
        "name": "引用正确率",
        "formula": "回答中来自正确文件的引用数 ÷ 引用总数，再按回答型用例平均",
        "meaning": "引用是否真的来自期望制度，而不是只看答案是否流畅。",
        "range": "0～1",
        "scope": "回答型用例；无引用记 0",
    },
    {
        "key": "keyword_coverage",
        "name": "关键词覆盖率",
        "formula": "答案命中的期望关键词数 ÷ 期望关键词总数，再按回答型用例平均",
        "meaning": "答案是否覆盖确定性关键事实；不等同于完整语义正确性。",
        "range": "0～1",
        "scope": "回答型用例",
    },
    {
        "key": "refusal_accuracy",
        "name": "拒答准确率",
        "formula": "正确输出 refuse 的无答案用例数 ÷ 应拒答用例数",
        "meaning": "知识库没有答案时，系统能否避免编造。",
        "range": "0～1",
        "scope": "expected_decision=refuse",
    },
    {
        "key": "clarify_accuracy",
        "name": "澄清准确率",
        "formula": "正确输出 clarify 的歧义用例数 ÷ 应澄清用例数",
        "meaning": "问题信息不足时，系统能否先追问而不是猜测。",
        "range": "0～1",
        "scope": "expected_decision=clarify",
    },
    {
        "key": "avg_retrieval_ms",
        "name": "平均检索耗时",
        "formula": "Σ 每次检索耗时 ÷ 检索用例数",
        "meaning": "检索链路的平均响应成本，单位毫秒。",
        "range": "≥ 0 ms",
        "scope": "检索用例",
    },
    {
        "key": "p95_retrieval_ms",
        "name": "P95 检索耗时",
        "formula": "排序后第 ceil(0.95 × N) 个真实检索耗时",
        "meaning": "95% 请求不会超过的检索耗时，更能反映慢请求。",
        "range": "≥ 0 ms",
        "scope": "nearest-rank，不插值",
    },
    {
        "key": "avg_generation_ms",
        "name": "平均生成耗时",
        "formula": "平均(端到端耗时 - 检索耗时)",
        "meaning": "非检索阶段的平均耗时，包含改写、路由、门控和回答。",
        "range": "≥ 0 ms",
        "scope": "近似值，不是首 Token 延迟",
    },
    {
        "key": "p95_generation_ms",
        "name": "P95 生成耗时",
        "formula": "非检索阶段耗时的 nearest-rank P95",
        "meaning": "生成相关慢请求的尾延迟。",
        "range": "≥ 0 ms",
        "scope": "近似值，不是首 Token 延迟",
    },
    {
        "key": "degraded_cases",
        "name": "降级用例数",
        "formula": "要求 Rerank 但 rerank_applied=false 的用例数量",
        "meaning": "识别模型加载失败等降级，避免把降级结果误当成完整实验。",
        "range": "≥ 0，越低越好",
        "scope": "vector_rerank / hybrid_rerank",
    },
]
