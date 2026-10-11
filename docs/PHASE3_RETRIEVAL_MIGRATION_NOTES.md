# 阶段 3：LlamaIndex 原生混合检索与 Rerank 变更说明

## 1. 本阶段结果

检索主流程现在是：

```text
原问题 + 可选改写问题
  ├─ LlamaIndex VectorIndexRetriever
  └─ LlamaIndex BM25Retriever
          ↓
  LlamaIndex QueryFusionRetriever（RRF）
          ↓
  LlamaIndex SentenceTransformerRerank（可选）
          ↓
  项目证据阈值 + Retrieval Trace
          ↓
  NodeWithScore
```

阶段 3 只迁移检索端。旧 `RAGChain` 暂时负责历史问题改写和答案生成，并通过薄适配器调用
新服务；阶段 4 再迁移到 LangGraph。

## 2. 原生组件与项目封装的边界

这是本次调整最重要的设计约束：**算法交给 LlamaIndex，业务编排留在项目内。**

| 能力 | 归属 | 当前实现 | 为什么放在这里 |
| --- | --- | --- | --- |
| 向量相似度召回 | LlamaIndex 原生 | `VectorStoreIndex.as_retriever()` | Chroma 查询、相似度和 Node 转换不重复实现 |
| BM25 建索引与打分 | LlamaIndex 官方集成 | `llama_index.retrievers.bm25.BM25Retriever` | 不维护另一套 BM25 公式 |
| 多通道 RRF 融合 | LlamaIndex 原生 | `QueryFusionRetriever` + `RECIPROCAL_RANK` | 不维护另一套融合公式；RRF 常数沿用框架实现 |
| CrossEncoder 重排 | LlamaIndex 原生 | `SentenceTransformerRerank` | 模型加载、批量预测和 Node 排序由框架负责 |
| 中文 token 边界 | 项目配置 | `core/retrieval/native.py` 中的正则 | 官方默认正则会把连续中文视为一个长 token，需要中文适配 |
| 多 query、多召回通道 | 项目编排 | `HybridRetrievalService` | 原问题和业务侧改写问题来自上游流程，必须保留独立 trace |
| BM25 缓存与知识库刷新 | 项目编排 | Node 指纹变化后清空 Retriever 缓存 | 这是本项目上传/更新流程的生命周期规则 |
| 零分占位项清理 | 项目规则 | BM25 结果进入 RRF 前移除 `score <= 0` | 官方 BM25 可能补齐零分项；零分不代表关键词命中 |
| 异常降级、证据阈值 | 项目规则 | 单通道失败继续；低证据返回空结果 | 这是产品可用性和拒答策略，不属于检索算法 |
| Retrieval Trace | 项目可观测性 | 记录通道、排名、耗时、错误与贡献来源 | 框架输出不足以直接回答线上排错和面试解释需求 |
| LangChain 兼容接口 | 临时项目适配 | `core/retriever.py` | 阶段 4 前保证旧问答链可运行，之后可删除 |

可以用一句话解释代码边界：

> `native.py` 只集中配置和适配 LlamaIndex 原生组件，`hybrid.py` 只编排它们；项目没有自行
> 计算 BM25、RRF、向量相似度或 CrossEncoder 分数。

### 为什么仍然需要 `HybridRetrievalService`

LlamaIndex 能提供单个组件，但不知道本项目的业务规则，例如：

- 原问题与上游改写问题都要召回，且不能互相覆盖；
- 向量通道失败时是否允许 BM25 继续回答；
- 上传或更新知识后，什么时候刷新进程内 BM25；
- 什么分数判定“证据不足”；
- trace 中需要展示哪些字段。

所以保留的是一个**应用服务/编排层**，不是第二套检索实现。

## 3. 主要文件

| 文件 | 作用 |
| --- | --- |
| `core/retrieval/native.py` | 官方 BM25、QueryFusionRetriever 的集中构造及中文 token 配置 |
| `core/retrieval/hybrid.py` | 多 query、通道编排、缓存刷新、降级、阈值和 trace |
| `core/llama_embedding.py` | 摄取端和检索端共用 Embedding 单例 |
| `core/retriever.py` | LlamaIndex Node 到旧 LangChain Document 的过渡适配 |
| `scripts/retrieve_demo.py` | 查看检索结果和完整轨迹 |
| `evaluation/run.py` | 最小 Recall@K/MRR 离线评测入口 |
| `tests/retrieval/` | 原生组件接入、业务编排和兼容接口测试 |

旧的项目自研 `bm25.py`、`fusion.py`、`reranker.py` 已删除。

## 4. 原生组件如何接入

### 4.1 VectorIndexRetriever

`VectorStoreIndex.from_vector_store()` 连接阶段 2 的 Chroma collection，再通过
`index.as_retriever()` 获取原生向量 Retriever。项目只传入 `top_k` 和 metadata filter。

### 4.2 BM25Retriever 与中文适配

新增官方依赖：

```text
llama-index-retrievers-bm25>=0.8,<1
```

官方默认 token 正则更适合英文，会把连续中文视为一个长 token。本项目只覆盖 token 边界：

```python
CHINESE_TOKEN_PATTERN = r"(?u)[\u4e00-\u9fff]|[A-Za-z0-9_]+"
```

中文按单字切分，英文、数字和下划线保持连续，因此 `HR-2024-01`、`第十五条`、`发票` 都能
参与关键词匹配。BM25 索引、IDF 与排序仍完全由官方 `BM25Retriever` 计算。

当前策略强调稳定和少依赖，不做自定义词典。若将来需要更强的中文词语边界，应优先选择官方
组件支持的 tokenizer/stemmer 扩展点或专用检索后端，而不是复制 BM25 公式。

官方 BM25 在过滤或无命中时可能返回 `score=0` 的补位结果。业务层会在 RRF 前过滤这些项，
否则“没有关键词命中”的文档会仅凭列表位置获得 RRF 分数。

### 4.3 QueryFusionRetriever

各向量/BM25 通道先执行一次并保留原始结果用于 trace，再由只读
`StaticResultsRetriever` 交给 `QueryFusionRetriever` 做 `RECIPROCAL_RANK` 融合。

这个适配器不计算 RRF，只解决两个工程问题：

1. 避免融合组件再次执行 Chroma/BM25；
2. 避免原生融合修改 score 后丢失通道原始分数。

`num_queries=1` 禁止融合器内部再次调用 LLM 改写；项目上游传入的 `alternate_queries` 已经是
明确的业务输入。RRF 常数由 LlamaIndex 固定实现，因此移除了项目配置 `RRF_RANK_CONSTANT`。

### 4.4 SentenceTransformerRerank

启用 `USE_RERANKER` 时，第一次真实查询才创建原生 `SentenceTransformerRerank`，避免服务
构造阶段就加载大模型。调用使用标准 `postprocess_nodes()` 接口。

`keep_retrieval_score=False`，防止框架把旧分数写进 Node metadata 后改变 `node.hash`，造成
BM25 指纹误判。进入重排器前还会复制 `NodeWithScore`：官方组件会原地覆盖 score，复制后
`fusion` trace 仍保留 RRF 分数，`rerank` trace 单独展示 CrossEncoder 分数，两者不会串义。

## 5. 多 Query、缓存刷新与降级

```python
result = service.retrieve(
    "那需要什么证明？",
    alternate_queries=["员工申请病假需要提供什么证明？"],
)
```

每个问题分别产生 `vector:n` 和 `bm25:n` 通道，再统一进入原生 RRF。原问题始终保留。

BM25 是进程内索引。每次检索前计算：

```text
fingerprint = SHA256(sorted(node_id + node_hash))
```

- 新增、删除或修改 Node：指纹变化，清空 BM25Retriever 缓存；
- 数据未变化：复用按 `top_k + filter` 缓存的官方 Retriever；
- 单个通道抛异常：记录 `errors`，其他通道继续；
- 所有通道都为空：返回证据不足。

该刷新方式适合当前演示规模。百万级语料应改为摄取任务发布索引版本，而不是每次查询扫描 Node。

## 6. 最低证据阈值

```dotenv
RETRIEVAL_MIN_RERANK_SCORE=0.2
RETRIEVAL_MIN_FUSION_SCORE=0.0
```

启用 rerank 时，最高 BGE 原始分数低于 `0.2` 会返回 `nodes=[]` 和
`insufficient_evidence=true`。trace 仍保留召回过程。

已做过的本地真实模型验证：

- “员工病假需要什么证明？”最高分约 `0.9744`，通过；
- “量子计算机纠错码是什么？”最高分约 `0.000037`，拒答。

这些是当前模型上的相关性原始分数，不是概率。正式阈值仍需由评测集标定。

## 7. Retrieval Trace

```json
{
  "query": "员工病假需要什么证明？",
  "queries": ["员工病假需要什么证明？"],
  "channels": {
    "vector:0": [{"rank": 1, "node_id": "...", "score": 0.81}],
    "bm25:0": [{"rank": 1, "node_id": "...", "score": 2.14}]
  },
  "fusion": [{
    "rank": 1,
    "node_id": "...",
    "score": 0.0327,
    "contributors": {"vector:0": 1, "bm25:0": 1}
  }],
  "rerank": [{"rank": 1, "node_id": "...", "score": 0.9744}],
  "score_type": "rerank_raw",
  "threshold": 0.2,
  "insufficient_evidence": false,
  "errors": [],
  "elapsed_ms": 38.2
}
```

`contributors` 由项目根据融合前排名派生，只用于解释；RRF score 本身由 LlamaIndex 计算。
trace 只保留 120 字预览，不包含密钥或调用栈。

## 8. 配置与依赖变化

```dotenv
SEARCH_TOP_K=5
RETRIEVAL_CANDIDATE_K=20
RETRIEVAL_MIN_FUSION_SCORE=0.0
RETRIEVAL_MIN_RERANK_SCORE=0.2
USE_RERANKER=true
RERANKER_MODEL_NAME=bge_reranker_base
```

- 新增 `llama-index-retrievers-bm25`；
- 删除直接依赖 `rank-bm25`、`jieba`；
- 删除 `RRF_RANK_CONSTANT`，使用 LlamaIndex 原生 RRF 行为；
- `USE_RERANKER=false` 时跳过 CrossEncoder，直接使用 RRF 结果。

## 9. 兼容现有问答链

`core/retriever.py` 暂时提供：

- `RAGRetriever.retrieve()`：返回 LangChain Document；
- `get_compression_retriever()`：返回 LangChain `BaseRetriever`；
- `get_compresstion_retriever()`：兼容旧代码中的拼写错误；
- `retrieve_with_trace()`：新代码直接取得 NodeWithScore 和 trace。

旧 `ContextualCompressionRetriever` 已从检索模块移除。旧 history-aware chain 到阶段 4 删除。

## 10. 测试与人工验收

自动测试：

```bash
.venv-v2/bin/python -m pytest tests/retrieval -q
.venv-v2/bin/python -m pytest tests/ingestion tests/retrieval -q
```

覆盖官方 BM25 类型与中文 token 配置、metadata filter、原生 RRF、双 query 通道、最低证据阈值、
同数量内容更新、原生 reranker 接口及旧 LangChain 适配契约。

人工验收：

```bash
.venv-v2/bin/python scripts/ingest_demo.py tests/sample_docs --category demo
.venv-v2/bin/python scripts/retrieve_demo.py \
  "员工全年事假最多可以请多少天？" --category demo
.venv-v2/bin/python scripts/retrieve_demo.py \
  "量子计算机纠错码是什么？" --category demo
.venv-v2/bin/python -m evaluation.run --mode retrieval --category demo
```

重点检查 `channels`、`fusion[].contributors`、`rerank_applied`、`score_type` 与
`insufficient_evidence`。

## 11. 已知边界

- 官方 BM25 当前仍是进程内全量索引，适合本地项目，不适合百万级 Node；
- 中文单字 token 稳定但粒度较细，领域词语召回质量需要通过评测持续观察；
- `0.2` 是演示阈值，需用正式评测集重新校准；
- HTTP debug trace 留到阶段 5；
- 旧 RAGChain 仍依赖 `langchain-classic`，阶段 4 后移除。

## 12. 回滚范围

恢复 `core/retriever.py`、`core/rag_chain.py`、`core/ingestion/service.py`、配置和依赖文件，并删除：

```text
core/retrieval/
core/llama_embedding.py
scripts/retrieve_demo.py
evaluation/
tests/retrieval/
docs/PHASE3_RETRIEVAL_MIGRATION_NOTES.md
```

本阶段没有改变 Chroma 数据格式，回滚代码不需要删除已有向量。
