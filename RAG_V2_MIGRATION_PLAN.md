# 企业知识库 RAG Agent V2 改造计划

> 项目：`rag-v1`  
> 目标分支：`main`  
> 当前基线：LangChain `0.3.7`、LangChain Community `0.3.5`、Chroma/FAISS、自定义文档加载与检索  
> 目标：在保留可运行基线的前提下，引入 LlamaIndex 负责数据摄取与检索，升级到 LangChain 1.x / LangGraph 1.x，并形成可演示、可评测、可恢复的企业知识库 Agent。

## 1. 改造原则

1. **始终保持可运行**：每一阶段只做一组内聚改动，自动测试通过后暂停，由项目负责人在本机实际运行并确认，再进入下一阶段。
2. **不同时重写全部模块**：先建立测试和兼容边界，再替换数据层、检索层和 Agent 编排层。
3. **两个框架职责分离**：LlamaIndex 负责数据与检索；LangChain 1.x 负责模型、工具与 Agent 接口；LangGraph 1.x 负责有状态流程和 checkpoint。
4. **避免长期依赖 legacy API**：不把 `langchain-classic` 作为目标架构。旧 LCEL chain 只在迁移期间作为对照，最终移除 `create_history_aware_retriever`、`create_stuff_documents_chain`、`create_retrieval_chain` 等旧链式实现。
5. **每阶段可回滚**：每次仅修改明确文件；项目负责人验收后再提交该阶段。已有未提交的 `config/settings.py` 修改必须保留，不覆盖、不丢弃。
6. **密钥不进入代码和日志**：现有明文 API Key 必须撤销并迁移到 `.env`；所有样例只保留占位符。

## 2. 技术职责边界

### 2.1 LlamaIndex：数据与检索层

LlamaIndex 负责：

- Reader 与多格式文档加载；
- `Document` / `Node` 及统一 metadata；
- `IngestionPipeline` transformations；
- 格式感知切片；
- ingestion cache；
- docstore、文档 hash、重复检测和增量更新；
- Chroma 适配和 `VectorStoreIndex`；
- `VectorIndexRetriever`；
- 中文 BM25 自定义 retriever；
- RRF/Query Fusion、rerank 和 NodeWithScore；
- 检索评测输入输出。

不让 LlamaIndex 负责最终 Agent 编排，避免与 LangGraph 重叠。

### 2.2 LangChain 1.x：模型和 Agent 标准接口

LangChain 1.x 负责：

- `init_chat_model` 或 provider 包中的 ChatModel；
- message、tool、structured output 等标准接口；
- `create_agent` 仅用于需要自主工具选择的 Agent 场景；
- middleware：模型调用错误、重试、日志、输入输出约束；
- ProviderStrategy / ToolStrategy 结构化输出。

本项目的确定性 RAG 主链不会强行使用 ReAct 循环；知识检索会包装成可复用 tool，供后续 Agent 使用。

### 2.3 LangGraph 1.x：确定性业务工作流

LangGraph 负责：

- Query Rewrite；
- 意图/路由判断；
- 检索；
- 证据评分；
- 回答、澄清和拒答分支；
- SQLite checkpoint；
- thread/session 恢复；
- 流式事件；
- 后续人工确认与多 Agent 扩展点。

### 2.4 应用层

- FastAPI：HTTP API、校验、错误码、健康检查、流式接口；
- Streamlit：演示与调试界面；
- SQLite：文档任务、会话元数据、评测运行记录；
- Chroma：向量与 Node 内容；
- pytest：单元、契约和端到端测试。

## 3. 目标架构

```text
文件上传
  ↓
LlamaIndex Readers
  ↓
IngestionPipeline
  ├─ 清洗
  ├─ 格式感知切片
  ├─ metadata / document_id / chunk_id
  ├─ HuggingFace Embedding
  └─ cache + docstore 去重/更新
  ↓
Chroma Vector Store + BM25 Index

用户问题 + thread_id
  ↓
LangGraph StateGraph
  ├─ load_context
  ├─ rewrite_query
  ├─ route_intent
  ├─ LlamaIndex hybrid_retrieve
  ├─ rerank
  ├─ grade_evidence
  ├─ answer / clarify / refuse
  └─ persist_result
  ↓
回答 + 引用 + 检索轨迹 + checkpoint
```

## 4. 依赖迁移策略

### 4.1 目标依赖族

实际版本号在阶段 1 中通过干净虚拟环境解析并锁定，不在规划阶段猜测补丁版本。目标约束为：

```text
Python >= 3.10,<3.13

langchain >= 1.0,<2.0
langchain-core >= 1.0,<2.0
langchain-openai >= 1.0,<2.0
langgraph >= 1.0,<2.0
langgraph-checkpoint-sqlite >= 1.0

llama-index-core
llama-index-readers-file
llama-index-vector-stores-chroma
llama-index-embeddings-huggingface
llama-index-llms-openai-like（如所选版本支持且确有需要）

chromadb
sentence-transformers
rank-bm25
jieba
fastapi
uvicorn
pydantic-settings
pytest
pytest-asyncio
httpx
```

### 4.2 不做原地盲升

当前 `.venv` 保留作为旧基线。阶段 1 新建干净的 `.venv-v2` 做兼容性验证：

1. 安装最小依赖；
2. 输出完整 `pip freeze`；
3. 先验证模型、Embedding、Chroma、FastAPI 能独立 import；
4. 再写锁定依赖；
5. 只有新环境冒烟通过后，才将其作为默认开发环境。

### 4.3 LangChain 0.3 → 1.x 重点风险

- `langchain` 顶级命名空间已精简；旧 chains、retrievers 和 indexing API 被移到 `langchain-classic`。
- 当前项目的 `create_history_aware_retriever`、`create_stuff_documents_chain`、`create_retrieval_chain` 不作为目标实现保留。
- 新 Agent 优先使用 `langchain.agents.create_agent`；自定义确定性图继续使用 LangGraph `StateGraph`。
- Agent 自定义 state 使用 `TypedDict`；不要使用 Pydantic model 充当 Agent state。
- 动态 prompt、模型前后处理和 tool error handling 使用 middleware。
- structured output 使用 ProviderStrategy / ToolStrategy，不再依赖纯 prompt 约定 JSON。
- Python 至少为 3.10；当前项目的 Python 3.10 可作为最低兼容环境，但推荐最终验证 Python 3.11。

## 5. 数据模型约定

每个 LlamaIndex Node 至少保存：

```json
{
  "document_id": "稳定文档 ID",
  "chunk_id": "稳定片段 ID",
  "file_name": "员工制度.pdf",
  "file_type": "pdf",
  "source_path": "upload/...",
  "page": 12,
  "title_path": ["考勤制度", "加班申请"],
  "category": "hr",
  "content_hash": "...",
  "ingested_at": "ISO-8601"
}
```

ID 原则：

- `document_id` 不依赖上传后的随机文件名；优先使用租户/知识库/原始文件名的规范化组合或业务 ID。
- `chunk_id` 由 document_id、页码/标题路径、chunk 序号和内容 hash 派生。
- 同一文档内容未变化时不重复 Embedding；变化时可定位更新和删除旧 Node。

## 6. 分阶段实施与人工验收门禁

> 规则：每个阶段完成后停止。Codex 提供变更摘要、自动测试结果和本机测试命令；项目负责人亲自运行并明确确认后，才进入下一阶段。

### 阶段 0：规划与基线保护

**改动**

- 新增本规划文档；
- 确认当前位于 `main`；
- 记录工作区已有未提交修改；
- 不修改现有业务代码。

**验收**

```bash
git branch --show-current
git status --short
```

预期：分支为 `main`；除原有 `config/settings.py` 修改外，只新增本规划文档。

---

### 阶段 1：安全、配置与依赖兼容性基线

**目标**：项目能在干净的新环境中启动，不泄露密钥，为双框架迁移建立稳定依赖集。

**改动范围**

- `requirements.txt` 拆为运行依赖与开发依赖，或迁移到 `pyproject.toml`；
- 建立锁定文件；
- 修复 Pydantic Settings 2.x 配置；
- 新增 `.env.example`；
- 删除代码中的明文 Key；
- 增加 `ALLOWED_ORIGINS`；
- 修复当前 API 方法名和参数不一致；
- 新增 `run_api.py`；
- 新增最小 import/smoke tests；
- 保留旧 RAG 主链，暂不引入行为变化。

**自动测试**

```bash
python -m pytest tests/test_config.py tests/test_api_smoke.py -q
python -c "import langchain, langgraph, llama_index.core; print('dependency imports ok')"
```

**人工验收**

```bash
python run_api.py
curl http://127.0.0.1:8000/api/v1/system/health
open http://127.0.0.1:8000/docs
```

**通过标准**

- API 正常启动；
- Swagger 可打开；
- 健康检查成功；
- 无 Key 出现在源码、响应和日志；
- 旧问答接口至少保持原有行为。

---

### 阶段 2：LlamaIndex 数据摄取层

**目标**：用 LlamaIndex 取代自定义的“加载 → 切片 → Embedding → 入库”主流程。

**改动范围**

- 新增 `core/ingestion/`；
- 使用 LlamaIndex Reader 和 `IngestionPipeline`；
- PDF 按页保留页码；
- Word 按标题/段落处理；
- Excel 按 sheet、表头与行组生成 Markdown Node；
- TXT/Markdown 按标题和段落处理；
- 接入本地 BGE HuggingFace Embedding；
- 接入 Chroma vector store；
- 配置 ingestion cache 与 docstore；
- 支持内容 hash、重复跳过、变更更新；
- 上传任务增加 `pending/running/succeeded/failed` 状态。

**自动测试**

```bash
python -m pytest tests/ingestion -q
```

覆盖：PDF 页码、Excel 表头、metadata、稳定 ID、重复导入不重复写入、修改后更新、失败任务状态。

**人工验收**

```bash
python scripts/ingest_demo.py tests/sample_docs
python scripts/list_nodes.py --limit 10
```

项目负责人检查：

- 文档数量和 Node 数量合理；
- PDF 页码正确；
- Excel chunk 能独立理解；
- 第二次摄取相同文件不会重复写入；
- 修改文件后只更新相关内容。

---

### 阶段 3：LlamaIndex 混合检索与 Rerank

**目标**：检索端完全由 LlamaIndex Node/Index 体系承载，同时保留中文关键词优势和可观察结果。

**改动范围**

- `VectorIndexRetriever` 负责语义召回；
- 自定义 `ChineseBM25Retriever` 使用 jieba + rank-bm25；
- 使用 Query Fusion/RRF，或实现等价可测试的 RRF 融合器；
- 原始问题与改写问题可分别召回再融合；
- BGE CrossEncoder 作为 postprocessor/reranker；
- 统一输出 NodeWithScore 和 retrieval trace；
- 增加最低证据阈值与无证据结果；
- 上传、删除、更新后同步刷新 BM25 索引。

**注意**

- 不直接加权 Chroma distance 与 BM25 score，因为两者量纲不同；采用 RRF 基于排名融合。
- rerank 分数和向量距离不伪装成概率；前端称为“相关性指标”。

**自动测试**

```bash
python -m pytest tests/retrieval -q
python -m evaluation.run --mode retrieval
```

**人工验收问题**

1. 语义改写问题；
2. 包含制度编号/专有名词的问题；
3. 表格数据问题；
4. 知识库不存在的问题。

检查每个问题的 vector、BM25、fusion、rerank 排名和最终引用。

---

### 阶段 4：LangChain 1.x 模型层与 LangGraph 1.x RAG Agent

**目标**：删除旧 Chain 依赖，建立有状态、可恢复、可解释的工作流。

**Graph 节点**

```text
load_context
  → rewrite_query
  → route_intent
  → retrieve
  → grade_evidence
      ├─ answer
      ├─ clarify
      └─ refuse
  → persist_result
```

**改动范围**

- 新增 TypedDict graph state；
- Query Rewrite 使用结构化输出；
- 保留 original_query、rewritten_query、rewrite_applied、rewrite_reason；
- 关键实体、数字、时间、否定约束做一致性检查；
- 原问题与改写问题双路检索，降低语义漂移风险；
- 检索封装为 LangChain tool，为后续 `create_agent` 留接口；
- SQLite checkpointer；
- 使用 `thread_id` 恢复多轮会话；
- 节点异常分类、重试与降级；
- 最终输出 answer/citations/decision/trace；
- 移除旧 `create_retrieval_chain` 主链。

**自动测试**

```bash
python -m pytest tests/agent -q
```

覆盖：独立问题不改写、指代问题正确改写、约束不丢失、检索失败拒答、歧义澄清、checkpoint 恢复、同一 thread 连续对话。

**人工验收**

```bash
python scripts/chat_cli.py --thread demo-001 --debug
```

对话：

1. “加班制度是什么？”
2. “那怎么申请？”
3. 重启程序；
4. 使用同一 thread 继续问“审批人是谁？”

预期：可看到 rewrite、检索轨迹、引用；重启后上下文仍可恢复。

---

### 阶段 5：API 契约、引用与流式事件

**目标**：前端和评测都通过稳定 API 使用 Agent。

**接口**

- `POST /api/v1/documents/upload`
- `GET /api/v1/documents/{id}/status`
- `DELETE /api/v1/documents/{id}`
- `POST /api/v1/qa/ask`
- `POST /api/v1/qa/stream`
- `GET /api/v1/sessions/{thread_id}/history`
- `DELETE /api/v1/sessions/{thread_id}`
- `GET /api/v1/knowledge/stats`
- `GET /api/v1/system/health`
- `GET /api/v1/system/ready`

**回答契约**

```json
{
  "answer": "……[1]",
  "decision": "answer",
  "thread_id": "demo-001",
  "original_query": "怎么申请？",
  "rewritten_query": "员工加班应该如何申请？",
  "citations": [
    {
      "citation_id": 1,
      "document_id": "...",
      "chunk_id": "...",
      "file_name": "员工制度.pdf",
      "page": 12,
      "content": "..."
    }
  ],
  "trace": null
}
```

默认不向普通用户返回完整 trace；仅在 debug 模式下返回经脱敏的检索轨迹。

**自动测试**

```bash
python -m pytest tests/api -q
```

**人工验收**

使用 Swagger 完成上传、轮询状态、提问、查看引用、继续追问、清理会话全流程。

---

### 阶段 6：Streamlit 面试演示界面

**目标**：5～8 分钟内完整演示系统能力。

**页面**

1. 智能问答：回答、引用、流式状态、thread_id；
2. 知识库：上传、处理状态、文档列表、删除、统计；
3. 调试面板：Query Rewrite、召回排名、RRF、rerank、耗时；
4. 评测页：指标、Bad Case 列表。

**人工验收**

```bash
python run_api.py
python run_frontend.py
```

浏览器完成一次端到端演示，并确认前端刷新后 thread 可恢复。

---

### 阶段 7：评测、Bad Case 与交付文档

**目标**：证明优化有效，而不只是展示一个能回答问题的 Demo。

**评测集**

- 直接事实问题；
- 语义改写；
- 编号/专有词；
- 表格问题；
- 多轮指代；
- 相似制度混淆；
- 无答案拒答；
- 信息不足澄清。

**指标**

- Recall@K；
- MRR；
- 来源命中率；
- 引用正确率；
- 关键词覆盖率；
- 拒答准确率；
- 平均/P95 检索耗时；
- 平均/P95 生成耗时。

**对比实验**

至少保留：

- vector only；
- vector + rerank；
- vector + BM25 + RRF + rerank。

**验收**

```bash
python -m evaluation.run --all --output reports/latest.json
python -m pytest -q
```

README 最终包含架构、快速启动、配置、演示脚本、评测结果、已知限制和生产化路线。

## 7. 今日范围与降级顺序

若要求单日完成并可运行，优先级如下：

### 必须完成

1. 阶段 1：安全、配置、依赖与现有接口修复；
2. 阶段 2：LlamaIndex 摄取主链；
3. 阶段 3：LlamaIndex 检索 + rerank；
4. 阶段 4：最小 LangGraph Agent + SQLite checkpoint；
5. 阶段 5：稳定问答 API；
6. 最小 Streamlit 问答/上传页；
7. 关键 pytest 和 10～15 条评测数据。

### 时间不足时依次推迟

1. Docker；
2. 流式回答；
3. 完整评测 UI；
4. Excel 高级表格处理；
5. ingestion 并行处理；
6. LLM-as-a-Judge；
7. 多 Agent。

不推迟：密钥治理、可运行基线、引用、拒答、Query Rewrite 可观察性、检索评测。

## 8. 提交与测试协作规则

每阶段采用相同节奏：

1. Codex 开始前报告本阶段文件范围；
2. Codex 修改代码；
3. Codex 运行该阶段自动测试；
4. Codex 汇报变更、测试结果、风险和本机命令；
5. **停止继续改造**；
6. 项目负责人实际运行；
7. 若失败，优先修复当前阶段；
8. 项目负责人确认通过后，再进入下一阶段；
9. 阶段提交只包含该阶段相关文件，不混入已有个人改动。

建议提交信息：

```text
chore: establish secure v2 dependency baseline
feat: migrate ingestion pipeline to llamaindex
feat: add llamaindex hybrid retrieval and reranking
feat: migrate rag workflow to langgraph v1
feat: expose stable rag agent api
feat: add streamlit interview demo
test: add rag evaluation and bad-case reports
```

## 9. 完成定义

项目达到 V2 完成状态，需要同时满足：

- 全新环境按 README 可以启动；
- 上传文件后能看到明确处理状态；
- 重复上传不会重复向量化；
- PDF/Excel metadata 和引用正确；
- Query Rewrite 前后可观察且约束不漂移；
- 混合检索与 rerank 有可解释 trace；
- 无证据时能拒答，不只依赖 prompt；
- 多轮会话重启后可以恢复；
- API 和 Streamlit 均能完成端到端演示；
- 评测脚本输出检索、引用和拒答指标；
- 源码、Git 历史新增内容和日志中没有有效密钥；
- 所有自动测试通过。

## 10. 官方参考

- LangChain v1 migration guide: <https://docs.langchain.com/oss/python/migrate/langchain-v1>
- LangGraph v1 migration guide: <https://docs.langchain.com/oss/python/migrate/langgraph-v1>
- LlamaIndex Ingestion Pipeline: <https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/>
- LlamaIndex Retriever: <https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/>

---

本文档是实施门禁，不代表所有阶段必须一次性连续执行。每阶段以“本机实际可运行”为最高优先级。
