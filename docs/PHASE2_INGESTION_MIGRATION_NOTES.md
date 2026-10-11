# 阶段 2：LlamaIndex 数据摄取层变更说明

## 1. 本阶段结果

文件上传后的主流程已由旧的 `DocumentLoader → RecursiveCharacterTextSplitter → LangChain Chroma.add_documents` 切换为：

```text
Upload API
  → EnterpriseDocumentReader（生成 LlamaIndex Document）
  → IngestionPipeline
      ├─ SentenceSplitter
      ├─ 稳定 chunk_id / metadata
      ├─ HuggingFaceEmbedding（本地 BGE）
      ├─ IngestionCache
      └─ SimpleDocumentStore（UPSERTS）
  → Chroma: langchain collection
```

阶段 2 只替换摄取端。现有问答、Retriever 和 RAGChain 暂时保留，阶段 3 再迁移检索端。

## 2. 主要文件

| 文件 | 作用 |
| --- | --- |
| `core/ingestion/readers.py` | PDF、DOCX、XLSX、TXT、Markdown 的格式感知读取 |
| `core/ingestion/ids.py` | 稳定 document/source/chunk ID 和 SHA-256 |
| `core/ingestion/service.py` | 组装 Pipeline、Chroma、Embedding、cache、docstore 和 manifest |
| `core/ingestion/jobs.py` | SQLite 上传任务状态仓储 |
| `api/routes/upload.py` | 新摄取服务接入、202 响应和状态查询 |
| `scripts/ingest_demo.py` | 命令行摄取目录或单个文件 |
| `scripts/list_nodes.py` | 查看 Chroma Node 与 metadata |
| `tests/ingestion/` | 阶段 2 自动测试 |

## 3. 格式处理规则

- PDF：每页一个来源单元，`page` 从 1 开始；后续 token 超限时继续切片。
- DOCX：`docx2txt` 提取自然段，每 20 段形成一个来源单元。
- XLSX：按 Sheet 和行组生成 Markdown；每组重复列名，单个召回片段也能理解列语义。
- Markdown：标题作为来源单元边界。
- TXT：先作为一个来源单元，再由 `SentenceSplitter` 切片。
- 旧二进制 `.doc`、`.xls`：本阶段不再把它们列为直接支持格式。建议先转换为 `.docx`、`.xlsx`，避免依赖 LibreOffice、antiword、xlrd 等不可控系统组件。

## 4. ID、去重与更新语义

### 4.1 稳定 ID

- `document_id`：由 `category + 原始文件名` 规范化后计算，不使用上传时生成的随机前缀。
- `source_id`：由 `document_id + 页码/Sheet 行组/段落组` 计算。
- `chunk_id`：由 `source_id + chunk 序号 + chunk 文本` 计算。
- `content_hash`：是来源单元正文的 SHA-256。

因此，同一 category 下再次上传同名文件会被视为更新；不同 category 的同名文件相互独立。

### 4.2 重复文件

`data/ingestion/manifest.json` 保存整个文件 hash。相同内容再次上传时，在加载模型和执行 Embedding 前直接返回 `skipped=true`，不会重复写入 Chroma。

### 4.3 内容更新

LlamaIndex `DocstoreStrategy.UPSERTS` 比较稳定 `source_id` 与 Node hash：

- 未变化的页/Sheet/段落组跳过；
- 发生变化的来源单元删除旧 Node 后重新切片和 Embedding；
- manifest 额外记录旧 `source_ids`，用于删除新版本中已经消失的页或 Sheet。

### 4.4 LlamaIndex 的保留字段

LlamaIndex Chroma 适配器会将 Chroma 顶层 metadata 的 `document_id` 用作 `ref_doc_id`。业务级 `document_id` 仍完整保存在序列化 Node 的 metadata 中；阶段 3 统一输出引用时，以 LlamaIndex Node 中的业务值为准。

## 5. 上传任务状态

接口：

```text
POST /api/v1/upload/file
GET  /api/v1/upload/status/{file_id}
POST /api/v1/upload/batch
```

上传成功返回 HTTP `202` 和 `pending`。状态保存在 `data/ingestion/jobs.db`：

```text
pending → running → succeeded
                  ↘ failed
```

成功状态包含 `node_count` 和 `skipped`。失败信息最多保存 500 字符，不返回内部堆栈。

## 6. 与旧检索链的兼容

- LlamaIndex 和旧 LangChain 适配器暂时共用 `langchain` collection。
- 修复了旧适配器传入 `client_settings` 后未自动启用 `is_persistent` 的问题。
- 阶段 2 新写入的向量已验证能由旧 LangChain `similarity_search` 读取。
- `langchain-community.Chroma` 的弃用警告属于阶段 3 待删除项。

## 7. 自动测试结果

阶段 2 专项测试：

```bash
.venv-v2/bin/python -m pytest tests/ingestion -q
```

阶段 2 专项用例为 7 项；连同阶段 1 回归契约共执行 16 项，全部通过。覆盖 PDF 页码、Excel 表头、稳定 ID、重复跳过、内容更新、任务生命周期、状态接口、配置与安全基线。

真实临时库验证结果：

- 本地 `bge_small_zh` 加载成功；
- 首次 TXT 摄取产生 4 个 Node；
- 第二次相同文件 `skipped=true`；
- Chroma 中共 4 个 Node；
- Node 包含 `document_id/chunk_id/content_hash/category/source_path` 等 metadata；
- 旧 LangChain 检索器返回 1 条结果。
- 两页 PDF 更新其中一页时仅重新生成 1 个 Node；删除该页后 Chroma 最终只保留 1 个 Node。

## 8. 人工验收步骤

### 8.1 专项测试

```bash
cd /Users/steinsgate/Desktop/Study/Agent-BZ/rag-v1
.venv-v2/bin/python -m pytest tests/ingestion -q
```

预期：`7 passed`。

### 8.2 使用独立测试目录（推荐）

为避免把面试演示数据写入现有知识库，可在 `.env` 临时指定：

```dotenv
DATA_DIR=./data-phase2-test
VECTOR_DB_DIR=./vector-db-phase2-test
UPLOAD_DIR=./upload-phase2-test
INGESTION_STORAGE_DIR=./data-phase2-test/ingestion
```

然后运行：

```bash
.venv-v2/bin/python scripts/ingest_demo.py tests/sample_docs
.venv-v2/bin/python scripts/ingest_demo.py tests/sample_docs
.venv-v2/bin/python scripts/list_nodes.py --limit 10
```

第一次应显示大于 0 的 `nodes`；第二次应显示 `nodes=0, skipped=True`。

### 8.3 API 验证

```bash
.venv-v2/bin/python run_api.py
curl -F "file=@tests/sample_docs/test.txt" -F "category=demo" \
  http://127.0.0.1:8000/api/v1/upload/file
curl http://127.0.0.1:8000/api/v1/upload/status/替换为上一步的file_id
```

状态最终应为 `succeeded`。再次上传同一文件，最终状态中的 `skipped` 应为 `true`。

## 9. 已知边界

- PDF 本阶段使用文本层提取；扫描 PDF/OCR 和复杂 PDF 表格解析留作后续增强。
- DOCX 当前按自然段分组，`docx2txt` 不保留完整标题样式树。
- 单进程内使用线程锁保护 manifest/docstore/cache；多进程部署前应把摄取 Worker 独立化，或增加数据库/分布式锁。
- 状态任务由 FastAPI `BackgroundTasks` 执行，适合本地演示；生产环境建议换成 Celery、RQ 或独立任务队列。
- `.doc`、`.xls` 需要先转换为现代 Office 格式。

## 10. 回滚范围

如需回滚阶段 2，只需恢复 `api/routes/upload.py`、`api/dependencies.py`、`core/vector_store.py`、`config/settings.py`，并删除：

```text
core/ingestion/
scripts/ingest_demo.py
scripts/list_nodes.py
tests/ingestion/
docs/PHASE2_INGESTION_MIGRATION_NOTES.md
```

运行时生成的 `data/ingestion/` 与新写入的 Chroma Node 需要按测试数据范围单独清理，不建议直接删除已有正式向量库。
