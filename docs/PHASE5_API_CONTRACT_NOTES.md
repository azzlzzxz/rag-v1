# 阶段 5：稳定 API 契约、引用与流式事件变更说明

## 1. 本阶段结果

阶段 5 为文档摄取、Agent 问答、会话恢复和知识库管理建立了稳定的 HTTP 边界：

```text
调用方
  ├─ /documents  → LlamaIndex 摄取与文档生命周期
  ├─ /qa         → LangGraph Agent 同步回答 / SSE 事件
  ├─ /sessions   → SQLite checkpoint 历史与清理
  ├─ /knowledge  → 知识库统计
  └─ /system     → 健康与就绪探针
```

Pydantic Schema 现在是 API 的唯一响应契约。内部 Node、LangGraph State、模型异常和完整检索
trace 不再直接透传给客户端。

旧 `/knowledge/delete` 与 `/knowledge/clear` 已从 HTTP 层移除：前者接受任意 source_path，
后者容易造成 Chroma、manifest 与 docstore 状态不一致。删除统一使用稳定 document_id。

## 2. 稳定接口

| 方法 | 路径 | ID 含义 | 说明 |
| --- | --- | --- | --- |
| POST | `/api/v1/documents/upload` | — | 上传文件，返回 `file_id` 和 `document_id` |
| GET | `/api/v1/documents/{file_id}/status` | 后台任务 ID | 查询 pending/running/succeeded/failed |
| DELETE | `/api/v1/documents/{document_id}` | 稳定业务文档 ID | 删除整篇文档及其 chunks |
| POST | `/api/v1/qa/ask` | body 中 `thread_id` | 同步问答 |
| POST | `/api/v1/qa/stream` | body 中 `thread_id` | SSE 生命周期事件 |
| GET | `/api/v1/sessions/{thread_id}/history` | LangGraph thread | 获取持久化消息 |
| DELETE | `/api/v1/sessions/{thread_id}` | LangGraph thread | 删除指定会话 |
| GET | `/api/v1/knowledge/stats` | — | 知识库统计 |
| GET | `/api/v1/system/health` | — | 进程存活检查 |
| GET | `/api/v1/system/ready` | — | 运行目录就绪检查 |

`file_id` 和 `document_id` 不同：

- `file_id` 标识一次上传/摄取任务，用于轮询状态；
- `document_id` 由分类和原始文件名稳定生成，用于更新和删除业务文档。

## 3. 主要文件

| 文件 | 作用 |
| --- | --- |
| `api/schemas/qa.py` | 问答、引用和历史的稳定 Pydantic 契约 |
| `api/schemas/documents.py` | 上传、状态和删除契约 |
| `api/presenters.py` | Agent 内部结果到公开响应的转换与 trace 脱敏 |
| `api/routes/qa.py` | 同步问答、SSE 和旧问答路径兼容 |
| `api/routes/documents.py` | 新版文档生命周期 API |
| `api/routes/sessions.py` | checkpoint 历史与 thread 删除 |
| `core/ingestion/service.py` | 增加按 document_id 的安全删除 |
| `tests/api/` | 阶段 5 API 契约测试 |

关键逻辑均添加了中文注释，重点解释线程池边界、SSE 已发出 200 后的错误处理、trace 脱敏、
`file_id/document_id` 区别和文件删除安全校验。

## 4. 问答契约

请求：

```json
{
  "question": "那怎么申请？",
  "thread_id": "demo-001",
  "debug": false
}
```

响应：

```json
{
  "answer": "员工应先提交加班申请，审批通过后执行。[1]",
  "decision": "answer",
  "thread_id": "demo-001",
  "original_query": "那怎么申请？",
  "rewritten_query": "员工加班应该怎么申请？",
  "rewrite_applied": true,
  "citations": [
    {
      "citation_id": 1,
      "document_id": "doc-...",
      "chunk_id": "chk-...",
      "file_name": "员工制度.pdf",
      "page": 12,
      "content": "员工提交加班申请……",
      "score": 0.9744
    }
  ],
  "trace": null
}
```

`decision` 只可能是：

- `answer`：证据通过门控；
- `clarify`：问题缺少指代对象；
- `refuse`：问题完整，但知识库证据不足。

请求中的 `session_id` 仍可作为 `thread_id` 的兼容别名，但 OpenAPI 和新代码统一使用
`thread_id`。空白问题、空 thread、超长输入和额外未知字段由 Pydantic 返回 422。

## 5. 引用契约

每条引用都有本次回答内从 1 开始的 `citation_id`，并同时返回：

- `document_id`：稳定业务文档；
- `chunk_id`：准确定位检索 chunk；
- `file_name`、`page`：用户可理解的来源；
- `content`：最多约 200 字的引用片段；
- `score`：当前 rerank/RRF 的相关性原始分数，不是概率。

引用数据来自阶段 2 写入的 Node metadata 和阶段 4 Agent citations，不再从文件名猜测 ID。

## 6. Trace 可见性与脱敏

普通请求：

```text
debug=false → trace=null
```

调试请求：

```text
debug=true → 返回脱敏 trace
```

允许返回：

- 实际检索 queries；
- vector/BM25/fusion/rerank 排名；
- node_id、分数、文件名、页码；
- contributors、阈值、耗时和证据判断；
- 通道/节点名与异常类型。

禁止返回：

- chunk `preview` 和完整 content；
- 本地绝对路径；
- 第三方 API 响应正文；
- 异常详细消息、调用栈、密钥和完整配置。

因此 debug trace 可以用于前端调试面板和面试展示，但不能替代服务端日志。

## 7. SSE 流式事件

请求：

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/qa/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"那怎么申请？","thread_id":"demo-001"}'
```

成功事件顺序：

```text
event: started
data: {"thread_id":"demo-001"}

event: completed
data: {完整 AnswerResponse}

event: done
data: {"thread_id":"demo-001"}
```

失败时发送：

```text
event: error
data: {"thread_id":"demo-001","message":"问答处理失败，请稍后重试"}
```

当前阶段流的是**生命周期状态和完整结果**，不是逐 token 输出。这已经能让阶段 6 前端展示
“处理中/完成/失败”状态。未来新增 `token` 事件不会破坏已有事件消费者。

响应包含：

```text
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

模型、检索和 SQLite 都是同步调用，路由使用线程池执行，避免阻塞 FastAPI 事件循环。

## 8. 文档生命周期

### 8.1 上传

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F 'file=@tests/sample_docs/test.txt' \
  -F 'category=demo'
```

返回 202：

```json
{
  "file_id": "ab12cd34",
  "document_id": "doc-...",
  "file_name": "test.txt",
  "category": "demo",
  "status": "pending"
}
```

### 8.2 状态

```bash
curl http://127.0.0.1:8000/api/v1/documents/ab12cd34/status
```

失败任务只返回统一公开错误文案；SQLite 和服务端日志保留详细异常，API 不暴露本地路径或第三方
响应正文。

### 8.3 删除

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/doc-...
```

删除会同步清理：

1. Chroma 中该 document_id 的全部 chunks；
2. LlamaIndex docstore 中的 source references；
3. ingestion manifest；
4. 仅当路径确实位于 `UPLOAD_DIR` 内时，删除上传原文件。

接口不接受 source_path。即使 manifest 被错误写入上传目录外的路径，代码也只记录告警，不会
删除该文件。

## 9. 会话 API

```bash
curl http://127.0.0.1:8000/api/v1/sessions/demo-001/history
curl -X DELETE http://127.0.0.1:8000/api/v1/sessions/demo-001
```

历史响应只包含：

```json
{
  "thread_id": "demo-001",
  "history": [
    {"role": "human", "content": "加班制度是什么？"},
    {"role": "assistant", "content": "……"}
  ]
}
```

删除会话只删除该 thread 的 LangGraph checkpoint，不删除文档和其他会话。

## 10. 旧接口兼容

| 旧接口 | 状态 | 新接口 |
| --- | --- | --- |
| `POST /api/v1/upload/file` | 保留 | `POST /api/v1/documents/upload` |
| `GET /api/v1/upload/status/{file_id}` | 保留 | `GET /api/v1/documents/{file_id}/status` |
| `POST /api/v1/qa/clear_memory` | 保留并标记 deprecated | `DELETE /api/v1/sessions/{thread_id}` |
| `GET /api/v1/qa/history` | 保留并标记 deprecated | `GET /api/v1/sessions/{thread_id}/history` |
| request `session_id` | 接受兼容别名 | request `thread_id` |

旧问答响应已经升级为阶段 5 契约。调用方如果严格校验旧 `sources/intent` 字段，需要迁移到
`citations/decision`。

## 11. 自动测试

阶段 5 专项测试：

```bash
.venv-v2/bin/python -m pytest tests/api -q --tb=short
```

覆盖：

- 默认隐藏 trace；
- 引用编号与 document/chunk ID；
- debug trace 删除正文、路径和异常细节；
- 旧 session_id 请求兼容；
- 空白输入返回 422；
- SSE 事件类型与顺序；
- 会话历史与删除；
- 文档上传、状态、删除和 404；
- 删除同步清理 Chroma/docstore/manifest/上传文件；
- 上传目录外文件永不删除。

阶段 1—5 联合测试：

```bash
.venv-v2/bin/python -m pytest \
  tests/api tests/agent tests/retrieval tests/ingestion \
  tests/test_config.py tests/test_phase1_contracts.py \
  tests/test_api_smoke.py tests/test_security_baseline.py \
  -q --tb=short
```

## 12. Swagger 人工验收

启动：

```bash
.venv-v2/bin/python run_api.py
```

打开 `http://127.0.0.1:8000/docs`，按顺序执行：

1. `POST /documents/upload`；
2. 使用返回的 `file_id` 轮询状态；
3. `POST /qa/ask`，确认引用；
4. 使用同一 thread 追问；
5. `GET /sessions/{thread_id}/history`；
6. `DELETE /sessions/{thread_id}`；
7. 使用 `document_id` 删除文档；
8. 检查 `/knowledge/stats` 和 `/system/ready`。

## 13. 已知边界

- SSE 当前不是 token 流，只提供生命周期事件和完整回答；
- FastAPI BackgroundTasks 适合单机演示，生产环境应换任务队列；
- 上传任务表暂未保存 document_id，客户端需保留 upload 响应中的两个 ID；
- API 尚未加入身份认证、租户隔离和限流；
- debug trace 虽已脱敏，生产环境仍应通过权限控制开放；
- 全库清空不再提供公开 HTTP 接口，如确有运维需要应增加鉴权后的管理命令。

## 14. 回滚范围

删除：

```text
api/schemas/
api/presenters.py
api/routes/documents.py
api/routes/sessions.py
tests/api/
docs/PHASE5_API_CONTRACT_NOTES.md
```

恢复：

```text
api/routes/qa.py
api/routes/upload.py
api/routes/__init__.py
api/main.py
core/agent/graph.py
core/ingestion/service.py
core/ingestion/__init__.py
README.md
```

回滚 API 代码不需要删除 Chroma 数据。若已经通过新版删除接口删除文档，数据需重新上传摄取。
