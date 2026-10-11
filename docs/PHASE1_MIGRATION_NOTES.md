# 阶段 1：RAG V2 依赖、安全与运行基线迁移说明

## 1. 文档信息

- 分支：`feat/rag-v2-migration`
- 基线分支：`main`
- 阶段目标：在不重写现有 RAG 业务链的前提下，升级框架依赖、修复启动断点、消除明文密钥，并建立后续 LlamaIndex/LangGraph 改造所需的可运行基线。
- 状态：已通过自动测试、真实 API 启动和旧 RAGChain 初始化验证；尚未提交 commit。

## 2. 本阶段架构决策

### 2.1 框架职责

后续架构将明确分工：

| 层级 | 框架 | 主要职责 |
|---|---|---|
| 数据摄取与检索 | LlamaIndex | Reader、Node、IngestionPipeline、缓存、docstore、Chroma、Retriever、Rerank |
| 模型与工具标准接口 | LangChain 1.x | ChatModel、Message、Tool、Structured Output、Agent Middleware |
| 有状态业务编排 | LangGraph 1.x | Query Rewrite、路由、证据判断、回答/澄清/拒答、checkpoint |
| API 与演示 | FastAPI / Streamlit | HTTP 契约、上传、流式响应、调试和评测展示 |

原则是避免 LlamaIndex 与 LangChain 同时封装完整 RAG 主链。LlamaIndex 专注数据与检索，LangGraph 专注业务流程。

### 2.2 为什么暂时保留 `langchain-classic`

LangChain 1.x 精简了顶级命名空间，当前项目使用的以下 API 已迁出核心包：

- `create_history_aware_retriever`
- `create_stuff_documents_chain`
- `create_retrieval_chain`
- `ConversationBufferWindowMemory`
- `ContextualCompressionRetriever`

阶段 1 的目标是先建立稳定升级基线，因此使用 `langchain-classic` 维持旧问答链可运行。它是明确的临时依赖：

- 阶段 2/3 用 LlamaIndex 替换旧文档、向量库和检索实现；
- 阶段 4 用 LangGraph 替换旧 RAG Chain 和 Memory；
- 完成阶段 4 后删除 `langchain-classic` 与 `langchain-community`。

## 3. 实际锁定版本

核心版本由 `uv.lock` 固定：

| 依赖 | 版本 | 说明 |
|---|---:|---|
| Python | 3.10.18 | 当前机器已有版本 |
| LangChain | 1.4.4 | 新模型/Agent 标准接口 |
| LangChain Core | 1.6.9 | Runnable、Message、Document 等核心类型 |
| LangChain Classic | 1.0.8 | 仅用于旧链过渡 |
| LangGraph | 1.2.14 | 后续工作流与 checkpoint |
| LlamaIndex Core | 0.14.25 | 后续摄取与检索层 |
| LlamaIndex Chroma | 0.6.0 | 后续连接 Chroma |
| ChromaDB | 0.6.3 | 与 Python 3.10 兼容 |
| ONNX Runtime | 1.23.2 | 最后支持当前 Python/平台的兼容版本 |
| PostHog | 5.4.0 | 与 Chroma 0.6.x 遥测接口兼容 |

### 3.1 为什么没有使用最新 Chroma 1.5.x

依赖解析时发现 Chroma 1.5.9 会安装 `onnxruntime 1.24.3`，该版本没有 macOS CPython 3.10 wheel，导致环境无法创建。为保证当天可运行，采用：

```text
chromadb>=0.5.17,<1
onnxruntime>=1.17,<1.24
```

后续如果项目升级到 Python 3.11，可单独评估 Chroma 1.x；不在本阶段强制升级系统 Python。

## 4. 配置与安全改造

### 4.1 Pydantic Settings 2.x

`config/settings.py` 已从旧式 `class Config` 迁移为 `SettingsConfigDict`：

- 从项目根目录 `.env` 加载配置；
- 忽略迁移期间多余的旧配置项；
- 支持大小写不敏感；
- 使用 `AliasChoices` 兼容旧变量名：
  - `VERSION` → `PROJECT_VERSION`
  - `OLLAMA_MODEL` → `OLLAMA_MODEL_NAME`
  - `OPENAI_MODEL` → `OPENAI_MODEL_NAME`
  - `SILICONFLOW_MODEL` → `SILICONFLOW_MODEL_NAME`

因此已有个人 `.env` 不需要立即重写。

### 4.2 密钥处理

- `OPENAI_API_KEY` 和 `SILICONFLOW_API_KEY` 改为 `SecretStr`；
- 源码默认值为空；
- 只有在创建 SDK 客户端时调用 `get_secret_value()`；
- 新增 `.env.example`，仅保留空占位符；
- `.env`、虚拟环境、日志和运行数据均由 `.gitignore` 排除。

注意：旧提交曾包含有效密钥。当前源码安全并不等于历史安全，仍需在服务商控制台撤销旧 Key，再将新 Key 仅写入 `.env`。本阶段未重写 Git 历史，避免未经确认执行破坏性操作。

## 5. 代码修复点

### 5.1 API 启动

- 增加缺失的 `ALLOWED_ORIGINS`；
- 新增 `run_api.py`；
- `/api/v1/system/config` 根据当前 provider 返回正确模型名；
- 新增 `/api/v1/system/ready`，区分进程存活与运行目录就绪。

### 5.2 上传链路

旧接口调用 `load_file(file_path, metadata)`，但加载器只接收 `file_path`。现在：

- `DocumentLoader.load_file()` 接受可选 metadata；
- category、原始文件名等属性传播到每个 chunk；
- 文件名先经过 `Path(...).name` 净化，避免路径穿越；
- 默认 chunk 参数统一读取全局 Settings。

### 5.3 知识库管理

修复 API 与 VectorStoreManager 方法名称不一致：

- 对外公开 `delete_by_source()`；
- 对外公开 `get_collection_stats()`；
- 暂时保留 `get_collection_count()` 兼容别名；
- 根据 Chroma/FAISS 类型在管理器内部完成删除分派。

### 5.4 FAISS 删除

原逻辑错误地把 document ID 列表传给 `FAISS.from_documents()`。现在改为：

- 收集实际 `Document` 对象；
- 删除目标来源后，用剩余文档重建索引；
- 全部删除时创建初始化占位文档，避免空索引构造失败。

### 5.5 日志与仓库卫生

- 日志初始化增加幂等保护，避免 reload/测试时重复 handler；
- 支持 `LOG_TO_FILE=false`；
- 关闭 Chroma 匿名遥测；
- `app.log` 和已提交的 `.pyc` 从 Git 索引移除，本地文件不删除；
- `upload/`、`data/`、`.venv-v2/` 加入忽略规则。

## 6. 新增文件

| 文件 | 用途 |
|---|---|
| `.env.example` | 无密钥的配置模板 |
| `pyproject.toml` | 项目元数据、直接依赖和 pytest 配置 |
| `uv.lock` | 完整可复现依赖锁 |
| `requirements-dev.txt` | 兼容 pip 的开发依赖入口 |
| `run_api.py` | API 标准启动入口 |
| `tests/test_config.py` | 配置兼容与密钥脱敏测试 |
| `tests/test_api_smoke.py` | health/ready API 冒烟测试 |
| `tests/test_phase1_contracts.py` | 上传 metadata 和向量库公开契约测试 |
| `tests/test_security_baseline.py` | 明文密钥回归测试 |

## 7. 验证记录

### 7.1 自动测试

```bash
source .venv-v2/bin/activate
python -m pytest \
  tests/test_config.py \
  tests/test_api_smoke.py \
  tests/test_phase1_contracts.py \
  tests/test_security_baseline.py \
  -q
```

验证结果：

```text
9 passed
```

### 7.2 依赖与旧链初始化

已验证以下包可同时导入：LangChain、LangGraph、LlamaIndex、ChromaDB。

旧 `RAGChain` 在不调用外部 LLM 的情况下初始化成功，说明过渡兼容层有效。

### 7.3 真实 API 启动

```bash
python run_api.py
```

实际验证：

- 本地 BGE Embedding 模型加载成功；
- 现有 Chroma 数据库加载成功；
- `/api/v1/system/health` 返回 200；
- `/api/v1/system/ready` 返回 200；
- `/api/v1/system/config` 返回 200 且不包含密钥；
- OpenAPI 正常生成。

## 8. 本机复现步骤

### 使用已经创建的环境

```bash
cd /Users/steinsgate/Desktop/Study/Agent-BZ/rag-v1
source .venv-v2/bin/activate
python run_api.py
```

### 从锁文件重新创建环境

```bash
cd /Users/steinsgate/Desktop/Study/Agent-BZ/rag-v1
UV_PROJECT_ENVIRONMENT=.venv-v2 uv sync --dev --frozen
source .venv-v2/bin/activate
python -m pytest tests/test_config.py tests/test_api_smoke.py tests/test_phase1_contracts.py tests/test_security_baseline.py -q
python run_api.py
```

## 9. 已知限制与技术债

1. `langchain-community.Chroma` 会显示弃用警告；阶段 2 改成 LlamaIndex Chroma 后解决。
2. `langchain-classic` 只是过渡依赖；阶段 4 必须删除。
3. 当前 Memory 仍为进程内窗口记忆，重启后丢失；阶段 4 使用 LangGraph SQLite checkpoint。
4. 文档解析、切片和索引仍是旧实现；阶段 2 替换为 LlamaIndex IngestionPipeline。
5. 当前检索仍是单路向量召回；阶段 3 增加中文 BM25、RRF 和 rerank trace。
6. 旧 Git 历史中的密钥尚未清理；在决定是否重写历史前，必须先撤销旧 Key。

## 10. 下一阶段入口

阶段 2 将完成：

1. 建立 `core/ingestion/`；
2. 引入 LlamaIndex `IngestionPipeline`；
3. 统一 Document/Node metadata；
4. 接入本地 BGE Embedding 与 LlamaIndex Chroma；
5. 增加 ingestion cache 和 docstore；
6. 支持稳定 ID、重复跳过和内容变化更新；
7. 保留旧摄取链作为短期对照，项目负责人验收后再删除。

阶段 2 仍遵循同一门禁：自动测试通过 → 停止 → 本机人工验证 → 明确确认后继续。
