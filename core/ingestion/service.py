"""LlamaIndex 摄取 Pipeline 的组装、执行与状态持久化。

本模块是摄取链路的“总控层”，负责把 Reader、切片器、Embedding、Docstore、Cache
和 Chroma 组装起来。它不解析 PDF/Excel 的具体格式，那是 readers.py 的职责。

一次调用的主流程：

    IngestionService.ingest_file()
        1. 计算业务 document_id 和整个文件的 SHA-256
        2. 查询 manifest，相同文件直接跳过
        3. 调用 EnterpriseDocumentReader 生成 Document
        4. 清理新版本中已消失的页、Sheet 或段落组
        5. IngestionPipeline 执行 SentenceSplitter → metadata → Embedding
        6. 将 Node 写入 Chroma
        7. 持久化 docstore、cache 和 manifest

命令行用法：

    service = IngestionService()
    result = service.ingest_file(
        "upload/ab12cd34_员工制度.pdf",
        category="hr",
        original_filename="员工制度.pdf",
    )
    print(result.document_id, result.node_count, result.skipped)

FastAPI 后台任务用法：

    service.jobs.create(file_id, original_filename)
    service.process_job(file_id, file_path, category, original_filename)

职责边界：阶段 2 中 LlamaIndex 只接管“数据进入知识库”的过程；问答检索仍由旧
LangChain 链处理，阶段 3 才会迁移检索端。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core.ingestion import DocstoreStrategy, IngestionCache, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from config.settings import Settings, settings
from core.ingestion.ids import build_document_id, sha256_file, stable_chunk_id
from core.ingestion.jobs import IngestionJobRepository
from core.ingestion.readers import EnterpriseDocumentReader
from core.llama_embedding import get_llama_embedding

logger = logging.getLogger(__name__)


class AttachStableChunkMetadata(TransformComponent):
    """将 SentenceSplitter 生成的稳定 node_id 同步写入业务 metadata。

    LlamaIndex 本身可以通过 ``node.node_id`` 访问 ID，但 API 引用、调试脚本和后续
    评测通常只读取 metadata，因此显式保存一份 ``chunk_id``。
    """

    def __call__(self, nodes: Sequence[BaseNode], **_: Any) -> Sequence[BaseNode]:
        for node in nodes:
            node.metadata["chunk_id"] = node.node_id
        return nodes


@dataclass(frozen=True)
class IngestionResult:
    """一次同步摄取的精简结果。

    Attributes:
        document_id: 稳定的业务文档 ID。
        node_count: 本次真正重新生成的 Node 数；不是该文档的历史总 Node 数。
        skipped: 整个文件内容未变化时为 True。
    """

    document_id: str
    node_count: int
    skipped: bool


@dataclass(frozen=True)
class DeletionResult:
    """按 document_id 删除知识库文档后的结果。"""

    document_id: str
    deleted: bool
    deleted_chunks: int


@dataclass(frozen=True)
class DocumentRecord:
    """可公开展示的 manifest 文档摘要。

    manifest 内部还保存 file_hash、source_ids、upload_path 等运行字段。这里主动挑选
    允许公开的字段，避免前端接口泄漏本地绝对路径或文件指纹。
    """

    document_id: str
    file_name: str
    category: str
    source_count: int
    ingested_at: str
    updated_at: str


class IngestionService:
    """协调文档读取、增量判断、Embedding、向量写入和摄取状态。

    ``pipeline`` 和 ``reader`` 可以从外部注入，主要用于单元测试：测试去重逻辑时使用
    FakePipeline，就不需要真正加载几百 MB 的 BGE 模型。
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        *,
        pipeline: IngestionPipeline | None = None,
        reader: EnterpriseDocumentReader | None = None,
    ) -> None:
        """创建摄取服务，但暂不加载 Embedding 模型。

        Args:
            app_settings: 路径、切片大小、模型名等运行配置。
            pipeline: 可选的自定义 Pipeline；不传时首次摄取再创建真实 Pipeline。
            reader: 可选的自定义 Reader；不传时使用 EnterpriseDocumentReader。

        延迟初始化非常重要：上传接口应先快速返回 file_id，模型加载和向量化在后台执行。
        """
        self.settings = app_settings
        self.reader = reader or EnterpriseDocumentReader(app_settings.EXCEL_ROWS_PER_DOCUMENT)
        self.jobs = IngestionJobRepository(app_settings.INGESTION_STORAGE_DIR / "jobs.db")
        self._pipeline = pipeline
        self._lock = threading.RLock()
        self._manifest_path = app_settings.INGESTION_STORAGE_DIR / "manifest.json"
        self._docstore_path = app_settings.INGESTION_STORAGE_DIR / "docstore.json"
        self._cache_path = app_settings.INGESTION_STORAGE_DIR / "cache.json"

    def _build_pipeline(self) -> IngestionPipeline:
        """创建真实的 LlamaIndex IngestionPipeline。

        transformations 严格按顺序执行：

        1. SentenceSplitter：把来源 Document 切成最终 Node；
        2. AttachStableChunkMetadata：补齐业务 chunk_id；
        3. HuggingFaceEmbedding：生成归一化向量。

        Pipeline 同时连接 docstore、cache 和 Chroma：docstore 判断内容是否变化，cache
        缓存 transformations 结果，Chroma 保存最终 Node 与向量。
        """
        self.settings.INGESTION_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        # 重启后从 JSON 恢复 docstore；没有历史文件时才创建空实例。
        docstore = (
            SimpleDocumentStore.from_persist_path(str(self._docstore_path))
            if self._docstore_path.exists()
            else SimpleDocumentStore()
        )
        # IngestionCache 避免相同输入重复执行切片和 Embedding transformation。
        cache = (
            IngestionCache.from_persist_path(str(self._cache_path))
            if self._cache_path.exists()
            else IngestionCache()
        )
        # PersistentClient 明确将 Chroma 数据写到磁盘，而不是只存在当前 Python 进程中。
        client = chromadb.PersistentClient(
            path=str(self.settings.VECTOR_DB_DIR),
            settings=ChromaSettings(
                anonymized_telemetry=self.settings.CHROMA_ANONYMIZED_TELEMETRY
            ),
        )
        collection = client.get_or_create_collection(self.settings.CHROMA_COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # 摄取端与阶段 3 检索端共用同一个模型实例，避免重复加载 BGE。
        embedding = get_llama_embedding(self.settings)
        # 与旧 LangChain 检索器共用 `langchain` collection，保证阶段 2 上传后旧问答链仍能查到数据。
        return IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self.settings.CHUNK_SIZE,
                    chunk_overlap=self.settings.CHUNK_OVERLAP,
                    id_func=stable_chunk_id,
                ),
                AttachStableChunkMetadata(),
                embedding,
            ],
            vector_store=vector_store,
            docstore=docstore,
            cache=cache,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

    @property
    def pipeline(self) -> IngestionPipeline:
        """按需创建并复用 Pipeline，避免每个上传任务重复加载模型。"""
        # 模型延迟到后台任务真正运行时加载，上传接口不会因首次加载模型而阻塞响应。
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """读取文档级 manifest；文件不存在表示知识库尚未摄取过文档。"""
        if not self._manifest_path.exists():
            return {}
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        """用临时文件加原子替换保存 manifest，避免写到一半留下损坏 JSON。"""
        temporary = self._manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self._manifest_path)

    def _persist_pipeline_state(self) -> None:
        """持久化增量判断所需的 docstore 和 transformation cache。"""
        # docstore 和 cache 必须与向量库一起落盘，否则重启后相同文档会再次 Embedding。
        if self.pipeline.docstore is not None:
            self.pipeline.docstore.persist(str(self._docstore_path))
        if self.pipeline.cache is not None:
            self.pipeline.cache.persist(str(self._cache_path))

    def ingest_file(
        self,
        file_path: str | Path,
        *,
        category: str = "general",
        original_filename: str | None = None,
    ) -> IngestionResult:
        """同步摄取一个文件，支持整文件去重和来源单元级增量更新。

        Args:
            file_path: 磁盘上的实际文件路径，可以包含随机上传前缀。
            category: 知识库分类，也是业务 document_id 的组成部分。
            original_filename: 用户上传时的原始文件名。若省略，则使用 file_path.name。

        Returns:
            IngestionResult。相同文件重复摄取时 node_count=0、skipped=True；内容变化时
            node_count 只表示本次重新处理的 Node 数量。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 格式不支持或文件没有有效文本。
            其他 Reader、Embedding 或 Chroma 异常会继续向上抛出，由 process_job
            捕获并记录为 failed。

        示例：

            first = service.ingest_file("policy.pdf", category="hr")
            second = service.ingest_file("policy.pdf", category="hr")
            assert second.skipped is True
        """
        path = Path(file_path)
        original_filename = original_filename or path.name
        document_id = build_document_id(category, original_filename)
        file_hash = sha256_file(path)

        # manifest、docstore、cache 是多个文件，单进程并发时必须串行更新，避免两个任务互相覆盖。
        with self._lock:
            manifest = self._load_manifest()
            previous = manifest.get(document_id)
            # 第一层快速去重：整个文件 hash 一致时，不创建 Pipeline，也不加载 Embedding 模型。
            if previous and previous.get("file_hash") == file_hash:
                return IngestionResult(document_id=document_id, node_count=0, skipped=True)

            first_ingested_at = (
                previous.get("ingested_at")
                if previous
                else datetime.now(timezone.utc).isoformat()
            )
            # Reader 只生成带格式语义边界的 Document；最终 token chunk 在 Pipeline 中生成。
            documents = self.reader.load_data(
                path,
                extra_info={
                    "category": category,
                    "original_filename": original_filename,
                    "document_id": document_id,
                    "ingested_at": first_ingested_at,
                },
            )
            if not documents:
                raise ValueError("文件中没有可摄取的有效文本")

            current_source_ids = {document.id_ for document in documents}
            previous_source_ids = set(previous.get("source_ids", [])) if previous else set()
            # 默认 UPSERT 会处理“存在且内容变化”的来源，但不会发现新版本中已经消失的页或 Sheet。
            for removed_source_id in previous_source_ids - current_source_ids:
                if self.pipeline.docstore is not None:
                    self.pipeline.docstore.delete_ref_doc(removed_source_id, raise_error=False)
                if self.pipeline.vector_store is not None:
                    self.pipeline.vector_store.delete(removed_source_id)

            # UPSERTS 会跳过 hash 未变化的来源单元，只对新增或变化单元执行 transformations。
            nodes = self.pipeline.run(documents=documents, show_progress=False)
            self._persist_pipeline_state()
            manifest[document_id] = {
                "file_hash": file_hash,
                "source_ids": sorted(current_source_ids),
                "file_name": original_filename,
                "category": category,
                "ingested_at": first_ingested_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                # 删除 API 需要清理上传原文件；使用前仍会校验路径必须位于 UPLOAD_DIR。
                "upload_path": str(path.resolve()),
            }
            self._save_manifest(manifest)
            return IngestionResult(document_id=document_id, node_count=len(nodes), skipped=False)

    def delete_document(self, document_id: str) -> DeletionResult:
        """删除文档的向量、docstore 引用、manifest 和安全范围内的上传原文件。

        删除接口只接受稳定 ``document_id``，不接受调用方给出的任意磁盘路径，从入口上避免
        旧 source_path 删除方式可能造成的路径越界。
        """
        with self._lock:
            manifest = self._load_manifest()
            record = manifest.get(document_id)
            if record is None:
                return DeletionResult(document_id=document_id, deleted=False, deleted_chunks=0)

            # 直接按 metadata 查询 Chroma，删除操作无需加载 Embedding 模型。
            client = chromadb.PersistentClient(
                path=str(self.settings.VECTOR_DB_DIR),
                settings=ChromaSettings(
                    anonymized_telemetry=self.settings.CHROMA_ANONYMIZED_TELEMETRY
                ),
            )
            collection = client.get_or_create_collection(self.settings.CHROMA_COLLECTION_NAME)
            stored = collection.get(where={"document_id": document_id}, include=[])
            node_ids = list(stored.get("ids") or [])
            if node_ids:
                collection.delete(ids=node_ids)

            source_ids = list(record.get("source_ids") or [])
            # Pipeline 已初始化时更新其内存 docstore；否则只加载轻量 JSON docstore。
            docstore = self._pipeline.docstore if self._pipeline is not None else None
            if docstore is None and self._docstore_path.exists():
                docstore = SimpleDocumentStore.from_persist_path(str(self._docstore_path))
            if docstore is not None:
                for source_id in source_ids:
                    docstore.delete_ref_doc(source_id, raise_error=False)
                docstore.persist(str(self._docstore_path))

            manifest.pop(document_id)
            self._save_manifest(manifest)

            # 旧 manifest 没有 upload_path 时跳过；越界路径只告警，绝不删除。
            upload_path_value = record.get("upload_path")
            if upload_path_value:
                upload_path = Path(upload_path_value).resolve()
                upload_root = self.settings.UPLOAD_DIR.resolve()
                if upload_path.is_relative_to(upload_root) and upload_path.is_file():
                    upload_path.unlink()
                elif not upload_path.is_relative_to(upload_root):
                    logger.warning("忽略 manifest 中越界的上传路径: document_id=%s", document_id)

            return DeletionResult(
                document_id=document_id,
                deleted=True,
                deleted_chunks=len(node_ids),
            )

    def get_knowledge_stats(self) -> dict[str, Any]:
        """从 LlamaIndex manifest 和 Chroma 返回稳定统计，不经过旧 LangChain VectorStore。"""
        with self._lock:
            manifest = self._load_manifest()
        categories: dict[str, int] = {}
        for record in manifest.values():
            category = str(record.get("category") or "general")
            categories[category] = categories.get(category, 0) + 1

        client = chromadb.PersistentClient(
            path=str(self.settings.VECTOR_DB_DIR),
            settings=ChromaSettings(
                anonymized_telemetry=self.settings.CHROMA_ANONYMIZED_TELEMETRY
            ),
        )
        collection = client.get_or_create_collection(self.settings.CHROMA_COLLECTION_NAME)
        return {
            "document_count": len(manifest),
            "node_count": collection.count(),
            "categories": categories,
        }

    def list_documents(self) -> list[DocumentRecord]:
        """读取可公开的文档摘要，并按最近更新时间倒序返回。

        文档列表以 manifest 为准，因为 manifest 是摄取事务完成后的业务索引；不能直接
        从 Chroma 聚合，否则一次文档更新过程中可能短暂看到不完整的 chunk 集合。
        """
        with self._lock:
            manifest = self._load_manifest()

        records = [
            DocumentRecord(
                document_id=document_id,
                file_name=str(item.get("file_name") or "未知文档"),
                category=str(item.get("category") or "general"),
                source_count=len(item.get("source_ids") or []),
                ingested_at=str(item.get("ingested_at") or ""),
                updated_at=str(item.get("updated_at") or item.get("ingested_at") or ""),
            )
            for document_id, item in manifest.items()
        ]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def process_job(
        self,
        file_id: str,
        file_path: str,
        category: str,
        original_filename: str,
    ) -> None:
        """执行一个后台任务，并把运行结果转换成可查询的任务状态。

        该方法不再向调用方抛出摄取异常：异常会写日志，并将 SQLite 任务状态更新为
        failed。这样客户端始终可以通过 ``GET /upload/status/{file_id}`` 得知结果。
        """
        self.jobs.update(file_id, "running")
        try:
            result = self.ingest_file(
                file_path,
                category=category,
                original_filename=original_filename,
            )
            self.jobs.update(
                file_id,
                "succeeded",
                node_count=result.node_count,
                skipped=result.skipped,
            )
        except Exception as exc:
            logger.exception("摄取任务失败: file_id=%s", file_id)
            self.jobs.update(file_id, "failed", error=f"{type(exc).__name__}: {exc}")


_ingestion_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """返回进程级单例，复用已经加载的 Embedding 模型和 Chroma 客户端。"""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service
