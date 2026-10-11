"""
向量库管理模块——封装chroma或者faiss操作，提供统一接口
通过embedding模块获取嵌入模型，转换为向量后，使用chroma或者faiss存储到本地向量库持久化存储，方便后续的检索与查询等操作。
"""

import logging
import shutil
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
# 迁移过渡：阶段 2 会切换到 llama-index-vector-stores-chroma。
# 目前保留旧实现，是为了让阶段 1 升级依赖后仍能读取现有向量库。
from langchain_community.vectorstores import Chroma, FAISS
from config.settings import settings
from core.embedding import get_embedding_model

logger = logging.getLogger(__name__)

class VectorStoreManager:
  """向量库管理器"""
  def __init__(self):
    self.embedding_model = get_embedding_model()
    self._store = None
    self._initialize_vector_store()
    
  def _initialize_vector_store(self):
    """初始化向量存储"""
    if settings.VECTOR_STORE_TYPE == "chroma":
      # 初始化Chroma向量存储
      self._store = Chroma(
        embedding_function=self.embedding_model,
        persist_directory=str(settings.VECTOR_DB_DIR),
        # 关闭匿名遥测，避免面试演示时出现 PostHog 网络请求和无关错误日志。
        client_settings=ChromaSettings(
          anonymized_telemetry=settings.CHROMA_ANONYMIZED_TELEMETRY
        ),
      )
      logger.info(f"Chroma向量库已初始化，存储路径: {settings.VECTOR_DB_DIR}")
    elif settings.VECTOR_STORE_TYPE == "faiss":
      # 初始化Faiss向量存储
      faiss_index_path = settings.VECTOR_DB_DIR / "index.faiss"
      if faiss_index_path.exists():
        self._store = FAISS.load_local(str(settings.VECTOR_DB_DIR), self.embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"Faiss向量库已初始化，存储路径: {faiss_index_path}")
      else:
        self._store = FAISS.from_documents([Document(page_content="初始化向量库", metadata={"source": "init"})], self.embedding_model)
        logger.info("Faiss向量库已初始化")
        self._save_faiss()
    else:
      raise ValueError(f"不支持的向量存储类型: {settings.VECTOR_DB_DIR}")
    
  def _save_faiss(self):
    """保存Faiss向量库"""
    if settings.VECTOR_STORE_TYPE == "faiss":
      self._store.save_local(str(settings.VECTOR_DB_DIR))
      logger.info(f"Faiss向量库已保存，存储路径: {settings.VECTOR_DB_DIR}")
  
  def add_documents(self, documents: list[Document]) -> int:
    """添加文档到向量库"""
    if not documents:
      return 0
    try: 
      self._store.add_documents(documents)
      if settings.VECTOR_STORE_TYPE == "faiss":
        self._save_faiss()
      logger.info(f"成功添加 {len(documents)} 个文档到向量库")
      return len(documents)
    except Exception as e:
      logger.error(f"添加文档到向量库失败: {e}")
      raise

  def _delete_by_source_chroma(self, source: str) -> int:
    """
    根据源文件删除文档
    :param source: 源文件路径
    :return: 是否删除成功
    """
    try:
      if settings.VECTOR_STORE_TYPE == "chroma":
        result = self._store.get(where={"source": source})
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
          self._store.delete(ids=ids_to_delete)
          logger.info(f"成功删除来源为 {source} 的文档")
          return len(ids_to_delete)
        return 0
    except Exception as e:
      logger.error(f"删除文档失败: {e}")
      raise

  def _delete_by_source_faiss(self, source: str) -> int:
    """
    FAISS模式的删除，是需要重构索引的
    """
    documents = list(self._store.docstore._dict.values())
    remaining_documents = []
    delete_count = 0
    for doc in documents:
      if doc.metadata.get("source") == source:
        delete_count += 1
      else:
        remaining_documents.append(doc)
    if delete_count > 0:
      if remaining_documents:
        self._store = FAISS.from_documents(remaining_documents, self.embedding_model)
      else:
        self._store = FAISS.from_documents(
          [Document(page_content="初始化向量库", metadata={"source": "init"})],
          self.embedding_model,
        )
      self._save_faiss()
      logger.info(f"成功删除来源为 {source} 的文档，共删除 {delete_count} 个文档")
    return delete_count

  def delete_by_source(self, source: str) -> int:
    """Delete all chunks associated with a source path."""
    # API 只依赖这个公开入口，底层根据存储类型分派，避免路由层调用私有方法。
    if settings.VECTOR_STORE_TYPE == "chroma":
      return self._delete_by_source_chroma(source)
    if settings.VECTOR_STORE_TYPE == "faiss":
      return self._delete_by_source_faiss(source)
    raise ValueError(f"不支持的向量存储类型: {settings.VECTOR_STORE_TYPE}")

  def similarity_search(self, query: str, k: int = 5, filter: dict = None) -> list[Document]:
    """相似度搜索"""
    k = k or settings.SEARCH_TOP_K
    if filter and settings.VECTOR_STORE_TYPE == 'chroma':
      return self._store.similarity_search(query, k=k, filter=filter)
    return self._store.similarity_search(query, k=k)

  def similarity_search_with_score(self, query: str, k: int = 5, filter: dict = None) -> list[tuple[Document, float]]:
    """相似度搜索，返回文档和分数"""
    k = k or settings.SEARCH_TOP_K
    if filter and settings.VECTOR_STORE_TYPE == 'chroma':
      return self._store.similarity_search_with_score(query, k=k, filter=filter)
    return self._store.similarity_search_with_score(query, k=k)

  def get_collection_stats(self) -> dict:
    """获取集合中的文档数量"""
    if settings.VECTOR_STORE_TYPE == 'chroma':
      count = self._store._collection.count()
      return {
        "count": count,
        "vector_store_type": "chroma",
        "persist_directory": str(settings.VECTOR_DB_DIR)
      }
    elif settings.VECTOR_STORE_TYPE == 'faiss':
      return {
        "count": len(self._store.docstore._dict),
        "vector_store_type": "faiss",
        "persist_directory": str(settings.VECTOR_DB_DIR)
      }
    else:
      raise ValueError(f"不支持的向量存储类型: {settings.VECTOR_STORE_TYPE}")

  def get_collection_count(self) -> dict:
    """Backward-compatible alias for the old method name."""
    # 暂时保留旧方法名，给已有脚本一个平滑迁移窗口。
    return self.get_collection_stats()

  def clear_all(self) -> None:
    """清空所有文档"""
    if settings.VECTOR_STORE_TYPE == 'chroma':
      self._store.delete_collection()
      self._initialize_vector_store()
      logger.info("成功清空所有文档")
    elif settings.VECTOR_STORE_TYPE == 'faiss':
      shutil.rmtree(settings.VECTOR_DB_DIR, ignore_errors=True)
      settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
      self._initialize_vector_store()
      logger.info("成功清空所有文档")
    else:
      raise ValueError(f"不支持的向量存储类型: {settings.VECTOR_STORE_TYPE}")


_vectore_store_instance: VectorStoreManager | None = None

def get_vector_store_manager() -> VectorStoreManager:
    """获取向量存储管理器实例（单例模式）"""
    global _vectore_store_instance
    if _vectore_store_instance is None:
        _vectore_store_instance = VectorStoreManager()
    return _vectore_store_instance
