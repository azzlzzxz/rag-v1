"""
检索模块 —— 支持相似度检索 + 可选的重排序
主要功能：
    1. 基于向量相似度进行文档检索
    2. 可选的文档重排序（使用Reranker）
        使用重排序的原因是为了提高检索结果的相关性和准确性。
        这属于对RAG技术的优化，主要就是为了提高检索结果的质量。
        但是会增加一些额外的开销。
"""
import logging
from config.settings import settings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from core.vector_store import get_vector_store_manager
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseDocumentCompressor):
    """基于交叉编码器的重排序器"""
    model_name: str
    top_k: int
    model: object = None  # 延迟加载模型

    def _initialize_model(self):
        """初始化模型"""
        if self.model is None:
            try:
                self.model = CrossEncoder(self.model_name)
                logger.info(f"交叉编码器模型加载成功: {self.model_name}")
            except Exception as e:
                logger.error(f"加载交叉编码器模型失败: {e}")
                raise

    def compress_documents(self, documents, query, **kwargs):
        """重排序核心方法：对文档列表按与查询query的相关性排序"""
        if not documents:
            return []
        # 初始化模型
        self._initialize_model()
        # 如果模型加载失败，直接返回前top_k个文档
        if self.model is None:
            return documents[:self.top_k]
        
        try:
          # 构建查询-文档对
          pairs = [[query, doc.page_content] for doc in documents]
          # 预测相关性分数
          scores = self.model.predict(pairs)
          # 按分数排序并返回前top_k个文档
          sorted_docs = [
            doc
            for _, doc in sorted(
              zip(scores, documents), key=lambda item: item[0], reverse=True
            )
          ]
          return sorted_docs[:self.top_k]
        except Exception as e:
          logger.error(f"重排序失败: {e}")
          return documents[:self.top_k]

class RAGRetriever:
    def __init__(self):
        self.vector_store = get_vector_store_manager()
        self.reranker = self._initialize_reranker()
    
    def _initialize_reranker(self):
        """初始化Reranker"""
        if not settings.USE_RERANKER:
            logging.info("未启用Reranker")
            return None

        try:
          model_name = settings.RERANKER_MODEL_NAME
          local_model_path = settings.MODELS_DIR / model_name.replace("/", "_")
          if local_model_path.exists():
            model_path = str(local_model_path)
            logger.info(f"使用本地交叉编码器模型: {model_path}")
          else:
            model_path = model_name
            logger.info(f"使用远程交叉编码器模型: {model_path}")
          reranker = CrossEncoderReranker(model_name=model_path, top_k=settings.SEARCH_TOP_K)
          logger.info(f"Reranker初始化成功")
          return reranker
        except Exception as e:
          logger.error(f"初始化Reranker失败: {e}")
          return None
          
    def retrieve(self, query: str, top_k: int = None, filter_dict: dict = None, return_score: bool = False) -> list:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            filter_dict: 过滤条件
            return_score: 是否返回分数
            
        Returns:
            检索到的文档列表
        """
        top_k = top_k or settings.SEARCH_TOP_K
        # 先进行向量检索
        doc_with_score = self.vector_store.similarity_search_with_score(query, k=top_k, filter=filter_dict)
        # 如果启用了Reranker，则进行重排序
        if self.reranker:
            # 提取文档
            row_docs = [doc for doc, _ in doc_with_score]
            # 使用Reranker进行重排序
            ranked_docs = self.reranker.compress_documents(row_docs, query)
            # 重新组合文档和分数
            ranked_docs_with_score = [
              (doc, next(score for d, score in doc_with_score if d == doc))
              for doc in ranked_docs
            ]            
        else:
            ranked_docs_with_score = doc_with_score
            
        if return_score:
            return ranked_docs_with_score
        else:
            return [doc for doc, _ in ranked_docs_with_score]


    def get_compresstion_retriever(self, search_kwargs: dict):
      search_kwargs = search_kwargs or {"k": settings.SEARCH_TOP_K}
      # 基础检索器
      # as_retriever 是 LangChain VectorStore 的内置方法；
      # self.vector_store 是 VectorStoreManager，因此应在其底层 store 上调用。
      base_retriever = self.vector_store._store.as_retriever(search_kwargs=search_kwargs)
      
      # 如果有Reranker，则使用ContextualCompressionRetriever
      if self.reranker:
        return ContextualCompressionRetriever(base_compressor=self.reranker, base_retriever=base_retriever)
      else:
        return base_retriever


_rag_retriever_instance = None

def get_rag_retriever() -> RAGRetriever:
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    return _rag_retriever_instance
