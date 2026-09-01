"""
embedding模型加载模块——使用单例模式，支持本地缓存 与 HuggingFace自动下载

单例模式：是一种设计模式，它保证一个类只有一个实例，并提供一个全局访问点。
在Python中，我们可以使用类方法来实现单例模式。 cls.__instance 是一个类变量，用于存储类的唯一实例。
目的：避免重复加载与初始化，提高性能 和 资源利用率。
"""

import logging
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingModel:
  """embedding模型加载类 单例模式"""
  _instance = None
  
  def __new__(cls):
    if cls._instance is None:
      cls._instance = super(EmbeddingModel, cls).__new__(cls)
      cls._instance._initialize()
    return cls._instance

  def _initialize(self):
    """初始化embedding模型"""
    # 尝试从本地先加载，本地没有再从HuggingFace上面下载
    model_name = settings.EMBEDDING_MODEL_NAME
    model_path = Path(settings.MODELS_DIR) / model_name.replace("/", "_")
    
    if model_path.exists():
      logger.info(f"从本地加载embedding模型: {model_path}")
      model_path = str(model_path)
    else:
      logger.info(f"从HuggingFace下载embedding模型: {model_name}")
      model_path = model_name
    
    # 加载模型
    try:
      self.model = HuggingFaceEmbeddings(
        model_name=model_path, 
        model_kwargs={"device": settings.EMBEDDING_DEVICE}, 
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
      )
      logger.info("Embedding模型初始化完成")
    except Exception as e:
      logger.error(f"加载embedding模型失败: {e}")
      raise

  def get_model(self):
    return self.model


def get_embedding_model():
  """获取embedding模型实例"""
  return EmbeddingModel().get_model()