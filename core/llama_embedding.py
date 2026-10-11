"""供 LlamaIndex 摄取端和检索端共用的 Embedding 单例。"""

from __future__ import annotations

import threading
from pathlib import Path

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.settings import Settings, settings

_embedding_cache: dict[tuple[str, str, str], BaseEmbedding] = {}
_embedding_lock = threading.Lock()


def get_llama_embedding(app_settings: Settings = settings) -> BaseEmbedding:
    """按“模型路径 + 设备 + 缓存目录”复用 LlamaIndex Embedding。

    摄取和检索如果分别加载 BGE，会在内存中保存两份模型。这里使用进程级缓存，既保证
    两端使用完全相同的向量空间，也降低本地演示时的内存占用和首次查询延迟。
    """
    local_path = app_settings.MODELS_DIR / app_settings.EMBEDDING_MODEL_NAME.replace("/", "_")
    model_name = str(local_path) if local_path.exists() else app_settings.EMBEDDING_MODEL_NAME
    key = (model_name, app_settings.EMBEDDING_DEVICE, str(app_settings.MODELS_DIR))
    with _embedding_lock:
        if key not in _embedding_cache:
            _embedding_cache[key] = HuggingFaceEmbedding(
                model_name=model_name,
                device=app_settings.EMBEDDING_DEVICE,
                normalize=True,
                cache_folder=str(app_settings.MODELS_DIR),
            )
        return _embedding_cache[key]
