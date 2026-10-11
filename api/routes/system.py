"""
系统状态API
"""
from fastapi import APIRouter
from config.settings import settings

# 必须定义名为 router 的 APIRouter 实例
router = APIRouter(prefix="/system", tags=["系统状态"])


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": settings.PROJECT_NAME, "version": settings.PROJECT_VERSION}


@router.get("/ready")
async def readiness_check():
    """Verify that required local storage paths are available."""
    # health 只说明进程存活；ready 额外确认运行所需目录已准备好，便于部署探针判断。
    required_directories = [settings.UPLOAD_DIR, settings.VECTOR_DB_DIR]
    ready = all(path.exists() and path.is_dir() for path in required_directories)
    return {
        "status": "ready" if ready else "not_ready",
        "checks": {path.name: path.exists() and path.is_dir() for path in required_directories},
    }


@router.get("/config")
async def get_config():
    """获取当前配置（隐藏敏感信息）"""
    return {
        "vector_store_type": settings.VECTOR_STORE_TYPE,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.active_model_name,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "search_top_k": settings.SEARCH_TOP_K,
    }
