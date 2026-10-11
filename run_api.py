"""Development entry point for the FastAPI service."""

import uvicorn

from config.settings import settings


if __name__ == "__main__":
    # 使用模块路径启动，确保 FastAPI 与脚本运行时的导入行为一致。
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )
