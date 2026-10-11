"""
日志配置模块
为什么配置日志？——因为日志可以帮助我们更好的去了解程序的运行情况，快速定位出现问题的地方。
日志的级别——DEBUG  INFO   WARNING   ERROR   CRITICAL
写好了日志配置模块之后，在主程序中调用，在其他的任意模块中可以继承全局日志。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from config.settings import settings

def setup_logging() -> None:
    """Configure console logging and optional rotating file logging once."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 根日志器
    root_logger = logging.getLogger()
    # FastAPI reload 或测试重复导入时避免重复添加 handler，防止同一日志打印多次。
    if getattr(root_logger, "_rag_logging_configured", False):
        return

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # 测试环境可通过 LOG_TO_FILE=false 关闭文件写入，保持 Git 工作区干净。
    if settings.LOG_TO_FILE:
        settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root_logger.addHandler(file_handler)

    # 减少第三方库的日志噪声 （可选）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    root_logger._rag_logging_configured = True
    root_logger.info("日志系统初始化完成, 级别是 %s", settings.LOG_LEVEL)
