"""Streamlit 前端启动入口。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config.settings import settings


def main() -> int:
    """用当前虚拟环境启动 Streamlit，避免误用系统 Python。"""
    app_path = Path(__file__).parent / "frontend" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(settings.STREAMLIT_PORT),
        "--browser.gatherUsageStats",
        "false",
    ]
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        # Ctrl+C 会同时传给 Streamlit 子进程和当前父进程；在这里吞掉父进程异常，
        # 让开发者正常停止服务时不会看到一段误导性的 Python traceback。
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
