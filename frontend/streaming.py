"""智能问答的前端流式展示辅助函数。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator


def typewriter_chunks(
    text: str,
    *,
    chunk_size: int = 2,
    delay_seconds: float = 0.025,
    sleeper: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """把完整回答切成小段，交给 ``st.write_stream`` 做打字机展示。

    当前后端 SSE 传输的是工作流生命周期事件，并不是模型 token。这里仅负责前端的渐进
    展示，不改变后端答案、引用编号或 checkpoint 中保存的内容。两个字符一段可以兼顾
    中文打字效果和长回答的展示速度；测试可注入空 sleeper，避免真实等待。
    """
    if chunk_size < 1:
        raise ValueError("chunk_size 必须大于等于 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds 不能小于 0")

    for offset in range(0, len(text), chunk_size):
        chunk = text[offset : offset + chunk_size]
        if delay_seconds:
            sleeper(delay_seconds)
        yield chunk
