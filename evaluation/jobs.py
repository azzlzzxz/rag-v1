"""页面评测任务的 SQLite 状态仓储。"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class EvaluationJob:
    run_id: str
    mode: str
    top_k: int
    category: str
    status: str
    report_name: str | None
    error: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class EvaluationJobRepository:
    """跨页面刷新保存任务状态；同一数据库只允许一个 active 任务。"""

    VALID_STATUSES = {"pending", "running", "succeeded", "failed"}

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_jobs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_name TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, run_id: str, mode: str, top_k: int, category: str) -> EvaluationJob:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_jobs VALUES (?, ?, ?, ?, 'pending', NULL, NULL, ?, ?)",
                (run_id, mode, top_k, category, now, now),
            )
        job = self.get(run_id)
        assert job is not None
        return job

    def update(
        self,
        run_id: str,
        status: str,
        *,
        report_name: str | None = None,
        error: str | None = None,
    ) -> EvaluationJob:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"非法评测任务状态: {status}")
        now = datetime.now(timezone.utc).isoformat()
        # 数据库保留简短错误类型便于排查；公开 API 仍只返回统一失败文案。
        safe_error = error[:300] if error else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_jobs
                SET status = ?, report_name = ?, error = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (status, report_name, safe_error, now, run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(run_id)
        job = self.get(run_id)
        assert job is not None
        return job

    def get(self, run_id: str) -> EvaluationJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return EvaluationJob(**dict(row)) if row else None

    def active(self) -> EvaluationJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM evaluation_jobs
                WHERE status IN ('pending', 'running')
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
        return EvaluationJob(**dict(row)) if row else None

    def recover_interrupted(self) -> int:
        """进程重启后把遗留 active 任务标记失败，避免永远阻塞新评测。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_jobs
                SET status = 'failed', error = 'ProcessInterrupted', updated_at = ?
                WHERE status IN ('pending', 'running')
                """,
                (now,),
            )
        return cursor.rowcount
