"""SQLite 摄取任务状态仓储。"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class IngestionJob:
    file_id: str
    filename: str
    status: str
    node_count: int
    skipped: bool
    error: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class IngestionJobRepository:
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
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    file_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    node_count INTEGER NOT NULL DEFAULT 0,
                    skipped INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, file_id: str, filename: str) -> IngestionJob:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ingestion_jobs VALUES (?, ?, 'pending', 0, 0, NULL, ?, ?)",
                (file_id, filename, now, now),
            )
        job = self.get(file_id)
        assert job is not None
        return job

    def update(
        self,
        file_id: str,
        status: str,
        *,
        node_count: int = 0,
        skipped: bool = False,
        error: str | None = None,
    ) -> IngestionJob:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"非法任务状态: {status}")
        # 错误写库前截断，既避免 SQLite 被超长堆栈污染，也不向状态接口泄露内部调用栈。
        safe_error = error[:500] if error else None
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, node_count = ?, skipped = ?, error = ?, updated_at = ?
                WHERE file_id = ?
                """,
                (status, node_count, int(skipped), safe_error, now, file_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(file_id)
        job = self.get(file_id)
        assert job is not None
        return job

    def get(self, file_id: str) -> IngestionJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE file_id = ?", (file_id,)
            ).fetchone()
        if row is None:
            return None
        values = dict(row)
        values["skipped"] = bool(values["skipped"])
        return IngestionJob(**values)
