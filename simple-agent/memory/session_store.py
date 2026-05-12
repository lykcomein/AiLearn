"""会话持久化：按 session_id 把每条消息写入 SQLite，支持重启恢复。"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class SessionStore:
    def __init__(self, db_path: str = "sessions.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False 允许 FastAPI 的 worker 线程访问同一个连接；
        # 配合 self._lock 串行化写入，避免并发写冲突。
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT    NOT NULL,
                    role       TEXT    NOT NULL,
                    content    TEXT,
                    extra      TEXT,
                    ts         REAL    NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sess ON messages(session_id)"
            )
            self.conn.commit()

    # ---------- 写 ----------
    def append(self, session_id: str, message: dict) -> None:
        extra = {k: v for k, v in message.items() if k not in ("role", "content")}
        with self._lock:
            self.conn.execute(
                "INSERT INTO messages(session_id, role, content, extra, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    message.get("role", ""),
                    message.get("content", "") or "",
                    json.dumps(extra, ensure_ascii=False, default=str) if extra else "",
                    time.time(),
                ),
            )
            self.conn.commit()

    # ---------- 读 ----------
    def load(self, session_id: str, limit: int = 200) -> list:
        """返回最近 limit 条消息（按时间升序）。"""
        with self._lock:
            cur = self.conn.execute(
                "SELECT role, content, extra FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cur.fetchall()
        rows.reverse()

        out: list = []
        for role, content, extra in rows:
            msg: dict = {"role": role, "content": content}
            if extra:
                try:
                    msg.update(json.loads(extra))
                except json.JSONDecodeError:
                    pass
            out.append(msg)
        return out

    # ---------- 管理 ----------
    def list_sessions(self) -> list:
        with self._lock:
            cur = self.conn.execute(
                "SELECT session_id, COUNT(*) AS n, MAX(ts) AS last_ts "
                "FROM messages GROUP BY session_id ORDER BY last_ts DESC"
            )
            rows = cur.fetchall()
        return [{"session_id": r[0], "count": r[1], "last_ts": r[2]} for r in rows]

    def delete(self, session_id: str) -> int:
        with self._lock:
            cur = self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self.conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self.conn.close()