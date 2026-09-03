"""SQLite 数据库管理：交互统计、待办事项。

零第三方依赖（标准库 sqlite3），写入轻量，不会明显阻塞事件循环。
"""
import sqlite3
import threading
import time
import json
from pathlib import Path


class DatabaseManager:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_path / "ltvm.db"
        self._lock = threading.Lock()
        self._conn = None
        self._init_db()

    def _connect(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    session_type TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    user_name TEXT,
                    character_key TEXT,
                    emotion TEXT,
                    llm_ms REAL DEFAULT 0,
                    tts_ms REAL DEFAULT 0,
                    sentence_count INTEGER DEFAULT 0,
                    ok INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    session_type TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    content TEXT NOT NULL,
                    remind_time REAL,
                    status TEXT DEFAULT 'pending',
                    source TEXT DEFAULT 'auto'
                );
                CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);
            """)
            conn.commit()

    def execute(self, sql, params=()):
        with self._lock:
            conn = self._connect()
            cur = conn.execute(sql, params)
            conn.commit()
            return cur

    def query_all(self, sql, params=()):
        with self._lock:
            conn = self._connect()
            cur = conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql, params=()):
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    # ---------- 交互统计 ----------
    def record_interaction(self, session_type, session_id, user_id, user_name,
                           character_key, emotion, llm_ms, tts_ms,
                           sentence_count, ok=True):
        try:
            self.execute(
                "INSERT INTO interactions (ts, session_type, session_id, user_id, user_name,"
                " character_key, emotion, llm_ms, tts_ms, sentence_count, ok)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), session_type, str(session_id), str(user_id or ""), str(user_name or ""),
                 str(character_key or ""), str(emotion or ""), float(llm_ms or 0),
                 float(tts_ms or 0), int(sentence_count or 0), 1 if ok else 0)
            )
        except Exception as e:
            print(f"记录交互统计失败: {e}")

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
