"""性能监控与统计：内存中的实时指标 + 数据库聚合查询。"""
import time
from collections import deque
from typing import Optional

from .database import DatabaseManager


class StatsManager:
    def __init__(self, db: DatabaseManager, start_time: float = None):
        self.db = db
        self.start_time = start_time or time.time()
        self.llm_times = deque(maxlen=100)   # 最近 100 次 LLM 耗时(ms)
        self.tts_times = deque(maxlen=200)   # 最近 200 次 TTS 耗时(ms)
        self.msg_count = 0
        self.err_count = 0
        self.active_sessions = set()

    def record_llm(self, ms: float):
        if ms and ms > 0:
            self.llm_times.append(float(ms))

    def record_tts(self, ms: float):
        if ms and ms > 0:
            self.tts_times.append(float(ms))

    def record_message(self, session_id: str, ok: bool = True):
        self.msg_count += 1
        if not ok:
            self.err_count += 1
        if session_id:
            self.active_sessions.add(session_id)

    def get_performance(self) -> dict:
        def stats(dq):
            if not dq:
                return {"count": 0, "avg": 0, "min": 0, "max": 0}
            return {"count": len(dq), "avg": round(sum(dq) / len(dq), 1),
                    "min": round(min(dq), 1), "max": round(max(dq), 1)}
        return {
            "uptime_seconds": int(time.time() - self.start_time),
            "llm": stats(self.llm_times),
            "tts": stats(self.tts_times),
            "messages_total": self.msg_count,
            "errors_total": self.err_count,
            "active_sessions": len(self.active_sessions),
        }

    def get_stats(self) -> dict:
        now = time.time()
        day = 86400
        out = {}
        try:
            out["totals"] = self.db.query_one(
                "SELECT COUNT(*) AS n, IFNULL(AVG(llm_ms),0) AS avg_llm, IFNULL(AVG(tts_ms),0) AS avg_tts,"
                " IFNULL(AVG(sentence_count),0) AS avg_sentences FROM interactions") or {}
            out["today"] = self.db.query_one(
                "SELECT COUNT(*) AS n FROM interactions WHERE ts >= ?", (now - now % day,)) or {}
            out["per_day"] = self.db.query_all(
                "SELECT CAST(ts/? AS INTEGER) AS day_idx, COUNT(*) AS n FROM interactions"
                " WHERE ts >= ? GROUP BY day_idx ORDER BY day_idx",
                (day, now - 13 * day))
            out["emotions"] = self.db.query_all(
                "SELECT emotion, COUNT(*) AS n FROM interactions"
                " WHERE ts >= ? AND IFNULL(emotion,'') != '' GROUP BY emotion ORDER BY n DESC LIMIT 12",
                (now - 30 * day,))
            out["emotion_trend"] = self.db.query_all(
                "SELECT CAST(ts/? AS INTEGER) AS day_idx, emotion, COUNT(*) AS n FROM interactions"
                " WHERE ts >= ? AND IFNULL(emotion,'') != '' GROUP BY day_idx, emotion ORDER BY day_idx",
                (day, now - 13 * day))
            out["top_sessions"] = self.db.query_all(
                "SELECT session_id, session_type, COUNT(*) AS n, MAX(ts) AS last_ts FROM interactions"
                " GROUP BY session_id ORDER BY n DESC LIMIT 10")
            out["top_users"] = self.db.query_all(
                "SELECT user_id, IFNULL(MAX(user_name),'') AS user_name, COUNT(*) AS n FROM interactions"
                " WHERE IFNULL(user_id,'') != '' GROUP BY user_id ORDER BY n DESC LIMIT 10")
            out["top_characters"] = self.db.query_all(
                "SELECT character_key, COUNT(*) AS n FROM interactions WHERE IFNULL(character_key,'') != ''"
                " GROUP BY character_key ORDER BY n DESC LIMIT 10")
        except Exception as e:
            out["error"] = str(e)
        out["performance"] = self.get_performance()
        return out
