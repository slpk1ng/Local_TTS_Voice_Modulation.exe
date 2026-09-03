"""待办/提醒管理：正则或 LLM 提取 → SQLite 存储 → 到点通过调度器发送提醒。

所有正则模式、关键词、提醒话术、检查行为均可在 WebUI 配置，无硬编码。
"""
import re
import time
from typing import Dict, List, Optional, Tuple

from .database import DatabaseManager
from .scheduler import SchedulerManager
from .llm_helpers import RoleContext, generate_json_reply

# 时段前缀（下午3点 / 晚上11点 等）+ 点分时间（支持中文数字与「半」点）
_CN_NUM = r"[一二两三四五六七八九十]+"
_TODAY_HHMM = (r"(?:凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里)?\s*"
               rf"(?:\d{{1,2}}|{_CN_NUM})\s*[点时:：]\s*(?:半|\d{{1,2}}\s*分?|\d{{0,2}})?")
# 相对时长（30分钟后 / 半小时之后 / 两小时后 / 3天后，数字支持中文写法）
_TODAY_DUR = (rf"(?:(?:\d{{1,3}}|{_CN_NUM})\s*分钟|"
              rf"(?:\d{{1,3}}|{_CN_NUM})\s*个?小时|"
              rf"(?:\d{{1,3}}|{_CN_NUM})\s*天|"
              rf"半\s*个?小时)(?:之|以)?[後后]")

DEFAULT_TODO_PATTERNS = [
    rf"(?:提醒|记得)(?:我)?\s*(?:在|于)?\s*({_TODAY_HHMM}|{_TODAY_DUR}|(?:明天|明日)\s*{_TODAY_HHMM})[,，。\s]*(.+)",
    rf"((?:明天|明日)?\s*{_TODAY_HHMM})\s*(?:提醒|叫我)[,，。\s]*(.+)",
    rf"({_TODAY_DUR})\s*(?:提醒|叫我)[,，。\s]*(.+)",
]

DEFAULT_TODO_KEYWORDS = ["提醒", "待办", "别忘了", "记得", "叫我"]

DEFAULT_EXTRACT_PROMPT = (
    "你是待办提取助手。判断用户消息是否包含一个明确的提醒/待办事项。"
    "如果有，输出JSON：{\"has_todo\": true, \"content\": \"要提醒的事项\", "
    "\"delay_minutes\": 相对当前时间的分钟数(整数,无法确定则为0), \"time\": \"HH:MM 格式的绝对时间(可选)\"}；"
    "如果没有明确的提醒事项，输出 {\"has_todo\": false}。只输出JSON。"
)


# 中文数字映射（一~九），两 归二
_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_num_to_int(text: str) -> Optional[int]:
    """中文数字转整数（一~九十九：十/十五/二十/二十三…）；无法解析返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left) if left else 1
        ones = _CN_DIGITS.get(right) if right else 0
        if (left and tens is None) or (right and ones is None):
            return None
        return tens * 10 + ones
    if len(text) == 1:
        return _CN_DIGITS.get(text)
    return None


def _expr_num(token: str) -> Optional[int]:
    """时间表达式里的数量词：阿拉伯数字或中文数字。"""
    token = (token or "").strip()
    if token.isdigit():
        return int(token)
    return _cn_num_to_int(token)


class TodoManager:
    def __init__(self, config, db: DatabaseManager, scheduler: SchedulerManager,
                 sender=None, emotions_provider=None):
        self.config = config
        self.db = db
        self.scheduler = scheduler
        self.sender = sender
        self.emotions_provider = emotions_provider  # 返回当前角色 emotions dict
        self.ctx_provider = None                    # 返回当前角色 RoleContext

    # ---------------- 提取 ----------------
    def _patterns(self) -> List[re.Pattern]:
        raw = self.config.get("todo_regex_patterns", DEFAULT_TODO_PATTERNS)
        if isinstance(raw, str):
            raw = [line for line in raw.splitlines() if line.strip()]
        patterns = []
        for p in raw or DEFAULT_TODO_PATTERNS:
            try:
                patterns.append(re.compile(p))
            except re.error as e:
                print(f"待办正则无效，已跳过: {p} ({e})")
        return patterns

    def _parse_time_expr(self, expr: str, now: float) -> Optional[float]:
        expr = str(expr).strip()
        m = re.search(rf"((?:\d{{1,3}}|{_CN_NUM}))\s*分钟(?:之|以)?[后後]", expr)
        if m:
            n = _expr_num(m.group(1))
            if n:
                return now + n * 60
        m = re.search(rf"((?:\d{{1,3}}|{_CN_NUM}))\s*个?小时(?:之|以)?[后後]", expr)
        if m:
            n = _expr_num(m.group(1))
            if n:
                return now + n * 3600
        if re.search(r"半\s*个?小时", expr):
            return now + 1800
        m = re.search(rf"((?:\d{{1,3}}|{_CN_NUM}))\s*天(?:之|以)?[后後]", expr)
        if m:
            n = _expr_num(m.group(1))
            if n:
                return now + n * 86400
        tomorrow = "明天" in expr or "明日" in expr
        m = re.search(r"(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里)?\s*((?:\d{1,2}|[一二两三四五六七八九十]{1,3}))\s*[点时:：]\s*(半|\d{1,2}\s*分?|\d{0,2})?", expr)
        if m:
            hour = _expr_num(m.group(2))
            if hour is None:
                return None
            # 时段换算：下午/晚上等 +12h（凌晨/上午不加）
            if m.group(1) in ("下午", "傍晚", "晚上", "夜里") and hour < 12:
                hour += 12
            minute_part = m.group(3) or ""
            if "半" in minute_part:
                minute = 30
            else:
                digits = re.sub(r"\D", "", minute_part)
                minute = int(digits) if digits else 0
            lt = time.localtime(now)
            candidate = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, minute, 0, 0, 0, -1))
            if candidate <= now:
                candidate += 86400
            if tomorrow:
                candidate += 86400
            return candidate
        return None

    @staticmethod
    def _keyword_list(keywords) -> List[str]:
        """配置可能是多行字符串（WebUI 保存格式）或列表，统一成词列表。

        不能直接迭代字符串——那会按单字符匹配，关键词门槛形同虚设。
        """
        if isinstance(keywords, str):
            return [kw.strip() for kw in keywords.splitlines() if kw.strip()]
        if isinstance(keywords, list):
            return [str(kw).strip() for kw in keywords if str(kw).strip()]
        return []

    def extract_sync(self, text: str) -> List[Tuple[str, float]]:
        """正则模式提取，返回 [(content, remind_ts), ...]。"""
        found = []
        now = time.time()
        keywords = self._keyword_list(self.config.get("todo_keywords", DEFAULT_TODO_KEYWORDS))
        if keywords and not any(kw in text for kw in keywords):
            return found
        for pattern in self._patterns():
            for m in pattern.finditer(text):
                try:
                    time_expr, content = m.group(1), m.group(2)
                except (IndexError, AttributeError):
                    print(f"待办正则需包含两个捕获组（时间、内容）: {pattern.pattern}")
                    break
                remind_ts = self._parse_time_expr(time_expr, now)
                content = (content or "").strip()[:200]
                # 循环剥离提醒内容开头连续的冗余代词/动词（"提醒我喝水"→"喝水"、
                # "叫我叫我起床"→"起床"），让提醒话术自然，也提升去重命中率
                while content:
                    stripped = re.sub(r"^(?:提醒我|叫我|提醒|记得|我)", "", content).strip()
                    if stripped == content:
                        break
                    content = stripped
                if remind_ts and content:
                    found.append((content, remind_ts))
        # 去重：多个正则可能同时命中同一句提醒（如「记得在8点提醒我吃药」
        # 会提取出「提醒我吃药」和「我吃药」），同一时间点仅保留最长内容
        best: Dict[float, str] = {}
        for content, ts in found:
            if ts not in best or len(content) > len(best[ts]):
                best[ts] = content
        seen, out = set(), []
        for content, ts in found:
            if ts in best and best[ts] == content and ts not in seen:
                seen.add(ts)
                out.append((content, ts))
        return out

    async def extract_llm(self, ctx: RoleContext, text: str) -> List[Tuple[str, float]]:
        now = time.time()
        lt = time.localtime(now)
        keywords = self._keyword_list(self.config.get("todo_keywords", DEFAULT_TODO_KEYWORDS))
        if keywords and not any(kw in text for kw in keywords):
            return []
        prompt = str(self.config.get("todo_extract_prompt", "") or DEFAULT_EXTRACT_PROMPT)
        system = (f"{prompt}\n当前时间: {time.strftime('%Y-%m-%d %H:%M', lt)}")
        try:
            data = await generate_json_reply(ctx, system, text, max_tokens=200)
        except Exception as e:
            print(f"LLM 待办提取失败: {e}")
            return []
        if not isinstance(data, dict) or not data.get("has_todo"):
            return []
        content = str(data.get("content", "")).strip()[:200]
        if not content:
            return []
        remind_ts = None
        delay = data.get("delay_minutes")
        try:
            delay = float(delay)
            if delay > 0:
                remind_ts = now + delay * 60
        except (TypeError, ValueError):
            pass
        if remind_ts is None and data.get("time"):
            remind_ts = self._parse_time_expr(str(data.get("time")), now)
        if remind_ts is None:
            return []
        return [(content, remind_ts)]

    async def extract_and_add(self, ctx: RoleContext, text: str, session_type: str,
                              session_id: str, user_id: str = "") -> List[Tuple[str, float]]:
        """LLM 模式提取并入库（后台任务调用）。"""
        found = await self.extract_llm(ctx, text)
        for content, remind_ts in found:
            self.add_todo(content, remind_ts, session_type, session_id, user_id, source="llm")
        return found

    # ---------------- 增删查 ----------------
    def add_todo(self, content: str, remind_ts: float, session_type: str,
                 session_id: str, user_id: str = "", source: str = "auto") -> Optional[dict]:
        try:
            cur = self.db.execute(
                "INSERT INTO todos (created_at, session_type, session_id, user_id, content,"
                " remind_time, status, source) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), session_type, str(session_id), str(user_id), content,
                 remind_ts, "pending", source))
            todo_id = cur.lastrowid
            self._schedule(todo_id, content, remind_ts, session_type, session_id)
            return {"id": todo_id, "content": content, "remind_time": remind_ts}
        except Exception as e:
            print(f"保存待办失败: {e}")
            return None

    def _schedule(self, todo_id: int, content: str, remind_ts: float,
                  session_type: str, session_id: str):
        async def _remind(todo_id=todo_id, content=content,
                          session_type=session_type, session_id=session_id):
            await self.fire_reminder(todo_id, content, session_type, session_id)
        self.scheduler.add_job(f"todo_{todo_id}", f"待办提醒: {content[:16]}",
                               {"type": "oneshot", "at": remind_ts}, _remind)

    async def fire_reminder(self, todo_id: int, content: str, session_type: str, session_id: str):
        self.db.execute("UPDATE todos SET status='done' WHERE id=?", (todo_id,))
        template = str(self.config.get("todo_remind_template", "") or "⏰ 提醒时间到啦：{content}")
        message = template.replace("{content}", content)
        if self.sender is None:
            print(f"[待办提醒] {message}")
            return
        emotions = self.emotions_provider() if self.emotions_provider else {}
        ctx = self.ctx_provider() if self.ctx_provider else None
        await self.sender.speak_and_send(session_type, session_id, message, emotions, ctx,
                                         use_voice=bool(self.config.get("todo_voice", False)))

    def restore_pending(self):
        """程序启动时恢复未完成的待办调度。"""
        try:
            rows = self.db.query_all("SELECT * FROM todos WHERE status='pending'")
            now = time.time()
            for row in rows:
                remind_ts = row.get("remind_time") or 0
                if remind_ts <= now:
                    self.db.execute("UPDATE todos SET status='missed' WHERE id=?", (row["id"],))
                    continue
                self._schedule(row["id"], row["content"], remind_ts,
                               row.get("session_type", "private"), row.get("session_id", ""))
            if rows:
                print(f"已恢复 {len(rows)} 条待办提醒调度。")
        except Exception as e:
            print(f"恢复待办失败: {e}")

    def list_todos(self, status: str = None) -> list:
        if status:
            return self.db.query_all("SELECT * FROM todos WHERE status=? ORDER BY remind_time", (status,))
        return self.db.query_all("SELECT * FROM todos ORDER BY id DESC LIMIT 200")

    def complete(self, todo_id: int) -> bool:
        self.db.execute("UPDATE todos SET status='done' WHERE id=?", (todo_id,))
        self.scheduler.remove_job(f"todo_{todo_id}")
        return True

    def delete(self, todo_id: int) -> bool:
        self.db.execute("DELETE FROM todos WHERE id=?", (todo_id,))
        self.scheduler.remove_job(f"todo_{todo_id}")
        return True
