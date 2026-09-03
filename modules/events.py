"""事件监听问候：节假日/纪念日/用户生日自动问候。

配置存于 data/events.json（WebUI「定时任务」页编辑）：
  [{"id": "newyear", "name": "元旦", "type": "date", "date": "01-01", "enabled": true,
    "mode": "template|llm", "template": "新年快乐！", "llm_prompt": "今天是元旦，向主人送上祝福",
    "use_voice": false, "targets": [{"session_type": "private", "session_id": "10001"}]}]

生日问候（type=birthday）可绑定 user_id（从用户画像读取生日）；
另提供全局「画像生日问候」：所有画像中生日匹配今天的用户都会收到私信祝福。
"""
import json
import time
from pathlib import Path
from typing import Callable, Optional

from .llm_helpers import RoleContext
from .jobs import generate_proactive_text, render_template


class EventManager:
    def __init__(self, config, data_path: Path, profiles=None):
        self.config = config
        self.file = Path(data_path) / "events.json"
        self.log_file = Path(data_path) / "greeting_log.json"
        self.profiles = profiles
        self.events: list = []
        self.load_events()
        self.load_log()

    # ---------------- 持久化 ----------------
    def load_events(self):
        try:
            if self.file.exists():
                self.events = json.loads(self.file.read_text(encoding="utf-8"))
                if not isinstance(self.events, list):
                    self.events = []
        except Exception as e:
            print(f"加载事件配置失败: {e}")
            self.events = []

    def save_events(self):
        try:
            self.file.write_text(json.dumps(self.events, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as e:
            print(f"保存事件配置失败: {e}")

    def load_log(self):
        self._log = {}
        try:
            if self.log_file.exists():
                self._log = json.loads(self.log_file.read_text(encoding="utf-8"))
        except Exception:
            self._log = {}

    def _mark_sent(self, key: str):
        today = time.strftime("%Y-%m-%d")
        self._log.setdefault(today, [])
        if key not in self._log[today]:
            self._log[today].append(key)
        # 只保留最近 30 天
        cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - 30 * 86400))
        self._log = {k: v for k, v in self._log.items() if k >= cutoff}
        try:
            self.log_file.write_text(json.dumps(self._log, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except Exception:
            pass

    def _sent(self, key: str) -> bool:
        return key in self._log.get(time.strftime("%Y-%m-%d"), [])

    # ---------------- 检查 ----------------
    async def check_and_greet(self, sender, ctx_provider: Callable[[], RoleContext],
                              emotions_provider: Callable[[], dict]):
        """每日问候检查（由调度器在 greeting_check_hour 触发）。"""
        if not self.config.get("greeting_events_enabled", False):
            return
        today_md = time.strftime("%m-%d")
        ctx = ctx_provider()
        emotions = emotions_provider()
        for event in self.events:
            if not event.get("enabled"):
                continue
            if str(event.get("date", "")).strip() != today_md:
                continue
            key = f"event:{event.get('id')}"
            if self._sent(key):
                continue
            text = await self._render_event_text(event, ctx)
            if text:
                for target in event.get("targets", []):
                    await sender.speak_and_send(
                        target.get("session_type", "private"), target.get("session_id", ""),
                        text, emotions, ctx,
                        use_voice=bool(event.get("use_voice", False)),
                        sticker=bool(self.config.get("proactive_sticker", False)))
            self._mark_sent(key)
        # 用户画像生日问候
        if self.config.get("birthday_greeting_enabled", False) and self.profiles:
            template = str(self.config.get("birthday_greet_template", "") or
                           "今天是 {nickname} 的生日，送上最真挚的生日祝福！")
            for user_id, profile in list(self.profiles.profiles.items()):
                if str(profile.get("birthday", "")).strip() != today_md:
                    continue
                key = f"birthday:{user_id}"
                if self._sent(key):
                    continue
                text = template.replace("{nickname}", profile.get("nickname") or "主人")
                if self.config.get("birthday_greet_mode", "template") == "llm":
                    text = await generate_proactive_text(ctx, text) or text
                await sender.speak_and_send("private", user_id, text, emotions, ctx,
                                            use_voice=bool(self.config.get("birthday_greet_voice", False)),
                                            sticker=bool(self.config.get("proactive_sticker", False)))
                self._mark_sent(key)

    async def _render_event_text(self, event: dict, ctx: RoleContext) -> str:
        mode = event.get("mode", "template")
        if mode == "llm":
            instruction = render_template(str(event.get("llm_prompt", "") or
                                              f"今天是{event.get('name', '节日')}，向主人送上问候"),
                                          character_name=ctx.character_name)
            try:
                return await generate_proactive_text(ctx, instruction)
            except Exception as e:
                print(f"节日问候 LLM 生成失败: {e}")
                return ""
        return render_template(str(event.get("template", "")), character_name=ctx.character_name)

    # ---------------- 手动触发（WebUI 测试） ----------------
    async def greet_event_now(self, event_id: str, sender, ctx_provider, emotions_provider):
        event = next((e for e in self.events if str(e.get("id")) == str(event_id)), None)
        if not event:
            return False
        ctx = ctx_provider()
        text = await self._render_event_text(event, ctx)
        if not text:
            return False
        for target in event.get("targets", []):
            await sender.speak_and_send(target.get("session_type", "private"),
                                        target.get("session_id", ""), text,
                                        emotions_provider(), ctx,
                                        use_voice=bool(event.get("use_voice", False)))
        return True
