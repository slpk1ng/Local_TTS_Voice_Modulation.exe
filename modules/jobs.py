"""用户自定义定时任务：interval / daily / weekly 触发，向指定会话发送模板或 LLM 生成消息。

配置存于 data/scheduled_jobs.json（WebUI「定时任务」页可视化编辑）：
  {
    "id": "morning_greet",
    "name": "每日早安",
    "enabled": true,
    "trigger": {"type": "daily", "time": "08:30"},        # 或 {"type":"interval","seconds":3600}
    "weekdays": [0,1,2,3,4],                              # daily 可选，0=周一
    "target": {"session_type": "private", "session_id": "10001"},
    "action": {"mode": "template", "template": "主人早上好呀～今天是 {date} {weekday}",
               "use_voice": true}
  }
mode=llm 时使用 action.llm_prompt 生成开场白（可使用 {character_name} 等占位符）。
"""
import json
import time
from pathlib import Path
from typing import Callable, Optional

from .scheduler import SchedulerManager
from .llm_helpers import RoleContext, generate_text_reply

JOB_PREFIX = "sched_"


def render_template(template: str, character_name: str = "") -> str:
    lt = time.localtime()
    try:
        weekday = "周" + "一二三四五六日"[lt.tm_wday % 7]
    except Exception:
        weekday = ""
    return (str(template)
            .replace("{date}", time.strftime("%Y-%m-%d", lt))
            .replace("{time}", time.strftime("%H:%M", lt))
            .replace("{weekday}", weekday)
            .replace("{character_name}", str(character_name or "")))


async def generate_proactive_text(ctx: RoleContext, instruction: str) -> str:
    """让角色以其人设生成一段主动开口的话（纯文本，无 JSON）。"""
    system = (
        f"{ctx.get('personality_prompt', '')}\n"
        "【输出要求】直接输出你要主动说的话本身，口语化、简短自然（1~3句）。"
        "禁止输出JSON、解释、动作描写或任何多余格式。"
    )
    text = await generate_text_reply(ctx, system, instruction, max_tokens=200)
    return text.strip().strip('"')


class ScheduledJobManager:
    def __init__(self, config, data_path: Path, scheduler: SchedulerManager,
                 sender=None, ctx_provider: Optional[Callable[[], RoleContext]] = None,
                 emotions_provider: Optional[Callable[[], dict]] = None):
        self.config = config
        self.file = Path(data_path) / "scheduled_jobs.json"
        self.scheduler = scheduler
        self.sender = sender
        self.ctx_provider = ctx_provider
        self.emotions_provider = emotions_provider
        self.jobs: list = []
        self.load()

    # ---------------- 持久化 ----------------
    def load(self):
        try:
            if self.file.exists():
                self.jobs = json.loads(self.file.read_text(encoding="utf-8"))
                if not isinstance(self.jobs, list):
                    self.jobs = []
        except Exception as e:
            print(f"加载定时任务失败: {e}")
            self.jobs = []

    def save(self):
        try:
            self.file.write_text(json.dumps(self.jobs, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as e:
            print(f"保存定时任务失败: {e}")

    # ---------------- 调度 ----------------
    def register_all(self):
        if not self.config.get("scheduler_enabled", True):
            print("定时任务功能未开启（scheduler_enabled=false），跳过注册。")
            return
        count = 0
        for job in self.jobs:
            if self._register(job):
                count += 1
        print(f"已注册 {count}/{len(self.jobs)} 个自定义定时任务。")

    def _register(self, job: dict) -> bool:
        if not job.get("enabled"):
            return False
        trigger = job.get("trigger") or {}
        ttype = trigger.get("type")
        if ttype == "weekly":
            # 调度器以 daily+weekdays 表示每周任务；兼容 weekly 类型避免退化为分钟级循环
            trigger = {"type": "daily", "time": trigger.get("time", "08:00"),
                       "weekdays": trigger.get("weekdays")}
            ttype = "daily"
        if ttype not in ("interval", "daily"):
            print(f"定时任务 {job.get('name')} 触发器类型无效: {trigger.get('type')}")
            return False
        job_id = JOB_PREFIX + str(job.get("id", ""))
        self.scheduler.add_job(job_id, job.get("name", job_id), trigger,
                               self._run_job, args=(job,))
        return True

    def reload(self):
        """WebUI 保存后重新注册全部任务。"""
        self.load()
        for job_id in [jid for jid in list(self.scheduler.jobs) if jid.startswith(JOB_PREFIX)]:
            self.scheduler.remove_job(job_id)
        self.register_all()

    def describe(self):
        """返回任务定义 + 实时运行状态。"""
        infos = []
        for job in self.jobs:
            info = dict(job)
            live = self.scheduler.jobs.get(JOB_PREFIX + str(job.get("id", "")))
            if live:
                info["runtime"] = live.describe()
            else:
                info["runtime"] = None
            infos.append(info)
        return infos

    # ---------------- 执行 ----------------
    async def _run_job(self, job: dict):
        target = job.get("target") or {}
        action = job.get("action") or {}
        session_type = target.get("session_type", "private")
        session_id = target.get("session_id", "")
        if not session_id or self.sender is None or self.sender.client is None:
            return
        mode = action.get("mode", "template")
        text = ""
        character_name = self.ctx_provider().character_name if self.ctx_provider else ""
        if mode == "llm":
            instruction = render_template(str(action.get("llm_prompt", "") or "主动打个招呼"),
                                          character_name=character_name)
            ctx = self.ctx_provider() if self.ctx_provider else RoleContext(self.config)
            try:
                text = await generate_proactive_text(ctx, instruction)
            except Exception as e:
                print(f"定时任务 LLM 生成失败: {e}")
        else:
            text = render_template(str(action.get("template", "")), character_name=character_name)
        if not text:
            return
        emotions = self.emotions_provider() if self.emotions_provider else {}
        ctx = self.ctx_provider() if self.ctx_provider else None
        await self.sender.speak_and_send(session_type, session_id, text, emotions, ctx,
                                         use_voice=bool(action.get("use_voice", False)),
                                         sticker=bool(self.config.get("proactive_sticker", False)))
