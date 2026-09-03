"""轻量级异步调度器（纯 asyncio 实现，零第三方依赖）。

替代 apscheduler，避免打包（PyInstaller）与依赖冲突问题，
同时保证所有用户开箱即用。支持：
- interval  循环间隔任务
- daily     每日固定时刻任务（可指定星期几）
- oneshot   一次性任务（指定 unix 时间戳）

所有任务回调异常都会被捕获并记录，不会中断调度循环。
"""
import asyncio
import time
from typing import Callable, Awaitable, Optional, Dict, List

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


class Job:
    def __init__(self, job_id: str, name: str, trigger: dict,
                 func: Callable[..., Awaitable], args: tuple = (), enabled: bool = True):
        self.id = job_id
        self.name = name
        self.trigger = trigger  # {"type": "interval", "seconds": n} / {"type":"daily","time":"HH:MM","weekdays":[..]} / {"type":"oneshot","at": ts}
        self.func = func
        self.args = args
        self.enabled = enabled
        self.next_run = 0.0
        self.last_run = None
        self.last_error = None
        self.run_count = 0
        self.compute_next_run(time.time())

    def compute_next_run(self, now: float):
        t = self.trigger or {}
        ttype = t.get("type", "interval")
        if ttype == "interval":
            seconds = max(5, int(t.get("seconds", 60)))
            self.next_run = now + seconds
        elif ttype == "daily":
            self.next_run = _next_daily(t.get("time", "08:00"), t.get("weekdays"), now)
        elif ttype == "oneshot":
            self.next_run = float(t.get("at", now))
        else:
            self.next_run = now + 60

    def describe(self) -> dict:
        t = self.trigger or {}
        desc = {"id": self.id, "name": self.name, "trigger": t, "enabled": self.enabled,
                "next_run": self.next_run, "last_run": self.last_run,
                "run_count": self.run_count, "last_error": self.last_error}
        ttype = t.get("type", "interval")
        if ttype == "daily":
            wd = t.get("weekdays")
            desc["trigger_text"] = ("每天 " if not wd else "每" + ",".join(WEEKDAY_NAMES[w] for w in wd) + " ") + t.get("time", "08:00")
        elif ttype == "interval":
            desc["trigger_text"] = f"每 {t.get('seconds', 60)} 秒"
        elif ttype == "oneshot":
            desc["trigger_text"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.next_run))
        return desc


def _next_daily(hhmm: str, weekdays: Optional[List[int]], now: float) -> float:
    try:
        hour, minute = [int(x) for x in str(hhmm).split(":")[:2]]
    except Exception:
        hour, minute = 8, 0
    lt = time.localtime(now)
    candidate = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, minute, 0, 0, 0, -1))
    if candidate <= now:
        candidate += 86400
    if weekdays is not None and len(weekdays) > 0:
        for _ in range(8):
            wday = time.localtime(candidate).tm_wday
            if wday in weekdays:
                return candidate
            candidate += 86400
    return candidate


class SchedulerManager:
    """在主事件循环内运行的轻量调度器。"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._wakeup = asyncio.Event()

    def add_job(self, job_id: str, name: str, trigger: dict,
                func: Callable[..., Awaitable], args: tuple = (), enabled: bool = True) -> Job:
        job = Job(job_id, name, trigger, func, args, enabled)
        self.jobs[job_id] = job
        self._wakeup.set()
        return job

    def remove_job(self, job_id: str):
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._wakeup.set()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def reschedule(self, job_id: str, trigger: dict = None, enabled: bool = None):
        job = self.jobs.get(job_id)
        if not job:
            return
        if trigger is not None:
            job.trigger = trigger
        if enabled is not None:
            job.enabled = enabled
        job.compute_next_run(time.time())
        self._wakeup.set()

    def describe_jobs(self) -> list:
        return [j.describe() for j in self.jobs.values()]

    async def _run_job(self, job: Job):
        try:
            await job.func(*job.args)
            job.last_error = None
        except Exception as e:
            job.last_error = f"{type(e).__name__}: {e}"
            print(f"调度任务 [{job.name}] 执行异常: {job.last_error}")
        job.run_count += 1
        job.last_run = time.time()
        if job.trigger and job.trigger.get("type") == "oneshot":
            self.remove_job(job.id)

    async def _loop(self):
        while self._running:
            now = time.time()
            next_wake = now + 30
            for job in list(self.jobs.values()):
                if not job.enabled:
                    continue
                if job.next_run <= now:
                    asyncio.create_task(self._run_job(job))
                    job.compute_next_run(now)
                    if job.next_run <= now:
                        # 间隔过短或已过期的一次性任务，强制前进避免死循环
                        job.next_run = now + 5
                if job.next_run < next_wake:
                    next_wake = job.next_run
            delay = max(0.5, min(next_wake - time.time(), 30))
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._wakeup.clear()

    def start(self):
        if self._running:
            return
        self._running = True
        try:
            self._task = asyncio.get_running_loop().create_task(self._loop())
        except RuntimeError:
            self._running = False

    async def stop(self):
        self._running = False
        self._wakeup.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3)
            except Exception:
                self._task.cancel()
            self._task = None


_scheduler: Optional[SchedulerManager] = None


def get_scheduler() -> SchedulerManager:
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerManager()
    return _scheduler
