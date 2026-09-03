"""工具调用（Function Calling）：内置工具 + 自定义 HTTP/命令工具，全部可在 WebUI 配置与授权。

工具定义存于 data/tools.json（WebUI「高级功能-工具调用」页可视化编辑）：
  {
    "name": "get_weather",
    "type": "builtin|http|command",
    "description": "给 LLM 看的功能描述",
    "parameters": {JSON Schema},
    "enabled": true,
    "allowed_users": [],           # 空 = 所有用户可用；填写 QQ 号则仅这些用户可触发
    "max_calls_per_reply": 1,      # 单次回复内最多调用次数
    # http 类型专用
    "url": "https://.../?city={city}",
    "method": "GET",
    "timeout": 10,
    # command 类型专用
    "command": "python {script} --arg {value}",
    # builtin 类型专用
    "builtin": "time|calculate|random|weather|web_fetch"
  }

权限控制：全局开关 tools_enabled → 工具级 enabled → 用户白名单 allowed_users
→ 单回复调用次数限制 → 命令类工具需额外打开 tools_allow_commands。
"""
import ast
import json
import math
import operator
import re
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple

import httpx

DEFAULT_TOOLS = [
    {
        "name": "get_current_time",
        "type": "builtin", "builtin": "time",
        "description": "获取当前的日期、时间和星期几。当用户询问现在的时间或日期时调用。",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "enabled": False, "allowed_users": [], "max_calls_per_reply": 2,
    },
    {
        "name": "calculate",
        "type": "builtin", "builtin": "calculate",
        "description": "计算一个数学表达式，例如 23*7+sqrt(144)。仅支持四则运算、幂、开方等安全运算。",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如 (3+4)*2"}}, "required": ["expression"]},
        "enabled": False, "allowed_users": [], "max_calls_per_reply": 3,
    },
    {
        "name": "get_weather",
        "type": "builtin", "builtin": "weather",
        "description": "查询某个城市的当前天气。需要提供城市名（中文或拼音）。",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名，如 北京"}}, "required": ["city"]},
        "enabled": False, "allowed_users": [], "max_calls_per_reply": 2,
    },
    {
        "name": "web_fetch",
        "type": "builtin", "builtin": "web_fetch",
        "description": "访问一个网页 URL 并返回其正文文本内容，用于查询资料。",
        "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "完整的 http/https 网址"}}, "required": ["url"]},
        "enabled": False, "allowed_users": [], "max_calls_per_reply": 1,
    },
]


# ------------------------- 内置工具实现 -------------------------

_CALC_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_CALC_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
    "pow": math.pow, "max": max, "min": min, "pi": math.pi, "e": math.e,
}


def _safe_calc(expression: str):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
            return _CALC_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
            return _CALC_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _CALC_FUNCS:
            return _CALC_FUNCS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCS:
            args = [_eval(a) for a in node.args]
            return _CALC_FUNCS[node.func.id](*args)
        raise ValueError("不支持的表达式")
    node = ast.parse(str(expression), mode="eval")
    value = _eval(node)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return value


def _strip_html(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ToolRegistry:
    def __init__(self, config, data_path: Path):
        self.config = config
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.file = self.data_path / "tools.json"
        self.tools: List[dict] = []
        self._call_counts: Dict[str, int] = {}
        self.load()

    # ---------------- 持久化 ----------------
    def load(self):
        try:
            if self.file.exists():
                self.tools = json.loads(self.file.read_text(encoding="utf-8"))
                if not isinstance(self.tools, list):
                    self.tools = []
            else:
                self.tools = [dict(t) for t in DEFAULT_TOOLS]
                self.save()
        except Exception as e:
            print(f"加载工具配置失败: {e}")
            self.tools = [dict(t) for t in DEFAULT_TOOLS]

    def save(self):
        try:
            self.file.write_text(json.dumps(self.tools, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as e:
            print(f"保存工具配置失败: {e}")

    # ---------------- 权限与模式 ----------------
    def has_enabled_tools(self) -> bool:
        return any(t.get("enabled") for t in self.tools)

    def get_schema(self) -> List[dict]:
        schema = []
        for t in self.tools:
            if not t.get("enabled"):
                continue
            schema.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            })
        return schema

    def check_permission(self, tool: dict, user_id: str) -> Tuple[bool, str]:
        if not self.config.get("tools_enabled", False):
            return False, "工具调用功能未开启"
        if not tool.get("enabled"):
            return False, "该工具未启用"
        allowed = tool.get("allowed_users") or []
        if allowed and str(user_id) not in [str(u) for u in allowed]:
            return False, "当前用户无权使用该工具"
        if tool.get("type") == "command" and not self.config.get("tools_allow_commands", False):
            return False, "命令类工具已被全局禁用（tools_allow_commands）"
        return True, ""

    # ---------------- 执行 ----------------
    async def execute(self, name: str, arguments, user_id: str = "") -> Tuple[bool, str]:
        tool = next((t for t in self.tools if t.get("name") == name), None)
        if tool is None:
            return False, f"未找到工具: {name}"
        ok, reason = self.check_permission(tool, user_id)
        if not ok:
            return False, reason
        used = self._call_counts.get(name, 0)
        limit = int(tool.get("max_calls_per_reply", 3))
        if used >= limit:
            return False, f"工具 {name} 本次回复调用次数已达上限({limit})"
        self._call_counts[name] = used + 1
        args = {}
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except Exception:
                return False, f"参数解析失败: {arguments[:100]}"
        elif isinstance(arguments, dict):
            args = arguments
        try:
            ttype = tool.get("type", "builtin")
            if ttype == "builtin":
                return await self._run_builtin(tool.get("builtin", ""), args, tool)
            elif ttype == "http":
                return await self._run_http(tool, args)
            elif ttype == "command":
                return await self._run_command(tool, args)
            return False, f"未知工具类型: {ttype}"
        except Exception as e:
            return False, f"工具执行异常: {type(e).__name__}: {e}"

    def begin_reply(self):
        """每次回复开始时重置调用计数。"""
        self._call_counts.clear()

    async def _run_builtin(self, builtin: str, args: dict, tool: dict) -> Tuple[bool, str]:
        builtin = (builtin or "").lower()
        if builtin == "time":
            lt = time.localtime()
            try:
                week = "周" + "一二三四五六日"[lt.tm_wday % 7]
            except Exception:
                week = ""
            return True, time.strftime(f"%Y-%m-%d %H:%M:%S {week}", lt)
        if builtin == "calculate":
            expr = str(args.get("expression", "")).strip()
            if not expr:
                return False, "缺少 expression 参数"
            try:
                return True, f"{expr} = {_safe_calc(expr)}"
            except Exception as e:
                return False, f"计算失败: {e}"
        if builtin == "random":
            lo = float(args.get("min", 0)); hi = float(args.get("max", 1))
            import random as _random
            val = _random.uniform(lo, hi)
            return True, str(int(val) if val.is_integer() else round(val, 6))
        if builtin == "weather":
            city = str(args.get("city", "")).strip()
            if not city:
                return False, "缺少 city 参数"
            from urllib.parse import quote
            url_tpl = tool.get("url", "https://wttr.in/{city}?format=j1")
            if "{city}" in url_tpl:
                url = url_tpl.replace("{city}", quote(city))
            else:
                url = f"https://wttr.in/{quote(city)}?format=j1"
            async with httpx.AsyncClient(timeout=int(tool.get("timeout", 10)), trust_env=False) as client:
                resp = await client.get(url, headers={"User-Agent": "curl/8.0"})
                resp.raise_for_status()
                data = resp.json()
            cur = (data.get("current_condition") or [{}])[0]
            desc = ""
            try:
                desc = cur.get("lang_zh", [{}])[0].get("value") or cur.get("weatherDesc", [{}])[0].get("value", "")
            except Exception:
                pass
            out = (f"{city} 当前天气: {desc}, 气温 {cur.get('temp_C', '?')}°C, "
                   f"体感 {cur.get('FeelsLikeC', '?')}°C, 湿度 {cur.get('humidity', '?')}%")
            return True, out
        if builtin == "web_fetch":
            url = str(args.get("url", "")).strip()
            if not re.match(r"^https?://", url):
                return False, "URL 必须以 http:// 或 https:// 开头"
            max_chars = int(tool.get("max_chars", 1200))
            async with httpx.AsyncClient(timeout=int(tool.get("timeout", 15)), trust_env=False,
                                         follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                text = _strip_html(resp.text)
            return True, text[:max_chars]
        return False, f"未知内置工具: {builtin}"

    async def _run_http(self, tool: dict, args: dict) -> Tuple[bool, str]:
        url_tpl = tool.get("url", "")
        if not url_tpl:
            return False, "未配置 url"
        url = url_tpl
        for k, v in args.items():
            url = url.replace("{" + str(k) + "}", str(v))
        method = str(tool.get("method", "GET")).upper()
        timeout = float(tool.get("timeout", 10))
        headers = tool.get("headers") or {}
        max_chars = int(tool.get("max_chars", 800))
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=True) as client:
            if method == "POST":
                resp = await client.post(url, json=args, headers=headers)
            else:
                resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "json" in ctype:
                try:
                    body = json.dumps(resp.json(), ensure_ascii=False)
                except Exception:
                    body = resp.text
            else:
                body = _strip_html(resp.text)
        return True, body[:max_chars]

    async def _run_command(self, tool: dict, args: dict) -> Tuple[bool, str]:
        cmd_tpl = tool.get("command", "")
        if not cmd_tpl:
            return False, "未配置 command"
        cmd = cmd_tpl
        for k, v in args.items():
            cmd = cmd.replace("{" + str(k) + "}", str(v))
        timeout = float(tool.get("timeout", 20))
        max_chars = int(tool.get("max_chars", 800))

        def _run():
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=timeout, creationflags=creationflags)
            return proc

        try:
            proc = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            return False, f"命令超时({timeout}s)"
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr.strip() else "")
        return proc.returncode == 0, output.strip()[:max_chars] or "(无输出)"
