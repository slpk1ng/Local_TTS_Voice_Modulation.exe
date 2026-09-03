"""LLM 交互核心：消息构建、流式句子解析、工具调用循环、文本/JSON 生成。

- RoleContext: 全局配置 + 角色覆盖字段的只读视图（多角色支持的基础）。
- SentenceStreamParser: 边接收增量输出边解析出完整句子对象，实现实时流式回复。
- chat_with_tools: Function Calling 循环（支持 Ollama 原生与 OpenAI 兼容接口）。
"""
import json
import re
import time
from typing import Optional, List, Dict, AsyncGenerator

import httpx


def extract_json_objects(text: str) -> List[dict]:
    """提取文本中所有顶层 JSON 对象（按出现顺序）；顶层数组会展开其中的对象元素。

    与 extract_json 只取第一个对象不同：模型常把每句话输出成独立的 JSON 块
    （提示词要求"至少两个JSON块"时尤其常见），只取第一块会丢掉其余句子，
    导致"生成了多个JSON块却只合成一条语音"。每个对象严格解析失败时
    用宽容解析器修复；输出被截断时自动补齐末尾未闭合的括号。
    """
    if not text:
        return []
    objs: List[dict] = []
    depth = 0
    in_string = False
    escape = False
    start = None
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in '{[':
            if depth == 0:
                start = i
            depth += 1
        elif ch in '}]':
            if depth == 0:
                continue  # 游离闭括号，忽略
            depth -= 1
            if depth == 0 and start is not None:
                data = _loads_lenient(text[start:i + 1])
                if isinstance(data, dict):
                    objs.append(data)
                elif isinstance(data, list):
                    objs.extend(x for x in data if isinstance(x, dict))
                start = None
    if depth > 0 and start is not None:
        # 截断输出：末尾未闭合的对象/数组补齐后解析，保住已完整生成的句子
        data = _loads_lenient(text[start:])
        if isinstance(data, dict):
            objs.append(data)
        elif isinstance(data, list):
            objs.extend(x for x in data if isinstance(x, dict))
    return objs


# 句子对象至少含有这些键之一（用于把句子块与无关 JSON 对象区分开）
_SENTENCE_KEYS = ("zh", "ja", "en", "lang", "display", "emotion", "text")
# 台词文本键（emotion 不算台词；不含任何台词键值的句子对象必须丢弃，
# 否则会用用户消息兜底，导致机器把用户刚说的话朗读出来）
_TEXT_KEYS = ("zh", "ja", "en", "lang", "display", "text")


def _is_sentence_like(obj) -> bool:
    """判断 JSON 对象是否像一条句子块（含 zh/ja/emotion 等键，且不是 sentences 包装）。"""
    return isinstance(obj, dict) and "sentences" not in obj and \
        any(k in obj for k in _SENTENCE_KEYS)


def sentence_obj_has_text(s) -> bool:
    """判断一个句子对象是否含实际台词文本（emotion 等元数据不算）。"""
    if not isinstance(s, dict):
        return bool(str(s).strip())
    return any(str(s.get(k, "")).strip() for k in _TEXT_KEYS)


class _TolerantJSONParser:
    """宽容 JSON 解析器：修复模型常见的 JSON 语法病，尽量避免整段输出报废。

    处理的问题（均为模型真实输出中高频出现的）：
    - 字符串值内未转义的引号（仅当引号后跟结构字符 ,}]: 时才视为字符串结束）
    - 字符串内的裸换行/控制字符（严格 JSON 不允许，模型经常直接断行）
    - 尾逗号 / 多余逗号
    - 输出被截断：未闭合的字符串、对象、数组在文本末尾自动补齐，
      保住已完整生成的句子（旧逻辑遇截断直接全盘失败）
    - 无引号的键名
    """

    def __init__(self, text: str):
        self.t = text
        self.n = len(text)
        self.i = 0

    def parse(self):
        self.i = 0
        return self._value()

    def _skip_ws(self):
        while self.i < self.n and self.t[self.i] in " \t\r\n":
            self.i += 1

    def _value(self):
        self._skip_ws()
        if self.i >= self.n:
            return None
        c = self.t[self.i]
        if c == '{':
            return self._object()
        if c == '[':
            return self._array()
        if c == '"':
            return self._string()
        return self._atom()

    def _object(self):
        obj = {}
        self.i += 1  # {
        while True:
            self._skip_ws()
            if self.i >= self.n:
                return obj  # 截断：返回已解析出的部分
            c = self.t[self.i]
            if c == '}':
                self.i += 1
                return obj
            if c == ',':  # 尾逗号/多余逗号
                self.i += 1
                continue
            if c == '"':
                key = self._string()
            else:  # 无引号键名：取到冒号/换行为止
                j = self.i
                while j < self.n and self.t[j] not in ':}\n':
                    j += 1
                key = self.t[self.i:j].strip()
                self.i = j
            self._skip_ws()
            if self.i < self.n and self.t[self.i] == ':':
                self.i += 1
            obj[str(key)] = self._value()
            self._skip_ws()
            if self.i >= self.n:
                return obj
            if self.t[self.i] == ',':
                self.i += 1
                continue
            if self.t[self.i] == '}':
                self.i += 1
                return obj
            self.i += 1  # 其他语法错误字符：跳过继续

    def _array(self):
        arr = []
        self.i += 1  # [
        while True:
            self._skip_ws()
            if self.i >= self.n:
                return arr
            c = self.t[self.i]
            if c == ']':
                self.i += 1
                return arr
            if c == ',':
                self.i += 1
                continue
            arr.append(self._value())
            self._skip_ws()
            if self.i >= self.n:
                return arr
            if self.t[self.i] == ',':
                self.i += 1
                continue
            if self.t[self.i] == ']':
                self.i += 1
                return arr
            self.i += 1

    def _string(self):
        self.i += 1  # 开引号
        out = []
        esc = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\',
               '/': '/', 'b': '\b', 'f': '\f'}
        while self.i < self.n:
            c = self.t[self.i]
            if c == '\\':
                if self.i + 1 < self.n:
                    nxt = self.t[self.i + 1]
                    if nxt == 'u' and self.i + 6 <= self.n:
                        try:
                            out.append(chr(int(self.t[self.i + 2:self.i + 6], 16)))
                            self.i += 6
                            continue
                        except ValueError:
                            pass
                    out.append(esc.get(nxt, nxt))
                    self.i += 2
                    continue
                self.i += 1  # 末尾孤立反斜杠
                continue
            if c == '"':
                # 宽容关键点：引号后跟结构字符（,}]:）或到结尾才算字符串结束；
                # 否则视为值内部忘记转义的引号，按普通字符保留
                j = self.i + 1
                while j < self.n and self.t[j] in ' \t\r\n':
                    j += 1
                if j >= self.n or self.t[j] in ',}]:':
                    self.i += 1
                    return ''.join(out)
                out.append(c)
                self.i += 1
                continue
            out.append(c)  # 裸控制字符（如换行）原样保留进值里
            self.i += 1
        return ''.join(out)  # 截断：字符串自动闭合

    def _atom(self):
        j = self.i
        while j < self.n and self.t[j] not in ',}]"\n':
            j += 1
        s = self.t[self.i:j].strip()
        self.i = j
        low = s.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        if low in ('null', 'none', ''):
            return None
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                pass
        return s


def _loads_lenient(text: str):
    """严格解析失败时用宽容解析器修复模型 JSON 语法病。"""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return _TolerantJSONParser(text).parse()
    except Exception:
        return None


def _looks_like_json(text: str) -> bool:
    """结构性判断一段文本是否形似 JSON（不匹配任何具体模型输出）。"""
    s = text.lstrip()
    if s[:1] in ('{', '['):
        return True
    # 文本中嵌着"带引号键名+冒号"的 JSON 对象碎片
    return bool(re.search(r'\{\s*"[^"]{1,40}"\s*:', text))


def extract_json(text: str) -> Optional[dict]:
    """从文本中提取第一个 JSON 对象；非对象（数字/字符串/数组等标量）视为提取失败。"""
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    # 宽容修复解析（模型 JSON 常见语法病：未转义引号/裸换行/尾逗号/截断）
    data = _loads_lenient(cleaned)
    if isinstance(data, dict):
        return data
    start_indices = [i for i, char in enumerate(cleaned) if char == '{']
    # 优先匹配最外层的完整 JSON 对象（模型常在 JSON 前后夹杂闲聊文本，
    # 若从最内层匹配会拿到句子对象而非 {"sentences": [...]} 整体）
    for start in start_indices:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            char = cleaned[i]
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(cleaned[start:i+1])
                            if isinstance(data, dict):
                                return data
                        except Exception:
                            break
    return None


class RoleContext:
    """全局配置 + 角色覆盖字段的只读视图。

    角色字段中非空的值优先；否则回退到全局配置。
    """

    def __init__(self, config, role: Optional[dict] = None):
        self.config = config  # ConfigLoader 或任何带 .get() 的对象
        self.role = role or {}

    def get(self, key, default=None):
        role_val = self.role.get(key)
        if role_val is not None and role_val != "":
            return role_val
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    @property
    def character_key(self) -> str:
        return self.role.get("character_key", "") or str(self.config.get("character_key", ""))

    @property
    def character_name(self) -> str:
        return self.role.get("character_name", "") or str(self.config.get("character_name", ""))


# ---------------------------------------------------------------------------
# 消息与提示词构建
# ---------------------------------------------------------------------------

def build_merged_history(history: list, ctx: RoleContext) -> List[dict]:
    """将持久化历史转换为对话消息列表（合并同角色相邻消息，附带说话人）。"""
    n = max(0, int(ctx.get("history_length", 8) or 0))
    history_data = history[-n:] if n > 0 else []
    merged = []
    for msg in history_data:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role not in ["user", "assistant"] or not content:
            continue
        speaker = msg.get("speaker") or msg.get("sender_name") if role == "user" else msg.get("speaker")
        if role == "assistant" and speaker:
            content = f"[{speaker}]: {content}"
        elif role == "user" and msg.get("sender_name"):
            content = f"[用户:{msg.get('sender_name')}] {content}"
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged


def build_system_prompt(ctx: RoleContext, emotions: dict, extra_parts: Optional[List[str]] = None) -> str:
    parts = [
        str(ctx.get("personality_prompt", "") or ""),
        str(ctx.get("json_prompt", "") or ""),
        str(ctx.get("supplement_prompt", "") or ""),
    ]
    emotion_keys = list(emotions.keys()) if emotions else []
    if ctx.get("llm_judge", True) and emotion_keys:
        parts.append(f"【情绪可选列表】{', '.join(emotion_keys)}")
    if ctx.get("enable_time_awareness", False):
        lt = time.localtime()
        try:
            weekday = "周" + "一二三四五六日"[lt.tm_wday % 7]
        except Exception:
            weekday = ""
        parts.append(f"【当前时间】{time.strftime('%Y-%m-%d %H:%M', lt)} {weekday}")
    for part in (extra_parts or []):
        if part:
            parts.append(str(part))
    return "\n".join(p for p in parts if p.strip())


def build_chat_messages(ctx: RoleContext, user_text: str, history: list, emotions: dict,
                        extra_parts: Optional[List[str]] = None,
                        history_extra_user_msg: str = "") -> List[dict]:
    messages = [{"role": "system", "content": build_system_prompt(ctx, emotions, extra_parts)}]
    n = max(0, int(ctx.get("history_length", 8) or 0))
    history_data = history[-n:] if n > 0 else []
    messages.extend(build_merged_history(history, ctx))
    # 重申最近的指令类消息（保留原有行为）
    task_keywords = ["提醒", "记住", "要求", "命令", "叫我", "以后", "别忘"]
    for msg in reversed(history_data):
        if msg.get("role") == "user":
            content = str(msg.get("content", ""))
            if any(kw in content for kw in task_keywords):
                messages.append({"role": "user", "content": f"（重申之前的指令）{content}"})
                break
    if history_extra_user_msg:
        messages.append({"role": "user", "content": history_extra_user_msg})
    messages.append({"role": "user", "content": user_text})
    return messages


# ---------------------------------------------------------------------------
# 底层请求
# ---------------------------------------------------------------------------

def _endpoint_and_payload(ctx: RoleContext, messages: list, stream: bool, tools=None):
    backend = ctx.get("llm_backend", "ollama")
    base_url = str(ctx.get("llm_base_url", "http://127.0.0.1:11434")).rstrip("/")
    model = ctx.get("llm_model_name", "")
    timeout = ctx.get("llm_timeout", 120)
    enable_think = ctx.get("enable_think", False)
    headers = {}
    if backend == "ollama":
        endpoint = f"{base_url}/api/chat"
        payload = {
            "model": model, "messages": messages, "stream": stream,
            "think": enable_think,
            "options": {
                "num_ctx": int(ctx.get("num_ctx", 8192)),
                "temperature": min(1.0, float(ctx.get("temperature", 1.0))),
            },
        }
        if tools:
            payload["tools"] = tools
    else:
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        endpoint = f"{base_url}/chat/completions"
        payload = {"model": model, "messages": messages, "stream": stream,
                   "temperature": min(1.0, float(ctx.get("temperature", 1.0)))}
        api_key = ctx.get("llm_api_key", "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if tools:
            payload["tools"] = tools
    return backend, endpoint, payload, headers, timeout


async def chat_once(ctx: RoleContext, messages: list, tools=None) -> Dict:
    """单次非流式对话。返回 {content, tool_calls, ms}。"""
    backend, endpoint, payload, headers, timeout = _endpoint_and_payload(ctx, messages, False, tools)
    enable_think = bool(payload.get("think", False))
    start = time.time()
    async with httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    ms = (time.time() - start) * 1000
    content = ""
    tool_calls = []
    if backend == "ollama":
        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        if enable_think and msg.get("thinking"):
            print(f"【模型思考】{str(msg.get('thinking'))[:200]}")
        tool_calls = msg.get("tool_calls") or []
    else:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []
    return {"content": content, "tool_calls": tool_calls, "ms": ms, "backend": backend}


async def stream_chat(ctx: RoleContext, messages: list) -> AsyncGenerator[Dict, None]:
    """流式对话。逐块 yield {"delta": str, "tool_calls": [...], "ms_done": float}。"""
    backend, endpoint, payload, headers, timeout = _endpoint_and_payload(ctx, messages, True)
    start = time.time()
    first_token_ms = None
    async with httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False) as client:
        async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            if backend == "ollama":
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    msg = chunk.get("message", {}) or {}
                    if first_token_ms is None and msg.get("content"):
                        first_token_ms = (time.time() - start) * 1000
                    yield {"delta": msg.get("content", "") or "",
                           "tool_calls": msg.get("tool_calls") or [],
                           "first_token_ms": first_token_ms,
                           "ms_done": (time.time() - start) * 1000 if chunk.get("done") else None}
            else:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                    if first_token_ms is None and delta.get("content"):
                        first_token_ms = (time.time() - start) * 1000
                    yield {"delta": delta.get("content", "") or "",
                           "tool_calls": delta.get("tool_calls") or [],
                           "first_token_ms": first_token_ms,
                           "ms_done": None}


def _merge_openai_tool_fragments(frags: list) -> list:
    """将 OpenAI 流式返回的 tool_calls 分片合并为完整对象。"""
    merged = {}
    for frag in frags:
        idx = frag.get("index", 0)
        slot = merged.setdefault(idx, {"id": "", "type": "function",
                                       "function": {"name": "", "arguments": ""}})
        if frag.get("id"):
            slot["id"] = frag["id"]
        fn = frag.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]
    return [merged[k] for k in sorted(merged)]


# ---------------------------------------------------------------------------
# 句子规整与流式解析
# ---------------------------------------------------------------------------

def normalize_single(obj, ctx: RoleContext, emotions: dict, user_text: str) -> Dict:
    """将单个句子对象规整为 {zh, lang, display, emotion}。"""
    s = obj if isinstance(obj, dict) else {"zh": str(obj)}
    text_lang = ctx.get("text_lang", "ja")
    display_lang = ctx.get("display_lang", "zh")
    default_voice = ctx.get("default_voice", "pingjing")
    zh = str(s.get("zh", "")).strip()
    lang = str(s.get(text_lang, "")).strip()
    if not zh:
        # 空台词不复读用户消息（user_text），改用安全台词
        zh = lang or FALLBACK_REPLY
    if not lang:
        lang = zh
    if display_lang == "auto":
        display = lang or zh
    else:
        display = str(s.get(display_lang, "")).strip() or zh
    emo = str(s.get("emotion", "")).strip()
    if not ctx.get("llm_judge", True) or emo not in emotions:
        emo = default_voice
    return {"zh": zh, "lang": lang, "display": display, "emotion": emo}


# 提示词约定「一个句号才算一句话」；模型偶尔把多句写进同一个数组元素，
# 导致多句内容合成进同一段语音。此处按句号/问号切分（不切感叹号/省略号，
# 避免把「哼！xxx。」这类单句拆碎），中日文子句数对齐时才拆，防止错位。
_SENT_BOUNDARY = re.compile(r'(?<=[。？])')


def _split_clauses(text: str) -> List[str]:
    if not text:
        return []
    return [p for p in _SENT_BOUNDARY.split(text) if p and p.strip()]


def split_multi_clause_sentences(sentences: List[Dict]) -> List[Dict]:
    out = []
    for s in sentences:
        zh_parts = _split_clauses(s.get("zh", ""))
        if not zh_parts:
            out.append(s)
            continue
        n = len(zh_parts)
        lang = s.get("lang", "")
        lang_parts = _split_clauses(lang)
        # 当 lang 为空或 lang 无法解析时，直接按 zh 拆分
        if not lang or not lang_parts:
            # 为每个拆分后的部分构造新的 sentence 对象
            for i, zh_part in enumerate(zh_parts):
                out.append({**s, "zh": zh_part, "lang": zh_part, "display": zh_part})
            continue
        if len(lang_parts) != n:
            # 暂时不拆，但可在此处补充逻辑
            out.append(s)
            continue
        display = s.get("display", "")
        disp_parts = _split_clauses(display)
        if display == lang and lang_parts:
            disp_list = lang_parts
        elif len(disp_parts) == n:
            disp_list = disp_parts
        else:
            disp_list = zh_parts
        for i in range(n):
            out.append({"zh": zh_parts[i],
                        "lang": lang_parts[i] if lang and lang_parts else lang,
                        "display": disp_list[i] if i < len(disp_list) else disp_list[-1],
                        "emotion": s.get("emotion", "")})
    return out


# 形似JSON但彻底无法修复时的安全台词（绝不把JSON语法当台词念出来）
FALLBACK_REPLY = "呜……刚才走神了，主人再说一遍好吗？"


def normalize_sentences(content: str, ctx: RoleContext, emotions: dict, user_text: str) -> List[Dict]:
    """将 LLM 最终输出规整为句子列表：[{zh, lang, display, emotion}]。

    兼容多种模型输出形态：
    - 标准包装 {"sentences": [{...}, {...}]}
    - 多个独立 JSON 块（模型不套包装时常见；旧版只取第一块，
      导致"提示词要求至少两个JSON块却只合成一条语音"）
    - 顶层 JSON 数组 [{...}, {...}]
    - JSON 语法病（未转义引号/裸换行/尾逗号/截断）：宽容修复后照常解析
    - 包装块之外又补了独立块、sentences 值为字符串、纯文本兜底
    """
    default_voice = ctx.get("default_voice", "pingjing")
    raw = (content or "").strip()
    objs = extract_json_objects(content)

    sentences = None
    wrapper = next((o for o in objs if isinstance(o, dict)
                    and isinstance(o.get("sentences"), list) and o["sentences"]), None)
    if wrapper is not None:
        # 包装块 + 文本中其他散落的句子块合并（模型有时在包装外又补一块）
        sentences = list(wrapper["sentences"]) + [
            o for o in objs if o is not wrapper and _is_sentence_like(o)]
    else:
        sentence_like = [o for o in objs if _is_sentence_like(o)]
        if sentence_like:
            sentences = sentence_like

    if sentences is not None:
        # 丢弃只有 emotion 等元数据、没有台词内容的句子对象，
        # 否则会被用户消息兜底，把用户刚说的话朗读出来
        sentences = [s for s in sentences if sentence_obj_has_text(s)]
        if not sentences:
            sentences = None

    if sentences is None:
        # 无句子块：单对象兜底（含 "sentences": "文本" 的错误格式）
        first = next((o for o in objs if isinstance(o, dict)), None)
        if first is not None and (first.get("sentences") is not None
                                  or any(k in first for k in _SENTENCE_KEYS)):
            s = first.get("sentences")
            if isinstance(s, str) and s.strip():
                sentences = [{"zh": s}]
            else:
                zh = str(first.get("zh", "") or "").strip()
                if not zh:
                    # 无任何台词内容（如 {"sentences": []}）：不复读用户消息
                    sentences = [{"zh": FALLBACK_REPLY, "lang": FALLBACK_REPLY,
                                  "display": FALLBACK_REPLY, "emotion": default_voice}]
                else:
                    sentences = [{"zh": zh, "emotion": first.get("emotion", default_voice)}]
        else:
            # 模型输出了纯文本或 JSON 标量（如裸数字"72"），按纯文本整句兜底
            if not raw:
                raw = "出错了。"
            elif _looks_like_json(raw):
                # 形似JSON但已无法修复：绝不能把JSON语法当台词念出来
                # （否则会出现"回复中含JSON块"的事故），改用安全台词
                raw = FALLBACK_REPLY
            sentences = [{"zh": raw, "lang": raw, "display": raw, "emotion": default_voice}]
    return split_multi_clause_sentences(
        [normalize_single(s, ctx, emotions, user_text) for s in sentences])


class SentenceStreamParser:
    """从增量文本中解析完整句子对象，支持两种模型输出形态：

    - 标准包装：{"sentences": [{...}, {...}]}，逐个产出数组内对象；
    - 裸块：模型把每句话输出成独立 JSON 对象（无 sentences 包装），
      逐块产出（旧版不认这种形态，只能等流结束兜底，且旧兜底只取
      第一块，导致"至少两个JSON块"只合成一条语音）。
    """

    _SENT_KEY = re.compile(r'"sentences"\s*:\s*\[')

    def __init__(self):
        self.buffer = ""
        self.scan_pos = 0
        self.in_array = False
        self.depth = 0
        self.in_string = False
        self.escape = False
        self.obj_start = None
        self.array_closed = False
        self.yielded = 0

    def feed(self, chunk: str) -> List[Dict]:
        if not chunk:
            return []
        self.buffer += chunk
        return self._scan()

    def _emit(self, obj) -> List[Dict]:
        """解析出的顶层对象 → 待产出的句子对象列表（避免重复产出）。"""
        if isinstance(obj, dict) and isinstance(obj.get("sentences"), list):
            # 整个包装对象被当作一块解析出来（如 "sentences": [ 之前有闲聊
            # 文本导致数组模式误触发，或流结束时才凑齐包装）：产出内部块。
            if self.yielded == 0:
                inner = [s for s in obj["sentences"] if isinstance(s, dict)]
                self.yielded += len(inner)
                return inner
            return []  # 内部块早已逐个产出
        if _is_sentence_like(obj):
            self.yielded += 1
            return [obj]
        return []  # 与句子无关的 JSON 对象（工具回显等），跳过

    def _scan(self) -> List[Dict]:
        buf = self.buffer
        if self.in_array:
            return self._scan_array(buf)
        # 数组模式未确认：先看 "sentences": [ 是否出现（回看24字符防关键字被分块截断）
        m = self._SENT_KEY.search(buf, max(0, self.scan_pos - 24))
        if m and (self.obj_start is None or m.start() >= self.obj_start):
            # 切入数组模式（丢弃包装对象自身的扫描状态）
            self.in_array = True
            self.scan_pos = m.end()
            self.depth = 0
            self.in_string = False
            self.escape = False
            self.obj_start = None
            return self._scan_array(buf)
        # 顶层裸块扫描（同时兼容顶层数组）
        out = []
        i = self.scan_pos
        n = len(buf)
        while i < n:
            ch = buf[i]
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif ch == '\\':
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
            elif ch == '"':
                self.in_string = True
            elif ch in '{[':
                if self.depth == 0:
                    self.obj_start = i
                self.depth += 1
            elif ch in '}]':
                self.depth = max(0, self.depth - 1)
                if self.depth == 0 and self.obj_start is not None:
                    obj = _loads_lenient(buf[self.obj_start:i + 1])
                    if obj is not None:
                        if isinstance(obj, list):
                            for s in obj:
                                out.extend(self._emit(s))
                        else:
                            out.extend(self._emit(obj))
                    self.obj_start = None
            i += 1
        self.scan_pos = i
        return out

    def _scan_array(self, buf: str) -> List[Dict]:
        out = []
        i = self.scan_pos
        n = len(buf)
        while i < n:
            ch = buf[i]
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif ch == '\\':
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
            elif ch == '"':
                self.in_string = True
            elif ch == '{':
                if self.depth == 0:
                    self.obj_start = i
                self.depth += 1
            elif ch == '}':
                self.depth = max(0, self.depth - 1)
                if self.depth == 0 and self.obj_start is not None:
                    obj = _loads_lenient(buf[self.obj_start:i + 1])
                    if obj is not None:
                        out.extend(self._emit(obj))
                    self.obj_start = None
            elif ch == ']' and self.depth == 0:
                self.array_closed = True
                i += 1
                break
            i += 1
        self.scan_pos = i
        return out

    def finish(self, ctx: RoleContext, user_text: str, emotions: dict) -> List[Dict]:
        """流结束后兜底：补齐尚未产出的句子。

        - 一句都没产出：整段内容走 normalize_sentences（含宽容修复/防线）；
        - 已产出过句子：尝试从末尾未闭合的截断对象中抢救最后一句
          （模型输出被掐断时，句子内容往往已完整，只是缺收尾括号）。
        """
        if self.yielded == 0:
            return normalize_sentences(self.buffer, ctx, emotions, user_text)
        if self.obj_start is None:
            return []
        obj = _loads_lenient(self.buffer[self.obj_start:])
        cand = []
        if isinstance(obj, dict):
            if isinstance(obj.get("sentences"), list):
                cand = [s for s in obj["sentences"] if isinstance(s, dict)]
            elif _is_sentence_like(obj):
                cand = [obj]
        # 截断对象可能拿到半截空台词，过滤掉没有实际内容的
        cand = [s for s in cand if str(s.get("zh", "")).strip()]
        if not cand:
            return []
        return split_multi_clause_sentences(
            [normalize_single(s, ctx, emotions, user_text) for s in cand])


# ---------------------------------------------------------------------------
# 工具调用（Function Calling）循环
# ---------------------------------------------------------------------------

async def chat_with_tools(ctx: RoleContext, messages: list, tool_registry,
                          stats=None, user_id: str = "") -> Dict:
    """带工具调用的完整对话循环，返回最终 {content, tool_trace, ms}。"""
    tools_schema = tool_registry.get_schema()
    trace = []
    total_ms = 0.0
    max_iter = max(1, int(ctx.get("tools_max_iterations", 3)))
    work = list(messages)
    final_content = ""
    for _ in range(max_iter):
        result = await chat_once(ctx, work, tools=tools_schema)
        total_ms += result["ms"]
        if stats:
            stats.record_llm(result["ms"])
        calls = result.get("tool_calls") or []
        if not calls:
            final_content = result["content"]
            break
        # 记录 assistant 的工具调用消息
        norm_calls = []
        for call in calls:
            fn = call.get("function", {}) or {}
            norm_calls.append({
                "id": call.get("id", ""),
                "name": fn.get("name", call.get("name", "")),
                "arguments": fn.get("arguments", call.get("arguments", "{}")),
            })
        work.append({"role": "assistant", "content": result["content"] or "",
                     "tool_calls": norm_calls})
        for call in norm_calls:
            ok, output = await tool_registry.execute(call["name"], call["arguments"], user_id)
            trace.append({"name": call["name"], "arguments": call["arguments"],
                          "ok": ok, "output": output})
            if result["backend"] == "ollama":
                work.append({"role": "tool", "content": str(output)})
            else:
                work.append({"role": "tool", "tool_call_id": call.get("id") or call["name"],
                             "content": str(output)})
        final_content = result["content"]
    else:
        # 达到最大迭代次数，做一次无工具的最终请求
        result = await chat_once(ctx, work)
        total_ms += result["ms"]
        if stats:
            stats.record_llm(result["ms"])
        final_content = result["content"]
    return {"content": final_content, "tool_trace": trace, "ms": total_ms}


# ---------------------------------------------------------------------------
# 面向主流程的完整入口
# ---------------------------------------------------------------------------

async def generate_text_reply(ctx: RoleContext, system_prompt: str, user_prompt: str,
                              max_tokens: int = 512) -> str:
    """简单的纯文本生成（用于开场白、摘要、画像提取等辅助任务）。"""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    try:
        result = await chat_once(ctx, messages)
        return (result.get("content") or "").strip()
    except Exception as e:
        print(f"文本生成失败: {type(e).__name__}: {e}")
        return ""


async def generate_json_reply(ctx: RoleContext, system_prompt: str, user_prompt: str,
                              max_tokens: int = 512) -> Optional[dict]:
    text = await generate_text_reply(ctx, system_prompt, user_prompt, max_tokens)
    return extract_json(text)


async def get_image_reply(ctx: RoleContext, user_text: str, history: list,
                          emotions: dict, image_urls: list,
                          extra_parts=None, stats=None) -> Optional[Dict]:
    """识图回复：读取本地或下载网络图片，交给识图模型生成句子。"""
    try:
        personality = ctx.get("personality_prompt", "")
        json_prompt = ctx.get("json_prompt", "")
        supplement = ctx.get("supplement_prompt", "")
        prompt_text = (
            f"用户发来了一张图片，请仔细观察图片内容，结合你的角色人设：{personality}；{json_prompt}；{supplement}，"
            f"根据图片内容回复（可以是吐槽、评价、撒娇等）。\n当前对话历史：{json.dumps(history[-10:], ensure_ascii=False)}\n"
            f"用户附加文字：{user_text}"
        )
        images_for_payload = []  # [(mime, base64)]
        seen_sources = set()
        for img_source in image_urls:
            src = str(img_source)
            if src in seen_sources:
                continue
            seen_sources.add(src)
            data = None
            try:
                if src.lower().startswith("file://"):
                    local = file_uri_to_path(src)
                    if os_path_exists(local):
                        with open(local, 'rb') as f:
                            data = f.read()
                    else:
                        print(f"file:// 图片不存在: {local}")
                elif os_path_exists(src):
                    with open(src, 'rb') as f:
                        data = f.read()
                elif src.startswith(("http://", "https://")):
                    data = await download_image(src)
                else:
                    print(f"未知图片路径格式: {src}")
            except Exception as e:
                print(f"获取图片失败 {src[:120]}: {e}")
            if not data:
                continue
            mime = sniff_image_mime(data)
            if not mime:
                # 图床防盗链/过期链接常返回 HTML 错误页，垃圾数据会让识图模型直接 400
                print(f"跳过非图片内容（链接可能已过期或被拦截）: {src[:120]}")
                continue
            data, mime = normalize_image_data(data, mime)
            if not data:
                print(f"图片格式转换失败，已跳过: {src[:120]}")
                continue
            images_for_payload.append((mime, base64_b64(data)))
        if not images_for_payload:
            print("没有有效的图片数据，使用默认回复")
            default_text = "啊嘞，看不清这张图呢。"
            return {"sentences": normalize_sentences(default_text, ctx, emotions, user_text), "ms": 0, "tool_trace": []}
        model = ctx.get("image_caption_model_name", "")
        if not model:
            print("未配置识图模型名称，无法处理图片")
            return None
        backend = ctx.get("llm_backend", "ollama")
        base_url = str(ctx.get("llm_base_url", "http://127.0.0.1:11434")).rstrip("/")
        timeout = ctx.get("image_caption_timeout", 90)
        system_content = build_system_prompt(ctx, emotions, extra_parts)
        start = time.time()
        if backend == "ollama":
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt_text,
                     "images": [b64 for _mime, b64 in images_for_payload]}
                ],
                "stream": False, "think": False,
                "options": {"temperature": float(ctx.get("temperature", 0.7)), "num_predict": 512}
            }
            endpoint = f"{base_url}/api/chat"
            headers = {}
        else:
            content_parts = []
            for mime, img_b64 in images_for_payload:
                content_parts.append({"type": "image_url",
                                      "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
            content_parts.append({"type": "text", "text": prompt_text})
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": content_parts}
                ],
                "stream": False,
                "temperature": float(ctx.get("temperature", 0.7)),
                "max_tokens": 512
            }
            endpoint = f"{base_url}/chat/completions"
            api_key = ctx.get("llm_api_key", "")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        ms = (time.time() - start) * 1000
        if backend == "ollama":
            content = data.get("message", {}).get("content", "")
        else:
            content = data["choices"][0]["message"]["content"]
        if stats:
            stats.record_llm(ms)
        sentences = normalize_sentences(content, ctx, emotions, user_text or "（图片）")
        return {"sentences": sentences, "ms": ms, "tool_trace": []}
    except Exception as e:
        print(f"识图模型处理失败: {e}")
        return None


# 避免在模块顶部重复导入 os/base64
def os_path_exists(path) -> bool:
    import os
    return os.path.exists(path)


def base64_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode('utf-8')


# 常见图片文件头。QQ 图床/防盗链经常返回 HTML 错误页或空响应，
# 垃圾数据直接喂给识图模型会触发 400（Failed to load image）。
_IMAGE_MAGICS = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def sniff_image_mime(data: bytes) -> str:
    """按文件头识别图片格式；非图片内容返回空串。"""
    for magic, mime in _IMAGE_MAGICS:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def file_uri_to_path(uri: str) -> str:
    """file:///C:/a/b.png → C:/a/b.png；非 file:// 原样返回。"""
    from urllib.parse import unquote, urlparse
    u = urlparse(str(uri))
    if u.scheme.lower() != "file":
        return str(uri)
    path = unquote(u.path)
    if re.match(r"^/[A-Za-z]:[\\/]", path):
        path = path[1:]  # Windows 盘符前的斜杠
    if u.netloc and u.netloc != "localhost":
        path = f"//{u.netloc}{path}"  # UNC 路径
    return path


async def download_image(url: str) -> bytes:
    """下载网络图片。QQ 图床会 302 跳 CDN，必须跟随重定向；带 Referer 应对防盗链。"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    if "://" in url:
        host = url.split("/", 3)[2]
        if any(d in host for d in ("qq.com", "qpic.cn", "gtimg.cn")):
            headers["Referer"] = "https://gchat.qpic.cn/"
    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True,
                                 proxy=None, trust_env=False) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def normalize_image_data(data: bytes, mime: str):
    """WebP 转成 JPEG/PNG（多数推理后端不支持），超大图等比缩小到 2048 内。

    Pillow 未安装或转码失败时：WebP 视为不可用（返回 (None, None)），
    其余格式原样返回，交由推理后端自行处理。
    """
    oversize = False
    try:
        import io
        from PIL import Image as PILImage
        with PILImage.open(io.BytesIO(data)) as img:
            oversize = max(img.size) > 2048
            if mime != "image/webp" and not oversize:
                return data, mime
            if oversize:
                img.thumbnail((2048, 2048))
            buf = io.BytesIO()
            if img.mode in ("RGBA", "LA", "P"):
                img.save(buf, format="PNG")
                return buf.getvalue(), "image/png"
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        return (None, None) if mime == "image/webp" else (data, mime)
