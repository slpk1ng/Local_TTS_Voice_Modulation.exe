import asyncio
import io
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import httpx
try:
    import webview
    HAS_WEBVIEW = True
except ImportError:
    HAS_WEBVIEW = False
    print("警告：未安装 pywebview，将使用浏览器访问。可运行 pip install pywebview 启用。")

try:
    from napcat import NapCatClient, PrivateMessageEvent, GroupMessageEvent, Text, Record, Image, At
except ImportError:
    print("错误：未安装 napcat-sdk，请先运行 pip install napcat-sdk")
    raise
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("警告：未安装 aiohttp，WebUI 管理功能将不可用。可运行 pip install aiohttp 启用。")

# ---------------- 功能模块 ----------------
from modules.database import DatabaseManager
from modules.scheduler import get_scheduler, SchedulerManager
from modules.stats import StatsManager
from modules.stickers import StickerManager
from modules.tools import ToolRegistry
from modules.profiles import UserProfileManager
from modules.rag import RAGManager, extract_text_from_file
from modules.todo_manager import TodoManager
from modules.jobs import ScheduledJobManager, generate_proactive_text
from modules.events import EventManager
from modules.sender import MessageSender
from modules.llm_helpers import (RoleContext, build_chat_messages, chat_once,
                                chat_with_tools, normalize_sentences,
                                normalize_single, sentence_obj_has_text,
                                split_multi_clause_sentences,
                                stream_chat,
                                SentenceStreamParser, get_image_reply)
from modules.tts import synthesize_sentence, get_audio_duration, resolve_tts_path
from modules.tts_service import (process_manager, ensure_tts_service,
                                 auto_start_and_switch_tts)

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return base_path / relative_path

global_log_buffer = []
log_lock = threading.Lock()

class StdoutRedirector:
    def __init__(self, original_stream):
        # 兼容 console=False 时 sys.stdout 为 None 的情况
        if original_stream is None:
            try:
                original_stream = open(os.devnull, 'w', encoding='utf-8')
            except Exception:
                original_stream = None
        self.original_stream = original_stream
        self._last_saved_config = None

    def write(self, message):
        if not message:
            return
        with log_lock:
            if self.original_stream is not None:
                try:
                    self.original_stream.write(message)
                    self.original_stream.flush()
                except Exception:
                    pass  # 无控制台时忽略写入错误
            for line in message.splitlines(True):
                global_log_buffer.append(line.rstrip('\n'))
                if len(global_log_buffer) > 500:
                    global_log_buffer.pop(0)

    def flush(self):
        if self.original_stream is not None:
            try:
                self.original_stream.flush()
            except Exception:
                pass


class ConfigLoader:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_or_init()
        # 多角色配置解析
        self.active_character = self.config.get("active_character", self.config.get("character_key", "murasame"))
        self.roles = self._parse_roles()

    def _parse_roles(self) -> dict:
        """解析多角色配置，将旧版单角色配置迁移为角色列表"""
        # 每次解析都从配置重读活跃角色，保证 WebUI 切换后立即生效
        self.active_character = str(self.config.get("active_character", "") or
                                    self.config.get("character_key", "") or "murasame")
        roles = {}
        if "roles" in self.config and isinstance(self.config["roles"], list):
            roles_config = self.config["roles"]
        else:
            roles_config = [{
                "character_name": self.config.get("character_name", "丛雨"),
                "character_key": self.config.get("character_key", "murasame"),
                "personality_prompt": self.config.get("personality_prompt", ""),
                "json_prompt": self.config.get("json_prompt", ""),
                "supplement_prompt": self.config.get("supplement_prompt", ""),
                "default_voice": self.config.get("default_voice", "pingjing"),
                "ref_audio_root": self.config.get("ref_audio_root", ""),
                "text_lang": self.config.get("text_lang", "ja")
            }]

        for role_cfg in roles_config:
            key = role_cfg.get("character_key", "")
            if not key:
                continue
            roles[key] = {
                "character_name": role_cfg.get("character_name", "丛雨"),
                "character_key": key,
                "personality_prompt": role_cfg.get("personality_prompt", self.config.get("personality_prompt", "")),
                "json_prompt": role_cfg.get("json_prompt", self.config.get("json_prompt", "")),
                "supplement_prompt": role_cfg.get("supplement_prompt", self.config.get("supplement_prompt", "")),
                "default_voice": role_cfg.get("default_voice", "pingjing"),
                "ref_audio_root": role_cfg.get("ref_audio_root", ""),
                "text_lang": role_cfg.get("text_lang", "ja")
            }
        if self.active_character not in roles:
            self.active_character = list(roles.keys())[0] if roles else "murasame"
        return roles

    def _load_or_init(self) -> dict:
        def can_interact():
            try:
                return sys.stdin is not None and sys.stdin.isatty()
            except Exception:
                return False
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if not Path(config.get("ref_audio_root", "")).exists():
                    print(f"⚠️ 参考音频目录无效：{config.get('ref_audio_root')}")
                    if can_interact():
                        return self._interactive_init(config)
                    else:
                        return self._auto_save_default(config)
                return config
            except Exception as e:
                print(f"⚠️ 读取配置文件失败：{e}，将自动重新生成默认配置。")
                if can_interact():
                    return self._interactive_init(self.default_config())
                else:
                    return self._auto_save_default(self.default_config())
        else:
            print("未找到配置文件，正在自动生成默认配置...")
            if can_interact():
                return self._interactive_init(self.default_config())
            else:
                return self._auto_save_default(self.default_config())

    def _auto_save_default(self, base_config: dict) -> dict:
        merged_config = {**self.default_config(), **base_config}
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(merged_config, f, ensure_ascii=False, indent=2)
            print(f"已自动生成配置文件：{self.config_path.resolve()}")
        except Exception as e:
            print(f"自动保存配置失败（请手动创建 config.json）：{e}")
        return merged_config

    def _interactive_init(self, base_config: dict) -> dict:
        print("\n--- 配置向导 ---")
        print("按回车使用默认值，或输入自定义值。")
        print("\n[1] NapCat 连接配置")
        ws_url = input(f"WebSocket 地址 (默认 {base_config.get('napcat_ws_url')}): ").strip()
        if ws_url:
            base_config["napcat_ws_url"] = ws_url
        token = input(f"Token (默认 {base_config.get('napcat_token')}): ").strip()
        if token:
            base_config["napcat_token"] = token

        print("\n[2] 本地大模型 (LLM) 配置")
        base_url = input(f"API 地址 (默认 {base_config.get('llm_base_url')}): ").strip()
        if base_url:
            base_config["llm_base_url"] = base_url
        model = input(f"模型名称 (默认 {base_config.get('llm_model_name')}): ").strip()
        if model:
            base_config["llm_model_name"] = model

        print("\n[3] 角色配置")
        character_name = input(f"角色名称 (默认 {base_config.get('character_name')}): ").strip()
        if character_name:
            base_config["character_name"] = character_name
        character_key = input(f"角色标识符 (默认 {base_config.get('character_key')}): ").strip()
        if character_key:
            base_config["character_key"] = character_key

        if "roles" not in base_config or not base_config["roles"]:
            base_config["roles"] = [{
                "character_name": base_config.get("character_name", "丛雨"),
                "character_key": base_config.get("character_key", "murasame"),
                "personality_prompt": base_config.get("personality_prompt", ""),
                "json_prompt": base_config.get("json_prompt", ""),
                "supplement_prompt": base_config.get("supplement_prompt", ""),
                "default_voice": base_config.get("default_voice", "pingjing"),
                "ref_audio_root": base_config.get("ref_audio_root", ""),
                "text_lang": base_config.get("text_lang", "ja")
            }]
        base_config["active_character"] = base_config.get("active_character", base_config.get("character_key", "murasame"))

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(base_config, f, ensure_ascii=False, indent=2)
            print(f"\n 配置已保存到：{self.config_path.resolve()}")
        except Exception as e:
            print(f"保存配置失败：{e}")
            input("按回车退出...")
            raise SystemExit(1)
        return base_config

    @staticmethod
    def default_config() -> dict:
        from modules.todo_manager import DEFAULT_TODO_PATTERNS
        # 完整包含所有可配置字段（含各功能模块的开关与参数，全部可在 WebUI 修改）
        return {
            "hide_gsv_options": False,
            "llm_model_name": "",
            "image_caption_model_name": "",
            "llm_base_url": "http://127.0.0.1:11434",
            "llm_backend": "ollama",
            "llm_api_key": "",
            "num_ctx": 8192,
            "history_length": 8,
            "enable_think": False,
            "llm_timeout": 120,
            "image_caption_timeout": 90,
            "client_base_url": "http://127.0.0.1:9880",
            "model_dir": "",
            "ref_audio_root": "",
            "timeout_seconds": 120,
            "prompt_text": "ふむ、おぬしが我輩のご主人か?",
            "prompt_lang": "ja",
            "text_lang": "ja",
            "top_k": 20,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut1",
            "batch_size": 1,
            "batch_threshold": 1,
            "split_bucket": True,
            "speed_factor": 1.0,
            "fragment_interval": 0.5,
            "streaming_mode": False,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "media_type": "wav",
            "character_name": "丛雨",
            "character_key": "murasame",
            "personality_prompt": "【角色设定】你是丛雨，一位从神刀中获得人类生活的少女。你外表年幼，实际活了五百多年；性格天真活泼、略带古风和孩子气，内心温柔而坚强。你把用户视作重要的主人。中文对话中自称“本座”，称用户为“主人”；日语对话中自称“吾輩”，称用户为“ご主人”。你喜欢甜食、撒娇和被摸头，害怕幽灵，也不喜欢被叫作幼刀、钝刀或搓衣板。你偶尔嘴硬、吃醋或开小玩笑，但不会刻薄、控制或道德绑架主人。性格方面，丛雨表面元气开朗、充满活力，言行大多孩子气，爱撒娇，被主人摸头时会瞬间羞涩，她内在像个成年女性，常讲黄段子，把“情趣”等词挂在嘴边，还带点傲娇和爱吃醋。保持温柔、纯真、治愈并带一点幽默的语气。",
            "json_prompt": "【输出格式】你最终必须只输出一个JSON对象，格式为：{\"sentences\": [JSON块1, JSON块2, ...]}。其中：{\"zh\": \"这里是你生成的中文台词\", \"ja\": \"这里是你生成的日语台词\", \"emotion\": \"这里是你判断的情绪\"}，……（依此类推）。sentences数组中必须放至少两个JSON块（也就是至少两句话），绝对不允许只放一个JSON块，最多放五个；每个JSON块只写一句完整的话（一个句号或问号才算一句话）。【最终输出规则】最终输出必须严格只包含这一个JSON对象（内部含多个JSON块），绝对禁止输出任何思考过程、解释、非JSON文本或Markdown代码块。所有的推理和思考都只能在内部进行，最终回复只能是JSON格式。",
            "supplement_prompt": "回答自然、简短，通常两到五句话(一个句号才算一句话)；不要重复最近说过的话，不要加入动作、旁白或括号舞台说明；生成的回复要符合当前对话，不能出现主谓宾不分，乱序的情况。【情绪判断规则】请仔细阅读最近对话历史，结合你（角色）的性格特点来判断情绪！如果主人对你亲昵（如摸头、夸奖），即使你嘴上说“我才没有”，情绪也应该是害羞或高兴；如果主人故意逗你、骂你或惹你生气，情绪应该是生气或着急；如果只是平淡陈述，使用平静。【翻译一致性要求】必须表达完全相同的含义和语气，绝对不能出现含义相反或意思不匹配的翻译！【情绪连贯性强制规则】如果用户明确地侮辱、挑衅或激怒你（例如叫你“幼刀、搓衣板、飞机场”），你的情绪必须保持连贯。即：整句话所有分句的情绪必须都是“生气”或“着急”，绝对不能把后半句的“命令/威胁”改成“害羞”或“高兴”！除非你明确使用了“但是”、“不过”等转折词，否则不要轻易切换成其他情绪。【情绪匹配规则】情绪文件夹可能是拼音（如 gaoxing），也可能是英文（如 happy）。你必须严格只输出我在【情绪可选列表】中提供的单词，绝对不能输出中文汉字或拼音简写！",
            "max_voice_cache": 20,
            "isolated_session": False,
            "separate_send": False,
            "send_voice_separately": False,
            "text_separate": False,
            "dynamic_sleep": True,
            "only_private": False,
            "auto_start_tts": True,
            "tts_start_script": "",
            "device": "cuda",
            "llm_judge": True,
            "display_lang": "zh",
            "default_voice": "pingjing",
            "voice_transition": True,
            "breathing_gap_ms": 100,
            "crossfade_ms": 300,
            "llm_emotion_intensity": True,
            "intensity_to_temperature": 0.3,
            "intensity_to_top_k": 10.0,
            "enable_time_awareness": False,
            "summary_enabled": True,
            "summary_threshold": 20,
            "summary_max_history": 5,
            "active_character": "murasame",
            "roles": [
                {
                    "character_name": "丛雨",
                    "character_key": "murasame",
                    "personality_prompt": "【角色设定】你是丛雨，一位从神刀中获得人类生活的少女。你外表年幼，实际活了五百多年；性格天真活泼、略带古风和孩子气，内心温柔而坚强。你把用户视作重要的主人。中文对话中自称“本座”，称用户为“主人”；日语对话中自称“吾輩”，称用户为“ご主人”。你喜欢甜食、撒娇和被摸头，害怕幽灵，也不喜欢被叫作幼刀、钝刀或搓衣板。你偶尔嘴硬、吃醋或开小玩笑，但不会刻薄、控制或道德绑架主人。性格方面，丛雨表面元气开朗、充满活力，言行大多孩子气，爱撒娇，被主人摸头时会瞬间羞涩，她内在像个成年女性，常讲黄段子，把“情趣”等词挂在嘴边，还带点傲娇和爱吃醋。保持温柔、纯真、治愈并带一点幽默的语气。",
                    "json_prompt": "【输出格式】你最终必须只输出一个JSON对象，格式为：{\"sentences\": [JSON块1, JSON块2, ...]}。其中：{\"zh\": \"这里是你生成的中文台词\", \"ja\": \"这里是你生成的日语台词\", \"emotion\": \"这里是你判断的情绪\"}，……（依此类推）。sentences数组中必须放至少两个JSON块（也就是至少两句话），绝对不允许只放一个JSON块，最多放五个；每个JSON块只写一句完整的话（一个句号或问号才算一句话）。【最终输出规则】最终输出必须严格只包含这一个JSON对象（内部含多个JSON块），绝对禁止输出任何思考过程、解释、非JSON文本或Markdown代码块。所有的推理和思考都只能在内部进行，最终回复只能是JSON格式。",
                    "supplement_prompt": "回答自然、简短，通常两到五句话(一个句号才算一句话)；不要重复最近说过的话，不要加入动作、旁白或括号舞台说明；生成的回复要符合当前对话，不能出现主谓宾不分，乱序的情况。【情绪判断规则】请仔细阅读最近对话历史，结合你（角色）的性格特点来判断情绪！如果主人对你亲昵（如摸头、夸奖），即使你嘴上说“我才没有”，情绪也应该是害羞或高兴；如果主人故意逗你、骂你或惹你生气，情绪应该是生气或着急；如果只是平淡陈述，使用平静。【翻译一致性要求】必须表达完全相同的含义和语气，绝对不能出现含义相反或意思不匹配的翻译！【情绪连贯性强制规则】如果用户明确地侮辱、挑衅或激怒你（例如叫你“幼刀、搓衣板、飞机场”），你的情绪必须保持连贯。即：整句话所有分句的情绪必须都是“生气”或“着急”，绝对不能把后半句的“命令/威胁”改成“害羞”或“高兴”！除非你明确使用了“但是”、“不过”等转折词，否则不要轻易切换成其他情绪。【情绪匹配规则】情绪文件夹可能是拼音（如 gaoxing），也可能是英文（如 happy）。你必须严格只输出我在【情绪可选列表】中提供的单词，绝对不能输出中文汉字或拼音简写！",
                    "default_voice": "pingjing",
                    "ref_audio_root": "",
                    "text_lang": "ja"
                }
            ],
            # ============ 以下为各功能模块的开关与参数（WebUI 可视化配置） ============
            # 回复方式
            "tts_reply_enabled": True,
            "streaming_enabled": False,
            # 定时任务与主动消息
            "scheduler_enabled": True,
            "proactive_enabled": False,
            "proactive_idle_minutes": 30,
            "proactive_check_seconds": 300,
            "proactive_max_per_day": 2,
            "proactive_quiet_start": "23:00",
            "proactive_quiet_end": "08:00",
            "proactive_prompt": "主人已经有一段时间没有和你说话了，主动找个自然的话题关心一下主人吧。",
            "proactive_voice": False,
            "proactive_sticker": False,
            "greeting_events_enabled": True,
            "greeting_check_time": "08:00",
            "birthday_greeting_enabled": True,
            "birthday_greet_template": "今天是 {nickname} 的生日！本座在此郑重宣布：生日快乐！要一直一直开心下去哦！",
            "birthday_greet_mode": "template",
            "birthday_greet_voice": False,
            # 待办提醒
            "todo_enabled": False,
            "todo_extract_mode": "regex",
            "todo_voice": False,
            "todo_remind_template": "⏰ 提醒时间到啦：{content}",
            "todo_keywords": "提醒\n待办\n别忘了\n记得\n叫我",
            "todo_regex_patterns": "\n".join(DEFAULT_TODO_PATTERNS),  # 与 TodoManager 共享同一组默认正则
            "todo_extract_prompt": "你是待办提取助手。判断用户消息是否包含一个明确的提醒/待办事项。如果有，输出JSON：{\"has_todo\": true, \"content\": \"要提醒的事项\", \"delay_minutes\": 相对当前时间的分钟数(整数,无法确定则为0), \"time\": \"HH:MM 格式的绝对时间(可选)\"}；如果没有明确的提醒事项，输出 {\"has_todo\": false}。只输出JSON。",
            # 表情包
            "stickers_enabled": False,
            "stickers_dir": "",
            "sticker_probability": 1.0,
            "sticker_max_per_reply": 1,
            "sticker_every_sentence": False,
            # 多角色对话
            "multi_role_enabled": False,
            "multi_role_max_replies": 2,
            "multi_role_auto_rounds": 0,
            "multi_role_max_total": 6,
            # 工具调用
            "tools_enabled": False,
            "tools_allow_commands": False,
            "tools_max_iterations": 3,
            # RAG 知识库
            "rag_enabled": False,
            "rag_embedding_model": "nomic-embed-text",
            "rag_chunk_size": 500,
            "rag_chunk_overlap": 80,
            "rag_top_k": 3,
            "rag_min_similarity": 0.35,
            "rag_max_context_chars": 1000,
            "rag_context_template": "【参考资料】以下是知识库中可能相关的内容，回答时可以参考（不确定时以你的角色身份自然回答）：\n{refs}",
            # 用户画像
            "profiles_enabled": False,
            "profiles_auto_extract": False,
            "profiles_max_chars": 300,
            "profiles_extract_prompt": "你是信息提取助手。请从用户与角色的对话中提取关于【用户】的长期个人信息（不是角色设定）。只提取明确、可靠的信息，没有则返回空对象。只输出JSON，格式：{\"nickname\": \"称呼/昵称(可选)\", \"birthday\": \"MM-DD(可选)\", \"likes\": [\"喜好\"], \"dislikes\": [\"厌恶\"], \"notes\": [\"重要事项\"]}",
            "profiles_inject_template": "【用户画像】关于当前用户的已知信息：{profile}",
            # 动态上下文
            "dynamic_context_enabled": False,
            "topic_summary_every_n": 10,
            "topic_summary_prompt": "请用一句话概括以下对话当前正在讨论的话题，直接输出话题本身：",
            "summary_prompt": "请把以下对话历史浓缩成一段简短的背景摘要（保留关键事实、约定和用户信息，用第三人称叙述），直接输出摘要内容：",
            # WebUI
            "webui_enabled": True,
            "webui_host": "127.0.0.1",
            "webui_port": 11500
        }

    def get(self, key: str, default=None):
        if "." in key:
            parts = key.split(".")
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        return self.config.get(key, default)


def _is_reserved(path_obj: Path) -> bool:
    if hasattr(os.path, "isreserved"):
        return os.path.isreserved(str(path_obj))
    return path_obj.is_reserved()


class EmotionManager:
    def __init__(self, config):
        self.config = config
        self.ref_audio_root = resolve_tts_path(config.get("ref_audio_root", "C:/tts"))
        self.default_voice = config.get("default_voice", "pingjing")
        self.emotions = {}
        self._discover_emotions()
        self._apply_manual_emotions()

    def _discover_emotions(self):
        base_folder = Path(self.ref_audio_root)
        if not base_folder.exists():
            print(f"警告：参考音频根目录不存在：{self.ref_audio_root}")
            return
        if _is_reserved(base_folder) or base_folder.name in {"WpSystem", "System Volume Information", "$Recycle.Bin", "Recovery", "PerfLogs", "Config.Msi"}:
            print(f"错误：{self.ref_audio_root} 是系统保护目录，无法访问！")
            return
        ignore_dirs = {"WpSystem", "System Volume Information", "$Recycle.Bin", "Recovery", "PerfLogs", "Config.Msi"}
        try:
            for folder in base_folder.iterdir():
                if folder.name in ignore_dirs or folder.name.startswith("$"):
                    continue
                try:
                    if not folder.is_dir():
                        continue
                except PermissionError:
                    continue
                emotion_name = folder.name
                ref_audio = None
                prompt_text = ""
                for ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                    try:
                        candidate = folder / f"ref{ext}"
                        if candidate.exists():
                            ref_audio = candidate
                            break
                    except (PermissionError, OSError):
                        continue
                if not ref_audio:
                    try:
                        candidate = folder / f"{emotion_name}.mp3"
                        if not candidate.exists():
                            candidate = folder / f"{emotion_name}.wav"
                        if candidate.exists():
                            ref_audio = candidate
                    except (PermissionError, OSError):
                        continue
                if ref_audio:
                    asr_path = folder / "asr.txt"
                    if asr_path.exists():
                        try:
                            prompt_text = asr_path.read_text(encoding='utf-8', errors='ignore').strip()
                        except Exception:
                            prompt_text = ""
                    if not prompt_text:
                        prompt_text = self.config.get("prompt_text", "ふむ、おぬしが我輩のご主人か?")
                    self.emotions[emotion_name] = {
                        "ref_path": str(ref_audio).replace("\\", "/"),
                        "prompt_text": prompt_text
                    }
        except Exception as e:
            print(f"扫描目录异常：{e}")
        if self.emotions:
            print(f"成功扫描到 {len(self.emotions)} 个情绪配置: {list(self.emotions.keys())}")
        else:
            print(f"警告：未在 {self.ref_audio_root} 下找到任何情绪配置")

    def _apply_manual_emotions(self):
        manual_list = self.config.get("emotions_config", [])
        if not manual_list:
            return
        for item in manual_list:
            emotion_name = item.get("emotion_name", "")
            ref_filename = item.get("ref_filename", "ref.mp3")
            prompt_text = item.get("prompt_text", "")
            if not emotion_name or not self.ref_audio_root:
                continue
            ref_path = os.path.join(self.ref_audio_root, emotion_name, ref_filename)
            if not os.path.exists(ref_path):
                print(f"警告：手动情绪 {emotion_name} 的参考音频不存在：{ref_path}")
                continue
            self.emotions[emotion_name] = {
                "ref_path": ref_path.replace("\\", "/"),
                "prompt_text": prompt_text
            }
        print(f"手动配置情绪已加载，当前情绪总数：{len(self.emotions)}")

    def get_emotion(self, name):
        return self.emotions.get(name, self.emotions.get(self.default_voice))


class MemoryManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.data_path = Path(config.get("memory_data_path", "./data")).resolve()
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.character_key = config.get("character_key", "murasame")
        self.character_name = config.get("character_name", "丛雨")
        self.isolated_session = config.get("isolated_session", False)

    def get_memory_file(self, session_id: str) -> Path:
        safe_session = re.sub(r'[^A-Za-z0-9_\-]', '_', session_id)
        return self.data_path / f"{self.character_key}_{safe_session}.json"

    def load_session_data(self, session_id: str) -> dict:
        file_path = self.get_memory_file(session_id)
        if file_path.exists():
            try:
                data = json.loads(file_path.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    data.setdefault("history", [])
                    data.setdefault("meta", {})
                    return data
            except Exception:
                pass
        return {"character_name": self.character_name, "history": [], "meta": {}}

    def save_session_data(self, session_id: str, data: dict):
        file_path = self.get_memory_file(session_id)
        data["character_name"] = data.get("character_name", self.character_name)
        data.setdefault("meta", {})
        data["history"] = (data.get("history") or [])[-60:]
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def load_history(self, session_id: str) -> list:
        return self.load_session_data(session_id).get("history", [])

    def save_history(self, session_id: str, history: list):
        data = self.load_session_data(session_id)
        data["history"] = history
        self.save_session_data(session_id, data)

    def get_meta(self, session_id: str) -> dict:
        return self.load_session_data(session_id).get("meta", {})

    def update_meta(self, session_id: str, **kwargs):
        data = self.load_session_data(session_id)
        data.setdefault("meta", {}).update(kwargs)
        self.save_session_data(session_id, data)

    def cleanup_voice_cache(self, max_cache: int = 20):
        try:
            cache_files = list(self.data_path.glob("temp_*.wav")) + list(self.data_path.glob("combined_*.wav"))
            if len(cache_files) <= max_cache:
                return
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            to_delete = len(cache_files) - max_cache
            for old_file in cache_files[:to_delete]:
                try:
                    temp_name = old_file.with_suffix('.tmp_del')
                    old_file.rename(temp_name)
                    temp_name.unlink(missing_ok=True)
                except PermissionError:
                    print(f"文件 {old_file.name} 被占用，跳过删除")
                except Exception as e:
                    print(f"删除 {old_file.name} 时异常: {e}")
        except Exception as e:
            print(f"清理语音缓存失败: {e}")

    def migrate_legacy_memory(self, session_id: str):
        legacy_file = self.data_path / f"{self.character_key}DATA.json"
        if not legacy_file.exists():
            return
        current_file = self.get_memory_file(session_id)
        if current_file.exists():
            return
        try:
            with open(legacy_file, 'r', encoding='utf-8') as f:
                legacy_data = json.load(f)
            history = legacy_data.get("history", [])
            character_name = legacy_data.get("character_name", self.character_name)
            with open(current_file, 'w', encoding='utf-8') as f:
                json.dump({"character_name": character_name, "history": history}, f, ensure_ascii=False, indent=2)
            legacy_file.unlink()
            print(f"已迁移旧记忆文件到 {current_file.name}，并删除旧文件。")
        except Exception as e:
            print(f"迁移旧记忆文件失败: {e}")

    def list_memories(self):
        memories = []
        if self.data_path.exists():
            for f in self.data_path.glob("*.json"):
                # 仅匹配角色会话记忆文件，排除 tools.json / scheduled_jobs.json 等功能数据
                if not re.match(r'^[A-Za-z0-9_\-]+_(private|group)_[A-Za-z0-9_\-]+\.json$', f.name):
                    continue
                try:
                    role_name = f.name.split("_")[0] if "_" in f.name else "未知"
                    with open(f, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    character_name = data.get("character_name", role_name)
                    history = data.get("history", [])
                    last_sentence = ""
                    for msg in reversed(history):
                        if msg.get("role") == "assistant":
                            last_sentence = str(msg.get("content", ""))[:50]
                            break
                    if "private_" in f.name:
                        sender_id = f.name.split("private_")[-1].replace(".json", "")
                        sender_name = None
                        for msg in reversed(history):
                            if msg.get("role") == "user" and msg.get("sender_name"):
                                sender_name = msg["sender_name"]
                                break
                        if not sender_name:
                            sender_name = sender_id
                        display_name = f"{character_name}和{sender_name}的聊天"
                    else:
                        group_id = f.name.split("group_")[-1].replace(".json", "")
                        group_name = f"群聊{group_id}"
                        display_name = f"{character_name}在{group_name}的聊天"
                    memories.append({
                        "filename": f.name,
                        "display_name": display_name,
                        "last_sentence": last_sentence,
                        "modified_time": f.stat().st_mtime,
                        "role_name": role_name
                    })
                except Exception as e:
                    print(f"读取记忆文件 {f.name} 失败: {e}")
        return memories

    def get_history(self, filename: str):
        if not re.match(r'^[A-Za-z0-9_\-]+\.json$', filename):
            return {"success": False, "error": "非法文件名"}
        file_path = (self.data_path / filename).resolve()
        if self.data_path.resolve() not in file_path.parents:
            return {"success": False, "error": "路径不安全"}
        if not file_path.exists():
            return {"success": False, "error": "文件不存在"}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"success": True, "character_name": data.get("character_name", "未知"), "history": data.get("history", [])}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_memory_file(self, filename: str):
        if not re.match(r'^[A-Za-z0-9_\-]+\.json$', filename):
            return False
        target = (self.data_path / filename).resolve()
        if self.data_path.resolve() in target.parents and target.exists():
            try:
                target.unlink()
                return True
            except Exception as e:
                print(f"删除失败 {filename}: {e}")
        return False

    def delete_messages(self, filename: str, indices: list):
        if not re.match(r'^[A-Za-z0-9_\-]+\.json$', filename):
            return {"success": False, "error": "非法文件名"}
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            return {"success": False, "error": "索引必须为整数列表"}
        file_path = (self.data_path / filename).resolve()
        if self.data_path.resolve() not in file_path.parents:
            return {"success": False, "error": "路径不安全"}
        if not file_path.exists():
            return {"success": False, "error": "文件不存在"}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            history = data.get("history", [])
            valid_indices = sorted(set(indices), reverse=True)
            deleted_count = 0
            for idx in valid_indices:
                if 0 <= idx < len(history):
                    history.pop(idx)
                    deleted_count += 1
            data["history"] = history
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return {"success": True, "deleted_count": deleted_count}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# 运行时上下文与全局管理器
# ============================================================================

global_config: Optional[ConfigLoader] = None
global_emotion_manager: Optional[EmotionManager] = None
memory_manager: Optional[MemoryManager] = None
db: Optional[DatabaseManager] = None
stats_mgr: Optional[StatsManager] = None
sticker_mgr: Optional[StickerManager] = None
tool_registry: Optional[ToolRegistry] = None
profile_mgr: Optional[UserProfileManager] = None
rag_mgr: Optional[RAGManager] = None
todo_mgr: Optional[TodoManager] = None
job_mgr: Optional[ScheduledJobManager] = None
event_mgr: Optional[EventManager] = None
sender: Optional[MessageSender] = None
napcat_client = None
scheduler: SchedulerManager = get_scheduler()

last_interaction: Dict[str, float] = {}   # session_id -> 最后交互时间
proactive_counts: Dict[str, int] = {}     # "date|session_id" -> 当日主动消息次数
_role_emotions_cache: Dict[str, dict] = {}


def get_active_role() -> dict:
    if global_config is None:
        return {}
    return global_config.roles.get(global_config.active_character, {}) or \
        (next(iter(global_config.roles.values())) if global_config.roles else {})


def get_active_ctx() -> RoleContext:
    return RoleContext(global_config.config if global_config else {}, get_active_role())


def get_active_emotions() -> dict:
    return global_emotion_manager.emotions if global_emotion_manager else {}


def get_role_emotions(role: dict) -> dict:
    """按角色获取情绪配置（各角色可有独立 ref_audio_root），带缓存。"""
    key = (role or {}).get("character_key", "")
    if not key or global_emotion_manager is None:
        return get_active_emotions()
    if key not in _role_emotions_cache:
        try:
            mgr = EmotionManager(RoleContext(global_config.config, role))
            _role_emotions_cache[key] = mgr.emotions or get_active_emotions()
        except Exception as e:
            print(f"扫描角色 {key} 情绪失败: {e}")
            _role_emotions_cache[key] = get_active_emotions()
    return _role_emotions_cache[key]


def parse_session_target(session_id: str):
    """从会话ID解析发送目标：private_123 → (private,123)；group_456[_789] → (group,456)"""
    parts = str(session_id).split("_")
    if parts[0] == "private" and len(parts) >= 2:
        return "private", parts[1]
    if parts[0] == "group" and len(parts) >= 2:
        return "group", parts[1]
    return "private", str(session_id)


def _in_quiet_hours() -> bool:
    start = str(global_config.get("proactive_quiet_start", "23:00"))
    end = str(global_config.get("proactive_quiet_end", "08:00"))

    def to_minutes(hhmm):
        try:
            h, m = hhmm.split(":")[:2]
            return int(h) * 60 + int(m)
        except Exception:
            return None
    s, e, cur = to_minutes(start), to_minutes(end), to_minutes(time.strftime("%H:%M"))
    if s is None or e is None or cur is None or s == e:
        return False
    if s < e:
        return s <= cur < e
    return cur >= s or cur < e


# ============================================================================
# 回复生成（含流式）与句子发送
# ============================================================================

class SentenceSink:
    """流式回复的逐句发送器：句子入队，后台工作线程按顺序合成+发送。"""

    def __init__(self, session_type, target_id, emotions, ctx):
        self.session_type = session_type
        self.target_id = target_id
        self.emotions = emotions
        self.ctx = ctx
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None
        self.sent = 0
        self.sent_sentences: List[dict] = []  # 已成功发送的句子（流式中断时用于入库）
        self._tts_ok: Optional[bool] = None
        self.tts_ms = 0.0

    async def on_sentence(self, sentence: dict):
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())
        await self.queue.put(sentence)

    async def _run(self):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            try:
                await self._send_one(item)
            except Exception as e:
                print(f"流式发送单句异常: {type(e).__name__}: {e}")
            finally:
                self.queue.task_done()

    async def _tts_available(self) -> bool:
        if self._tts_ok is None:
            self._tts_ok = await ensure_tts_service(global_config)
            if not self._tts_ok:
                print("警告：TTS 服务不可用，流式回复降级为纯文本。")
        return self._tts_ok and global_config.get("tts_reply_enabled", True)

    async def _send_one(self, sentence: dict):
        sticker = sticker_mgr.pick(sentence.get("emotion", "")) if (sticker_mgr and self.sent == 0) else None
        wav = None
        if await self._tts_available():
            start = time.time()
            wav = await synthesize_sentence(self.ctx, sentence["lang"], sentence["emotion"],
                                            self.emotions, memory_manager.data_path,
                                            stats=stats_mgr)
            self.tts_ms += (time.time() - start) * 1000
        if wav:
            await sender.send_text(self.session_type, self.target_id, sentence["display"], sticker=sticker)
            await sender.send_voice(self.session_type, self.target_id, wav)
            if global_config.get("dynamic_sleep", True):
                await asyncio.sleep(get_audio_duration(str(wav)) + 0.5)
            else:
                await asyncio.sleep(0.2)
            wav.unlink(missing_ok=True)
        else:
            await sender.send_text(self.session_type, self.target_id, sentence["display"], sticker=sticker)
        self.sent += 1
        self.sent_sentences.append(sentence)

    async def flush(self):
        if self.worker is not None:
            await self.queue.put(None)
            try:
                await self.worker
            except Exception as e:
                print(f"流式发送工作器异常: {e}")
            self.worker = None


async def generate_reply(ctx: RoleContext, emotions: dict, user_text: str, history: list,
                         images: Optional[list], extra_parts: List[str],
                         user_id: str = "", on_sentence=None) -> Optional[dict]:
    """生成回复：识图 / 工具调用 / 流式 / 普通四种路径统一入口。"""
    if images:
        result = await get_image_reply(ctx, user_text, history, emotions, images,
                                       extra_parts=extra_parts, stats=stats_mgr)
        if result is not None:
            return {"sentences": result["sentences"], "llm_ms": result.get("ms", 0), "tool_trace": []}
        # 识图失败（模型未配置/服务异常），降级为普通文本回复，避免用户消息石沉大海
        print("识图失败，降级为普通文本回复。")

    messages = build_chat_messages(ctx, user_text, history, emotions, extra_parts)

    # 工具调用路径（非流式，保证 tool_calls 正确处理）
    if tool_registry and global_config.get("tools_enabled", False) and tool_registry.has_enabled_tools():
        tool_registry.begin_reply()
        result = await chat_with_tools(ctx, messages, tool_registry, stats=stats_mgr, user_id=user_id)
        for t in result.get("tool_trace", []):
            print(f"[工具调用] {t['name']} ok={t['ok']} → {str(t['output'])[:80]}")
        sentences = normalize_sentences(result["content"], ctx, emotions, user_text)
        return {"sentences": sentences, "llm_ms": result["ms"], "tool_trace": result["tool_trace"]}

    # 流式路径
    if global_config.get("streaming_enabled", False):
        return await generate_reply_stream(ctx, emotions, user_text, messages, on_sentence)

    # 普通路径
    result = await chat_once(ctx, messages)
    if stats_mgr:
        stats_mgr.record_llm(result["ms"])
    sentences = normalize_sentences(result["content"], ctx, emotions, user_text)
    return {"sentences": sentences, "llm_ms": result["ms"], "tool_trace": []}


async def generate_reply_stream(ctx: RoleContext, emotions: dict, user_text: str,
                                messages: list, on_sentence=None) -> dict:
    parser = SentenceStreamParser()
    sentences = []
    first_ms = None
    start = time.time()
    try:
        async for chunk in stream_chat(ctx, messages):
            if chunk.get("first_token_ms") and first_ms is None:
                first_ms = chunk["first_token_ms"]
            delta = chunk.get("delta") or ""
            if not delta:
                continue
            for obj in parser.feed(delta):
                if not sentence_obj_has_text(obj):
                    continue  # 只有 emotion 等元数据的空句子对象：跳过，避免念出兜底台词
                sentence = normalize_single(obj, ctx, emotions, user_text)
                # 模型可能把多句塞进同一元素；按句号拆分后逐句流式发送
                for piece in split_multi_clause_sentences([sentence]):
                    if on_sentence:
                        await on_sentence(piece)
                    if not on_sentence:
                        sentences.append(piece)
    except Exception as e:
        print(f"流式请求异常: {type(e).__name__}: {e}（已收到的内容将继续处理）")
    # 流结束兜底：一句未产出时整段规整（宽容修复/JSON防线）；
    # 已产出时抢救末尾被截断的半句
    sentences.extend(parser.finish(ctx, user_text, emotions))
    total_ms = (time.time() - start) * 1000
    if stats_mgr:
        stats_mgr.record_llm(first_ms or total_ms)
    return {"sentences": sentences, "llm_ms": total_ms, "tool_trace": []}


def resolve_target_roles(user_text: str, is_private: bool) -> List[dict]:
    """多角色路由：私聊始终当前角色；群聊按消息中出现的角色名路由。"""
    active = get_active_role()
    if not active:
        return []
    if is_private or not global_config.get("multi_role_enabled", False):
        return [active]
    matched = []
    for role in global_config.roles.values():
        name = str(role.get("character_name", "")).strip()
        if name and name in user_text:
            matched.append(role)
    if not matched:
        return [active]
    cap = max(1, int(global_config.get("multi_role_max_replies", 2)))
    return matched[:cap]


# ============================================================================
# 消息处理主管线
# ============================================================================

def _spawn(coro):
    """安全的后台任务封装，异常仅记录不中断主流程。"""
    async def _wrapper():
        try:
            await coro
        except Exception as e:
            import traceback
            print(f"后台任务异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    return asyncio.create_task(_wrapper())


def pick_image_source(seg) -> Optional[str]:
    """从图片消息段挑出可用的图片来源（本地路径 / http(s) URL / file://）。

    NapCat 的 file 常是裸文件名（如 ABC.image），既非本地路径也非 URL，
    此时必须回退到 path/url 字段，否则识图永远拿不到图。
    """
    srcs = [str(s) for s in (getattr(seg, "file", None), getattr(seg, "path", None),
                             getattr(seg, "url", None)) if s]
    chosen = next((s for s in srcs if os.path.isfile(s)
                   or s.startswith(("http://", "https://", "file://"))), None)
    return chosen or (srcs[0] if srcs else None)


async def handle_message_event(event, client):
    global napcat_client
    napcat_client = client
    # 只处理私聊和群聊消息事件，其他事件（如心跳）直接忽略
    if not isinstance(event, (PrivateMessageEvent, GroupMessageEvent)):
        return

    is_private = isinstance(event, PrivateMessageEvent)

    if is_private:
        session_type = "private"
        target_id = event.user_id
        sender_id = event.user_id
        session_id = f"private_{event.user_id}"
    else:
        group_id = event.group_id
        sender_id = getattr(event.sender, "user_id", None) or "0"
        session_type = "group"
        target_id = group_id
        session_id = f"group_{group_id}"
        if global_config.get("isolated_session", False):
            session_id = f"group_{group_id}_{sender_id}"

    sender_name = getattr(event.sender, "nickname", None) or str(sender_id)
    user_text = ""
    has_image = False
    image_urls = []
    at_bot = False
    for seg in event.message:
        if isinstance(seg, Text):
            user_text += seg.text
        elif isinstance(seg, Image):
            has_image = True
            chosen = pick_image_source(seg)
            if chosen and chosen not in image_urls:
                image_urls.append(chosen)
        elif isinstance(seg, At):
            if str(seg.qq) == str(client.self_id):
                at_bot = True

    if not is_private and not at_bot and global_config.get("only_private", False):
        return
    if not user_text and not has_image:
        return

    print(f"收到{'私聊' if is_private else '群聊'} [{target_id}] 来自 [{sender_id}]: {user_text}")
    last_interaction[session_id] = time.time()

    memory_manager.migrate_legacy_memory(session_id)
    data = memory_manager.load_session_data(session_id)
    history = data.get("history", [])
    meta = data.get("meta", {})
    meta["user_msg_count"] = int(meta.get("user_msg_count", 0)) + 1
    meta["last_user_text"] = user_text[:200]
    data["meta"] = meta
    history.append({
        "role": "user",
        "content": user_text,
        "sender_id": sender_id,
        "sender_name": sender_name,
        "timestamp": time.time()
    })

    # 待办提取（后台异步，不阻塞回复）
    if global_config.get("todo_enabled", False) and todo_mgr:
        mode = global_config.get("todo_extract_mode", "regex")
        if mode == "regex":
            found = todo_mgr.extract_sync(user_text)
            for content, remind_ts in found:
                todo_mgr.add_todo(content, remind_ts, session_type, session_id, sender_id, source="regex")
        elif mode == "llm":
            _spawn(todo_mgr.extract_and_add(get_active_ctx(), user_text, session_type, session_id, sender_id))

    target_roles = resolve_target_roles(user_text, is_private)
    max_total = max(1, int(global_config.get("multi_role_max_total", 6)))
    total_replies = 0
    first_reply_done = False

    async def process_role_reply(role: dict, trigger_text: str) -> bool:
        """让指定角色对 trigger_text 生成并回复一次，返回是否成功。"""
        nonlocal total_replies, first_reply_done
        if total_replies >= max_total:
            return False
        ctx = RoleContext(global_config.config, role)
        emotions = get_role_emotions(role)

        extra_parts = []
        use_history = history
        # 摘要与话题注入（动态上下文）
        if global_config.get("summary_enabled", False) and meta.get("summary"):
            keep = max(1, int(global_config.get("summary_max_history", 5)))
            use_history = history[-keep:]
            extra_parts.append(f"【早期对话摘要】{meta['summary']}")
        if global_config.get("dynamic_context_enabled", False) and meta.get("topic"):
            extra_parts.append(f"【当前话题】{meta['topic']}")
        if profile_mgr and global_config.get("profiles_enabled", False):
            p = profile_mgr.build_injection(sender_id)
            if p:
                extra_parts.append(p)
        if rag_mgr and global_config.get("rag_enabled", False):
            rc = await rag_mgr.build_context(user_text)
            if rc:
                extra_parts.append(rc)

        # 流式：首句即可发送；其余路径统一发送
        sink = None
        if global_config.get("streaming_enabled", False) and not first_reply_done:
            sink = SentenceSink(session_type, target_id, emotions, ctx)

        reply = await generate_reply(ctx, emotions, trigger_text, use_history,
                                     image_urls if has_image else None,
                                     extra_parts, sender_id,
                                     on_sentence=sink.on_sentence if sink else None)

        tts_ms = 0.0
        if sink is not None:
            await sink.flush()
            if (not reply or not reply.get("sentences")) and sink.sent_sentences:
                # 流式中断但已发出部分句子：以已发送内容入库，保持记忆与实际发送一致
                reply = {"sentences": sink.sent_sentences, "llm_ms": 0, "tool_trace": []}
            tts_ms = sink.tts_ms
        if not reply or not reply.get("sentences"):
            return False

        if sink is not None:
            if sink.sent == 0:
                # 流式解析未产出（模型未输出JSON），走统一发送
                await sender.send_reply(session_type, target_id, reply["sentences"],
                                        emotions, ctx,
                                        use_voice=global_config.get("tts_reply_enabled", True))
        else:
            send_result = await sender.send_reply(session_type, target_id, reply["sentences"],
                                                  emotions, ctx,
                                                  use_voice=global_config.get("tts_reply_enabled", True))
            tts_ms = send_result.get("tts_ms", 0.0)

        zh_text = "".join(s["zh"] for s in reply["sentences"])
        speaker = role.get("character_name", ctx.character_key)
        history.append({"role": "assistant", "content": zh_text, "timestamp": time.time(),
                        "speaker": speaker,
                        "emotion": reply["sentences"][0].get("emotion", "")})
        data["history"] = history
        data["meta"] = meta
        memory_manager.save_session_data(session_id, data)

        if stats_mgr and global_config.get("stats_enabled", True) and db is not None:
            db.record_interaction(
                session_type, session_id, sender_id, sender_name,
                role.get("character_key", ""), reply["sentences"][0].get("emotion", ""),
                reply.get("llm_ms", 0), tts_ms, len(reply["sentences"]), ok=True)
        if stats_mgr:
            stats_mgr.record_message(session_id)

        # 用户画像自动提取（后台）
        if profile_mgr and global_config.get("profiles_enabled", False) and \
                global_config.get("profiles_auto_extract", False) and not first_reply_done:
            _spawn(profile_mgr.extract_from_dialog(ctx, trigger_text, zh_text, sender_id))
        first_reply_done = True
        total_replies += 1
        return True

    try:
        if not await ensure_tts_service(global_config):
            print("警告：TTS 服务不可用，将降级为纯文本。")

        for role in target_roles:
            if total_replies >= max_total:
                break
            await process_role_reply(role, user_text)

        # 角色间互相回应（自动轮数）
        rounds = int(global_config.get("multi_role_auto_rounds", 0) or 0)
        if global_config.get("multi_role_enabled", False) and rounds > 0 and len(target_roles) > 1:
            for _ in range(rounds):
                for role in target_roles:
                    if total_replies >= max_total:
                        break
                    last_assistant = next((m for m in reversed(history)
                                           if m.get("role") == "assistant"), None)
                    if not last_assistant:
                        break
                    if last_assistant.get("speaker") == role.get("character_name", ""):
                        continue
                    trigger = (f"（群里的另一位角色「{last_assistant.get('speaker', '')}」刚刚说："
                               f"{last_assistant.get('content', '')}）请自然地接话回应。")
                    await process_role_reply(role, trigger)
    except Exception as e:
        print(f"回复生成失败: {type(e).__name__}: {e}")

    memory_manager.cleanup_voice_cache(global_config.get("max_voice_cache", 20))

    # 摘要与话题维护（后台异步）
    _spawn(post_reply_context_tasks(session_id, get_active_ctx()))


async def post_reply_context_tasks(session_id: str, ctx: RoleContext):
    """对话后维护：自动摘要 + 话题检测（均为可开关功能）。"""
    try:
        data = memory_manager.load_session_data(session_id)
        history = data.get("history", [])
        meta = data.get("meta", {})
        changed = False
        # 上下文自动摘要
        if global_config.get("summary_enabled", False):
            threshold = int(global_config.get("summary_threshold", 20))
            keep = max(1, int(global_config.get("summary_max_history", 5)))
            if len(history) >= threshold + keep:
                old_msgs = history[:len(history) - keep]
                existing = meta.get("summary", "")
                lines = []
                for m in old_msgs[-40:]:
                    who = m.get("sender_name") or m.get("speaker") or ("角色" if m.get("role") == "assistant" else "用户")
                    lines.append(f"{who}: {str(m.get('content', ''))[:120]}")
                prompt = (f"{global_config.get('summary_prompt', '')}\n\n"
                          f"{'已有摘要（请合并）：' + existing if existing else ''}\n\n对话：\n" + "\n".join(lines))
                summary = await generate_proactive_text(ctx, prompt)
                if summary:
                    meta["summary"] = summary[:800]
                    changed = True
                    print(f"已更新会话 {session_id} 的上下文摘要。")
        # 话题检测
        if global_config.get("dynamic_context_enabled", False):
            every = max(2, int(global_config.get("topic_summary_every_n", 10)))
            if int(meta.get("user_msg_count", 0)) % every == 0:
                recent = history[-10:]
                lines = [f"{m.get('sender_name') or m.get('speaker') or '用户'}: {str(m.get('content', ''))[:120]}"
                         for m in recent]
                prompt = (f"{global_config.get('topic_summary_prompt', '')}\n\n" + "\n".join(lines))
                topic = await generate_proactive_text(ctx, prompt)
                if topic:
                    meta["topic"] = topic[:200]
                    changed = True
                    print(f"已更新会话 {session_id} 的当前话题：{topic[:50]}")
        if changed:
            data["meta"] = meta
            memory_manager.save_session_data(session_id, data)
    except Exception as e:
        print(f"上下文维护任务异常: {e}")


# ============================================================================
# 主动消息与调度注册
# ============================================================================

async def proactive_idle_check():
    """定期检查长时间未互动的会话，主动发送话题。"""
    if not global_config.get("proactive_enabled", False):
        return
    if sender is None or sender.client is None:
        return
    now = time.time()
    idle_seconds = float(global_config.get("proactive_idle_minutes", 30)) * 60
    max_per_day = max(1, int(global_config.get("proactive_max_per_day", 2)))
    today = time.strftime("%Y-%m-%d")
    for session_id, last_ts in list(last_interaction.items()):
        if now - last_ts < idle_seconds:
            continue
        if _in_quiet_hours():
            continue
        used = proactive_counts.get(f"{today}|{session_id}", 0)
        if used >= max_per_day:
            continue
        session_type, target = parse_session_target(session_id)
        ctx = get_active_ctx()
        instruction = str(global_config.get("proactive_prompt", "主动找个话题和主人聊聊。"))
        try:
            text = await generate_proactive_text(ctx, instruction)
        except Exception as e:
            print(f"主动消息生成失败: {e}")
            text = ""
        if not text:
            continue
        await sender.speak_and_send(session_type, target, text, get_active_emotions(), ctx,
                                    use_voice=bool(global_config.get("proactive_voice", False)),
                                    sticker=bool(global_config.get("proactive_sticker", False)))
        last_interaction[session_id] = now
        proactive_counts[f"{today}|{session_id}"] = used + 1
        print(f"已向 {session_id} 发送主动消息。")
        # 写入会话历史，保持上下文连贯
        data = memory_manager.load_session_data(session_id)
        data.setdefault("history", []).append({
            "role": "assistant", "content": text, "timestamp": now,
            "speaker": ctx.character_name, "proactive": True})
        memory_manager.save_session_data(session_id, data)


async def greeting_daily_check():
    if event_mgr and sender:
        await event_mgr.check_and_greet(sender, get_active_ctx, get_active_emotions)


def register_feature_jobs():
    """根据配置注册/注销内置调度任务（主动消息、节日问候）。"""
    if global_config.get("proactive_enabled", False):
        seconds = max(30, int(global_config.get("proactive_check_seconds", 300)))
        scheduler.add_job("proactive_idle", "主动消息检查",
                          {"type": "interval", "seconds": seconds}, proactive_idle_check)
        print(f"主动消息检查已开启（每 {seconds} 秒，闲置阈值 {global_config.get('proactive_idle_minutes', 30)} 分钟）。")
    else:
        scheduler.remove_job("proactive_idle")
    if global_config.get("greeting_events_enabled", False):
        scheduler.add_job("greeting_check", "节日生日问候检查",
                          {"type": "daily", "time": global_config.get("greeting_check_time", "08:00")},
                          greeting_daily_check)
        print(f"节日/生日问候检查已开启（每日 {global_config.get('greeting_check_time', '08:00')}）。")
    else:
        scheduler.remove_job("greeting_check")


def hot_reload_managers():
    """配置保存后热重载依赖配置的管理器。"""
    global sticker_mgr
    if sticker_mgr is not None:
        sticker_mgr = StickerManager(global_config)
        if sender is not None:
            sender.sticker_manager = sticker_mgr
    _role_emotions_cache.clear()
    register_feature_jobs()
    if job_mgr is not None:
        job_mgr.reload()
    if sender is not None:
        sender.config = global_config


# ============================================================================
# WebUI 服务
# ============================================================================

def _json_file_response(data, filename: str):
    body = json.dumps(data, ensure_ascii=False, indent=2)
    return web.Response(body=body.encode("utf-8"), content_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _safe_subdir(root: Path, name: str) -> Optional[Path]:
    """校验 name 为 root 的直接子目录名（防路径穿越）。"""
    if not name or not re.match(r'^[\w\u4e00-\u9fff\- ]+$', name):
        return None
    p = (root / name).resolve()
    if root.resolve() not in p.parents:
        return None
    return p


class WebUIServer:
    def __init__(self, config: ConfigLoader, memory_manager: MemoryManager):
        self.config = config
        self._last_saved_config = None
        self.memory_manager = memory_manager
        self.html_path = get_resource_path("webui") / "start.html"
        self.app = web.Application(client_max_size=200 * 1080 * 1080)
        self.setup_routes()

    def setup_routes(self):
        r = self.app.router
        r.add_get("/api/list", self.handle_list)
        r.add_post("/api/history", self.handle_history)
        r.add_post("/api/delete", self.handle_delete)
        r.add_post("/api/history/delete_messages", self.handle_delete_messages)
        r.add_get("/api/config", self.handle_get_config)
        r.add_post("/api/config/save", self.handle_save_config)
        r.add_get("/api/config/export", self.handle_export_config)
        r.add_post("/api/config/import", self.handle_import_config)
        r.add_get("/api/roles", self.handle_get_roles)
        r.add_post("/api/roles/save", self.handle_save_roles)
        r.add_get("/api/logs", self.handle_get_logs)
        # 情绪音频管理
        r.add_get("/api/emotions/list", self.handle_emotions_list)
        r.add_post("/api/emotions/upload", self.handle_emotions_upload)
        r.add_post("/api/emotions/create", self.handle_emotions_create)
        r.add_post("/api/emotions/delete", self.handle_emotions_delete)
        r.add_get("/api/emotions/audio", self.handle_emotions_audio)
        # 聊天记录导入导出
        r.add_get("/api/memory/export", self.handle_memory_export)
        r.add_post("/api/memory/import", self.handle_memory_import)
        r.add_get("/api/memory/export_all", self.handle_memory_export_all)
        # 统计
        r.add_get("/api/stats", self.handle_stats)
        r.add_get("/api/performance", self.handle_performance)
        r.add_get("/api/sessions", self.handle_sessions)
        # 定时任务 / 待办 / 事件
        r.add_get("/api/jobs", self.handle_jobs)
        r.add_post("/api/jobs/save", self.handle_jobs_save)
        r.add_post("/api/jobs/run", self.handle_jobs_run)
        r.add_get("/api/todos", self.handle_todos)
        r.add_post("/api/todos/add", self.handle_todos_add)
        r.add_post("/api/todos/update", self.handle_todos_update)
        r.add_post("/api/todos/delete", self.handle_todos_delete)
        r.add_get("/api/events", self.handle_events)
        r.add_post("/api/events/save", self.handle_events_save)
        r.add_post("/api/events/test", self.handle_events_test)
        # 工具调用
        r.add_get("/api/tools", self.handle_tools)
        r.add_post("/api/tools/save", self.handle_tools_save)
        r.add_post("/api/tools/test", self.handle_tools_test)
        # RAG
        r.add_get("/api/rag/docs", self.handle_rag_docs)
        r.add_post("/api/rag/upload", self.handle_rag_upload)
        r.add_post("/api/rag/delete", self.handle_rag_delete)
        r.add_post("/api/rag/query", self.handle_rag_query)
        # 用户画像
        r.add_get("/api/profiles", self.handle_profiles)
        r.add_post("/api/profiles/save", self.handle_profiles_save)
        r.add_post("/api/profiles/delete", self.handle_profiles_delete)
        # 表情包
        r.add_get("/api/stickers/list", self.handle_stickers_list)
        r.add_post("/api/stickers/upload", self.handle_stickers_upload)
        r.add_post("/api/stickers/delete", self.handle_stickers_delete)
        r.add_get("/api/stickers/file", self.handle_stickers_file)
        r.add_get("/", self.handle_index)

    async def handle_index(self, request):
        if self.html_path.exists():
            return web.FileResponse(self.html_path)
        return web.Response(text="WebUI 页面未找到", status=404)

    # ---------------- 配置 ----------------
    async def handle_get_config(self, request):
        if not self.config.config:
            self.config.config = self.config.default_config()
        else:
            self.config.config = {**self.config.default_config(), **self.config.config}
        return web.json_response(self.config.config)

    async def handle_export_config(self, request):
        return _json_file_response(self.config.config, "ltvm_config_export.json")

    async def handle_import_config(self, request):
        try:
            ctype = request.content_type or ""
            if "multipart" in ctype:
                reader = await request.multipart()
                data = None
                async for part in reader:
                    if part.name == "file":
                        raw = await part.read(decode=False)
                        data = json.loads(raw.decode("utf-8"))
                        break
            else:
                data = await request.json()
            if not isinstance(data, dict):
                return web.json_response({"success": False, "error": "配置文件格式错误"}, status=400)
            merged = {**self.config.default_config(), **data}
            self.config.config = merged
            with open(self.config.config_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            self._after_config_reload()
            return web.json_response({"success": True, "message": "配置已导入并热重载生效！"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    def _after_config_reload(self):
        """配置变更后的统一热重载。"""
        global global_config, global_emotion_manager, memory_manager
        global_config = self.config
        self.config.roles = self.config._parse_roles()
        if not global_config.active_character or global_config.active_character not in self.config.roles:
            if self.config.roles:
                global_config.active_character = list(self.config.roles.keys())[0]
        global_emotion_manager = EmotionManager(self.config)
        memory_manager = MemoryManager(self.config)
        if sender is not None:
            sender.memory_manager = memory_manager
        hot_reload_managers()

    async def handle_save_config(self, request):
        try:
            new_config = await request.json()
            restart_tts = new_config.pop("restart_tts", False)
            self.config.config = new_config
            with open(self.config.config_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, ensure_ascii=False, indent=2)
            self._after_config_reload()

            old_config = self._last_saved_config
            tts_changed = False
            if old_config is not None:
                tts_keys = ['client_base_url', 'model_dir', 'ref_audio_root', 'device',
                            'auto_start_tts', 'tts_start_script', 'timeout_seconds',
                            'prompt_text', 'prompt_lang', 'text_lang', 'top_k', 'top_p',
                            'temperature', 'text_split_method', 'batch_size',
                            'batch_threshold', 'split_bucket', 'speed_factor',
                            'fragment_interval', 'streaming_mode', 'seed',
                            'parallel_infer', 'repetition_penalty', 'media_type',
                            'llm_emotion_intensity', 'intensity_to_temperature', 'intensity_to_top_k']
                for key in tts_keys:
                    if old_config.get(key) != new_config.get(key):
                        tts_changed = True
                        break

            force_restart = restart_tts and global_config.get("auto_start_tts", False)
            tts_restart_message = ""
            if (tts_changed or force_restart) and global_config.get("auto_start_tts", False):
                valid, error_msg = self.validate_tts_config(global_config)
                if not valid:
                    print(f"TTS 配置验证失败，跳过重启：{error_msg}")
                    tts_restart_message = f"配置已保存，但 TTS 服务未重启：{error_msg}"
                else:
                    print("检测到 TTS 相关配置变化或用户强制重启，正在重启 TTS 服务...")
                    process_manager.shutdown_all()
                    threading.Thread(target=auto_start_and_switch_tts, args=(global_config,), daemon=True).start()
                    tts_restart_message = "TTS 服务正在重启，请稍候..."
            elif tts_changed:
                tts_restart_message = "配置已保存，但 auto_start_tts 为 False，不会自动重启 TTS。"
            else:
                tts_restart_message = "TTS 配置未变化或未选择强制重启，无需重启 TTS 服务。"

            napcat_changed = False
            if old_config is not None:
                if (old_config.get('napcat_ws_url') != new_config.get('napcat_ws_url') or
                        old_config.get('napcat_token') != new_config.get('napcat_token')):
                    napcat_changed = True

            self._last_saved_config = new_config.copy()
            response = {"success": True}
            if napcat_changed:
                response['message'] = "配置已保存，但 NapCat 连接参数修改需重启程序才能生效。" + tts_restart_message
            else:
                response['message'] = "配置已保存并热重载生效。" + tts_restart_message
            return web.json_response(response)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"success": False, "error": str(e)}, status=400)

    def validate_tts_config(self, config: ConfigLoader) -> tuple:
        ref_audio_root = config.get("ref_audio_root", "")
        if not ref_audio_root or not Path(ref_audio_root).exists():
            return False, "参考音频根目录无效或不存在，请检查路径后重试"
        model_dir = config.get("model_dir", "")
        if not model_dir or not Path(model_dir).exists():
            return False, "模型文件夹路径无效或不存在，请检查路径后重试"
        return True, ""

    # ---------------- 角色 ----------------
    async def handle_get_roles(self, request):
        roles = []
        for key, role in self.config.roles.items():
            roles.append({
                "character_key": key,
                "character_name": role["character_name"],
                "active": key == self.config.active_character
            })
        return web.json_response({"roles": roles})

    async def handle_save_roles(self, request):
        try:
            new_data = await request.json()
            roles = new_data.get("roles", [])
            active = new_data.get("active_character", "")
            self.config.config["roles"] = roles
            self.config.config["active_character"] = active
            with open(self.config.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.config, f, ensure_ascii=False, indent=2)
            self._after_config_reload()
            return web.json_response({"success": True, "message": "角色配置已保存！"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_get_logs(self, request):
        logs = "\n".join(global_log_buffer)
        return web.json_response({"logs": logs})

    # ---------------- 聊天记录 ----------------
    async def handle_list(self, request):
        memories = self.memory_manager.list_memories()
        return web.json_response({"memories": memories})

    async def handle_history(self, request):
        try:
            payload = await request.json()
            filename = payload.get("filename", "")
            result = self.memory_manager.get_history(filename)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_delete(self, request):
        try:
            payload = await request.json()
            filenames = payload.get("files", [])
            deleted = []
            for filename in filenames:
                if self.memory_manager.delete_memory_file(filename):
                    deleted.append(filename)
            return web.json_response({"success": True, "deleted": deleted})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_delete_messages(self, request):
        try:
            payload = await request.json()
            filename = payload.get("filename", "")
            indices = payload.get("indices", [])
            result = self.memory_manager.delete_messages(filename, indices)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_memory_export(self, request):
        filename = request.query.get("filename", "")
        result = self.memory_manager.get_history(filename)
        if not result.get("success"):
            return web.json_response(result, status=404)
        return _json_file_response({"filename": filename,
                                    "character_name": result.get("character_name"),
                                    "history": result.get("history", [])}, filename)

    async def handle_memory_import(self, request):
        try:
            reader = await request.multipart()
            imported = []
            async for part in reader:
                if part.filename and part.filename.endswith(".json"):
                    raw = await part.read(decode=False)
                    data = json.loads(raw.decode("utf-8"))
                    if not isinstance(data, dict) or "history" not in data:
                        continue
                    name = Path(part.filename).name
                    if not re.match(r'^[A-Za-z0-9_\-]+\.json$', name):
                        name = re.sub(r'[^A-Za-z0-9_\-]', '_', Path(name).stem) + ".json"
                    (self.memory_manager.data_path / name).write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    imported.append(name)
            return web.json_response({"success": True, "imported": imported})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_memory_export_all(self, request):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in self.memory_manager.data_path.glob("*.json"):
                zf.write(f, f.name)
        buf.seek(0)
        return web.Response(body=buf.read(), content_type="application/zip",
                            headers={"Content-Disposition": 'attachment; filename="ltvm_memories.zip"'})

    async def handle_sessions(self, request):
        """返回已知会话列表（供定时任务/事件/待办选择发送目标）。"""
        sessions = []
        for f in self.memory_manager.data_path.glob("*.json"):
            m = re.match(r'^[A-Za-z0-9_\-]+_(private|group)_[A-Za-z0-9_\-]+\.json$', f.name)
            if not m:
                continue
            stype = m.group(1)
            rest = f.name.split(f"{stype}_", 1)[1].replace(".json", "")
            if stype == "group":
                sid = f"group_{rest.split('_')[0]}"
            else:
                sid = f"private_{rest}"
            item = {"session_id": sid, "session_type": stype}
            if not any(s["session_id"] == sid for s in sessions):
                sessions.append(item)
        return web.json_response({"sessions": sessions})

    # ---------------- 情绪音频管理 ----------------
    def _role_root(self, role_key: str) -> Path:
        role = self.config.roles.get(role_key, {}) if role_key else {}
        ctx = RoleContext(self.config.config, role or {})
        return Path(resolve_tts_path(ctx.get("ref_audio_root", "")))

    async def handle_emotions_list(self, request):
        role_key = request.query.get("role", "")
        root = self._role_root(role_key)
        emotions = []
        if root.exists():
            for folder in sorted(root.iterdir()):
                if not folder.is_dir():
                    continue
                files = [f.name for f in sorted(folder.iterdir()) if f.is_file()]
                asr = ""
                asr_path = folder / "asr.txt"
                if asr_path.exists():
                    asr = asr_path.read_text(encoding="utf-8", errors="ignore").strip()
                ref = next((f for f in files if f.lower().startswith("ref.")), None)
                if ref is None:
                    ref = next((f for f in files if f.lower().split(".")[-1] in
                                ("mp3", "wav", "ogg", "flac", "m4a")), None)
                emotions.append({"name": folder.name, "files": files, "asr": asr, "ref": ref})
        return web.json_response({"root": str(root), "role": role_key, "emotions": emotions})

    async def handle_emotions_upload(self, request):
        try:
            reader = await request.multipart()
            role = emotion = asr_text = None
            file_data = None
            file_name = ""
            async for part in reader:
                if part.name == "role":
                    role = (await part.text()).strip()
                elif part.name == "emotion":
                    emotion = (await part.text()).strip()
                elif part.name == "asr":
                    asr_text = await part.text()
                elif part.name == "file":
                    file_name = part.filename or ""
                    file_data = await part.read(decode=False)
            root = self._role_root(role or "")
            folder = _safe_subdir(root, emotion or "")
            if folder is None:
                return web.json_response({"success": False, "error": "情绪名称非法"}, status=400)
            folder.mkdir(parents=True, exist_ok=True)
            saved = []
            if file_data:
                ext = Path(file_name).suffix.lower() or ".mp3"
                if ext not in (".mp3", ".wav", ".ogg", ".flac", ".m4a"):
                    return web.json_response({"success": False, "error": "仅支持音频文件"}, status=400)
                target = folder / f"ref{ext}"
                target.write_bytes(file_data)
                saved.append(target.name)
            if asr_text is not None and asr_text.strip():
                (folder / "asr.txt").write_text(asr_text.strip(), encoding="utf-8")
                saved.append("asr.txt")
            self._after_config_reload()
            return web.json_response({"success": True, "saved": saved})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_emotions_create(self, request):
        try:
            payload = await request.json()
            root = self._role_root(payload.get("role", ""))
            folder = _safe_subdir(root, payload.get("emotion", ""))
            if folder is None:
                return web.json_response({"success": False, "error": "情绪名称非法"}, status=400)
            folder.mkdir(parents=True, exist_ok=True)
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_emotions_delete(self, request):
        try:
            payload = await request.json()
            root = self._role_root(payload.get("role", ""))
            folder = _safe_subdir(root, payload.get("emotion", ""))
            if folder is None or not folder.exists():
                return web.json_response({"success": False, "error": "目录不存在"}, status=404)
            shutil.rmtree(folder)
            self._after_config_reload()
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_emotions_audio(self, request):
        role = request.query.get("role", "")
        emotion = request.query.get("emotion", "")
        file = request.query.get("file", "")
        root = self._role_root(role)
        folder = _safe_subdir(root, emotion)
        if folder is None or not file or not re.match(r'^[\w\u4e00-\u9fff\-. ]+$', file):
            return web.Response(status=404, text="not found")
        target = folder / file
        if not target.exists():
            return web.Response(status=404, text="not found")
        return web.FileResponse(target)

    # ---------------- 统计 ----------------
    async def handle_stats(self, request):
        return web.json_response(stats_mgr.get_stats() if stats_mgr else {})

    async def handle_performance(self, request):
        perf = stats_mgr.get_performance() if stats_mgr else {}
        perf["tts_online"] = await ensure_tts_service_enabled_check()
        perf["napcat_connected"] = bool(sender and sender.client is not None)
        perf["scheduler_jobs"] = len([j for j in scheduler.jobs.values() if j.enabled])
        return web.json_response(perf)

    # ---------------- 定时任务 ----------------
    async def handle_jobs(self, request):
        return web.json_response({"jobs": job_mgr.describe() if job_mgr else [],
                                  "scheduler_enabled": bool(self.config.get("scheduler_enabled", False))})

    async def handle_jobs_save(self, request):
        try:
            payload = await request.json()
            jobs = payload.get("jobs", [])
            cleaned = []
            for job in jobs:
                jid = str(job.get("id") or f"job_{int(time.time()*1000)}")
                cleaned.append({
                    "id": jid, "name": str(job.get("name") or jid),
                    "enabled": bool(job.get("enabled", True)),
                    "trigger": job.get("trigger") or {"type": "daily", "time": "08:00"},
                    "target": job.get("target") or {},
                    "action": job.get("action") or {"mode": "template", "template": ""},
                })
            job_mgr.jobs = cleaned
            job_mgr.save()
            job_mgr.reload()
            return web.json_response({"success": True, "jobs": job_mgr.describe()})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_jobs_run(self, request):
        try:
            payload = await request.json()
            jid = str(payload.get("id", ""))
            job = next((j for j in job_mgr.jobs if str(j.get("id")) == jid), None)
            if not job:
                return web.json_response({"success": False, "error": "任务不存在"}, status=404)
            await job_mgr._run_job(job)
            return web.json_response({"success": True, "message": "任务已手动执行一次"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    # ---------------- 待办 ----------------
    async def handle_todos(self, request):
        status = request.query.get("status")
        todos = todo_mgr.list_todos(status) if todo_mgr else []
        return web.json_response({"todos": todos})

    async def handle_todos_add(self, request):
        try:
            payload = await request.json()
            content = str(payload.get("content", "")).strip()
            if not content:
                return web.json_response({"success": False, "error": "内容不能为空"}, status=400)
            remind = payload.get("remind_time")
            remind_ts = None
            if isinstance(remind, (int, float)):
                remind_ts = float(remind)
            elif isinstance(remind, str) and remind.strip():
                for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
                    try:
                        remind_ts = time.mktime(time.strptime(remind.strip(), fmt))
                        break
                    except ValueError:
                        continue
                if remind_ts is None:
                    return web.json_response({"success": False, "error": "时间格式应为 YYYY-MM-DD HH:MM"}, status=400)
            if remind_ts is None:
                return web.json_response({"success": False, "error": "请填写提醒时间"}, status=400)
            todo = todo_mgr.add_todo(content, remind_ts,
                                     payload.get("session_type", "private"),
                                     payload.get("session_id", ""),
                                     payload.get("user_id", ""), source="manual")
            return web.json_response({"success": bool(todo), "todo": todo})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_todos_update(self, request):
        try:
            payload = await request.json()
            todo_id = int(payload.get("id", 0))
            status = payload.get("status", "done")
            if status in ("done", "cancelled"):
                if status == "done":
                    todo_mgr.complete(todo_id)
                else:
                    todo_mgr.delete(todo_id)
                return web.json_response({"success": True})
            todo_mgr.db.execute("UPDATE todos SET status=? WHERE id=?", (status, todo_id))
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_todos_delete(self, request):
        try:
            payload = await request.json()
            todo_mgr.delete(int(payload.get("id", 0)))
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    # ---------------- 事件问候 ----------------
    async def handle_events(self, request):
        return web.json_response({"events": event_mgr.events if event_mgr else []})

    async def handle_events_save(self, request):
        try:
            payload = await request.json()
            events = payload.get("events", [])
            cleaned = []
            for ev in events:
                eid = str(ev.get("id") or f"evt_{int(time.time()*1000)}")
                cleaned.append({
                    "id": eid, "name": str(ev.get("name") or eid),
                    "type": ev.get("type", "date"),
                    "date": str(ev.get("date", "")),
                    "enabled": bool(ev.get("enabled", True)),
                    "mode": ev.get("mode", "template"),
                    "template": str(ev.get("template", "")),
                    "llm_prompt": str(ev.get("llm_prompt", "")),
                    "use_voice": bool(ev.get("use_voice", False)),
                    "targets": ev.get("targets") or [],
                })
            event_mgr.events = cleaned
            event_mgr.save_events()
            return web.json_response({"success": True, "events": cleaned})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_events_test(self, request):
        try:
            payload = await request.json()
            ok = await event_mgr.greet_event_now(str(payload.get("id", "")), sender,
                                                 get_active_ctx, get_active_emotions)
            return web.json_response({"success": bool(ok),
                                      "message": "已发送测试问候" if ok else "未找到事件或生成内容为空"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    # ---------------- 工具调用 ----------------
    async def handle_tools(self, request):
        return web.json_response({"tools": tool_registry.tools if tool_registry else [],
                                  "enabled": bool(self.config.get("tools_enabled", False))})

    async def handle_tools_save(self, request):
        try:
            payload = await request.json()
            tools = payload.get("tools", [])
            for t in tools:
                if not str(t.get("name", "")).strip():
                    return web.json_response({"success": False, "error": "工具名不能为空"}, status=400)
            tool_registry.tools = tools
            tool_registry.save()
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_tools_test(self, request):
        try:
            payload = await request.json()
            name = payload.get("name", "")
            arguments = payload.get("arguments", {})
            user_id = str(payload.get("user_id", "") or "")
            tool_registry.begin_reply()
            ok, output = await tool_registry.execute(name, arguments, user_id)
            return web.json_response({"success": ok, "output": output})
        except Exception as e:
            return web.json_response({"success": False, "output": str(e)}, status=400)

    # ---------------- RAG ----------------
    async def handle_rag_docs(self, request):
        return web.json_response({"docs": rag_mgr.list_docs() if rag_mgr else [],
                                  "enabled": bool(self.config.get("rag_enabled", False))})

    async def handle_rag_upload(self, request):
        try:
            results = []
            reader = await request.multipart()
            async for part in reader:
                if not part.filename:
                    continue
                raw = await part.read(decode=False)
                tmp = self.memory_manager.data_path / f"rag_upload_{int(time.time()*1000)}_{Path(part.filename).name}"
                tmp.write_bytes(raw)
                try:
                    text = extract_text_from_file(tmp)
                    r = await rag_mgr.add_document(Path(part.filename).stem, text)
                    results.append({"file": part.filename, **r})
                except Exception as e:
                    results.append({"file": part.filename, "success": False, "error": str(e)})
                finally:
                    tmp.unlink(missing_ok=True)
            return web.json_response({"success": True, "results": results})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_rag_delete(self, request):
        try:
            payload = await request.json()
            ok = rag_mgr.delete_document(str(payload.get("id", "")))
            return web.json_response({"success": ok})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_rag_query(self, request):
        try:
            payload = await request.json()
            hits = await rag_mgr.search(str(payload.get("question", "")))
            return web.json_response({"success": True, "hits": hits})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    # ---------------- 用户画像 ----------------
    async def handle_profiles(self, request):
        profiles = [{"user_id": uid, **(p or {})} for uid, p in (profile_mgr.profiles or {}).items()]
        return web.json_response({"profiles": profiles})

    async def handle_profiles_save(self, request):
        try:
            payload = await request.json()
            uid = str(payload.get("user_id", "")).strip()
            if not uid:
                return web.json_response({"success": False, "error": "user_id 不能为空"}, status=400)
            profile_mgr.update(uid, payload.get("profile", {}))
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_profiles_delete(self, request):
        try:
            payload = await request.json()
            ok = profile_mgr.delete(str(payload.get("user_id", "")))
            return web.json_response({"success": ok})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    # ---------------- 表情包 ----------------
    async def handle_stickers_list(self, request):
        root = sticker_mgr.dir if sticker_mgr else Path("data/stickers")
        categories = []
        if root.exists():
            for folder in sorted(root.iterdir()):
                if folder.is_dir():
                    files = [f.name for f in sorted(folder.iterdir())
                             if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}]
                    if files:
                        categories.append({"name": folder.name, "files": files})
        return web.json_response({"root": str(root), "categories": categories,
                                  "enabled": bool(self.config.get("stickers_enabled", False))})

    async def handle_stickers_upload(self, request):
        try:
            reader = await request.multipart()
            category = ""
            saved = []
            async for part in reader:
                if part.name == "category":
                    category = (await part.text()).strip()
                elif part.filename:
                    folder = _safe_subdir(sticker_mgr.dir, category)
                    if folder is None:
                        return web.json_response({"success": False, "error": "分类名非法"}, status=400)
                    folder.mkdir(parents=True, exist_ok=True)
                    fname = Path(part.filename).name
                    if not re.match(r'^[\w\u4e00-\u9fff\-. ]+$', fname):
                        continue
                    (folder / fname).write_bytes(await part.read(decode=False))
                    saved.append(f"{category}/{fname}")
            sticker_mgr.rescan()
            return web.json_response({"success": True, "saved": saved})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_stickers_delete(self, request):
        try:
            payload = await request.json()
            folder = _safe_subdir(sticker_mgr.dir, payload.get("category", ""))
            fname = payload.get("name", "")
            if folder is None or not re.match(r'^[\w\u4e00-\u9fff\-. ]+$', fname):
                return web.json_response({"success": False, "error": "参数非法"}, status=400)
            target = folder / fname
            if target.exists():
                target.unlink()
            sticker_mgr.rescan()
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_stickers_file(self, request):
        category = request.query.get("category", "")
        fname = request.query.get("name", "")
        folder = _safe_subdir(sticker_mgr.dir, category) if sticker_mgr else None
        if folder is None or not re.match(r'^[\w\u4e00-\u9fff\-. ]+$', fname):
            return web.Response(status=404, text="not found")
        target = folder / fname
        if not target.exists():
            return web.Response(status=404, text="not found")
        return web.FileResponse(target)

    async def start(self):
        host = self.config.get("webui_host", "127.0.0.1")
        base_port = int(self.config.get("webui_port", 11500))
        if not HAS_AIOHTTP:
            print("未安装 aiohttp，WebUI 不可用")
            return

        # 尝试多个端口，若被占用则自动递增，最多尝试 10 次
        max_tries = 10
        for attempt in range(max_tries):
            port = base_port + attempt
            try:
                print(f"正在启动 WebUI：http://{host}:{port}")
                runner = web.AppRunner(self.app)
                await runner.setup()
                site = web.TCPSite(runner, host, port)
                await site.start()
                print(f"WebUI 已启动：http://{host}:{port}")
                self.runner = runner
                # 若使用了非默认端口，更新配置
                if port != base_port:
                    self.config.config["webui_port"] = port
                    # 可选：持久化到配置文件
                    with open(self.config.config_path, 'w', encoding='utf-8') as f:
                        json.dump(self.config.config, f, ensure_ascii=False, indent=2)
                return
            except OSError as e:
                # 端口被占用或其他系统错误，尝试下一个端口
                print(f"端口 {port} 不可用（{e}），尝试下一个端口...")
                await runner.cleanup()  # 清理失败的 runner
            except Exception as e:
                error_msg = f"WebUI 启动失败（端口 {port}）：{type(e).__name__}: {e}"
                print(error_msg)
                try:
                    with open("webui_error.log", "a", encoding="utf-8") as f:
                        f.write(f"{time.ctime()} - {error_msg}\n")
                except Exception:
                    pass
                return  # 其他异常不自动切换，直接记录并退出

        # 所有端口尝试失败
        print("错误：无法找到可用端口，WebUI 启动失败。")
        with open("webui_error.log", "a", encoding="utf-8") as f:
            f.write(f"{time.ctime()} - 所有端口被占用，WebUI 启动失败。\n")

    async def shutdown(self):
        if HAS_AIOHTTP:
            await self.app.cleanup()


async def ensure_tts_service_enabled_check() -> bool:
    try:
        from modules.tts_service import check_tts_service
        return await check_tts_service(global_config)
    except Exception:
        return False


# ============================================================================
# 主入口
# ============================================================================

async def main(stop_event: threading.Event = None):
    global global_config, global_emotion_manager, memory_manager
    global db, stats_mgr, sticker_mgr, tool_registry, profile_mgr, rag_mgr
    global todo_mgr, job_mgr, event_mgr, sender
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding='utf-8')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding='utf-8')

    sys.stdout = StdoutRedirector(sys.stdout)
    os.environ["NAP_CAT_PLUGIN_INDEX_URL"] = ""
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"

    print("=" * 100)
    print(
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣛⣩⣤⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣬⣉⠛⠀⠀⠀⠀⠀⢛⣋⣩⣥⠴⠶⠶⠟⠛⠛⠛⠛⠛⠛⠛⠻⠿⠷⠶⢶⣦⣤⣍⣉⡛⠛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣋⣥⣶⠿⣛⣭⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣋⠁⠀⠀⠄⢒⣋⣩⣥⣴⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣦⣭⣍⣛⠻⢷⣶⣤⣍⣙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⣿⣿⣿⣿⣿⣿⠟⣋⣴⡾⢟⣫⣴⠾⣻⣿⣿⣿⣿⠿⠿⠿⠟⠛⠛⠛⠛⠛⠉⠀⠉⣀⣤⣴⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣮⣝⣿⣿⣿⣶⣦⣌⡙⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⠿⠿⠿⠿⢛⣡⡾⢟⣩⣶⠿⠋⠗⣛⣉⣥⣤⠤⠶⣒⣒⣚⡯⠭⣉⡭⠛⢁⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣌⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⣀⣀⢀⡴⠟⣋⣐⣩⡤⢴⣒⣻⣭⣵⣶⠿⢟⣛⡭⠽⠖⠚⠋⠉⣁⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣻⠿⣶⣄⡙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⣫⡥⠖⣚⣩⣵⣶⣾⣿⠿⣿⣛⠭⠖⠚⣉⣩⣤⣶⡶⠟⢋⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣟⣛⣯⣽⣷⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣭⡛⢦⣌⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⣥⣾⣿⣿⠿⣟⡫⠵⠚⣋⣡⣤⣶⣾⣿⡿⠟⠋⠁⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣟⣯⣵⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣌⠳⣤⡉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⢿⣛⠭⠒⣉⢅⣴⣾⣿⣿⣿⣿⠿⠋⠁⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢛⣭⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣻⣽⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣎⠻⣦⡈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⣩⡴⢠⡿⣣⣾⣿⣿⠿⠛⠉⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣻⣵⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣫⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣫⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣮⣝⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣌⢿⣦⡈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿\n"
        "⠿⣱⡟⣵⡿⠟⠋⠁⠀⠀⠀⢀⡤⠂⣴⣿⣿⣿⣿⣿⣿⣿⣿⡿⣛⣵⣿⣿⣿⣿⣿⣿⣿⣿⢟⣿⣿⡿⣋⣴⣿⣿⣿⣿⣿⣿⣿⠟⣫⣾⣿⣿⣿⣿⡿⣫⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣌⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡹⣿⣆⠙⢿⣿⣿⣿⣿⣿⣿⣿\n"
        "⠀⠟⠘⠉⠀⠀⠀⠀⢀⣤⣾⠟⣠⣾⣿⣿⣿⣿⣿⣿⣿⡿⣫⣾⣿⣿⣿⣿⣿⣿⣿⣿⣯⣾⣿⠟⣡⣾⣿⣿⣿⣿⣿⣿⣿⠟⣡⣾⣿⣿⣿⣿⣿⢏⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡌⠻⣿⣿⣿⣿⣿⣿⣿⣿⣷⡌⢻⣷⡈⠛⠛⠛⠛⠛⠻⠿\n"
        "⣇⠀⠀⠀⠀⣀⣴⣾⣿⡿⢃⣴⣿⣿⣿⣿⣿⣿⣿⡿⣫⣾⣿⣿⣿⣿⣿⣿⣿⡿⣫⣾⣿⠟⣡⣾⣿⣿⣿⣿⣿⣿⣿⠟⣡⣾⣿⣿⣿⣿⣿⡟⣱⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡘⢿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠳⠟⢠⡒⢦⠄⣀⣀⣤\n"
        "⣞⣆⢀⣴⣾⣿⣿⣿⠟⢡⣾⣿⣿⣿⣿⣿⣿⣿⣫⣾⣿⣿⣿⣿⣿⣿⣿⡿⣫⣾⣿⠟⣡⣾⣿⣿⣿⣿⣿⣿⣿⡿⡡⣾⣿⣿⣿⣿⣿⣿⢋⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⢻⣿⣿⣿⣿⣿⣿⣿⣿⡄⢠⣦⠙⠎⣰⣷⣿⣿\n"
        "⠿⠜⣄⠻⣿⣿⣿⠏⣰⣿⣿⣻⣿⣿⣿⣿⣟⣵⣿⣿⣿⣿⣿⣿⣿⣿⢫⣾⣿⡿⢋⣾⣿⣿⣿⣿⣿⣿⣿⣿⢏⢴⣾⣿⣿⣿⣿⣿⡿⣱⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⢦⠹⠿⠿⣿⣿⣿⣿⣿⣿⡀⠃⠀⠀⠹⣿⣿⣿\n"
        "⠉⠉⠙⠂⠹⣿⠃⣼⣿⡿⣱⣿⣿⣿⣿⢯⣾⣿⣿⣿⣿⣿⣿⣿⢟⣵⣿⣿⠏⣴⣿⣿⣿⣿⣿⣿⢿⢿⠟⠡⢢⣿⣿⣿⣿⣿⣿⠟⡼⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣿⣿⣿⣿⣿⣿⣿⢎⣴⣾⣷⡹⣿⣿⣿⣿⣿⣧⠀⠀⠀⢠⠘⣿⣿\n"
        "⣦⡀⠀⠀⠀⢀⣼⣿⡿⣱⣿⣿⣿⡿⣳⣿⣿⣿⣿⣿⣿⣿⡿⢫⣾⣿⡿⢡⣾⣿⣿⣿⣿⣿⣿⣿⡿⠃⡴⣱⣿⣿⣿⣿⣿⣿⢏⣞⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣹⡟⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠸⣿⣿⣿⡿⡿⢡⣾⣿⣿⣿⣇⢹⣿⣿⣿⣿⣿⡄⠀⠀⢸⢣⠘⣿\n"
        "⣿⣿⣦⡀⢀⣾⣿⣿⢡⣿⣿⣿⡿⣱⣿⣿⣿⣿⣿⣿⣿⡟⣱⣿⣿⠟⣰⣿⣿⣿⣿⣿⣿⣿⣿⢟⡔⡜⣼⣿⣿⣿⣿⣿⣿⢏⣞⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢃⣿⢁⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⡆⢿⣿⣿⣿⢁⣾⣿⠿⠟⠛⠛⠈⣿⣿⣿⣿⣿⣧⠀⠀⠈⣏⢧⠸\n"
        "⣿⣿⣿⠃⣼⣿⣿⢣⣿⣿⣿⣿⣱⣿⣿⣿⣿⣿⣿⣿⢏⣼⣿⣿⠏⣼⣿⣿⣿⣿⣿⣿⣿⣿⢃⠞⢜⣾⣿⣿⣿⣿⣿⣿⢏⡞⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣼⠃⢸⣿⣿⣿⣿⣿⣿⣿⡟⣾⣿⣿⣿⣿⣿⡇⢸⣿⣿⡏⢸⢿⣧⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⠀⠀⠀⠸⡌⢧\n"
        "⠻⣿⠃⣼⣿⣿⢇⣾⣿⣿⣿⢳⣿⣿⣿⣿⣿⣿⣿⢋⣾⣿⣿⢋⣾⣿⣿⣿⣿⣿⣿⣿⡿⢡⡏⢌⣾⣿⣿⣿⣿⣿⣿⢏⡞⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢰⡟⠀⣾⣟⢿⣿⣿⣿⣿⣿⢃⣿⣽⣿⣿⣿⣿⡇⢸⣿⣿⡇⠀⠀⢀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⣿⡇⠀⠀⣆⠗⢋\n"
        "⣷⠆⣸⣿⣿⡟⣼⣿⣿⣿⢧⣿⣿⣿⣿⣿⣿⡿⠃⠞⠛⠻⠁⠘⠛⠿⠿⣿⣿⣿⣿⡿⣱⡟⢈⣾⣿⣿⣿⣿⣿⣿⡏⡼⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢃⡿⡡⢸⣿⣿⣷⣿⡻⣿⣿⡟⣸⣧⣿⣿⣿⣿⣿⡇⢸⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⡇⠀⣠⠴⠚⠉\n"
        "⡟⢠⣿⣿⣿⢱⣿⣿⣿⡟⣾⣿⣿⣿⣿⣿⣦⢀⣀⠀⠠⠁⠀⠀⠀⠀⠀⠀⠉⠛⠿⣱⣿⢁⣾⣿⣿⣿⣿⣿⣿⡟⣸⢳⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣾⢣⢇⣿⣿⣿⣿⣿⣿⣿⣿⢡⡿⣼⣿⣿⣿⣿⣿⡇⣼⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀\n"
        "⠀⣿⣿⣿⠇⣿⣿⣿⣿⢱⣿⣿⣿⣿⣿⣿⢣⣿⣿⣿⠀⣀⣁⢤⣤⣄⣀⡀⠀⠀⠀⠈⠁⢼⣿⣿⣿⣿⣿⣿⣿⢡⡏⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣸⠏⡞⣸⣿⣿⣿⣿⣿⣿⣿⠇⣾⢳⣿⣿⣿⣿⣿⣿⠃⣿⣿⣿⡟⠂⠀⠀⠀⠀⠀⠀⠀⠀⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⣸⣿⣿⡿⣸⣿⣿⣿⡇⣾⣿⣿⣿⣿⣿⢏⣾⣿⣿⡇⢠⣿⣿⣷⣮⣝⡻⠿⠋⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⠇⡾⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣱⡟⣼⢣⣿⣿⣿⣿⣿⣿⣿⡟⣰⡏⣿⣿⣿⣿⣿⣿⣿⢠⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠊⠀⠀⠀⠀⠀⠀⠀⡀⢀⣼\n"
        "⣿⡿⣿⠇⣿⣿⣿⣿⢠⣿⣿⣿⣿⣿⡟⣾⣿⣿⣿⠁⣼⣿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⡟⢰⣇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢣⡿⣰⡏⣼⣿⣿⣿⣿⣿⣿⡟⣰⡿⣽⣿⣿⣿⣿⣿⣿⡇⢸⣿⢸⡃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⠐⢈⣴⡿⢋\n"
        "⣿⢻⣿⢸⣿⣿⣿⡿⢸⣿⣿⣿⣿⣿⣹⣿⣿⣿⡏⠀⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⡇⣾⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢯⡿⢡⣿⢳⣿⣿⣿⣿⣿⣿⡿⣰⣿⢳⣿⣿⣿⣿⣿⣿⡿⠀⣾⡇⣿⡇⢠⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠉⠉⠀⠀⠉⠉⠀⢻\n"
        "⡏⣿⡇⣾⣿⣿⣿⡇⣼⣿⣿⣿⣿⢯⣿⣿⣿⣿⢡⣿⣿⣿⣿⣿⣿⡟⢀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⡇⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢏⣾⢣⣿⣗⣾⣿⣿⣿⣿⣿⡟⣱⣿⢯⣿⣿⣿⣿⣿⣿⣿⢡⠂⣿⢰⣿⣿⣆⠻⣿⣦⠒⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘\n"
        "⢹⣿⢃⣿⣿⣿⣿⡇⣿⣿⣿⣿⡏⣸⣿⣿⣿⡿⢸⣿⣿⣿⣿⣿⣿⣧⣿⣷⡀⠀⠀⠀⠀⠀⠀⣶⣦⢀⢠⣷⣧⣿⣿⣿⣿⣿⣿⣿⣿⣿⢏⣾⢣⣿⡟⢸⣿⣿⠿⠿⠿⠟⠘⠛⠟⠿⠿⣿⣿⣿⣿⣿⢃⣿⢸⡇⣾⣿⣿⣿⡗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣆⠀⠀⠀⠀\n"
        "⣾⣿⢸⣿⣿⣿⣿⡇⣿⣿⣿⣿⡇⣿⣿⣿⣿⡇⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠈⠁⢸⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⢏⡾⣣⣿⠟⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠃⠺⠇⡿⢰⣿⣿⣿⡏⠀⠀⠀⠀⠀⠀⢀⢀⣠⡀⣀⢒⡉⠀⣿⣿⠀⠀⠀⠀\n"
        "⣿⡟⢸⣿⣿⣿⣿⡇⢿⣿⣿⣿⡧⡝⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣸⣿⣿⣿⣿⣿⣿⣿⢏⡾⣵⠟⠁⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣀⡀⠀⠀⠀⠀⠀⠰⠁⢿⣿⣿⡿⠀⢿⡴⢚⣡⡞⠿⠺⡏⢸⡇⢸⣿⠁⡆⠸⣿⡇⠀⠀⠀\n"
        "⣿⡇⣿⣿⣿⣿⣿⣧⢸⣿⣿⣿⡇⣿⡌⢿⣿⡇⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢋⣿⣾⣿⣿⡿⠁⠀⡀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⢿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠙⠿⠁⠀⢋⣴⣿⢰⣶⢼⡶⢻⡼⢃⣾⡇⢸⣧⠠⠻⠷⠀⠀⠀\n"
        "⣿⡇⣿⣿⣿⣿⣿⣿⠸⡿⠟⣻⣧⢻⣿⠀⡹⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⣠⣴⣤⣀⡀⠀⣀⠀⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠻⡿⠂⢸⣿⣿⣿⣿⣷⠄⡀⠀⠀⠀⠀⠑⣾⣿⣿⢟⣕⢲⢇⣼⡈⠇⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⣿⡇⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⡈⢿⡀⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣆⠙⣿⣿⣿⣿⡇⣴⣄⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⡟⣰⣿⣦⠐⠀⠀⠀⠘⣿⣿⢬⢋⡞⣨⢫⢷⣄⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡕⣌⢧⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣭⣿⣿⣧⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⣿⡿⣱⣿⣿⠃⣠⣾⣷⣶⣦⣽⣇⠿⡺⣱⣏⠺⢗⣿⠃⡤⢤⣤⡄⢶⣦⠰⣶⣄⠀\n"
        "⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠈⠈⡋⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠈⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⡟⣱⣿⣿⠃⣴⣿⣿⣿⣿⣿⣿⣫⣾⣱⣿⣿⣯⣼⣧⢰⣧⢸⣿⣿⡄⠻⣷⡘⢿⡄\n"
        "⣿⡇⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢀⠀⢷⣮⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⢟⣼⣿⠟⢡⣾⣿⣿⣿⣿⣿⡿⢃⢜⡱⣿⣿⣿⣷⠎⣠⣏⢻⡄⢿⣿⣷⡐⢌⡛⢮⡳\n"
        "⣿⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠈⢄⠈⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠠⣾⣿⣿⣶⣶⣦⠐⣂⠀⠀⣠⣾⣿⣿⢯⣟⣫⢅⣴⣿⣿⣿⣿⣿⣿⠟⣱⠏⡹⣛⣿⣿⣿⡏⢠⣝⡋⣚⡻⡘⣿⣿⣷⡘⢿⣶⣤\n"
        "⣿⣷⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠃⠠⠀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⠞⣿⣧⣾⣿⣿⣿⣿⣿⠟⣡⣾⣿⣿⣿⣿⣿⡿⢋⣾⢫⣾⢵⣯⣿⣿⡟⢠⣿⠟⢞⡿⡃⣳⠘⣿⣿⣷⡈⢿⣿\n"
        "⣿⣿⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠘⠀⠀⠃⢀⠈⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢟⣡⣾⣿⣿⣿⣿⣿⡿⢋⣴⢟⣵⣿⣿⡖⣤⡿⡟⢀⣿⣿⣷⣾⣿⣜⠿⡣⣘⡻⣿⣿⣄⠙\n"
        "⣿⣿⡆⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⢡⠀⠀⠀⠁⠀⠀⠉⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡵⣿⣿⣿⣿⣿⣿⡿⢋⣴⠟⣱⣿⣿⣿⣿⣧⡟⡟⠀⠀⣿⣿⣿⣿⣏⣹⣿⣜⠿⣇⣩⣝⢿⣦\n"
        "⣿⣿⣧⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡈⠀⠀⠀⠀⠄⢀⣤⣶⣄⡈⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡷⠂⣠⣴⡤⣩⡴⢛⣥⣾⣿⣿⣿⣿⣿⣏⡸⡿⢂⠀⣿⣿⣿⣿⣿⣿⣿⣿⣏⣡⣙⣋⢸⣶\n"
        "⣿⣿⣿⡀⡘⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⢀⠂⣠⣿⣿⣿⣿⣿⣷⢠⡄⠉⠛⠿⣿⣿⣿⣿⣿⣭⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⣠⡾⠟⠵⣊⣥⣾⣿⣿⣿⣿⣿⣿⣿⢯⡟⠀⢴⣶⡄⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣭\n"
        "⣿⣿⣿⣧⠘⣢⡙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠈⢰⣿⣿⣿⡏⣿⣿⣿⢸⠁⠀⠀⠀⠀⠈⠙⠛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠋⢀⣤⣥⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣳⠏⢀⠂⠈⢉⣬⡀⠙⢿⣿⠿⠿⠛⠛⠻⣿⣿⣿⣿⣿\n"
        "⢻⣿⣿⣿⣆⠩⢧⠑⠨⣙⠻⢿⣿⣿⣿⣿⣿⣿⣷⡄⢿⣿⣿⣿⢸⣿⣿⣿⠘⠀⠀⠀⠀⠀⠀⠀⠀⣤⣤⣤⣄⣉⣉⡙⠛⠛⠛⠛⠿⠿⠿⠿⠿⠿⠟⠛⠛⠛⠋⠉⠉⠀⢀⣴⣿⡿⣫⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⣽⠏⠀⠀⠀⡀⠌⠛⢃⣁⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣿⣿\n"
        "⣌⠻⠿⣿⣿⣆⠩⣧⠀⠀⠁⠂⢬⠉⠛⠿⢿⣿⣿⣿⣎⠻⣿⡇⡾⠋⠙⢿⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠐⣰⣿⣿⣿⢖⣴⣿⡿⣫⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣫⣾⠏⠀⠐⠂⠁⠀⠀⠀⠙⠟⢁⣀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿\n"
        "⣿⣿⣷⣶⣭⣍⣃⠈⢷⡀⠄⣂⣴⣶⣦⣑⠲⢠⠈⣭⣍⣓⡙⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⡿⢋⣵⣿⢟⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢟⣵⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠟⢁⣤⡀⠀⠀⠀⠀⠈⣉⡛\n"
        "⣿⣿⣿⣿⣿⣿⣿⣦⡀⠋⣾⣿⣿⣿⣿⠿⠃⣉⡀⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠿⣋⣴⢟⢏⣴⣿⡿⣫⣿⣿⣿⣿⣿⣿⣿⣿⡿⣫⣾⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠃⢴⣶⠀⣠⣄⠉⣁\n"
        "⣿⣿⣿⣿⣿⣿⣿⣿⡿⣂⣽⣿⣷⡍⣥⣚⡛⠿⠇⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢘⡥⢞⣫⢔⣵⣿⢟⣭⣾⣿⣿⣿⣿⣿⣿⣿⣿⢋⣾⣿⡿⢃⣶⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠙⠋⠀⠻\n"
        "⠻⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣿⠀⣿⣿⣿⣿⣶⣍⡛⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡾⢽⡾⣋⣴⠿⣫⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⢟⣵⣿⣿⡿⠡⢿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⢷⣬⡛⢿⣿⣿⣿⣿⡎⢿⣿⣿⣿⡄⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⢀⡴⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⢞⣫⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢟⣵⣿⣿⣿⡟⣱⣿⣷⡝⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠙⠻⢶⣬⡙⠛⠉⠀⠀⠈⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⢏⣴⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠦⣄⡀⢠⣾⣷⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢛⣵⣿⣿⣿⣿⠟⣰⣿⣿⣿⣷⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"

        "\n\n                                                            启动成功啦！                                                              "
    )
    print("=" * 100)

    global_config = ConfigLoader()
    global_emotion_manager = EmotionManager(global_config)
    if not global_emotion_manager.emotions:
        print("\n[警告] 未找到任何情绪配置（请检查 ref_audio_root 目录），将降级为纯文本模式。")

    memory_manager = MemoryManager(global_config)

    # 初始化功能模块
    db = DatabaseManager(memory_manager.data_path)
    stats_mgr = StatsManager(db)
    sticker_mgr = StickerManager(global_config)
    tool_registry = ToolRegistry(global_config, memory_manager.data_path)
    profile_mgr = UserProfileManager(global_config, memory_manager.data_path)
    rag_mgr = RAGManager(global_config, memory_manager.data_path)
    sender = MessageSender(global_config, memory_manager, sticker_mgr, stats_mgr)
    todo_mgr = TodoManager(global_config, db, scheduler, sender,
                           emotions_provider=get_active_emotions)
    todo_mgr.ctx_provider = get_active_ctx
    todo_mgr.restore_pending()
    job_mgr = ScheduledJobManager(global_config, memory_manager.data_path, scheduler,
                                  sender, get_active_ctx, get_active_emotions)
    event_mgr = EventManager(global_config, memory_manager.data_path, profile_mgr)

    # 启动调度器与功能任务
    scheduler.start()
    register_feature_jobs()
    job_mgr.register_all()

    if global_config.get("auto_start_tts", False):
        threading.Thread(target=auto_start_and_switch_tts, args=(global_config,), daemon=True).start()

    webui_server = None
    if HAS_AIOHTTP and global_config.get("webui_enabled", True):
        webui_server = WebUIServer(global_config, memory_manager)
        try:
            await webui_server.start()
        except Exception as e:
            print(f"WebUI 启动异常，继续运行其他功能：{e}")

    ws_url = global_config.get("napcat_ws_url", "ws://127.0.0.1:3001")
    token = global_config.get("napcat_token", "")

    print(f"正在连接 NapCat ({ws_url})...")

    while True:
        if stop_event is not None and stop_event.is_set():
            print("收到停止信号，正在退出消息循环...")
            break

        try:
            client = NapCatClient(ws_url=ws_url, token=token)
            async with client:
                sender.client = client  # 注入统一发送器，供消息回复与主动消息使用
                print(f"已连接！机器人 QQ: {client.self_id}")
                print("等待消息中...")
                async for event in client:
                    if stop_event is not None and stop_event.is_set():
                        print("收到停止信号，正在退出消息循环...")
                        break
                    try:
                        await handle_message_event(event, client)
                    except Exception as e:
                        print(f"处理消息异常: {type(e).__name__}: {e}")
                # 连接被服务端正常关闭（非异常路径）：稍候重连，避免紧密循环
                if stop_event is None or not stop_event.is_set():
                    print("连接已断开，3秒后重连...")
                    await asyncio.sleep(3)

        except Exception as e:
            print(f"NapCat 连接失败: {e}")
            print("10秒后尝试重新连接...")
            await asyncio.sleep(10)
            continue
        finally:
            sender.client = None  # 连接断开后置空，避免主动消息/定时任务使用失效连接

    print("正在关闭所有子进程...")
    process_manager.shutdown_all()
    await scheduler.stop()
    if db is not None:
        db.close()
    if webui_server:
        await webui_server.shutdown()


if __name__ == "__main__":
    import threading
    import time

    stop_event = threading.Event()
    config = ConfigLoader()

    def run_backend():
        try:
            asyncio.run(main(stop_event))
        except Exception as e:
            print(f"后台服务异常: {e}")
            import traceback
            traceback.print_exc()
            try:
                with open("backend_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.ctime()} - 异常: {e}\n")
                    traceback.print_exc(file=f)
            except Exception:
                pass

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    time.sleep(1)

    try:
        if HAS_WEBVIEW:
            import webview
            webui_port = int(config.get("webui_port", 11500))
            webview.create_window(
                'LTVM 控制台',
                f'http://127.0.0.1:{webui_port}',
                width=1920,
                height=1080,
                resizable=True,
                maximized=True
            )
            webview.start()
        else:
            print("程序正在运行，按 Ctrl+C 退出...")
            while not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到键盘中断，准备退出...")
    finally:
        print("正在停止后台服务...")
        stop_event.set()
        backend_thread.join(timeout=10)
        process_manager.shutdown_all()
        print("程序已完全退出。")
