import asyncio
import os
import time
import json
import wave
import re
import hashlib
import threading
import subprocess
import base64
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import httpx
import numpy as np
import logging
try:
    import webview
    HAS_WEBVIEW = True
except ImportError:
    HAS_WEBVIEW = False
    print("警告：未安装 pywebview，将使用浏览器访问。可运行 pip install pywebview 启用。")

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent
    return base_path / relative_path

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
            # 尝试写入原始流，失败则忽略
            if self.original_stream is not None:
                try:
                    self.original_stream.write(message)
                    self.original_stream.flush()
                except Exception:
                    pass  # 无控制台时忽略写入错误

            # 始终保存到日志缓冲区
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

class ProcessManager:
    _instance = None
    _lock = threading.Lock()
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.processes = []
        return cls._instance
    def register(self, proc, name=""):
        if proc is not None:
            self.processes.append({"proc": proc, "name": name})
            print(f"已注册子进程: {name or 'unnamed'} (PID: {proc.pid})")
    def shutdown_all(self):
        for entry in self.processes:
            proc = entry["proc"]
            name = entry["name"]
            try:
                if os.name == 'nt':
                    # 使用 taskkill /F /T 结束整个进程树
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)], capture_output=True)
                else:
                    proc.terminate()
                    proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception as e:
                    print(f"关闭子进程 {name} 失败: {e}")
        self.processes.clear()

process_manager = ProcessManager()

global_config = None
global_emotion_manager = None

class ConfigLoader:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_or_init()
        # 多角色配置解析
        self.active_character = self.config.get("active_character", self.config.get("character_key", "murasame"))
        self.roles = self._parse_roles()

    def _parse_roles(self) -> dict:
        """解析多角色配置，将旧版单角色配置迁移为角色列表"""
        roles = {}
        # 如果已有 roles 配置则直接使用
        if "roles" in self.config and isinstance(self.config["roles"], list):
            roles_config = self.config["roles"]
        else:
            # 兼容旧版单角色配置，构造一个默认角色
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
        # 确保默认角色存在
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
        # 简化的交互初始化
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
        
        # 配置角色列表
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
        # 完整包含 _conf_schema.json 中的所有字段（除已删除的三个）
        return {
            "hide_gsv_options": False,
            "llm_model_name": "",
            "image_caption_model_name": "",
            "llm_base_url": "http://127.0.0.1:11434",
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
            "json_prompt": "【输出格式】你必须严格只返回一个紧凑的JSON对象，格式为：{\"sentences\": [{\"zh\": \"这里是你生成的中文台词\", \"ja\": \"这里是你生成的日语台词\", \"emotion\": \"这里是你判断的情绪\"}, {\"zh\": \"第二句中文\", \"ja\": \"第二句日语\", \"emotion\": \"另一种情绪\"}]}等更多情绪均可。【最终输出规则】最终输出必须严格只包含JSON对象，绝对禁止输出任何思考过程、解释、非JSON文本或Markdown代码块。所有的推理和思考都只能在内部进行，最终回复只能是JSON格式。",
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
                    "json_prompt": "【输出格式】你必须严格只返回一个紧凑的JSON对象，格式为：{\"sentences\": [{\"zh\": \"这里是你生成的中文台词\", \"ja\": \"这里是你生成的日语台词\", \"emotion\": \"这里是你判断的情绪\"}, {\"zh\": \"第二句中文\", \"ja\": \"第二句日语\", \"emotion\": \"另一种情绪\"}]}等更多情绪均可。【最终输出规则】最终输出必须严格只包含JSON对象，绝对禁止输出任何思考过程、解释、非JSON文本或Markdown代码块。所有的推理和思考都只能在内部进行，最终回复只能是JSON格式。",
                    "supplement_prompt": "回答自然、简短，通常两到五句话(一个句号才算一句话)；不要重复最近说过的话，不要加入动作、旁白或括号舞台说明；生成的回复要符合当前对话，不能出现主谓宾不分，乱序的情况。【情绪判断规则】请仔细阅读最近对话历史，结合你（角色）的性格特点来判断情绪！如果主人对你亲昵（如摸头、夸奖），即使你嘴上说“我才没有”，情绪也应该是害羞或高兴；如果主人故意逗你、骂你或惹你生气，情绪应该是生气或着急；如果只是平淡陈述，使用平静。【翻译一致性要求】必须表达完全相同的含义和语气，绝对不能出现含义相反或意思不匹配的翻译！【情绪连贯性强制规则】如果用户明确地侮辱、挑衅或激怒你（例如叫你“幼刀、搓衣板、飞机场”），你的情绪必须保持连贯。即：整句话所有分句的情绪必须都是“生气”或“着急”，绝对不能把后半句的“命令/威胁”改成“害羞”或“高兴”！除非你明确使用了“但是”、“不过”等转折词，否则不要轻易切换成其他情绪。【情绪匹配规则】情绪文件夹可能是拼音（如 gaoxing），也可能是英文（如 happy）。你必须严格只输出我在【情绪可选列表】中提供的单词，绝对不能输出中文汉字或拼音简写！",
                    "default_voice": "pingjing",
                    "ref_audio_root": "",
                    "text_lang": "ja"
                }
            ]
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

def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        pass
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except:
        pass
    start_indices = [i for i, char in enumerate(cleaned) if char == '{']
    for start in reversed(start_indices):
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
                            return json.loads(cleaned[start:i+1])
                        except:
                            break
    return None

def get_audio_duration(file_path: str) -> float:
    try:
        import wave
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / rate
    except Exception:
        pass
    return 1.0

def resolve_tts_path(input_path: str) -> str:
    if not input_path:
        return "C:/tts"
    input_path = input_path.strip().replace("/", "\\").rstrip("\\")
    if len(input_path) <= 3 and input_path[1] == ":":
        target_dir = f"{input_path}\\tts"
    else:
        target_dir = input_path
    try:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        target_dir = "C:/tts"
    return target_dir

class EmotionManager:
    def __init__(self, config: ConfigLoader):
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
                        except:
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
        if not self.config.get("enable_default_emotions", True) and self.config.get("emotions_config", []):
            self.emotions = {}
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
    def load_history(self, session_id: str) -> list:
        file_path = self.get_memory_file(session_id)
        if file_path.exists():
            try:
                data = json.loads(file_path.read_text(encoding='utf-8'))
                return data.get("history", [])
            except:
                return []
        return []
    def save_history(self, session_id: str, history: list):
        file_path = self.get_memory_file(session_id)
        history = history[-60:]
        data = {"character_name": self.character_name, "history": history}
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
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
            for f in self.data_path.glob("*_*.json"):
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

async def get_llm_reply(config: ConfigLoader, user_text: str, history: list, emotions: dict, images: list = None):
    if images:
        return await get_image_reply(config, user_text, history, emotions, images)
    emotion_keys = list(emotions.keys())
    system_prompt = (
        config.get("personality_prompt", "") + "\n" +
        config.get("json_prompt", "") + "\n" +
        config.get("supplement_prompt", "") + "\n" +
        f"【情绪可选列表】{', '.join(emotion_keys)}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    history_length = config.get("history_length", 8)
    history_data = history[-history_length:]
    merged_history = []
    for msg in history_data:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role not in ["user", "assistant"]:
            continue
        if role == "user" and msg.get("sender_id"):
            content = f"[用户ID:{msg['sender_id']}] {content}"
        if merged_history and merged_history[-1]["role"] == role:
            merged_history[-1]["content"] += "\n" + content
        else:
            merged_history.append({"role": role, "content": content})
    messages.extend(merged_history)
    task_context = ""
    task_keywords = ["提醒", "记住", "要求", "命令", "叫我", "以后", "别忘"]
    for msg in reversed(history_data):
        if msg.get("role") == "user":
            content = str(msg.get("content", ""))
            if any(kw in content for kw in task_keywords):
                task_context = content
                break
    if task_context:
        messages.append({"role": "user", "content": f"（重申之前的指令）{task_context}"})
    messages.append({"role": "user", "content": user_text})
    backend = config.get("llm_backend", "ollama")
    base_url = config.get("llm_base_url", "http://127.0.0.1:11434")
    model = config.get("llm_model_name", "")
    timeout = config.get("llm_timeout", 120)
    enable_think = config.get("enable_think", False)
    payload = {"model": model, "messages": messages, "stream": False}
    if backend == "ollama":
        payload["options"] = {
            "num_ctx": config.get("num_ctx", 8192),
            "temperature": min(1.0, float(config.get("temperature", 1.2))),
        }
        payload["think"] = enable_think
        endpoint = f"{base_url}/api/chat"
        headers = {}
    else:
        payload["temperature"] = min(1.0, float(config.get("temperature", 1.0)))
        api_key = config.get("llm_api_key", "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        endpoint = f"{base_url}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if backend == "ollama":
                content = data.get("message", {}).get("content", "")
                if enable_think and data.get("message", {}).get("thinking"):
                    print(f"【模型思考】: {data['message']['thinking']}")
            else:
                content = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM 调用失败: {type(e).__name__}: {e}")
        default_text = "啊嘞，刚才走神了，请稍后重试。"
        return default_text, [default_text], [config.get("default_voice", "pingjing")], [default_text], [default_text]
    json_obj = extract_json(content)
    if not json_obj:
        print(f"LLM 输出格式错误，无法解析 JSON。原始内容：{content[:200]}")
        if content.strip():
            zh_text = content.strip()
        else:
            zh_text = "出错了。"
        lang_text = zh_text
        emotion = config.get("default_voice", "pingjing")
        return zh_text, [lang_text], [emotion], [zh_text], [lang_text]
    data = json_obj
    sentences = data.get("sentences", [])
    if not sentences:
        text_lang = config.get("text_lang", "ja")
        sentences = [{
            "zh": data.get("zh", user_text),
            text_lang: data.get(text_lang, user_text),
            "emotion": data.get("emotion", config.get("default_voice", "pingjing"))
        }]
    zh_list, lang_list, emo_list, display_list = [], [], [], []
    text_lang = config.get("text_lang", "ja")
    display_lang = config.get("display_lang", "zh")
    for s in sentences:
        zh_text_cur = str(s.get("zh", "")).strip()
        lang_text_cur = str(s.get(text_lang, "")).strip()
        if not zh_text_cur:
            zh_text_cur = lang_text_cur or user_text
        if not lang_text_cur:
            lang_text_cur = zh_text_cur
        zh_list.append(zh_text_cur)
        lang_list.append(lang_text_cur)
        if display_lang == "auto":
            display_cur = lang_text_cur if lang_text_cur else zh_text_cur
        else:
            display_cur = str(s.get(display_lang, "")).strip() or zh_text_cur
        display_list.append(display_cur)
        emo = s.get("emotion", config.get("default_voice", "pingjing"))
        if emo not in emotions:
            emo = config.get("default_voice", "pingjing")
        emo_list.append(emo)
    return "".join(zh_list), lang_list, emo_list, display_list, lang_list

async def get_image_reply(config: ConfigLoader, user_text: str, history: list, emotions: dict, image_urls: list):
    try:
        emotion_keys = list(emotions.keys())
        system_content = f"【情绪可选列表】{', '.join(emotion_keys)}"
        personality = config.get("personality_prompt", "")
        json_prompt = config.get("json_prompt", "")
        supplement = config.get("supplement_prompt", "")
        prompt_text = f"用户发来了一张图片，请仔细观察图片内容，结合你的角色人设：{personality}；{json_prompt}；{supplement}，根据图片内容回复（可以是吐槽、评价、撒娇等）。\n当前对话历史：{json.dumps(history[-10:], ensure_ascii=False)}\n用户附加文字：{user_text}"
        images_for_payload = []
        for img_path in image_urls:
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        b64_data = base64.b64encode(f.read()).decode('utf-8')
                    images_for_payload.append(b64_data)
                except Exception as e:
                    print(f"读取本地图片失败: {e}")
            elif img_path.startswith("http"):
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    }
                    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
                        resp = await client.get(img_path)
                        resp.raise_for_status()
                        b64_data = base64.b64encode(resp.content).decode('utf-8')
                    images_for_payload.append(b64_data)
                except Exception as e:
                    print(f"下载图片失败: {e}")
            else:
                print(f"未知图片路径格式: {img_path}")
        if not images_for_payload:
            print("没有有效的图片数据，使用默认回复")
            default_text = "啊嘞，看不清这张图呢。"
            return default_text, [default_text], [config.get("default_voice", "pingjing")], [default_text], [default_text]
        backend = config.get("llm_backend", "ollama")
        base_url = config.get("llm_base_url", "http://127.0.0.1:11434")
        model = config.get("image_caption_model_name", "")
        if not model:
            print("未配置识图模型名称，无法处理图片")
            return None, None, None, None, None
        timeout = config.get("image_caption_timeout", 90)
        if backend == "ollama":
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt_text, "images": images_for_payload}
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": float(config.get("temperature", 0.7)), "num_predict": 512}
            }
            endpoint = f"{base_url}/api/chat"
            headers = {}
        else:
            content_parts = []
            for img_b64 in images_for_payload:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            content_parts.append({"type": "text", "text": prompt_text})
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": content_parts}
                ],
                "stream": False,
                "temperature": float(config.get("temperature", 0.7)),
                "max_tokens": 512
            }
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            endpoint = f"{base_url}/chat/completions"
            api_key = config.get("llm_api_key", "")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if backend == "ollama":
                content = data.get("message", {}).get("content", "")
            else:
                content = data["choices"][0]["message"]["content"]
        json_obj = extract_json(content)
        if not json_obj:
            default_text = "看到图片啦，不过还没想好说什么呢～"
            return default_text, [default_text], [config.get("default_voice", "pingjing")], [default_text], [default_text]
        data = json_obj
        sentences = data.get("sentences", [])
        if not sentences:
            text_lang = config.get("text_lang", "ja")
            sentences = [{"zh": data.get("zh", "嗯嗯"), text_lang: data.get(text_lang, "嗯嗯"), "emotion": data.get("emotion", config.get("default_voice", "pingjing"))}]
        zh_list, lang_list, emo_list, display_list = [], [], [], []
        text_lang = config.get("text_lang", "ja")
        display_lang = config.get("display_lang", "zh")
        for s in sentences:
            zh_text_cur = str(s.get("zh", "")).strip()
            lang_text_cur = str(s.get(text_lang, "")).strip()
            zh_list.append(zh_text_cur)
            lang_list.append(lang_text_cur)
            display_cur = str(s.get(display_lang, "")).strip() or zh_text_cur
            display_list.append(display_cur)
            emo = s.get("emotion", config.get("default_voice", "pingjing"))
            if emo not in emotions:
                emo = config.get("default_voice", "pingjing")
            emo_list.append(emo)
        return "".join(zh_list), lang_list, emo_list, display_list, lang_list
    except Exception as e:
        print(f"识图模型处理失败: {e}")
        return None, None, None, None, None

_tts_restart_lock = threading.Lock()

async def check_tts_service(config: ConfigLoader) -> bool:
    base_url = config.get("client_base_url", "http://127.0.0.1:9880")
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get(f"{base_url}/docs")
            return resp.status_code < 500
    except Exception:
        return False

async def ensure_tts_service(config: ConfigLoader) -> bool:
    if await check_tts_service(config):
        return True
    if not config.get("auto_start_tts", False):
        return False
    threading.Thread(target=auto_start_and_switch_tts, args=(config,), daemon=True).start()
    for _ in range(12):
        await asyncio.sleep(5)
        if await check_tts_service(config):
            return True
    return False

async def synthesize_sentence(config: ConfigLoader, text: str, emotion: str, emotions: dict, data_path: Path) -> Optional[Path]:
    emotion_data = emotions.get(emotion, emotions.get(config.get("default_voice", "pingjing")))
    if not emotion_data:
        print(f"找不到情绪配置: {emotion}")
        return None
    ref_path = emotion_data["ref_path"]
    prompt_text = emotion_data["prompt_text"]
    clean_text = re.sub(r'^[\s。，！？、,.!?…～~]+$', '', text)
    if not clean_text:
        print(f"检测到纯标点或空句子，已跳过 TTS 合成: '{text}'")
        return None
    params = {
        "text": clean_text,
        "text_lang": config.get("text_lang", "ja"),
        "ref_audio_path": ref_path,
        "prompt_text": prompt_text,
        "prompt_lang": config.get("prompt_lang", "ja"),
        "device": config.get("device", "cuda"),
        "top_k": config.get("top_k", 20),
        "top_p": config.get("top_p", 1),
        "temperature": config.get("temperature", 1),
        "text_split_method": config.get("text_split_method", "cut1"),
        "batch_size": config.get("batch_size", 1),
        "batch_threshold": config.get("batch_threshold", 1),
        "split_bucket": config.get("split_bucket", True),
        "speed_factor": config.get("speed_factor", 1.0),
        "fragment_interval": config.get("fragment_interval", 0.5),
        "streaming_mode": config.get("streaming_mode", False),
        "seed": config.get("seed", -1),
        "parallel_infer": config.get("parallel_infer", True),
        "repetition_penalty": config.get("repetition_penalty", 1.35),
        "media_type": config.get("media_type", "wav")
    }
    base_url = config.get("client_base_url", "http://127.0.0.1:9880")
    timeout = config.get("timeout_seconds", 120)
    max_retries = 3
    retry_delay = 1.0
    for attempt in range(max_retries):
        try:
            print(f"正在合成: 情绪={emotion} | 文本={clean_text} (尝试 {attempt+1}/{max_retries})")
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{base_url}/tts", params=params)
                if resp.status_code == 200:
                    temp_path = data_path / f"temp_{emotion}_{int(time.time()*1000)}.wav"
                    temp_path.write_bytes(resp.content)
                    print(f"合成完成: {emotion} | {clean_text[:30]}...")
                    return temp_path
                else:
                    print(f"TTS 合成失败: {resp.status_code} - {resp.text} | 文本={clean_text}")
                    return None
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            print(f"TTS 连接异常 ({type(e).__name__})，等待 {retry_delay} 秒后重试...")
            await asyncio.sleep(retry_delay)
            retry_delay += 1.0
            if attempt == max_retries - 1:
                await ensure_tts_service(config)
        except Exception as e:
            print(f"TTS 连接异常 ({e})，等待 {retry_delay} 秒后重试...")
            await asyncio.sleep(retry_delay)
            retry_delay += 1.0
    print(f"TTS 合成在 {max_retries} 次尝试后仍失败: {clean_text}")
    return None

def merge_wavs(wav_paths: list, config: ConfigLoader, data_path: Path) -> Optional[Path]:
    if not wav_paths:
        return None
    output_path = data_path / f"combined_{int(time.time() * 1000)}.wav"
    if not config.get("voice_transition", True):
        try:
            data = []
            for wav_path in wav_paths:
                with wave.open(str(wav_path), 'rb') as wf:
                    data.append([wf.getparams(), wf.readframes(wf.getnframes())])
            with wave.open(str(output_path), 'wb') as out:
                out.setparams(data[0][0])
                for params, frames in data:
                    out.writeframes(frames)
            return output_path
        except Exception as e:
            print(f"合并音频失败: {e}")
            return None
    try:
        import numpy as np
        with wave.open(str(wav_paths[0]), 'rb') as wf:
            params = wf.getparams()
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            all_frames = wf.readframes(wf.getnframes())
        all_audio = np.frombuffer(all_frames, dtype=np.int16).copy().reshape(-1, n_channels)
        breathing_gap_ms = config.get("breathing_gap_ms", 100)
        breathing_gap_samples = int(sample_rate * breathing_gap_ms / 1000)
        crossfade_ms = config.get("crossfade_ms", 300)
        crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        for i in range(1, len(wav_paths)):
            with wave.open(str(wav_paths[i]), 'rb') as wf:
                if wf.getframerate() != sample_rate or wf.getnchannels() != n_channels:
                    print("检测到不同采样率的音频，已跳过渐变处理。")
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).copy().reshape(-1, n_channels)
                    all_audio = np.concatenate((all_audio, audio), axis=0)
                    continue
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).copy().reshape(-1, n_channels)
            breathing_gap = np.zeros((breathing_gap_samples, n_channels), dtype=np.int16)
            all_audio = np.concatenate((all_audio, breathing_gap), axis=0)
            if len(audio) < crossfade_samples:
                all_audio = np.concatenate((all_audio, audio), axis=0)
                continue
            fade_out = all_audio[-crossfade_samples:].astype(np.float32)
            fade_in = audio[:crossfade_samples].astype(np.float32)
            fade_in_gradient = (1 - np.cos(np.linspace(0, np.pi, crossfade_samples))) / 2
            fade_in_gradient = fade_in_gradient.reshape(-1, 1)
            fade_out_gradient = 1.0 - fade_in_gradient
            mixed = fade_out * fade_out_gradient + fade_in * fade_in_gradient
            all_audio[-crossfade_samples:] = mixed.astype(np.int16)
            all_audio = np.concatenate((all_audio, audio[crossfade_samples:]), axis=0)
        with wave.open(str(output_path), 'wb') as out:
            out.setnchannels(n_channels)
            out.setsampwidth(sampwidth)
            out.setframerate(sample_rate)
            out.writeframes(all_audio.tobytes())
        return output_path
    except ImportError:
        print("未安装 numpy，正在使用基础拼接。建议 pip install numpy 以启用平滑语气渐变。")
        try:
            data = []
            for wav_path in wav_paths:
                with wave.open(str(wav_path), 'rb') as wf:
                    data.append([wf.getparams(), wf.readframes(wf.getnframes())])
            with wave.open(str(output_path), 'wb') as out:
                out.setparams(data[0][0])
                for params, frames in data:
                    out.writeframes(frames)
            return output_path
        except Exception as e:
            print(f"合并音频失败: {e}")
            return None
    except Exception as e:
        print(f"合并音频失败: {e}")
        return None

_tts_started_lock = False

def auto_start_and_switch_tts(config: ConfigLoader):
    global _tts_started_lock
    if _tts_started_lock:
        return
    _tts_started_lock = True
    try:
        if not config.get("auto_start_tts", False):
            return
        base_url = config.get("client_base_url", "http://127.0.0.1:9880")
        try:
            resp = httpx.get(f"{base_url}/docs", timeout=2)
            if resp.status_code < 500:
                print("TTS 服务已在线，跳过自动启动。")
                return
        except Exception:
            pass

        script_path = config.get("tts_start_script", "")
        model_dir = config.get("model_dir", "")
        if not script_path or not model_dir:
            print("未配置 tts_start_script 或 model_dir，无法自动启动 TTS。")
            return

        script_path = Path(script_path)
        if script_path.is_dir():
            script_path = script_path / "api_v2.py"
        elif not script_path.exists() and script_path.suffix == "":
            candidate = script_path.parent / "api_v2.py"
            if candidate.exists():
                script_path = candidate
        if not script_path.exists():
            print(f"启动脚本不存在：{script_path}")
            return

        root_dir = str(script_path.parent).replace("\\", "/")
        python_candidates = [
            Path(root_dir) / "runtime" / "python.exe",
            Path(root_dir) / "runtime" / "python",
            Path("python.exe"),
            Path("python"),
        ]
        python_exe = None
        for candidate in python_candidates:
            if candidate.exists():
                python_exe = str(candidate)
                break
        if not python_exe:
            try:
                import shutil
                python_exe = shutil.which("python") or shutil.which("python3")
            except Exception:
                python_exe = None
        if not python_exe:
            print("找不到 Python 可执行文件，无法启动 TTS。")
            return

        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 9880

        config_candidates = [
            Path(root_dir) / "GPT_SoVITS" / "configs" / "tts_infer.yaml",
            Path(root_dir) / "configs" / "tts_infer.yaml",
            Path(root_dir) / "tts_infer.yaml",
        ]
        config_path = None
        for candidate in config_candidates:
            if candidate.exists():
                config_path = str(candidate)
                break
        if not config_path:
            print("未找到 tts_infer.yaml 配置文件，无法启动。")
            return

        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        cmd = [
            python_exe,
            str(script_path),
            "-a", host,
            "-p", str(port),
            "-c", config_path,
        ]
        try:
            tts_proc = subprocess.Popen(
                cmd,
                cwd=root_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            process_manager.register(tts_proc, name="GPT-SoVITS TTS")
            print("已启动 TTS 服务，等待就绪...")
        except Exception as e:
            print(f"启动 TTS 失败: {e}")
            return

        for _ in range(12):
            time.sleep(5)
            try:
                resp = httpx.get(f"{base_url}/docs", timeout=2)
                if resp.status_code < 500:
                    print("TTS 服务已就绪。")
                    break
            except Exception:
                continue
        else:
            print("TTS 服务在 60 秒内未就绪，请检查日志。")
            return

        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            print(f"模型目录不存在：{model_dir}")
            return
        gpt_file = sovits_file = None
        for f in model_dir_path.iterdir():
            if f.suffix == ".ckpt" and gpt_file is None:
                gpt_file = f.name
            if f.suffix == ".pth" and sovits_file is None:
                sovits_file = f.name
        if not gpt_file or not sovits_file:
            print(f"模型目录 {model_dir} 中未找到 .ckpt 或 .pth 文件！")
            return

        model_gpt = f"{model_dir}/{gpt_file}".replace("\\", "/")
        model_sovits = f"{model_dir}/{sovits_file}".replace("\\", "/")
        model_name = Path(gpt_file).stem
        try:
            resp = httpx.get(f"{base_url}/set_gpt_weights", params={"weights_path": model_gpt}, timeout=120)
            if resp.status_code == 200:
                print(f"[ {model_name} ] GPT 权重切换成功！")
            else:
                print(f"GPT 权重切换失败: {resp.text}")
            resp = httpx.get(f"{base_url}/set_sovits_weights", params={"weights_path": model_sovits}, timeout=120)
            if resp.status_code == 200:
                print(f"[ {model_name} ] SoVITS 权重切换成功！")
            else:
                print(f"SoVITS 权重切换失败: {resp.text}")
            print(f"[ {model_name} ] 模型加载完毕，可以开始使用了！")
        except Exception as e:
            print(f"调用 API 切换模型权重失败: {e}")
    finally:
        _tts_started_lock = False

class WebUIServer:
    def __init__(self, config: ConfigLoader, memory_manager: MemoryManager):
        self.config = config
        self._last_saved_config = None
        self.memory_manager = memory_manager
        self.html_path = get_resource_path("webui") / "start.html"
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get("/api/list", self.handle_list)
        self.app.router.add_post("/api/history", self.handle_history)
        self.app.router.add_post("/api/delete", self.handle_delete)
        self.app.router.add_post("/api/history/delete_messages", self.handle_delete_messages)
        self.app.router.add_get("/api/config", self.handle_get_config)
        self.app.router.add_post("/api/config/save", self.handle_save_config)
        self.app.router.add_get("/api/roles", self.handle_get_roles)
        self.app.router.add_post("/api/roles/save", self.handle_save_roles)
        self.app.router.add_get("/api/logs", self.handle_get_logs)
        self.app.router.add_get("/", self.handle_index)

    async def handle_index(self, request):
        if self.html_path.exists():
            return web.FileResponse(self.html_path)
        else:
            return web.Response(text="WebUI 页面未找到", status=404)

    async def handle_get_config(self, request):
        if not self.config.config:
            self.config.config = self.config.default_config()
        else:
            self.config.config = {**self.config.default_config(), **self.config.config}
        return web.json_response(self.config.config)

    async def handle_get_roles(self, request):
        """获取角色列表"""
        roles = []
        for key, role in self.config.roles.items():
            roles.append({
                "character_key": key,
                "character_name": role["character_name"],
                "active": key == self.config.active_character
            })
        return web.json_response({"roles": roles})

    async def handle_save_roles(self, request):
        """保存角色列表（前端已转换为json数组格式）"""
        try:
            new_data = await request.json()
            roles = new_data.get("roles", [])
            active = new_data.get("active_character", "")
            self.config.config["roles"] = roles
            self.config.config["active_character"] = active
            # 保存到文件
            with open(self.config.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.config, f, ensure_ascii=False, indent=2)
            # 重新初始化角色和配置
            global global_config, global_emotion_manager, memory_manager
            global_config = self.config
            self.config.roles = self.config._parse_roles()
            self.config.active_character = active
            # 重新初始化情绪管理器
            global_emotion_manager = EmotionManager(self.config)
            memory_manager = MemoryManager(self.config)
            return web.json_response({"success": True, "message": "角色配置已保存！"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    def validate_tts_config(self, config: ConfigLoader) -> tuple[bool, str]:
        ref_audio_root = config.get("ref_audio_root", "")
        if not ref_audio_root or not Path(ref_audio_root).exists():
            return False, "参考音频根目录无效或不存在，请检查路径后重试"
        model_dir = config.get("model_dir", "")
        if not model_dir or not Path(model_dir).exists():
            return False, "模型文件夹路径无效或不存在，请检查路径后重试"
        return True, ""

    async def handle_save_config(self, request):
        try:
            new_config = await request.json()
            
            # 提取重启标志并移除，防止写入配置文件
            restart_tts = new_config.pop("restart_tts", False)
            
            # 保存配置到文件
            self.config.config = new_config
            with open(self.config.config_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, ensure_ascii=False, indent=2)
            global global_config, global_emotion_manager, memory_manager
            global_config = self.config
            global_emotion_manager = EmotionManager(self.config)
            memory_manager = MemoryManager(self.config)

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
                            'llm_emotion_intensity', 'intensity_to_temperature', 'intensity_to_top_k']  # 补充了情绪强度参数
                for key in tts_keys:
                    if old_config.get(key) != new_config.get(key):
                        tts_changed = True
                        break

            tts_restart_success = False
            tts_restart_message = ""
            
            # 逻辑修正：如果用户手动点“保存并重启TTS”，强制重启；否则按照原有的变更检测逻辑重启
            force_restart = restart_tts and global_config.get("auto_start_tts", False)
            
            if (tts_changed or force_restart) and global_config.get("auto_start_tts", False):
                valid, error_msg = self.validate_tts_config(global_config)
                if not valid:
                    print(f"TTS 配置验证失败，跳过重启：{error_msg}")
                    tts_restart_message = f"配置已保存，但 TTS 服务未重启：{error_msg}"
                else:
                    print("检测到 TTS 相关配置变化或用户强制重启，正在重启 TTS 服务...")
                    process_manager.shutdown_all()
                    threading.Thread(target=auto_start_and_switch_tts, args=(global_config,), daemon=True).start()
                    tts_restart_success = True
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

    async def handle_get_logs(self, request):
        logs = "\n".join(global_log_buffer)
        return web.json_response({"logs": logs})

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

    async def start(self):
        host = self.config.get("webui_host", "127.0.0.1")
        port = int(self.config.get("webui_port", 11500))
        if not HAS_AIOHTTP:
            print("未安装 aiohttp，WebUI 不可用")
            return
        try:
            print(f"正在启动 WebUI：http://{host}:{port}")
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            print(f"WebUI 已启动：http://{host}:{port}")
            # 保存 runner 引用以便后续关闭
            self.runner = runner
        except Exception as e:
            error_msg = f"WebUI 启动失败：{type(e).__name__}: {e}"
            print(error_msg)
            # 写入日志文件，方便排查
            try:
                with open("webui_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.ctime()} - {error_msg}\n")
            except:
                pass
            # 不要抛出异常，让主程序继续运行

    async def shutdown(self):
        if HAS_AIOHTTP:
            await self.app.cleanup()

async def main(stop_event: threading.Event = None):
    global global_config, global_emotion_manager, memory_manager
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
    
    if global_config.get("auto_start_tts", False):
        threading.Thread(target=auto_start_and_switch_tts, args=(global_config,), daemon=True).start()
    
    webui_server = None
    if HAS_AIOHTTP:
        webui_server = WebUIServer(global_config, memory_manager)
        try:
            await webui_server.start()
        except Exception as e:
            print(f"WebUI 启动异常，继续运行其他功能：{e}")
    
    # 获取配置
    ws_url = global_config.get("napcat_ws_url", "ws://127.0.0.1:3001")
    token = global_config.get("napcat_token", "")
    
    print(f"正在连接 NapCat ({ws_url})...")
    
    # 循环尝试连接
    while True:
        if stop_event is not None and stop_event.is_set():
            print("收到停止信号，正在退出消息循环...")
            break
        
        try:
            client = NapCatClient(ws_url=ws_url, token=token)
            async with client:
                print(f"已连接！机器人 QQ: {client.self_id}")
                print("等待消息中...")
                async for event in client:
                    if stop_event is not None and stop_event.is_set():
                        print("收到停止信号，正在退出消息循环...")
                        break
                    
                    try:
                        if not await ensure_tts_service(global_config):
                            print("警告：TTS 服务不可用，将降级为纯文本。")
                        
                        if isinstance(event, PrivateMessageEvent):
                            user_id = event.user_id
                            session_id = f"private_{user_id}"
                            user_text = ""
                            has_image = False
                            image_urls = []
                            
                            for seg in event.message:
                                if isinstance(seg, Text):
                                    user_text += seg.text
                                elif isinstance(seg, Image):
                                    has_image = True
                                    path = getattr(seg, "file", None) or getattr(seg, "url", None)
                                    if path:
                                        image_urls.append(path)
                            
                            if not user_text and not has_image:
                                continue
                            
                            print(f"收到私聊 [{user_id}]: {user_text}")
                            
                            if global_config.get("isolated_session", False):
                                session_id = f"private_{user_id}"
                            
                            memory_manager.migrate_legacy_memory(session_id)
                            history = memory_manager.load_history(session_id)
                            
                            if has_image:
                                if global_config.get("image_caption_model_name", ""):
                                    result = await get_llm_reply(global_config, user_text, history, global_emotion_manager.emotions, images=image_urls)
                                else:
                                    print("未配置识图模型，忽略图片")
                                    result = None
                            else:
                                result = await get_llm_reply(global_config, user_text, history, global_emotion_manager.emotions, images=None)
                            
                            if not result or result[0] is None:
                                continue
                            
                            zh_text, lang_list, emo_list, display_list, lang_list = result
                            
                            history.append({
                                "role": "user",
                                "content": user_text,
                                "sender_id": user_id,
                                "sender_name": event.sender.nickname if hasattr(event.sender, 'nickname') else str(user_id),
                                "timestamp": time.time()
                            })
                            history.append({"role": "assistant", "content": zh_text, "timestamp": time.time()})
                            memory_manager.save_history(session_id, history)
                            
                            data_path = memory_manager.data_path
                            tasks = [synthesize_sentence(global_config, lang_list[i], emo_list[i], global_emotion_manager.emotions, data_path) for i in range(len(lang_list))]
                            wavs = await asyncio.gather(*tasks)
                            valid_wavs = [w for w in wavs if w]
                            
                            separate_send = global_config.get("separate_send", False)
                            send_voice_separately = global_config.get("send_voice_separately", False)
                            dynamic_sleep = global_config.get("dynamic_sleep", True)
                            
                            if separate_send and send_voice_separately:
                                for idx, wav in enumerate(valid_wavs):
                                    if not wav or not wav.exists():
                                        continue
                                    sentence_text = display_list[idx] if idx < len(display_list) else zh_text
                                    if sentence_text:
                                        await client.send_private_msg(user_id=user_id, message=[Text(text=sentence_text)])
                                    await client.send_private_msg(user_id=user_id, message=[Record(file=str(wav.resolve()))])
                                    if dynamic_sleep:
                                        await asyncio.sleep(get_audio_duration(str(wav)) + 0.5)
                                    else:
                                        await asyncio.sleep(0.2)
                                    wav.unlink(missing_ok=True)
                            else:
                                combined_audio = merge_wavs(valid_wavs, global_config, data_path)
                                combined_text = "".join(display_list) if display_list else zh_text
                                if combined_audio:
                                    if separate_send and global_config.get("text_separate", False):
                                        await client.send_private_msg(user_id=user_id, message=[Record(file=str(combined_audio.resolve()))])
                                        for text in display_list:
                                            await client.send_private_msg(user_id=user_id, message=[Text(text=text)])
                                            await asyncio.sleep(0.2)
                                    else:
                                        if combined_text:
                                            await client.send_private_msg(user_id=user_id, message=[Text(text=combined_text)])
                                        await client.send_private_msg(user_id=user_id, message=[Record(file=str(combined_audio.resolve()))])
                                    for w in valid_wavs:
                                        w.unlink(missing_ok=True)
                                    combined_audio.unlink(missing_ok=True)
                                else:
                                    print("TTS 合成失败，降级为纯文本。")
                                    await client.send_private_msg(user_id=user_id, message=[Text(text=zh_text)])
                            
                            memory_manager.cleanup_voice_cache(global_config.get("max_voice_cache", 20))
                        
                        elif isinstance(event, GroupMessageEvent):
                            group_id = event.group_id
                            sender_id = event.sender.user_id
                            session_id = f"group_{group_id}"
                            
                            if global_config.get("isolated_session", False):
                                session_id = f"group_{group_id}_{sender_id}"
                            
                            user_text = ""
                            has_image = False
                            image_urls = []
                            at_bot = False
                            
                            for seg in event.message:
                                if isinstance(seg, Text):
                                    user_text += seg.text
                                elif isinstance(seg, Image):
                                    has_image = True
                                    path = getattr(seg, "file", None) or getattr(seg, "url", None)
                                    if path:
                                        image_urls.append(path)
                                elif isinstance(seg, At):
                                    if seg.qq == str(client.self_id):
                                        at_bot = True
                            
                            if not at_bot and global_config.get("only_private", False):
                                continue
                            if not user_text and not has_image:
                                continue
                            
                            print(f"收到群聊 [{group_id}] 来自 [{sender_id}]: {user_text}")
                            
                            memory_manager.migrate_legacy_memory(session_id)
                            history = memory_manager.load_history(session_id)
                            
                            if has_image:
                                if global_config.get("image_caption_model_name", ""):
                                    result = await get_llm_reply(global_config, user_text, history, global_emotion_manager.emotions, images=image_urls)
                                else:
                                    print("未配置识图模型，忽略图片")
                                    result = None
                            else:
                                result = await get_llm_reply(global_config, user_text, history, global_emotion_manager.emotions, images=None)
                            
                            if not result or result[0] is None:
                                continue
                            
                            zh_text, lang_list, emo_list, display_list, lang_list = result
                            
                            history.append({
                                "role": "user",
                                "content": user_text,
                                "sender_id": sender_id,
                                "sender_name": event.sender.nickname if hasattr(event.sender, 'nickname') else str(sender_id),
                                "timestamp": time.time()
                            })
                            history.append({"role": "assistant", "content": zh_text, "timestamp": time.time()})
                            memory_manager.save_history(session_id, history)
                            
                            data_path = memory_manager.data_path
                            tasks = [synthesize_sentence(global_config, lang_list[i], emo_list[i], global_emotion_manager.emotions, data_path) for i in range(len(lang_list))]
                            wavs = await asyncio.gather(*tasks)
                            valid_wavs = [w for w in wavs if w]
                            
                            separate_send = global_config.get("separate_send", False)
                            send_voice_separately = global_config.get("send_voice_separately", False)
                            dynamic_sleep = global_config.get("dynamic_sleep", True)
                            
                            if separate_send and send_voice_separately:
                                for idx, wav in enumerate(valid_wavs):
                                    if not wav or not wav.exists():
                                        continue
                                    sentence_text = display_list[idx] if idx < len(display_list) else zh_text
                                    if sentence_text:
                                        await client.send_group_msg(group_id=group_id, message=[Text(text=sentence_text)])
                                    await client.send_group_msg(group_id=group_id, message=[Record(file=str(wav.resolve()))])
                                    if dynamic_sleep:
                                        await asyncio.sleep(get_audio_duration(str(wav)) + 0.5)
                                    else:
                                        await asyncio.sleep(0.2)
                                    wav.unlink(missing_ok=True)
                            else:
                                combined_audio = merge_wavs(valid_wavs, global_config, data_path)
                                combined_text = "".join(display_list) if display_list else zh_text
                                if combined_audio:
                                    if separate_send and global_config.get("text_separate", False):
                                        await client.send_group_msg(group_id=group_id, message=[Record(file=str(combined_audio.resolve()))])
                                        for text in display_list:
                                            await client.send_group_msg(group_id=group_id, message=[Text(text=text)])
                                            await asyncio.sleep(0.2)
                                    else:
                                        if combined_text:
                                            await client.send_group_msg(group_id=group_id, message=[Text(text=combined_text)])
                                        await client.send_group_msg(group_id=group_id, message=[Record(file=str(combined_audio.resolve()))])
                                    for w in valid_wavs:
                                        w.unlink(missing_ok=True)
                                    combined_audio.unlink(missing_ok=True)
                                else:
                                    print("TTS 合成失败，降级为纯文本。")
                                    await client.send_group_msg(group_id=group_id, message=[Text(text=zh_text)])
                            
                            memory_manager.cleanup_voice_cache(global_config.get("max_voice_cache", 20))
                    
                    except Exception as e:
                        print(f"处理消息异常: {type(e).__name__}: {e}")
        
        except Exception as e:
            print(f"NapCat 连接失败: {e}")
            print("10秒后尝试重新连接...")
            await asyncio.sleep(10)
            continue

    # 循环外正常清理
    print("正在关闭所有子进程...")
    process_manager.shutdown_all()
    if webui_server:
        await webui_server.shutdown()

if __name__ == "__main__":
    import threading
    import time

    # 创建停止事件，用于通知后台线程退出
    stop_event = threading.Event()

    # 获取初始配置（用于 WebUI 端口等）
    config = ConfigLoader()

    def run_backend():
        try:
            asyncio.run(main(stop_event))
        except Exception as e:
            print(f"后台服务异常: {e}")
            import traceback
            traceback.print_exc()
            # 写入日志文件
            try:
                with open("backend_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.ctime()} - 异常: {e}\n")
                    traceback.print_exc(file=f)
            except:
                pass

    # 启动后台线程，运行主逻辑
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
                width=1200,
                height=800,
                resizable=True
            )
            webview.start()  # 此函数会在窗口关闭后返回
        else:
            # 无 WebView 时，等待键盘中断
            print("程序正在运行，按 Ctrl+C 退出...")
            while not stop_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到键盘中断，准备退出...")
    finally:
        # 无论窗口关闭还是中断，都执行清理
        print("正在停止后台服务...")
        stop_event.set()                     # 通知后台线程退出
        backend_thread.join(timeout=10)      # 等待后台线程结束
        process_manager.shutdown_all()       # 强制清理所有子进程
        print("程序已完全退出。")