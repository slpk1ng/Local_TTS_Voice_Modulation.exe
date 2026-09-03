"""GPT-SoVITS 服务生命周期管理（自 main.py 迁出）。"""
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


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

_tts_started_lock = threading.Lock()
_tts_started_flag = False


async def check_tts_service(config) -> bool:
    base_url = config.get("client_base_url", "http://127.0.0.1:9880")
    try:
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            resp = await client.get(f"{base_url}/docs")
            return resp.status_code < 500
    except Exception:
        return False


_ensure_fail_until = 0.0  # 启动失败后的冷却截止时间，避免每条消息都阻塞重试 60 秒


async def ensure_tts_service(config) -> bool:
    global _ensure_fail_until
    if await check_tts_service(config):
        _ensure_fail_until = 0.0
        return True
    if time.time() < _ensure_fail_until:
        return False  # 冷却期内直接失败，不阻塞消息管线
    if not config.get("auto_start_tts", False):
        return False
    threading.Thread(target=auto_start_and_switch_tts, args=(config,), daemon=True).start()
    import asyncio
    for _ in range(12):
        await asyncio.sleep(5)
        if await check_tts_service(config):
            return True
    _ensure_fail_until = time.time() + 60
    return False


def auto_start_and_switch_tts(config):
    global _tts_started_flag
    with _tts_started_lock:
        if _tts_started_flag:
            return
        _tts_started_flag = True
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
        with _tts_started_lock:
            _tts_started_flag = False
