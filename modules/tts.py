"""TTS 合成与音频处理工具（自 main.py 迁出，供主流程与主动消息共用）。"""
import asyncio
import re
import time
import wave
from pathlib import Path
from typing import Optional

import httpx
import numpy as np


def get_audio_duration(file_path: str) -> float:
    try:
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


async def check_tts_service(config) -> bool:
    base_url = config.get("client_base_url", "http://127.0.0.1:9880")
    try:
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            resp = await client.get(f"{base_url}/docs")
            return resp.status_code < 500
    except Exception:
        return False


async def synthesize_sentence(config, text: str, emotion: str, emotions: dict,
                              data_path: Path, stats=None) -> Optional[Path]:
    """合成一句话语音，返回 wav 路径；纯标点/空句子返回 None。"""
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
    start = time.time()
    for attempt in range(max_retries):
        try:
            print(f"正在合成: 情绪={emotion} | 文本={clean_text} (尝试 {attempt+1}/{max_retries})")
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.get(f"{base_url}/tts", params=params)
                if resp.status_code == 200:
                    temp_path = data_path / f"temp_{emotion}_{int(time.time()*1000)}.wav"
                    temp_path.write_bytes(resp.content)
                    if stats:
                        stats.record_tts((time.time() - start) * 1000)
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
                from .tts_service import ensure_tts_service
                await ensure_tts_service(config)
        except Exception as e:
            print(f"TTS 连接异常 ({e})，等待 {retry_delay} 秒后重试...")
            await asyncio.sleep(retry_delay)
            retry_delay += 1.0
    print(f"TTS 合成在 {max_retries} 次尝试后仍失败: {clean_text}")
    return None


def merge_wavs(wav_paths: list, config, data_path: Path) -> Optional[Path]:
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
