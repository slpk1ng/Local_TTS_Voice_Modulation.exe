"""统一消息发送器：文本/语音/表情包发送，供主消息管线与定时、主动消息共用。"""
import asyncio
import time
from pathlib import Path
from typing import List, Optional

from .llm_helpers import RoleContext
from .tts import synthesize_sentence, merge_wavs, get_audio_duration
from .tts_service import check_tts_service


def _normalize_target(session_type: str, target_id) -> tuple:
    """兼容传入完整会话ID的情形：private_10001 / group_456 / group_456_789
    （待办提醒等模块保存的是 session_id，而发送需要纯数字号码）。"""
    s = str(target_id)
    if s.startswith("private_"):
        return "private", s.split("_", 1)[1]
    if s.startswith("group_"):
        return "group", s.split("_", 1)[1].split("_")[0]
    return session_type, target_id


class MessageSender:
    """封装 NapCat 客户端发送行为。client 由主程序注入（连接后设置）。"""

    def __init__(self, config, memory_manager, sticker_manager=None, stats=None):
        self.config = config
        self.memory_manager = memory_manager
        self.sticker_manager = sticker_manager
        self.stats = stats
        self.client = None  # NapCatClient，主循环连接后注入

    # ---------------- 基础发送 ----------------
    async def send_segments(self, session_type: str, target_id, segments: list):
        if self.client is None:
            print("发送失败：NapCat 客户端尚未连接")
            return
        session_type, target_id = _normalize_target(session_type, target_id)
        try:
            if session_type == "private":
                await self.client.send_private_msg(user_id=int(target_id), message=segments)
            else:
                await self.client.send_group_msg(group_id=int(target_id), message=segments)
        except Exception as e:
            print(f"发送消息失败 ({session_type} {target_id}): {type(e).__name__}: {e}")

    async def send_text(self, session_type: str, target_id, text: str, sticker=None):
        from napcat import Text, Image
        segments = []
        if text:
            segments.append(Text(text=text))
        if sticker is not None:
            try:
                segments.append(Image(file=str(Path(sticker).resolve())))
            except Exception as e:
                print(f"表情包发送失败: {e}")
        if not segments:
            return
        await self.send_segments(session_type, target_id, segments)

    async def send_voice(self, session_type: str, target_id, wav_path):
        from napcat import Record
        await self.send_segments(session_type, target_id,
                                 [Record(file=str(Path(wav_path).resolve()))])

    # ---------------- 回复合送（保持原有分合逻辑 + 表情包） ----------------
    async def send_reply(self, session_type: str, target_id: str, sentences: List[dict],
                        emotions: dict, ctx: RoleContext, use_voice: bool = True) -> dict:
        """按配置发送整组句子（文本+语音），返回 {tts_ms, voice_ok}。

        sentences: [{zh, lang, display, emotion}]
        """
        result = {"tts_ms": 0.0, "voice_ok": False}
        if not sentences:
            return result
        data_path = self.memory_manager.data_path
        separate_send = self.config.get("separate_send", False)
        send_voice_separately = self.config.get("send_voice_separately", False)
        text_separate = self.config.get("text_separate", False)
        dynamic_sleep = self.config.get("dynamic_sleep", True)
        use_tts = use_voice and self.config.get("tts_reply_enabled", True)

        # 1. 合成语音（全部句子一次性合成，或逐个合成，取决于是否需要分开）
        wavs: List[Optional[Path]] = [None] * len(sentences)
        if use_tts and self.client is not None:
            tasks = [synthesize_sentence(ctx, s["lang"], s["emotion"], emotions, data_path,
                                        stats=self.stats)
                    for s in sentences]
            synth = await asyncio.gather(*tasks)
            wavs = synth
        valid_wavs = [w for w in wavs if w]
        result["voice_ok"] = bool(valid_wavs)

        # 表情包选择（基于第一句情绪）
        sticker = None
        if self.sticker_manager is not None and self.client is not None:
            sticker = self.sticker_manager.pick(sentences[0].get("emotion", ""))

        # ========== 分开发送 + 语音分开发送 ==========
        if separate_send and send_voice_separately:
            missing = []
            for idx, wav in enumerate(wavs):
                sentence_text = sentences[idx]["display"]
                # ① 语音失败的句子先记下，稍后统一补发文本，避免文本重复发送
                if not wav or not wav.exists():
                    missing.append(idx)
                    continue
                # ② 逐句发送文本 + 对应语音
                if sentence_text:
                    await self.send_text(session_type, target_id, sentence_text)
                await self.send_voice(session_type, target_id, wav)
                if dynamic_sleep:
                    await asyncio.sleep(get_audio_duration(str(wav)) + 0.5)
                else:
                    await asyncio.sleep(0.2)
                wav.unlink(missing_ok=True)

            # ③ 如果所有语音都失败，降级发送合并文本
            if not valid_wavs:
                await self.send_text(session_type, target_id,
                                    "".join(s["display"] for s in sentences))
            else:
                # ④ 处理语音失败的句子（补发文本，只发一次）
                for idx in missing:
                    text = sentences[idx]["display"]
                    if text:
                        await self.send_text(session_type, target_id, text)

            # ⑤ 最后发送表情包（单独发送）
            if sticker:
                await self.send_text(session_type, target_id, "", sticker=sticker)

        # ========== 合并发送（默认或文字分开但语音合并） ==========
        else:
            combined_audio = merge_wavs(valid_wavs, self.config, data_path) if valid_wavs else None
            combined_text = "".join(s["display"] for s in sentences)

            if combined_audio:
                # 如果开启了文字分开发送（但语音合并），则先发语音，再逐句发文字
                if separate_send and text_separate:
                    await self.send_voice(session_type, target_id, combined_audio)
                    for i, s in enumerate(sentences):
                        await self.send_text(session_type, target_id, s["display"])
                        # 文字之间采用固定间隔（如果希望基于语音时长，可改为语音时长）
                        await asyncio.sleep(0.2)
                else:
                    # 正常合并发送：文字+语音一起发
                    await self.send_text(session_type, target_id, combined_text)
                    await self.send_voice(session_type, target_id, combined_audio)
                    # 合并语音发送后，可以等待其播放完毕（可选）
                    if dynamic_sleep:
                        await asyncio.sleep(get_audio_duration(str(combined_audio)) + 0.5)
                    else:
                        await asyncio.sleep(0.2)

                # 清理临时文件
                for w in valid_wavs:
                    w.unlink(missing_ok=True)
                combined_audio.unlink(missing_ok=True)
            else:
                # 语音合成失败，降级纯文本
                print("TTS 合成失败或未启用，降级为纯文本。")
                await self.send_text(session_type, target_id, combined_text)

            # 最后发送表情包
            if sticker:
                await self.send_text(session_type, target_id, "", sticker=sticker)

        self.memory_manager.cleanup_voice_cache(self.config.get("max_voice_cache", 20))
        return result

    # ---------------- 主动消息（定时/提醒/问候） ----------------
    async def speak_and_send(self, session_type: str, target_id, text: str,
                             emotions: dict, ctx: Optional[RoleContext] = None,
                             use_voice: bool = None, sticker: bool = False) -> bool:
        """发送一条主动消息：按配置决定是否合成语音。"""
        if not text:
            return False
        if ctx is None:
            ctx = RoleContext(self.config)
        if use_voice is None:
            use_voice = bool(self.config.get("proactive_voice", False))
        sticker_path = self.sticker_manager.pick("") if (sticker and self.sticker_manager) else None
        if use_voice and self.client is not None and await check_tts_service(self.config):
            wav = await synthesize_sentence(ctx, text,
                                            ctx.get("default_voice", "pingjing"),
                                            emotions, self.memory_manager.data_path,
                                            stats=self.stats)
            if wav:
                await self.send_text(session_type, target_id, text, sticker=sticker_path)
                await self.send_voice(session_type, target_id, wav)
                wav.unlink(missing_ok=True)
                return True
        await self.send_text(session_type, target_id, text, sticker=sticker_path)
        return True
