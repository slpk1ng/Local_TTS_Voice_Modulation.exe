"""表情包管理：按情绪目录扫描本地表情图片，回复时按概率附加。

目录结构（可配置 stickers_dir，默认 <data>/stickers）：
  stickers/
    happy/1.jpg 2.gif ...
    angry/*.png
    default/*.jpg      ← 可选，其它情绪未命中时的兜底
    any/*.jpg          ← 可选，任意情绪都可能随机抽取
"""
import random
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


class StickerManager:
    def __init__(self, config):
        self.config = config
        self.enabled = bool(config.get("stickers_enabled", False))
        self.dir = Path(config.get("stickers_dir", "") or Path("data/stickers"))
        self.probability = float(config.get("sticker_probability", 1.0))
        self.max_per_reply = max(1, int(config.get("sticker_max_per_reply", 1)))
        self.map = {}          # emotion -> [Path, ...]
        self.default_pool = []  # default/ 目录
        self.any_pool = []      # any/ 目录
        self._scan()

    def _scan(self):
        self.map.clear()
        self.default_pool.clear()
        self.any_pool.clear()
        root = self.dir
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"表情包目录不可用: {e}")
            return
        if not root.exists():
            return
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not images:
                continue
            name = folder.name.lower()
            if name == "default":
                self.default_pool = images
            elif name == "any":
                self.any_pool = images
            else:
                self.map[name] = images
        total = sum(len(v) for v in self.map.values()) + len(self.default_pool) + len(self.any_pool)
        if total:
            print(f"表情包扫描完成：{len(self.map)} 个情绪分类，共 {total} 张图片（目录：{root}）")
        else:
            print(f"表情包目录为空（{root}），可在 WebUI「表情包」页上传图片。")

    def rescan(self):
        self.__init__(self.config)

    def pick(self, emotion: str):
        """根据情绪返回一张表情图片路径；未启用/未命中/概率未触发返回 None。"""
        if not self.enabled:
            return None
        if self.probability < 1.0 and random.random() > self.probability:
            return None
        candidates = []
        key = str(emotion or "").strip().lower()
        if key and key in self.map:
            candidates = list(self.map[key])
        if not candidates and self.any_pool:
            candidates = list(self.any_pool)
        if not candidates:
            candidates = list(self.default_pool)
        if not candidates:
            return None
        return random.choice(candidates)
