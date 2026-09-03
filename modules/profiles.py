"""用户画像：按 user_id 记录昵称、生日、喜好、备注等，支持 LLM 自动提取与提示词注入。

存于 data/user_profiles.json：
  { "10001": {"nickname": "小明", "birthday": "05-20", "likes": [...], "notes": [...], "updated_at": ts} }
"""
import json
import time
from pathlib import Path
from typing import Optional

from .llm_helpers import generate_json_reply, RoleContext

DEFAULT_EXTRACT_PROMPT = (
    "你是信息提取助手。请从用户与角色的对话中提取关于【用户】的长期个人信息（不是角色设定）。"
    "只提取明确、可靠的信息，没有则返回空对象。只输出JSON，格式："
    '{"nickname": "称呼/昵称(可选)", "birthday": "MM-DD(可选)", "likes": ["喜好"], "dislikes": ["厌恶"], "notes": ["重要事项"]}'
)


class UserProfileManager:
    def __init__(self, config, data_path: Path):
        self.config = config
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.file = self.data_path / "user_profiles.json"
        self.profiles = {}
        self._dirty = False
        self.load()

    def load(self):
        try:
            if self.file.exists():
                self.profiles = json.loads(self.file.read_text(encoding="utf-8"))
                if not isinstance(self.profiles, dict):
                    self.profiles = {}
        except Exception as e:
            print(f"加载用户画像失败: {e}")
            self.profiles = {}

    def save(self):
        try:
            self.file.write_text(json.dumps(self.profiles, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as e:
            print(f"保存用户画像失败: {e}")

    def get(self, user_id: str) -> dict:
        return dict(self.profiles.get(str(user_id), {}))

    def update(self, user_id: str, data: dict):
        uid = str(user_id)
        if not uid or not isinstance(data, dict):
            return
        profile = self.profiles.setdefault(uid, {"likes": [], "dislikes": [], "notes": []})
        for key in ("nickname", "birthday"):
            val = str(data.get(key, "") or "").strip()
            if val:
                profile[key] = val
        for key in ("likes", "dislikes", "notes"):
            items = data.get(key)
            if isinstance(items, list):
                bucket = profile.setdefault(key, [])
                for item in items:
                    item = str(item).strip()
                    if item and item not in bucket:
                        bucket.append(item)
                profile[key] = bucket[-50:]
        profile["updated_at"] = time.time()
        self.save()

    def delete(self, user_id: str) -> bool:
        uid = str(user_id)
        if uid in self.profiles:
            del self.profiles[uid]
            self.save()
            return True
        return False

    # ---------------- LLM 自动提取 ----------------
    async def extract_from_dialog(self, ctx: RoleContext, user_text: str, reply_text: str,
                                  user_id: str):
        """对话后异步提取用户信息（失败静默）。"""
        prompt = str(self.config.get("profiles_extract_prompt", "") or DEFAULT_EXTRACT_PROMPT)
        system = (
            f"{prompt}\n当前用户ID: {user_id}\n已有画像(供去重参考): "
            f"{json.dumps(self.get(user_id), ensure_ascii=False)}"
        )
        user_prompt = f"用户说：{user_text}\n角色回复：{reply_text}"
        try:
            data = await generate_json_reply(ctx, system, user_prompt, max_tokens=256)
        except Exception as e:
            print(f"用户画像提取失败: {e}")
            return
        if not isinstance(data, dict):
            return
        # 空 nickname/生日时忽略对应字段，避免覆盖
        data.pop("nickname", None) if not str(data.get("nickname", "")).strip() else None
        self.update(user_id, data)

    # ---------------- 注入提示词 ----------------
    def build_injection(self, user_id: str) -> str:
        if not self.config.get("profiles_enabled", False):
            return ""
        profile = self.profiles.get(str(user_id))
        if not profile:
            return ""
        template = str(self.config.get("profiles_inject_template", "") or
                       "【用户画像】关于当前用户的已知信息：{profile}")
        lines = []
        if profile.get("nickname"):
            lines.append(f"昵称: {profile['nickname']}")
        if profile.get("birthday"):
            lines.append(f"生日: {profile['birthday']}")
        for key, label in (("likes", "喜好"), ("dislikes", "厌恶"), ("notes", "重要事项")):
            items = profile.get(key) or []
            if items:
                lines.append(f"{label}: " + "、".join(str(i) for i in items[:10]))
        if not lines:
            return ""
        text = "\n".join(lines)
        max_chars = int(self.config.get("profiles_max_chars", 300))
        return template.replace("{profile}", text[:max_chars])
