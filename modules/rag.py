"""RAG 检索增强生成：本地轻量向量库（JSON 分块 + numpy 向量，零外部向量数据库依赖）。

文档存于 data/rag/：
  index.json             文档索引
  docs/{id}.json         分块文本
  docs/{id}.npy          分块向量矩阵 (float32)
嵌入模型可配置（默认使用 Ollama 的 nomic-embed-text 或任何已拉取的 embedding 模型）。
"""
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import httpx

CHUNK_SIZE_DEFAULT = 500
OVERLAP_DEFAULT = 80


class RAGManager:
    def __init__(self, config, data_path: Path):
        self.config = config
        self.root = Path(data_path) / "rag"
        self.docs_dir = self.root / "docs"
        self.index_file = self.root / "index.json"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index: List[dict] = []
        self.load_index()

    # ---------------- 索引管理 ----------------
    def load_index(self):
        try:
            if self.index_file.exists():
                self.index = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"加载 RAG 索引失败: {e}")
            self.index = []

    def save_index(self):
        try:
            self.index_file.write_text(json.dumps(self.index, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        except Exception as e:
            print(f"保存 RAG 索引失败: {e}")

    def list_docs(self) -> list:
        return [{k: d.get(k) for k in ("id", "name", "chunks", "size", "added_at")}
                for d in self.index]

    # ---------------- 嵌入 ----------------
    async def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        model = self.config.get("rag_embedding_model", "nomic-embed-text")
        backend = self.config.get("llm_backend", "ollama")
        base_url = str(self.config.get("llm_base_url", "http://127.0.0.1:11434")).rstrip("/")
        api_key = self.config.get("llm_api_key", "")
        try:
            if backend == "ollama":
                async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                    errors = []
                    # 新版 Ollama：批量嵌入接口
                    resp = await client.post(f"{base_url}/api/embed",
                                             json={"model": model, "input": texts})
                    if resp.status_code == 200:
                        embs = resp.json().get("embeddings")
                        if embs:
                            return np.array(embs, dtype=np.float32)
                        errors.append("/api/embed[200]: 响应缺少 embeddings 字段")
                    else:
                        errors.append(f"/api/embed[{resp.status_code}]: {_response_error(resp)}")
                    # 兼容旧版 Ollama：/api/embeddings 逐条嵌入
                    vecs = []
                    for t in texts:
                        r = await client.post(f"{base_url}/api/embeddings",
                                              json={"model": model, "prompt": t})
                        if r.status_code != 200:
                            errors.append(f"/api/embeddings[{r.status_code}]: {_response_error(r)}")
                            vecs = []
                            break
                        vecs.append(r.json().get("embedding", []))
                    if vecs and vecs[0]:
                        return np.array(vecs, dtype=np.float32)
                    msg = "；".join(errors)
                    low = msg.lower()
                    if model.lower() in low or "try pulling" in low:
                        msg += f"（提示：嵌入模型未安装，请先执行: ollama pull {model}）"
                    elif "does not support" in low:
                        msg += "（提示：该模型不支持生成嵌入，请在 WebUI 将 rag_embedding_model 换成嵌入模型，如 nomic-embed-text 或 bge-m3）"
                    raise RuntimeError(msg)
            else:
                if not base_url.endswith("/v1"):
                    base_url += "/v1"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                    resp = await client.post(f"{base_url}/embeddings",
                                             json={"model": model, "input": texts},
                                             headers=headers)
                    resp.raise_for_status()
                    data = sorted(resp.json().get("data", []), key=lambda x: x.get("index", 0))
                    vecs = [d.get("embedding") for d in data]
                    if not vecs or not vecs[0]:
                        return None
                    return np.array(vecs, dtype=np.float32)
        except Exception as e:
            print(f"RAG 嵌入失败: {type(e).__name__}: {e}")
            return None

    # ---------------- 分块 ----------------
    @staticmethod
    def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        text = re_sub_spaces(text)
        chunks = []
        step = max(1, chunk_size - max(0, overlap))
        for i in range(0, len(text), step):
            piece = text[i:i + chunk_size].strip()
            if len(piece) >= 20:
                chunks.append(piece)
            if i + chunk_size >= len(text):
                break
        return chunks or ([text.strip()] if text.strip() else [])

    # ---------------- 文档操作 ----------------
    async def add_document(self, name: str, raw_text: str) -> dict:
        chunk_size = int(self.config.get("rag_chunk_size", CHUNK_SIZE_DEFAULT))
        overlap = int(self.config.get("rag_chunk_overlap", OVERLAP_DEFAULT))
        chunks = self.chunk_text(raw_text, chunk_size, overlap)
        if not chunks:
            return {"success": False, "error": "文档内容为空"}
        vectors = await self.embed_texts(chunks)
        if vectors is None or len(vectors) != len(chunks):
            return {"success": False, "error": "向量化失败，请检查嵌入模型是否可用"}
        doc_id = uuid.uuid4().hex[:12]
        norm = _normalize(vectors)
        np.save(self.docs_dir / f"{doc_id}.npy", norm)
        (self.docs_dir / f"{doc_id}.json").write_text(
            json.dumps({"name": name, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
        entry = {"id": doc_id, "name": name, "chunks": len(chunks),
                 "size": len(raw_text), "added_at": time.time()}
        self.index.append(entry)
        self.save_index()
        return {"success": True, "doc": entry}

    def delete_document(self, doc_id: str) -> bool:
        before = len(self.index)
        self.index = [d for d in self.index if d.get("id") != doc_id]
        if len(self.index) == before:
            return False
        for suffix in (".npy", ".json"):
            try:
                (self.docs_dir / f"{doc_id}{suffix}").unlink(missing_ok=True)
            except Exception:
                pass
        self.save_index()
        return True

    # ---------------- 检索 ----------------
    async def search(self, query: str, top_k: int = None, min_sim: float = None) -> List[dict]:
        if not self.index:
            return []
        top_k = int(top_k or self.config.get("rag_top_k", 3))
        min_sim = float(min_sim if min_sim is not None else self.config.get("rag_min_similarity", 0.35))
        qvec = await self.embed_texts([query])
        if qvec is None:
            return []
        q = _normalize(qvec)[0]
        hits = []
        for doc in self.index:
            try:
                mat = np.load(self.docs_dir / f"{doc['id']}.npy")
                chunk_data = json.loads((self.docs_dir / f"{doc['id']}.json").read_text(encoding="utf-8"))
            except Exception as e:
                print(f"读取 RAG 文档失败 {doc.get('name')}: {e}")
                continue
            sims = mat @ q
            order = np.argsort(-sims)[:top_k]
            for idx in order:
                sim = float(sims[idx])
                if sim >= min_sim:
                    text = chunk_data["chunks"][int(idx)] if int(idx) < len(chunk_data["chunks"]) else ""
                    hits.append({"doc": doc.get("name", ""), "sim": round(sim, 3), "text": text})
        hits.sort(key=lambda h: -h["sim"])
        return hits[:top_k]

    async def build_context(self, query: str) -> str:
        """检索并构建注入系统提示词的参考资料段落。"""
        if not self.config.get("rag_enabled", False) or not self.index:
            return ""
        hits = await self.search(query)
        if not hits:
            return ""
        max_chars = int(self.config.get("rag_max_context_chars", 1000))
        lines, used = [], 0
        for h in hits:
            piece = f"- ({h['doc']} 相关度{h['sim']}) {h['text']}"
            if used + len(piece) > max_chars:
                break
            lines.append(piece)
            used += len(piece)
        if not lines:
            return ""
        template = str(self.config.get("rag_context_template", "") or
                       "【参考资料】以下是知识库中可能相关的内容，回答时可以参考（不确定时以你的角色身份自然回答）：\n{refs}")
        return template.replace("{refs}", "\n".join(lines))


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _response_error(resp) -> str:
    """提取 HTTP 错误响应中的可读信息（Ollama 返回 {"error": "..."}），避免只报状态码。"""
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except Exception:
        pass
    text = (resp.text or "").strip()
    return text[:200] or f"HTTP {resp.status_code}"


def re_sub_spaces(text: str) -> str:
    import re
    return re.sub(r"[ \t\r]+", " ", text).strip()


def extract_text_from_file(path: Path) -> str:
    """从上传的文件提取纯文本（txt/md/code/json/csv 直接读取；pdf 依赖 pypdf）。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except ImportError:
            raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
