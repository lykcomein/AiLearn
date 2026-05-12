"""长期记忆：基于 Chroma 的向量库，用于跨会话语义检索。

依赖（可选，未安装则本模块退化为 no-op，不影响主流程）：
    pip install -i https://mirrors.aliyun.com/pypi/simple/ chromadb

Embedding 方案（按 .env 切换）：
- EMBED_PROVIDER=openai   （默认，需 EMBED_API_KEY + EMBED_BASE_URL + EMBED_MODEL）
- EMBED_PROVIDER=hash     （无依赖、纯本地哈希嵌入，仅用于 demo，不保证语义质量）
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

# ---------- 可选依赖 ----------
try:
    import chromadb  # type: ignore

    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False

_col = None  # chroma collection 单例


def _get_collection():
    global _col
    if _col is not None:
        return _col
    if not _CHROMA_OK:
        return None
    path = os.getenv("LONG_TERM_DB_PATH", "./chroma_store")
    client = chromadb.PersistentClient(path=path)
    _col = client.get_or_create_collection("agent_memory")
    return _col


# ---------- Embedding ----------
def _embed(text: str) -> list:
    provider = os.getenv("EMBED_PROVIDER", "openai").lower()
    if provider == "hash":
        return _hash_embed(text)
    return _openai_compatible_embed(text)


def _openai_compatible_embed(text: str) -> list:
    from openai import OpenAI  # noqa: WPS433

    api_key = os.getenv("EMBED_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("EMBED_BASE_URL") or os.getenv("LLM_BASE_URL")
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    if not api_key:
        raise RuntimeError("long_term: 未配置 EMBED_API_KEY，无法生成 embedding")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def _hash_embed(text: str, dim: int = 256) -> list:
    """无依赖的 demo 级嵌入：把文本哈希桶化为固定维向量。
    注意：语义能力很弱，仅在没有 embedding API 时用于打通链路。
    """
    vec = [0.0] * dim
    for tok in text.split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 归一化
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


# ---------- 对外 API ----------
def is_enabled() -> bool:
    """外部可用它判断长期记忆是否可用。"""
    return _CHROMA_OK


def remember(text: str, meta: Optional[dict] = None, mem_id: Optional[str] = None) -> bool:
    """写入一条长期记忆。未安装 chromadb 时静默跳过并返回 False。"""
    col = _get_collection()
    if col is None or not text.strip():
        return False
    mid = mem_id or "mem-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    # chromadb 要求 metadata 非空，没有则补一个占位字段
    safe_meta = dict(meta) if meta else {}
    if not safe_meta:
        safe_meta = {"_": ""}
    col.upsert(
        documents=[text],
        embeddings=[_embed(text)],
        metadatas=[safe_meta],
        ids=[mid],
    )
    return True


def recall(query: str, k: int = 3) -> list:
    """按语义相似度召回最相关的 k 条记忆文本。"""
    col = _get_collection()
    if col is None or not query.strip():
        return []
    try:
        r = col.query(query_embeddings=[_embed(query)], n_results=k)
    except Exception:
        return []
    docs = r.get("documents") or []
    return docs[0] if docs else []


def forget(mem_id: str) -> bool:
    col = _get_collection()
    if col is None:
        return False
    col.delete(ids=[mem_id])
    return True


def clear_all() -> None:
    """清空整个长期记忆集合（慎用，一般仅测试用）。"""
    col = _get_collection()
    if col is None:
        return
    # chromadb 不接受空 where，改用"取回所有 id 再删除"的方式
    try:
        data = col.get()
    except Exception:
        return
    ids = data.get("ids") or []
    if ids:
        col.delete(ids=ids)