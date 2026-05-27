"""FastAPI Web 服务：把 Simple Agent 包成 HTTP 接口 + 简易聊天 UI。

启动：
    cd simple-agent && source .venv/bin/activate
    uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload

接口：
    GET  /                         → 聊天页面
    POST /api/chat                 → 同步返回最终答案
    POST /api/chat/stream          → SSE 流式返回工具调用过程与答案
    GET  /api/sessions             → 列出已持久化的会话
    DELETE /api/sessions/{sid}     → 删除某会话
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

# 让 `python web/server.py` 或 uvicorn 启动时都能 import 到同级的 tools/memory/llm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from llm import chat_with_retry  # noqa: E402
from memory import SessionStore, ShortTermMemory  # noqa: E402
from memory import long_term  # noqa: E402
from prompts import CURRENT as SYSTEM_PROMPT  # noqa: E402
from tools import TOOL_MAP, TOOLS, safe_call  # noqa: E402
from skill_loader import activate_skill, parse_skill_command  # noqa: E402

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SHORT_TERM_TURNS = int(os.getenv("SHORT_TERM_TURNS", "10"))
TOOL_WORKERS = int(os.getenv("TOOL_WORKERS", "4"))

# ---------- 持久化：所有会话共用一个 store ----------
_store = SessionStore(os.getenv("SESSION_DB_PATH", "sessions.db"))


def _build_memory_for(session_id: str) -> ShortTermMemory:
    """根据 session_id 把历史从 SQLite 灌到短期记忆里。"""
    mem = ShortTermMemory(max_turns=SHORT_TERM_TURNS)
    mem.set_system(SYSTEM_PROMPT)
    for m in _store.load(session_id, limit=40):
        mem.add(m)
    return mem


def _normalize_assistant(msg) -> dict:
    tool_calls = []
    for tc in (msg.tool_calls or []):
        tool_calls.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
        )
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": tool_calls or None,
    }


def _run_once(session_id: str, user_input: str):
    """一次完整的 Agent 推理，返回 (final_answer, trace_list)。"""
    from concurrent.futures import ThreadPoolExecutor

    mem = _build_memory_for(session_id)

    parsed = parse_skill_command(user_input)
    if parsed:
        skill_name, remaining = parsed
        body = activate_skill(skill_name)
        mem.add({"role": "system", "content": body})
        if not remaining:
            confirm = f"已加载技能: {skill_name}"
            trace = [{"type": "final", "content": confirm}]
            return confirm, trace
        user_input = remaining
    user_msg = {"role": "user", "content": user_input}
    mem.add(user_msg)
    _store.append(session_id, user_msg)

    trace = []
    for step in range(1, MAX_STEPS + 1):
        try:
            resp = chat_with_retry(mem.build_messages(), tools=TOOLS)
        except Exception as e:  # noqa: BLE001
            err = f"[LLM 调用失败] {type(e).__name__}: {e}"
            trace.append({"type": "final", "content": err})
            return err, trace

        msg = resp.choices[0].message
        asst = _normalize_assistant(msg)
        mem.add(asst)
        _store.append(session_id, asst)

        if not msg.tool_calls:
            trace.append({"type": "final", "content": asst["content"]})
            return asst["content"], trace

        # —— 并发执行同一轮的多个 tool_calls ——
        def _run_one(tc):
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                return tc, name, {}, f"[参数 JSON 解析失败] {e}"
            fn = TOOL_MAP.get(name)
            if fn is None:
                return tc, name, args, f"未知工具: {name}"
            result = safe_call(name, fn, **args)
            return tc, name, args, result

        if len(msg.tool_calls) == 1:
            results = [_run_one(msg.tool_calls[0])]
        else:
            with ThreadPoolExecutor(max_workers=TOOL_WORKERS) as ex:
                results = list(ex.map(_run_one, msg.tool_calls))

        for tc, name, args, result in results:
            trace.append({"type": "tool_call", "name": name, "arguments": args})
            trace.append({"type": "tool_result", "name": name, "content": str(result)})
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            mem.add(tool_msg)
            _store.append(session_id, tool_msg)

    return "[已达到最大步数，未得到最终答案]", trace


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="Simple Agent Web", version="0.1")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatOut(BaseModel):
    session_id: str
    answer: str
    trace: list


@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatOut)
def api_chat(body: ChatIn):
    sid = body.session_id or f"web-{uuid.uuid4().hex[:8]}"
    if not body.message.strip():
        raise HTTPException(400, "message 不能为空")
    answer, trace = _run_once(sid, body.message)
    return ChatOut(session_id=sid, answer=answer, trace=trace)


@app.post("/api/chat/stream")
async def api_chat_stream(body: ChatIn):
    """SSE：把 trace 的每一步实时推给前端。"""
    sid = body.session_id or f"web-{uuid.uuid4().hex[:8]}"
    if not body.message.strip():
        raise HTTPException(400, "message 不能为空")

    async def gen() -> AsyncGenerator[str, None]:
        yield _sse({"type": "session", "session_id": sid})
        loop = asyncio.get_running_loop()
        answer, trace = await loop.run_in_executor(None, _run_once, sid, body.message)
        for ev in trace:
            yield _sse(ev)
            await asyncio.sleep(0)  # 让出事件循环
        yield _sse({"type": "done", "answer": answer})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/sessions")
def list_sessions():
    return _store.list_sessions()


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    n = _store.delete(sid)
    return {"deleted": n}


@app.get("/api/skills")
def list_skills_api():
    from skill_loader import load_skills
    skills = load_skills()
    return [{"name": s.name, "description": s.description} for s in sorted(skills.values(), key=lambda s: s.name)]


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "long_term_enabled": long_term.is_enabled(),
        "max_steps": MAX_STEPS,
        "short_term_turns": SHORT_TERM_TURNS,
    }