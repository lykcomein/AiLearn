"""Agent 主循环（含三层记忆 + 重试 + 并发工具）：

- 短期记忆：始终启用
- 会话持久化：ENABLE_SESSION_STORE=1 时启用
- 长期向量记忆：ENABLE_LONG_TERM=1 时启用（需 chromadb）
- LLM 失败自动重试；工具异常不会中断主循环
- 一轮多个 tool_calls 时用线程池并发执行
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from llm import chat_with_retry
from memory import SessionStore, ShortTermMemory, long_term
from prompts import CURRENT as SYSTEM_PROMPT
from skill_loader import activate_skill, parse_skill_command
from tools import TOOL_MAP, TOOLS, safe_call

load_dotenv()

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TOOL_WORKERS = int(os.getenv("TOOL_WORKERS", "4"))


# ---------- 初始化三层记忆 ----------
short_mem = ShortTermMemory(max_turns=int(os.getenv("SHORT_TERM_TURNS", "10")))
short_mem.set_system(SYSTEM_PROMPT)

SESSION_ID = os.getenv("SESSION_ID") or f"sess-{uuid.uuid4().hex[:8]}"
_session_store = None
if os.getenv("ENABLE_SESSION_STORE", "0") == "1":
    _session_store = SessionStore(os.getenv("SESSION_DB_PATH", "sessions.db"))
    history = _session_store.load(SESSION_ID, limit=int(os.getenv("SESSION_LOAD_LIMIT", "40")))
    for m in history:
        short_mem.add(m)
    if history:
        print(f"[会话 {SESSION_ID}] 已恢复 {len(history)} 条历史消息")

_long_term_on = os.getenv("ENABLE_LONG_TERM", "0") == "1" and long_term.is_enabled()
if os.getenv("ENABLE_LONG_TERM", "0") == "1" and not long_term.is_enabled():
    print("[提示] 已启用长期记忆但未安装 chromadb，已自动降级为不可用。")


# ---------- 工具函数 ----------
def _persist(msg: dict) -> None:
    if _session_store is not None:
        _session_store.append(SESSION_ID, msg)


def _recall_and_inject(user_input: str) -> None:
    if not _long_term_on:
        return
    hits = long_term.recall(user_input, k=3)
    if hits:
        text = "【相关历史记忆】\n" + "\n".join(f"- {h}" for h in hits)
        short_mem.add({"role": "system", "content": text})


def _maybe_remember(user_input: str, assistant_reply: str) -> None:
    if not _long_term_on or not assistant_reply:
        return
    KEYWORDS = ("我叫", "我住", "我喜欢", "我的", "偏好", "记住")
    if any(k in user_input for k in KEYWORDS) or len(assistant_reply) > 200:
        long_term.remember(
            text=f"用户: {user_input}\n助手: {assistant_reply}",
            meta={"session_id": SESSION_ID},
        )


def _execute_tools_parallel(tool_calls, verbose: bool = False):
    """并发执行 tool_calls，返回 [(tc, result_str), ...] 保持原顺序。"""
    def _run(tc):
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as e:
            return tc, f"[参数 JSON 解析失败] {e}"
        if verbose:
            print(f"  → 调用工具 {name}({args})")
        fn = TOOL_MAP.get(name)
        if fn is None:
            return tc, f"未知工具: {name}"
        return tc, safe_call(name, fn, **args)

    if len(tool_calls) == 1:
        # 单个直接跑，省掉线程开销
        return [_run(tool_calls[0])]

    with ThreadPoolExecutor(max_workers=TOOL_WORKERS) as ex:
        return list(ex.map(_run, tool_calls))


# ---------- 主循环 ----------
def run(user_input: str, verbose: bool = True) -> str:
    _recall_and_inject(user_input)

    user_msg = {"role": "user", "content": user_input}
    short_mem.add(user_msg)
    _persist(user_msg)

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n[第 {step} 轮] 调用 LLM ...")

        try:
            resp = chat_with_retry(short_mem.build_messages(), tools=TOOLS)
        except Exception as e:  # noqa: BLE001
            err = f"[LLM 调用失败] {type(e).__name__}: {e}"
            if verbose:
                print(err)
            return err

        msg = resp.choices[0].message

        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ] or None,
        }
        short_mem.add(assistant_msg)
        _persist(assistant_msg)

        if not msg.tool_calls:
            if verbose:
                print(f"[最终答案] {msg.content}")
            _maybe_remember(user_input, msg.content or "")
            return msg.content or ""

        for tc, result in _execute_tools_parallel(msg.tool_calls, verbose=verbose):
            if verbose:
                print(f"  ← 结果: {result}")
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            short_mem.add(tool_msg)
            _persist(tool_msg)

    return "[已达到最大步数，未得到最终答案]"


if __name__ == "__main__":
    print("=" * 50)
    print(f"Simple Agent 已启动 | session_id = {SESSION_ID}")
    print(
        "短期记忆: ON | "
        f"会话持久化: {'ON' if _session_store else 'OFF'} | "
        f"长期记忆: {'ON' if _long_term_on else 'OFF'} | "
        f"并发工具: {TOOL_WORKERS}"
    )
    print("输入 q 退出。")
    print("=" * 50)
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue

        parsed = parse_skill_command(q)
        if parsed:
            skill_name, remaining = parsed
            body = activate_skill(skill_name)
            short_mem.add({"role": "system", "content": body})
            print(f"\n[已加载技能: {skill_name}]")
            if not remaining:
                continue
            q = remaining

        ans = run(q)
        print(f"\nAgent: {ans}")