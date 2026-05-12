"""Agent 主循环测试：mock llm.chat，离线驱动多轮推理。

必须在 import agent 之前把环境准备好：
- LLM_API_KEY 置为占位，避免 llm.py 在模块加载时报 RuntimeError
- 关闭会话持久化与长期记忆
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# —— 必须在 import agent 之前 ——
os.environ.setdefault("LLM_API_KEY", "sk-test-dummy")
os.environ["ENABLE_SESSION_STORE"] = "0"
os.environ["ENABLE_LONG_TERM"] = "0"


# ---------- 工具：构造假 LLM 响应 ----------
def _make_tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _make_response(content: str = "", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class AgentTest(unittest.TestCase):
    def setUp(self):
        # 每个用例前重置短期记忆
        import agent

        agent.short_mem.clear()

    def test_direct_answer(self):
        """LLM 不调用工具，直接给出答案。"""
        import agent

        with patch.object(agent, "chat_with_retry", return_value=_make_response("你好！")):
            ans = agent.run("在吗？", verbose=False)
        self.assertEqual(ans, "你好！")

    def test_tool_call_flow(self):
        """第一轮调用 calculator，第二轮给出最终答案。"""
        import agent

        responses = [
            _make_response(
                content="",
                tool_calls=[_make_tool_call("c1", "calculator", {"expression": "1+2"})],
            ),
            _make_response(content="结果是 3"),
        ]

        with patch.object(agent, "chat_with_retry", side_effect=responses) as mocked:
            ans = agent.run("算一下 1+2", verbose=False)

        self.assertEqual(ans, "结果是 3")
        self.assertEqual(mocked.call_count, 2)

        second_call_msgs = mocked.call_args_list[1].args[0]
        roles = [m["role"] for m in second_call_msgs]
        self.assertIn("tool", roles)
        tool_msg = next(m for m in second_call_msgs if m["role"] == "tool")
        self.assertEqual(tool_msg["content"], "3")

    def test_unknown_tool_graceful(self):
        """LLM 调用了不存在的工具，应当返回 '未知工具: xxx' 而不是崩溃。"""
        import agent

        responses = [
            _make_response(
                content="",
                tool_calls=[_make_tool_call("c1", "not_exist", {})],
            ),
            _make_response(content="抱歉，我处理不了。"),
        ]

        with patch.object(agent, "chat_with_retry", side_effect=responses) as mocked:
            ans = agent.run("做件事", verbose=False)

        self.assertEqual(ans, "抱歉，我处理不了。")
        second_msgs = mocked.call_args_list[1].args[0]
        tool_msg = next(m for m in second_msgs if m["role"] == "tool")
        self.assertIn("未知工具", tool_msg["content"])

    def test_max_steps_guard(self):
        """LLM 一直返回 tool_calls，防死循环：应当在 MAX_STEPS 后优雅退出。"""
        import agent

        endless = _make_response(
            content="",
            tool_calls=[_make_tool_call("cX", "calculator", {"expression": "1+1"})],
        )

        with patch.object(agent, "chat_with_retry", return_value=endless) as mocked:
            ans = agent.run("无限循环", verbose=False)

        self.assertIn("最大步数", ans)
        self.assertEqual(mocked.call_count, agent.MAX_STEPS)

    def test_short_memory_updated(self):
        """一轮对话后，短期记忆里应当包含 user 与 assistant 两类消息。"""
        import agent

        with patch.object(agent, "chat_with_retry", return_value=_make_response("hi")):
            agent.run("hello", verbose=False)
        roles = [m["role"] for m in agent.short_mem.recent]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_llm_failure_returns_error_string(self):
        """LLM 重试全部失败时，run 应优雅返回错误串，不崩溃。"""
        import agent

        with patch.object(agent, "chat_with_retry", side_effect=RuntimeError("network down")):
            ans = agent.run("hi", verbose=False)
        self.assertIn("LLM 调用失败", ans)
        self.assertIn("network down", ans)

    def test_parallel_tool_calls(self):
        """一轮内多个 tool_calls 应当都被执行，结果都进短期记忆。"""
        import agent

        responses = [
            _make_response(
                content="",
                tool_calls=[
                    _make_tool_call("c1", "calculator", {"expression": "1+1"}),
                    _make_tool_call("c2", "calculator", {"expression": "2+2"}),
                    _make_tool_call("c3", "get_current_time", {}),
                ],
            ),
            _make_response(content="好了"),
        ]
        with patch.object(agent, "chat_with_retry", side_effect=responses) as mocked:
            ans = agent.run("同时算两题并报时", verbose=False)
        self.assertEqual(ans, "好了")
        second_msgs = mocked.call_args_list[1].args[0]
        tool_msgs = [m for m in second_msgs if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 3)
        contents = {m["content"] for m in tool_msgs}
        self.assertIn("2", contents)   # 1+1
        self.assertIn("4", contents)   # 2+2


if __name__ == "__main__":
    unittest.main()