"""ShortTermMemory 测试：注入假摘要函数，避免联网。"""

import unittest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.short_term import ShortTermMemory


def fake_summarize(old_msgs):
    """把 N 条消息压成一个固定串，方便断言。"""
    return f"SUMMARY({len(old_msgs)})"


class ShortTermMemoryTest(unittest.TestCase):
    def test_set_system_and_build(self):
        m = ShortTermMemory(max_turns=3, summarize_fn=fake_summarize)
        m.set_system("you are helpful")
        self.assertEqual(m.build_messages(), [{"role": "system", "content": "you are helpful"}])

    def test_add_below_threshold_no_summary(self):
        m = ShortTermMemory(max_turns=3, summarize_fn=fake_summarize)
        m.set_system("sys")
        for i in range(5):
            m.add({"role": "user", "content": f"msg{i}"})
        self.assertEqual(m.summary, "")
        self.assertEqual(len(m.recent), 5)

    def test_add_triggers_summary(self):
        """max_turns=2 → 阈值 4，加到第 5 条触发压缩。"""
        m = ShortTermMemory(max_turns=2, summarize_fn=fake_summarize)
        m.set_system("sys")
        for i in range(5):
            m.add({"role": "user", "content": f"m{i}"})

        # 应当压缩了前一半
        self.assertTrue(m.summary.startswith("SUMMARY("))
        # 剩下的应该是后一半 + 新加的最后一条
        self.assertTrue(len(m.recent) <= 4)

    def test_build_messages_order(self):
        m = ShortTermMemory(max_turns=2, summarize_fn=fake_summarize)
        m.set_system("sys")
        for i in range(6):
            m.add({"role": "user", "content": f"m{i}"})

        msgs = m.build_messages()
        # 顺序：system → 历史摘要 → recent
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("【历史摘要】", msgs[1]["content"])
        # recent 部分都是 user
        self.assertTrue(all(x["role"] == "user" for x in msgs[2:]))

    def test_clear(self):
        m = ShortTermMemory(max_turns=2, summarize_fn=fake_summarize)
        m.set_system("sys")
        for i in range(6):
            m.add({"role": "user", "content": f"m{i}"})
        m.clear()
        self.assertEqual(m.summary, "")
        self.assertEqual(m.recent, [])
        # system 仍在
        self.assertEqual(m.build_messages(), [{"role": "system", "content": "sys"}])

    def test_summary_accumulates(self):
        """连续两次触发压缩，summary 应当累加，而不是覆盖。"""
        m = ShortTermMemory(max_turns=2, summarize_fn=fake_summarize)
        for i in range(20):
            m.add({"role": "user", "content": f"m{i}"})
        # 多次 SUMMARY( 出现，证明累加
        self.assertGreaterEqual(m.summary.count("SUMMARY("), 2)


if __name__ == "__main__":
    unittest.main()