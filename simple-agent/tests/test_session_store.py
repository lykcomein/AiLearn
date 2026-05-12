"""SessionStore 测试：每个用例使用临时 db，互不干扰。"""

import os
import tempfile
import unittest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.session_store import SessionStore


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = SessionStore(self.db_path)

    def tearDown(self):
        self.store.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_append_and_load(self):
        self.store.append("s1", {"role": "user", "content": "hi"})
        self.store.append("s1", {"role": "assistant", "content": "hello"})
        msgs = self.store.load("s1")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["content"], "hello")

    def test_session_isolation(self):
        self.store.append("a", {"role": "user", "content": "in-a"})
        self.store.append("b", {"role": "user", "content": "in-b"})
        self.assertEqual(len(self.store.load("a")), 1)
        self.assertEqual(self.store.load("b")[0]["content"], "in-b")

    def test_extra_field_roundtrip(self):
        """tool_call_id 这类非 role/content 字段应当可以来回。"""
        self.store.append(
            "s",
            {"role": "tool", "tool_call_id": "tc-123", "content": "result"},
        )
        msg = self.store.load("s")[0]
        self.assertEqual(msg["tool_call_id"], "tc-123")

    def test_load_limit_and_order(self):
        for i in range(10):
            self.store.append("s", {"role": "user", "content": f"m{i}"})
        msgs = self.store.load("s", limit=4)
        self.assertEqual(len(msgs), 4)
        # load 返回升序，最后一条应是 m9
        self.assertEqual(msgs[-1]["content"], "m9")

    def test_delete(self):
        self.store.append("s", {"role": "user", "content": "x"})
        self.store.append("s", {"role": "user", "content": "y"})
        n = self.store.delete("s")
        self.assertEqual(n, 2)
        self.assertEqual(self.store.load("s"), [])

    def test_list_sessions(self):
        self.store.append("a", {"role": "user", "content": "1"})
        self.store.append("b", {"role": "user", "content": "2"})
        ids = {s["session_id"] for s in self.store.list_sessions()}
        self.assertEqual(ids, {"a", "b"})


if __name__ == "__main__":
    unittest.main()