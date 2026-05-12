"""长期记忆测试：用 hash embedding，避免依赖外部 embedding 服务。
若未安装 chromadb，整套用例自动跳过。"""

import os
import shutil
import tempfile
import unittest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import long_term


@unittest.skipUnless(long_term.is_enabled(), "未安装 chromadb，跳过长期记忆测试")
class LongTermTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 整个测试类共享一个临时持久化目录
        cls.tmp_dir = tempfile.mkdtemp(prefix="chroma_test_")
        os.environ["LONG_TERM_DB_PATH"] = cls.tmp_dir
        os.environ["EMBED_PROVIDER"] = "hash"
        # 重置单例，让上面的环境变量生效
        long_term._col = None  # type: ignore[attr-defined]

    @classmethod
    def tearDownClass(cls):
        long_term.clear_all()
        long_term._col = None  # type: ignore[attr-defined]
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_remember_and_recall(self):
        ok = long_term.remember("用户喜欢喝美式咖啡", meta={"tag": "preference"})
        self.assertTrue(ok)
        long_term.remember("北京今天下雨", meta={"tag": "weather"})

        hits = long_term.recall("用户偏好什么咖啡？", k=2)
        self.assertTrue(any("咖啡" in h for h in hits))

    def test_remember_empty_text(self):
        self.assertFalse(long_term.remember(""))

    def test_recall_empty_query(self):
        self.assertEqual(long_term.recall(""), [])

    def test_forget(self):
        long_term.remember("用户住在深圳", mem_id="mem-test-forget")
        # 召回应该能命中
        hits1 = long_term.recall("用户住哪里？", k=3)
        self.assertTrue(any("深圳" in h for h in hits1))
        # 删除后再召回不应再有这条（用唯一关键词避免命中其它残留）
        long_term.forget("mem-test-forget")
        hits2 = long_term.recall("住在深圳", k=3)
        self.assertFalse(any("用户住在深圳" == h for h in hits2))


class LongTermDisabledTest(unittest.TestCase):
    """模拟 chromadb 未安装的退化路径。"""

    def test_remember_returns_false_when_disabled(self):
        # 直接强制把 _col 置为 None 并且把 _CHROMA_OK 视为 False 不太干净，
        # 用替身函数验证：当 _get_collection 返回 None 时，remember 应安全跳过
        original = long_term._get_collection
        long_term._get_collection = lambda: None  # type: ignore[assignment]
        try:
            self.assertFalse(long_term.remember("x"))
            self.assertEqual(long_term.recall("x"), [])
        finally:
            long_term._get_collection = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()