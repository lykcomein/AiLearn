"""chat_with_retry 测试：mock 底层 chat，验证重试策略。"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LLM_API_KEY", "sk-test-dummy")

import llm  # noqa: E402


class _FakeResp:
    pass


class _FakeAPIError(Exception):
    """模拟 openai.APIError 的可重试子类，status_code 可配置。"""
    def __init__(self, status_code):
        super().__init__(f"status={status_code}")
        self.status_code = status_code


class LLMRetryTest(unittest.TestCase):
    def setUp(self):
        # 加速测试：不要真的 sleep
        self._orig_sleep = llm.time.sleep
        llm.time.sleep = lambda *_: None

    def tearDown(self):
        llm.time.sleep = self._orig_sleep

    def test_success_no_retry(self):
        fake = _FakeResp()
        with patch.object(llm, "chat", return_value=fake) as m:
            out = llm.chat_with_retry(["msg"])
        self.assertIs(out, fake)
        self.assertEqual(m.call_count, 1)

    def test_retry_on_connection_error_then_success(self):
        calls = [llm.APIConnectionError(request=None), _FakeResp()]
        def side(*a, **kw):
            r = calls.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        with patch.object(llm, "chat", side_effect=side) as m:
            out = llm.chat_with_retry(["msg"], retries=3)
        self.assertIsInstance(out, _FakeResp)
        self.assertEqual(m.call_count, 2)

    def test_give_up_after_max_retries(self):
        with patch.object(llm, "chat", side_effect=llm.APITimeoutError(request=None)):
            with self.assertRaises(llm.APITimeoutError):
                llm.chat_with_retry(["msg"], retries=2)

    def test_non_retryable_4xx_raises_immediately(self):
        err = _FakeAPIError(400)
        # 直接 patch _is_retryable_apierror 让它根据 status_code 判断
        with patch.object(llm, "chat", side_effect=err), \
             patch.object(llm, "APIError", _FakeAPIError):
            with self.assertRaises(_FakeAPIError):
                llm.chat_with_retry(["msg"], retries=3)


if __name__ == "__main__":
    unittest.main()