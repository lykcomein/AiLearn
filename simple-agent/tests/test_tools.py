"""工具函数纯单测。"""

import re
import unittest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import TOOLS, TOOL_MAP, calculator, get_current_time, get_weather, safe_call


class ToolsTest(unittest.TestCase):
    def test_get_weather(self):
        ans = get_weather("北京")
        self.assertIn("北京", ans)

    def test_get_current_time_format(self):
        ans = get_current_time()
        self.assertRegex(ans, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

    def test_calculator_basic(self):
        self.assertEqual(calculator("1+2*3"), "7")
        self.assertEqual(calculator("(12+8)*5/4"), "25.0")

    def test_calculator_reject_illegal(self):
        self.assertIn("非法", calculator("__import__('os').system('ls')"))

    def test_calculator_division_error(self):
        result = calculator("1/0")
        self.assertTrue(result.startswith("计算失败"))

    def test_tools_schema_consistency(self):
        for t in TOOLS:
            name = t["function"]["name"]
            self.assertIn(name, TOOL_MAP, f"工具 {name} 未在 TOOL_MAP 注册")

    def test_tools_schema_has_required(self):
        for t in TOOLS:
            self.assertEqual(t["type"], "function")
            self.assertIn("description", t["function"])
            self.assertIn("parameters", t["function"])


class SafeCallTest(unittest.TestCase):
    def test_normal_returns_str(self):
        self.assertEqual(safe_call("calc", calculator, expression="1+2"), "3")

    def test_none_returns_empty(self):
        self.assertEqual(safe_call("x", lambda: None), "")

    def test_type_error_is_caught(self):
        # 少传必需参数
        out = safe_call("calc", calculator)
        self.assertIn("工具参数错误", out)

    def test_runtime_error_is_caught(self):
        def boom():
            raise RuntimeError("炸了")

        out = safe_call("boom", boom)
        self.assertIn("工具异常", out)
        self.assertIn("炸了", out)


if __name__ == "__main__":
    unittest.main()