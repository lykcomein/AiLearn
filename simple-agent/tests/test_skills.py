"""Skill 系统的单元测试。

覆盖点：
1. 扫描空目录 → 返回空索引
2. 正常 frontmatter 解析（name + description）
3. 缺 frontmatter 时用目录名兜底
4. activate_skill 正常返回正文
5. activate_skill 未找到时返回友好错误串
6. build_skill_index_prompt 输出格式正确
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skill_loader import (  # noqa: E402
    activate_skill,
    build_skill_index_prompt,
    load_skills,
    parse_skill_command,
)


class SkillLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.skills_dir = self.tmpdir.name

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_skill(self, dir_name: str, content: str) -> None:
        skill_dir = os.path.join(self.skills_dir, dir_name)
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(content)

    # ---------- load_skills ----------
    def test_empty_dir_returns_empty(self) -> None:
        self.assertEqual(load_skills(self.skills_dir), {})

    def test_nonexistent_dir_returns_empty(self) -> None:
        """目录根本不存在时也不应抛异常。"""
        self.assertEqual(load_skills("/tmp/__not_exist_xxxxx__"), {})

    def test_parse_valid_frontmatter(self) -> None:
        self._write_skill(
            "my-skill",
            "---\nname: my-skill\ndescription: 测试用 skill\n---\n# 正文\n你好",
        )
        skills = load_skills(self.skills_dir)
        self.assertIn("my-skill", skills)
        self.assertEqual(skills["my-skill"].description, "测试用 skill")
        self.assertTrue(skills["my-skill"].path.endswith("SKILL.md"))

    def test_missing_frontmatter_uses_dirname(self) -> None:
        """没有 frontmatter 时，用目录名作为 name 兜底，description 为空。"""
        self._write_skill("no-meta", "# 仅有正文，没 frontmatter")
        skills = load_skills(self.skills_dir)
        self.assertIn("no-meta", skills)
        self.assertEqual(skills["no-meta"].description, "")

    def test_skip_dirs_without_skill_md(self) -> None:
        """目录里没有 SKILL.md 的应被跳过，不影响其他 skill。"""
        os.makedirs(os.path.join(self.skills_dir, "empty-dir"))
        self._write_skill("good", "---\nname: good\ndescription: ok\n---\n正文")
        skills = load_skills(self.skills_dir)
        self.assertEqual(set(skills.keys()), {"good"})

    # ---------- activate_skill ----------
    def test_activate_returns_body(self) -> None:
        self._write_skill(
            "demo",
            "---\nname: demo\ndescription: 演示\n---\n# 标题\n这是正文内容",
        )
        body = activate_skill("demo", self.skills_dir)
        self.assertIn("这是正文内容", body)
        # frontmatter 不应出现在正文里
        self.assertNotIn("name: demo", body)

    def test_activate_unknown_returns_friendly_error(self) -> None:
        self._write_skill("demo", "---\nname: demo\ndescription: x\n---\n正文")
        msg = activate_skill("not-there", self.skills_dir)
        self.assertIn("未找到", msg)
        # 应列出可用 skill 提示
        self.assertIn("demo", msg)

    def test_activate_on_empty_dir(self) -> None:
        msg = activate_skill("any", self.skills_dir)
        self.assertIn("未找到", msg)
        self.assertIn("(无)", msg)

    # ---------- build_skill_index_prompt ----------
    def test_build_index_prompt_format(self) -> None:
        self._write_skill("a", "---\nname: a\ndescription: A skill\n---\n")
        self._write_skill("b", "---\nname: b\ndescription: B skill\n---\n")
        skills = load_skills(self.skills_dir)
        prompt = build_skill_index_prompt(skills)
        self.assertIn("可用 skills", prompt)
        self.assertIn("- a: A skill", prompt)
        self.assertIn("- b: B skill", prompt)

    def test_build_index_prompt_empty(self) -> None:
        self.assertEqual(build_skill_index_prompt({}), "")

    # ---------- parse_skill_command ----------
    def test_parse_skill_command_valid(self) -> None:
        self._write_skill("demo", "---\nname: demo\ndescription: x\n---\n正文")
        result = parse_skill_command("/demo", self.skills_dir)
        self.assertIsNotNone(result)
        self.assertEqual(result, ("demo", ""))

    def test_parse_skill_command_with_message(self) -> None:
        self._write_skill("demo", "---\nname: demo\ndescription: x\n---\n正文")
        result = parse_skill_command("/demo 帮我分析数据", self.skills_dir)
        self.assertIsNotNone(result)
        self.assertEqual(result, ("demo", "帮我分析数据"))

    def test_parse_skill_command_unknown_skill(self) -> None:
        self._write_skill("demo", "---\nname: demo\ndescription: x\n---\n正文")
        result = parse_skill_command("/not-exist", self.skills_dir)
        self.assertIsNone(result)

    def test_parse_skill_command_not_a_command(self) -> None:
        result = parse_skill_command("普通消息", self.skills_dir)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()