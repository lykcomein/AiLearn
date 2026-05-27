"""
Skill 加载器：扫描 skills/ 目录，解析 SKILL.md 的 frontmatter 并提供激活能力。

SKILL.md 格式：
    ---
    name: my-skill
    description: 什么场景下使用这个 skill
    ---
    # 详细说明
    正文内容...

两阶段设计：
- 启动时仅读取 frontmatter（name + description），构建轻量索引注入 system prompt
- LLM 调用 activate_skill(name) 时再把完整正文回灌给下一轮对话
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


# 默认 skills 根目录（模块所在目录的 skills/ 子目录）
DEFAULT_SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


@dataclass
class SkillMeta:
    """Skill 的轻量元信息，用于索引。"""

    name: str
    description: str
    path: str  # SKILL.md 绝对路径


# frontmatter 正则：匹配文件开头的 --- ... --- 块
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    从 SKILL.md 内容中分离 frontmatter 和正文。

    Returns:
        (frontmatter_dict, body) —— 没有 frontmatter 时返回 ({}, 全文)
    """
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}, content

    raw = m.group(1)
    body = content[m.end():]

    # 简易 YAML 解析：只支持 key: value 单行格式（skill 场景已够用）
    meta: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")

    return meta, body


def load_skills(skills_dir: str | None = None) -> dict[str, SkillMeta]:
    """
    扫描 skills_dir 下所有子目录，读取 SKILL.md 的 frontmatter，返回 {name: SkillMeta} 索引。

    扫描规则：
    - 只认目录形式：skills/<any-name>/SKILL.md
    - frontmatter 中必须有 name 字段，否则该 skill 被跳过
    - 解析失败的 skill 不中断流程，仅忽略

    Args:
        skills_dir: skills 根目录，默认 ./skills/
    """
    skills_dir = skills_dir or DEFAULT_SKILLS_DIR
    index: dict[str, SkillMeta] = {}

    if not os.path.isdir(skills_dir):
        return index

    for entry in sorted(os.listdir(skills_dir)):
        skill_md = os.path.join(skills_dir, entry, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue

        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                content = f.read()
            meta, _ = _parse_frontmatter(content)
        except Exception:
            continue

        name = meta.get("name") or entry  # 没写 name 就用目录名兜底
        description = meta.get("description", "")

        index[name] = SkillMeta(
            name=name,
            description=description,
            path=skill_md,
        )

    return index


def activate_skill(name: str, skills_dir: str | None = None) -> str:
    """
    激活一个 skill：返回其 SKILL.md 完整正文（含 frontmatter 以下部分）。

    Args:
        name: skill 名称（对应 frontmatter 里的 name）

    Returns:
        正文字符串，找不到时返回错误提示字符串（不抛异常，方便回灌给 LLM）
    """
    index = load_skills(skills_dir)
    meta = index.get(name)
    if not meta:
        available = ", ".join(sorted(index.keys())) or "(无)"
        return f"[skill 未找到] 请求的 skill={name}；当前可用：{available}"

    try:
        with open(meta.path, "r", encoding="utf-8") as f:
            content = f.read()
        _, body = _parse_frontmatter(content)
        return body.strip() or "[skill 正文为空]"
    except Exception as e:
        return f"[skill 读取失败] {name}: {e}"


def parse_skill_command(user_input: str, skills_dir: str | None = None) -> tuple[str, str] | None:
    """
    解析 /skill_name 命令。

    Args:
        user_input: 用户原始输入，如 "/csv-analyzer 分析数据"

    Returns:
        匹配到已注册的 skill 时返回 (skill_name, remaining_message)，否则返回 None。
    """
    stripped = user_input.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped[1:].split(None, 1)
    if not parts:
        return None

    name = parts[0]
    remaining = parts[1] if len(parts) > 1 else ""

    index = load_skills(skills_dir)
    if name not in index:
        return None

    return name, remaining


def build_skill_index_prompt(skills: dict[str, SkillMeta]) -> str:
    """
    构造注入 system prompt 的 skill 列表，每个 skill 一行。

    示例输出：
        可用 skills（需要时调用 activate_skill(name=...) 加载详细说明）：
        - csv-analyzer: 当用户需要分析 CSV 文件...
        - sql-formatter: 当用户需要格式化 SQL...
    """
    if not skills:
        return ""
    lines = ["可用 skills（需要时调用 activate_skill(name=...) 加载详细说明）："]
    for meta in sorted(skills.values(), key=lambda s: s.name):
        lines.append(f"- {meta.name}: {meta.description}")
    return "\n".join(lines)