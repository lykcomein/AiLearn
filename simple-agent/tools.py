'''
Author: lyk
Date: 2026-05-07 15:51:11
LastEditTime: 2026-05-09 18:01:57
brief: Do not edit
details: Do not edit
'''
"""工具定义：每个工具 = 一个 Python 函数 + 一份 JSON Schema 描述。"""

from datetime import datetime

from skill_loader import activate_skill as _activate_skill


# ===== 工具实现 =====
def get_weather(city: str) -> str:
    """模拟天气查询（演示用，实际可接入和风/心知等 API）。"""
    return f"{city} 今天晴，气温 25℃，微风。"


def get_current_time() -> str:
    """返回当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    """执行简单数学表达式，例如 '1+2*3'。"""
    try:
        # 仅允许数字和基本运算符，安全起见
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "表达式包含非法字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算失败: {e}"


def activate_skill(name: str) -> str:
    """加载指定 skill 的详细说明，把内容回灌给 LLM 作为后续指令。

    Args:
        name: skill 名称（对应 skills/<dir>/SKILL.md 中 frontmatter 的 name 字段）
    """
    return _activate_skill(name)



# ===== 异常安全的调度 =====
def safe_call(name, fn, **kwargs) -> str:
    """统一把工具执行异常转成字符串回灌给 LLM，避免主循环崩溃。

    - 返回值强制转 str，满足 OpenAI tool message 的 content 字段要求；
    - 异常不抛出，LLM 能拿着错误信息再决定下一步（换工具 / 问用户 / 认错）。
    """
    try:
        result = fn(**kwargs)
        return "" if result is None else str(result)
    except TypeError as e:
        # 参数不匹配（LLM 瞎给参数）
        return f"[工具参数错误] {name}: {e}"
    except Exception as e:  # noqa: BLE001
        return f"[工具异常] {name} {type(e).__name__}: {e}"


# ===== 工具 Schema（给 LLM 看的说明书）=====
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学表达式计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_skill",
            "description": (
                "加载一个技能（skill）的详细操作说明。"
                "当你在 system prompt 中看到某个 skill 的描述匹配当前任务时，"
                "调用此工具把该 skill 的完整指令加载进来，然后严格按照返回的指令执行。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "skill 名称，必须是 system prompt 中列出的可用 skill 之一",
                    }
                },
                "required": ["name"],
            },
        },
    },
]

# 名称 -> 函数 映射，供 agent 调度使用
TOOL_MAP = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "calculator": calculator,
    "activate_skill": activate_skill,
}