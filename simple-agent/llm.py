"""LLM 调用封装：兼容 OpenAI / DeepSeek / 通义等所有 OpenAI 协议的服务。

特性：
- 自动从 .env 读配置；
- chat() 默认带超时；
- chat_with_retry() 对网络/限流/5xx 做指数退避重试。
"""

from __future__ import annotations

import os
import random
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

# 默认使用 DeepSeek（国内可直连），可在 .env 中覆盖
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
RETRIES = int(os.getenv("LLM_RETRIES", "3"))

if not API_KEY:
    raise RuntimeError("请在 .env 中设置 LLM_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)


# ---------- 可重试错误集合 ----------
# 网络问题、限流、超时、5xx 都属于"可重试"
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)


def _is_retryable_apierror(err: APIError) -> bool:
    """APIError 里只有 5xx 算可重试，4xx 是参数问题，重试也没用。"""
    status = getattr(err, "status_code", None)
    return status is None or status >= 500


def chat(messages, tools=None, **extra):
    """单次调用，不重试。供测试或调用方自己控制重试用。"""
    kwargs = {"model": MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    kwargs.update(extra)
    return client.chat.completions.create(**kwargs)


def chat_with_retry(messages, tools=None, retries: int | None = None, **extra):
    """带指数退避的安全调用。

    - 仅对网络/限流/超时/5xx 重试；
    - 退避：1s, 2s, 4s ...，最多 ±0.5s 抖动。
    """
    n = RETRIES if retries is None else retries
    last_err: Exception | None = None
    for i in range(n):
        try:
            return chat(messages, tools=tools, **extra)
        except _RETRYABLE as e:
            last_err = e
        except APIError as e:
            if not _is_retryable_apierror(e):
                raise
            last_err = e
        wait = (2 ** i) + random.random() * 0.5
        print(f"[llm.retry] 第 {i + 1}/{n} 次失败：{type(last_err).__name__} → {wait:.1f}s 后重试")
        time.sleep(wait)
    # 全部失败 → 抛出最后一次异常
    assert last_err is not None
    raise last_err