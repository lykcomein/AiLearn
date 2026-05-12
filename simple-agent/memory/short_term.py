"""短期记忆：带窗口 + 自动摘要压缩。

设计要点：
- 始终保留 system prompt
- 最近 N 轮消息原样保留
- 更早的消息用 LLM 摘要成要点，拼在 system 之后
"""

from typing import Optional


class ShortTermMemory:
    def __init__(self, max_turns: int = 10, summarize_fn=None):
        """
        :param max_turns: 保留的最近"轮"数（一轮≈一次 user+assistant）
        :param summarize_fn: 注入的摘要函数，签名 f(old_msgs: list[dict]) -> str
                             默认延迟从 llm.chat 动态构造，避免循环导入
        """
        self.max_turns = max_turns
        self.system: Optional[dict] = None
        self.summary: str = ""
        self.recent: list = []
        self._summarize_fn = summarize_fn

    # ---------- 对外 API ----------
    def set_system(self, prompt: str) -> None:
        self.system = {"role": "system", "content": prompt}

    def add(self, message: dict) -> None:
        """追加一条消息，必要时触发摘要压缩。"""
        self.recent.append(message)
        # 用消息条数近似判断：超过 2*max_turns 就压缩前一半
        if len(self.recent) > self.max_turns * 2:
            half = len(self.recent) // 2
            old = self.recent[:half]
            self.recent = self.recent[half:]
            self.summary = self._do_summarize(old)

    def build_messages(self) -> list:
        """构造本次调用 LLM 需要的 messages 列表。"""
        msgs: list = []
        if self.system:
            msgs.append(self.system)
        if self.summary:
            msgs.append({"role": "system", "content": f"【历史摘要】\n{self.summary}"})
        msgs.extend(self.recent)
        return msgs

    def clear(self) -> None:
        self.summary = ""
        self.recent = []

    # ---------- 内部 ----------
    def _do_summarize(self, old_msgs: list) -> str:
        if self._summarize_fn is not None:
            new_sum = self._summarize_fn(old_msgs)
        else:
            new_sum = _default_summarize(old_msgs)
        return (self.summary + "\n" + new_sum).strip() if self.summary else new_sum


def _default_summarize(old_msgs: list) -> str:
    """默认摘要实现：调用 llm.chat。延迟 import 避免循环依赖。"""
    from llm import chat  # noqa: WPS433

    text_lines = []
    for m in old_msgs:
        role = m.get("role", "")
        content = m.get("content") or ""
        if content:
            text_lines.append(f"{role}: {content}")
    if not text_lines:
        return ""

    prompt = (
        "请将下列对话压缩为 200 字以内的要点摘要，"
        "重点保留：用户身份/偏好、关键事实、尚未完成的承诺或待办。"
    )
    resp = chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n".join(text_lines)},
        ]
    )
    return (resp.choices[0].message.content or "").strip()