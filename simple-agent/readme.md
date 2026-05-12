# Simple Agent 项目介绍文档

> 一个从零搭建、可扩展的中文智能 Agent。具备工具调用、三层记忆、流式 Web 界面、错误重试、并发执行、完整单测。本文档逐文件、逐功能详细介绍。

---

## 目录

- [一、项目概览](#一项目概览)
- [二、整体架构](#二整体架构)
- [三、目录结构](#三目录结构)
- [四、核心能力速览](#四核心能力速览)
- [五、文件详解](#五文件详解)
  - [5.1 入口与编排](#51-入口与编排)
  - [5.2 LLM 与 Prompt](#52-llm-与-prompt)
  - [5.3 工具系统](#53-工具系统)
  - [5.4 记忆三层](#54-记忆三层)
  - [5.5 Web 服务与前端](#55-web-服务与前端)
  - [5.6 测试套件](#56-测试套件)
  - [5.7 配置与运行时文件](#57-配置与运行时文件)
- [六、运行方式](#六运行方式)
- [七、配置项总览](#七配置项总览)
- [八、扩展指南](#八扩展指南)
- [九、常见问题](#九常见问题)

---

## 一、项目概览

**Simple Agent** 是一个最小但完整的 LLM Agent 实现，强调"看得懂、改得动、能上线"。

| 维度 | 说明 |
|---|---|
| 语言 | Python 3.10+（实际开发于 3.13.9） |
| LLM 协议 | OpenAI Chat Completions（兼容 DeepSeek / 通义 / Kimi / 自部署 vLLM 等） |
| 默认模型 | `deepseek-chat`（国内可直连） |
| 依赖核心 | `openai` · `python-dotenv` · `fastapi` · `uvicorn` · `chromadb`(可选) |
| 代码量 | 约 900 行 Python + 600 行 HTML/CSS/JS |
| 测试 | 39 个单测，0 失败，<200ms 跑完 |

**对话能力**：
- LLM 推理 → 工具调用 → 工具结果 → 再推理，多轮自动循环。
- 一轮内多个工具调用可并行执行。
- 工具异常 / LLM 失败均不会让进程崩溃，会以错误信息回灌给 LLM 自我纠错。

**记忆能力**（三层独立可开关）：
- 短期记忆：进程内对话窗口 + 自动摘要压缩。
- 会话持久化：SQLite 按 `session_id` 存取，进程重启可恢复。
- 长期向量记忆：Chroma 语义检索，跨会话"记得用户偏好"。

**界面能力**：
- 命令行交互（`python agent.py`）。
- HTTP API + 浏览器 Web UI（FastAPI + SSE 流式）。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  入口：CLI (agent.py)  /  Web (web/server.py)                   │
│           │                                                      │
│           ▼                                                      │
│  Agent 主循环                                                     │
│   ├─ 召回长期记忆 ──── memory/long_term.py (Chroma)              │
│   ├─ 装载短期记忆 ──── memory/short_term.py (窗口+摘要)          │
│   ├─ 调 LLM (重试) ─── llm.py + prompts.py                       │
│   ├─ 执行工具 (并发) ── tools.py (safe_call)                     │
│   ├─ 持久化每条消息 ── memory/session_store.py (SQLite)          │
│   └─ 写入长期记忆 ──── memory/long_term.py                       │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计原则**：
1. **分层清晰**：LLM 层、工具层、记忆层、Agent 层、接口层互不耦合。
2. **配置走环境变量**：所有可调参数通过 `.env` 控制（12-Factor）。
3. **防御式编程**：每一环都有兜底——LLM 重试、工具异常包装、记忆按需开关、CDN 失败本地降级。
4. **可测试性**：所有外部依赖（LLM、向量库）均可 mock；测试 100% 离线运行。

---

## 三、目录结构

```
simple-agent/
├── agent.py                 # CLI 入口 + Agent 主循环
├── llm.py                   # LLM 客户端 + 重试封装
├── prompts.py               # System prompt 版本管理
├── tools.py                 # 工具实现 + Schema + safe_call
│
├── memory/                  # 三层记忆模块
│   ├── __init__.py          # 统一导出
│   ├── short_term.py        # 短期记忆（窗口 + 摘要）
│   ├── session_store.py     # SQLite 会话持久化
│   └── long_term.py         # Chroma 长期向量记忆
│
├── web/                     # FastAPI Web 服务
│   ├── __init__.py
│   ├── server.py            # HTTP/SSE 接口
│   └── static/
│       └── index.html       # 单文件聊天 UI
│
├── tests/                   # 单测套件（39 用例）
│   ├── __init__.py          # sys.path 注入
│   ├── test_agent.py        # 主循环 mock 测
│   ├── test_llm_retry.py    # 重试策略
│   ├── test_long_term.py    # 长期记忆
│   ├── test_session_store.py# 会话持久化
│   ├── test_short_term.py   # 短期记忆
│   └── test_tools.py        # 工具 + safe_call
│
├── .env                     # 本地配置（不入库）
├── .env.example             # 配置模板（入库）
├── sessions.db              # SQLite 数据文件（运行时生成）
├── chroma_store/            # 向量库目录（启用 LONG_TERM 时生成）
├── readme.md                # 本文档
└── .venv/                   # Python 虚拟环境
```

---

## 四、核心能力速览

| 能力 | 实现位置 | 关键点 |
|---|---|---|
| 多轮对话 | [`agent.py`](agent.py) | LLM ↔ 工具循环，最多 `MAX_STEPS` 轮防死循环 |
| 工具调用 | [`tools.py`](tools.py) + LLM `tools` 字段 | 函数 + JSON Schema 双声明，`TOOL_MAP` 映射 |
| 并发执行工具 | [`agent.py:_execute_tools_parallel`](agent.py) | `ThreadPoolExecutor`，单工具不开线程 |
| LLM 重试 | [`llm.py:chat_with_retry`](llm.py) | 指数退避 1s→2s→4s + jitter，仅重试网络/限流/5xx |
| 异常隔离 | [`tools.py:safe_call`](tools.py) | 工具抛异常转字符串，回灌 LLM 自我纠错 |
| 短期记忆 | [`memory/short_term.py`](memory/short_term.py) | 窗口 + 自动摘要压缩，避免 token 爆炸 |
| 会话持久化 | [`memory/session_store.py`](memory/session_store.py) | SQLite，按 `session_id` 存取，跨重启恢复 |
| 长期向量记忆 | [`memory/long_term.py`](memory/long_term.py) | Chroma，按需召回 + 启发式写入 |
| Prompt 版本管理 | [`prompts.py`](prompts.py) | `v1`/`v2`/`current`，便于 A/B 测试 |
| Web API | [`web/server.py`](web/server.py) | REST + SSE，多会话隔离 |
| 流式 UI | [`web/static/index.html`](web/static/index.html) | Markdown 渲染、代码高亮、暗色主题、抽屉响应式 |
| 配置 | `.env` + `python-dotenv` | 所有参数环境变量化 |
| 测试 | `tests/` | 39 用例完全离线，<200ms |

---

## 五、文件详解

### 5.1 入口与编排

#### [`agent.py`](agent.py) — CLI 入口 + 主循环

**职责**：把 LLM、工具、记忆、Prompt 串起来，提供命令行交互。

**关键函数**：
- `run(user_input, verbose=True) -> str`
  - 召回长期记忆 → 写入短期/持久化 → 进入 `MAX_STEPS` 轮循环。
  - 每轮：调 `chat_with_retry` → 若有 `tool_calls` 就并发执行 → 否则返回最终答案。
- `_execute_tools_parallel(tool_calls, verbose)`
  - 单调用直接跑，多调用用线程池（`TOOL_WORKERS` 控制并发数）。
- `_recall_and_inject(user_input)` / `_maybe_remember(user_input, reply)`
  - 长期记忆的"读路径"和"写路径"（详见 5.4）。

**初始化逻辑**（模块加载即执行）：
1. 从 `.env` 读 `MAX_STEPS / TOOL_WORKERS / SHORT_TERM_TURNS` 等。
2. 构造 `ShortTermMemory`，灌入 `prompts.CURRENT`。
3. 若 `ENABLE_SESSION_STORE=1`，初始化 `SessionStore` 并恢复历史。
4. 若 `ENABLE_LONG_TERM=1` 且 chromadb 可用，启用长期记忆。

**异常处理**：LLM 失败时返回 `"[LLM 调用失败] ..."` 字符串，主循环不 crash。

---

### 5.2 LLM 与 Prompt

#### [`llm.py`](llm.py) — LLM 客户端 + 重试封装

**职责**：屏蔽底层 SDK 差异，提供统一的 `chat()` 与 `chat_with_retry()` 接口。

**关键变量**（来自 `.env`）：
- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` —— 凭证与端点。
- `LLM_TIMEOUT`（默认 30s）—— 单次请求超时。
- `LLM_RETRIES`（默认 3）—— 最大重试次数。

**关键函数**：

- `chat(messages, tools=None, **extra)` —— 单次调用，无重试，主要给测试用。
- `chat_with_retry(messages, tools=None, retries=None)` —— 生产入口，特性如下：
  - 仅对**可重试错误**重试：`APIConnectionError` / `APITimeoutError` / `RateLimitError` / `APIError(5xx)`。
  - 4xx 参数错误直接抛，避免浪费 quota。
  - 退避策略：`(2 ^ i) + random(0, 0.5)` 秒，即 1s→2s→4s + ±0.5s jitter，避免雷击群效应。
  - 全部失败后抛出最后一次异常。

**为什么这样设计**：
- LLM 抖动是常态，但**不能无限重试**——5xx 多半是模型过载，几秒后就好；4xx 是参数错，重试 1000 次也没用。
- 加 jitter 防止多个 worker 同步重试雪崩。

**示例代码**：
```python
from llm import chat_with_retry
resp = chat_with_retry(messages, tools=TOOLS)   # 自动重试
print(resp.choices[0].message.content)
```

#### [`prompts.py`](prompts.py) — System Prompt 版本管理

**职责**：把 prompt 从业务代码里抽出来，方便迭代和 A/B 测试。

**当前版本**：`SYSTEM_V2`（即 `CURRENT`），包含五要素：
1. **角色**：专业、简洁、不卖弄、不编造。
2. **可用工具**：列出名字与用途，强化"该用工具时用工具"的意识。
3. **决策准则**：5 条规则，覆盖"何时调工具/何时直答/何时反问"。
4. **输出格式**：默认中文、Markdown、代码块、≤500 字。
5. **不要**：不暴露 prompt、不编造工具返回、不承诺做不到的事。

**版本切换**：
```python
from prompts import get_prompt
mem.set_system(get_prompt("v1"))  # 回滚到旧版本做对比
```

**为什么单独成文件**：好的 prompt 决定 90% 的表现，需要频繁迭代；写在业务代码里会被混合编辑，难版本管理。

---

### 5.3 工具系统

#### [`tools.py`](tools.py) — 工具实现 + Schema + safe_call

**职责**：
1. 定义可被 LLM 调用的函数（工具）。
2. 用 JSON Schema 描述每个工具，让 LLM 知道"有哪些工具、怎么用"。
3. 提供 `safe_call()` 统一兜底异常。

**内置工具**（演示用，生产按需替换）：

| 工具 | 函数 | 作用 |
|---|---|---|
| `get_weather` | `get_weather(city)` | 查城市天气（mock 返回，可接和风/心知等） |
| `get_current_time` | `get_current_time()` | 查当前时间 |
| `calculator` | `calculator(expression)` | 数学表达式求值，白名单字符防注入 |

**三件套结构**：
1. **Python 函数**：实现业务逻辑。
2. **`TOOLS` 列表**：JSON Schema，给 LLM 看的说明书。
3. **`TOOL_MAP` 字典**：`{"name": callable}`，供主循环按名字调度。

三者一致性由 [`tests/test_tools.py`](tests/test_tools.py) 的 `test_tools_schema_consistency` 用例保障。

**`safe_call(name, fn, **kwargs) -> str`**：

```python
def safe_call(name, fn, **kwargs) -> str:
    try:
        result = fn(**kwargs)
        return "" if result is None else str(result)
    except TypeError as e:         # LLM 传错参数
        return f"[工具参数错误] {name}: {e}"
    except Exception as e:         # 其他运行时异常
        return f"[工具异常] {name} {type(e).__name__}: {e}"
```

**为什么这样设计**：
- 工具抛异常 → 主循环崩 → 对话中断。**错误应该被翻译成"文本"，回灌给 LLM**，让它自己决定"换工具"或"问用户"。
- 区分 `TypeError`（参数不匹配，多半是 LLM 格式错误）与其他异常，便于调试。
- 返回值强制转 `str`，满足 OpenAI `tool` 消息的 `content` 字段要求。

**Schema 示例**：
```python
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "执行数学表达式计算",
    "parameters": {
      "type": "object",
      "properties": {"expression": {"type": "string"}},
      "required": ["expression"]
    }
  }
}
```

**新增一个工具的步骤**：
1. 写 Python 函数（注意参数加 type hint，写清 docstring）。
2. 在 `TOOLS` 里追加 JSON Schema。
3. 在 `TOOL_MAP` 里注册 `"name": fn`。
4. 在 `prompts.py` 的"可用工具"列表里也加一行（可选，但强烈建议）。
5. 写一个 `test_` 单测。

---

### 5.4 记忆三层

三层记忆各司其职，按需组合。

#### 分层对比

| 层级 | 存储 | 生命周期 | 用途 | 默认 | 开关 |
|---|---|---|---|---|---|
| 短期 | 进程内 list | 单次会话 | 多轮上下文 | ON | 始终启用 |
| 会话持久化 | SQLite | 跨进程重启 | 按 `session_id` 恢复 | OFF | `ENABLE_SESSION_STORE=1` |
| 长期向量 | Chroma | 永久 | 跨会话语义检索 | OFF | `ENABLE_LONG_TERM=1` |

#### [`memory/short_term.py`](memory/short_term.py) — `ShortTermMemory`

**原理**：窗口 + 自动摘要。当消息数 > `2 × max_turns` 时，把前一半交给 LLM 压成要点摘要，保留后一半原文。

**关键字段**：
- `system`：系统提示词，永远在 prompt 最前，摘要时不会被吞。
- `summary`：历史摘要（累积拼接）。
- `recent`：最近的消息列表。

**关键方法**：
- `add(message)` —— 追加消息，必要时触发摘要。
- `build_messages()` —— 构造给 LLM 的 `messages`：`[system] + [【历史摘要】] + recent`。
- `clear()` —— 重置摘要与 recent，保留 system。

**摘要 prompt 精心设计**：显式要求保留"用户身份/偏好、关键事实、待办"，避免把"我叫张三"这类重要信息吞了。

**测试友好**：`summarize_fn` 可外部注入，测试时用 `lambda msgs: "SUMMARY:"+str(len(msgs))` 绕开真实 LLM。

#### [`memory/session_store.py`](memory/session_store.py) — `SessionStore`

**原理**：SQLite 单表 `messages(id, session_id, role, content, extra, ts)`，按 `session_id` 过滤。

**关键设计**：
- `check_same_thread=False` + `threading.Lock` —— 允许 FastAPI worker 线程跨线程访问同一连接。
- `extra` 字段存 JSON，保留 `tool_call_id` 等非 role/content 字段。
- 索引 `idx_sess` 让按 `session_id` 查询 O(log n)。

**关键方法**：
- `append(session_id, message)` —— 写入一条。
- `load(session_id, limit=200)` —— 按时间升序取最近 N 条。
- `list_sessions()` —— 列出所有会话及消息数。
- `delete(session_id)` —— 删除整个会话。

**使用模式**：Web 端每个请求用 `session_id` 从 SQLite 加载 → 灌入一份临时 `ShortTermMemory`，实现"多会话隔离"。

#### [`memory/long_term.py`](memory/long_term.py) — 长期向量记忆

**原理**：用 Chroma 存向量化的对话片段，按语义相似度召回。

**关键函数**：
- `is_enabled() -> bool` —— chromadb 是否可用。
- `remember(text, meta=None, mem_id=None)` —— 写入。自动生成 `mem-<md5前16位>` 作 id，防重复。
- `recall(query, k=3)` —— 按语义召回最相关的 k 条。
- `forget(mem_id)` / `clear_all()` —— 删除。

**Embedding 切换**（通过 `EMBED_PROVIDER`）：
- `openai`（默认）—— 调 `text-embedding-3-small` 或任何 OpenAI 协议的 embedding 服务。
- `hash` —— 本地哈希嵌入，**语义能力弱**，仅用于单测与演示（无需联网）。

**降级保护**：未装 chromadb 时所有函数 no-op 返回空/False，不影响主流程。

**写入策略**（在 [`agent.py`](agent.py) 里）：启发式判断——用户说"我叫/我住/偏好/记住..."或助手回答超过 200 字时才写，避免库膨胀。生产环境建议改用**独立 LLM 调用抽取事实**，质量更高。

**召回策略**：每次用户输入先 `recall()`，命中的旧对话作为 `system` 消息临时注入短期记忆，LLM 自然看到"这个用户以前说过..."。

---

### 5.5 Web 服务与前端

#### [`web/server.py`](web/server.py) — FastAPI 应用

**职责**：把 Agent 包成 HTTP/SSE 服务，支持多会话隔离。

**接口一览**：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 返回聊天 UI（`static/index.html`） |
| `POST` | `/api/chat` | 同步对话，返回 `{session_id, answer, trace}` |
| `POST` | `/api/chat/stream` | SSE 流式，逐事件推送 trace |
| `GET` | `/api/sessions` | 列出所有持久化会话 |
| `DELETE` | `/api/sessions/{sid}` | 删除指定会话 |
| `GET` | `/api/health` | 探活 + 返回 long_term/MAX_STEPS 等配置 |

**关键设计**：
- **多会话**：每个请求按 `session_id` 现 build 一份 `ShortTermMemory`，互不串。
- **同步 SDK → 异步路由**：用 `loop.run_in_executor(None, _run_once, ...)` 把阻塞调用扔到线程池，不阻塞 event loop。
- **SSE 事件类型**：`session` / `tool_call` / `tool_result` / `final` / `done`。
- **错误兜底**：LLM 异常通过 trace 推送 `final` 事件，前端不会卡住。
- **静态资源**：`/static/*` 由 `StaticFiles` 提供。

**SSE 事件格式示例**：
```json
{"type": "session",     "session_id": "web-abcd1234"}
{"type": "tool_call",   "name": "calculator", "arguments": {"expression": "1+2"}}
{"type": "tool_result", "name": "calculator", "content": "3"}
{"type": "final",       "content": "结果是 3"}
{"type": "done",        "answer": "结果是 3"}
```

#### [`web/static/index.html`](web/static/index.html) — 单文件聊天 UI

**职责**：零构建、纯原生（HTML/CSS/JS）实现 ChatGPT 风格界面。

**功能清单**：

| 模块 | 实现 |
|---|---|
| 双栏布局 | CSS Grid `260px + 1fr`，`.backdrop { display: none }` 防 grid 占格 |
| 暗色主题 | CSS 变量切换 + `localStorage.theme` 持久化 |
| Markdown 渲染 | `marked` CDN，断网时降级为转义+换行 |
| 代码高亮 | `highlight.js` v11，明/暗双主题切换 |
| 流式打字机 | `requestAnimationFrame` 分帧追加 |
| 会话侧栏 | 拉 `/api/sessions`，点切换 / hover 删除 |
| 工具调用卡片 | `<details>` 折叠组件，结果到达后自动收起 |
| 复制按钮 | `navigator.clipboard.writeText` + Toast 反馈 |
| 推荐问题芯片 | 空状态时展示 4 条快捷输入 |
| 输入框自适应 | `input` 事件动态调 `height`（最高 180px） |
| 中文输入法保护 | 按键事件检测 `e.isComposing` 防误发 |
| 移动端抽屉 | `@media (max-width: 768px)` 切换为遮罩抽屉 |
| iOS 安全区 | `env(safe-area-inset-bottom)` 防被 Home 条遮挡 |
| 折叠侧栏 | 桌面端顶栏 `☰` 按钮，状态进 `localStorage` |
| 健康指示 | 拉 `/api/health` 显示 "长期记忆·ON/OFF" |

**响应式断点**：
- `>1024px`：侧栏 260px
- `≤1024px`：侧栏 220px
- `≤768px`：变抽屉

---

### 5.6 测试套件

**统一约定**：
- 标准库 `unittest`，零新依赖。
- [`tests/__init__.py`](tests/__init__.py) 注入 `sys.path`，让用例能直接 `import tools / memory / agent`。
- 全部用例**完全离线**：mock LLM、用临时 SQLite、长期记忆走 `EMBED_PROVIDER=hash`。
- 单次跑完 < 200ms。

**用例分布**：

| 文件 | 用例数 | 覆盖点 |
|---|---|---|
| [`test_tools.py`](tests/test_tools.py) | 11 | 三个工具的正确性 / 非法输入 / Schema 一致性 / `safe_call` 4 种异常 |
| [`test_short_term.py`](tests/test_short_term.py) | 6 | 注入假摘要函数验证窗口触发、build 顺序、累加、clear |
| [`test_session_store.py`](tests/test_session_store.py) | 6 | append/load、session 隔离、extra 字段往返、limit 顺序、delete、list |
| [`test_long_term.py`](tests/test_long_term.py) | 5 | hash embedding 离线跑 chroma；含 `is_enabled()` 守卫与降级路径 |
| [`test_llm_retry.py`](tests/test_llm_retry.py) | 4 | 成功不重试 / 可重试错误重试成功 / 耗尽抛出 / 4xx 不重试 |
| [`test_agent.py`](tests/test_agent.py) | 7 | mock `chat_with_retry`：直接答 / 工具调用流 / 未知工具 / MAX_STEPS / 短期记忆更新 / LLM 失败优雅返回 / 并发多工具 |

**运行**：
```bash
cd simple-agent && source .venv/bin/activate
python -m unittest discover -s tests -v        # 全量 + verbose
python -m unittest tests.test_agent -v         # 单文件
python -m unittest tests.test_agent.AgentTest.test_tool_call_flow -v  # 单用例
```

---

### 5.7 配置与运行时文件

#### [`.env.example`](.env.example) — 配置模板（入库）

包含所有可配置项，复制为 `.env` 后填入真实 key：
```bash
cp .env.example .env
```

#### `.env` —— 实际配置（`.gitignore`，不入库）

最小可用：
```bash
LLM_API_KEY=sk-你的deepseek-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

#### `sessions.db` —— SQLite 数据文件

启用 `ENABLE_SESSION_STORE=1` 后由 [`SessionStore`](memory/session_store.py) 自动创建。可用任何 SQLite 客户端（如 `sqlite3` CLI、DB Browser）查看历史。

#### `chroma_store/` —— Chroma 向量库目录

启用 `ENABLE_LONG_TERM=1` 后由 [`long_term`](memory/long_term.py) 自动创建。删除目录即清空所有长期记忆。

#### `.venv/` —— Python 虚拟环境

由 `python3 -m venv .venv` 生成，含独立 python/pip/site-packages，与系统 Python 完全隔离。**不要跨机器复制**，应通过 `pip freeze > requirements.txt` 重建。

---

## 六、运行方式

项目提供三种运行形态，共享同一份 Agent 核心代码。

### 6.1 CLI 交互模式（最快体验）

```bash
cd simple-agent
source .venv/bin/activate
python agent.py
```

特性：
- 命令行多轮对话，输入 `exit` / `quit` 退出。
- 使用默认 session（`default`），开启 `ENABLE_SESSION_STORE=1` 后跨次运行历史会自动恢复。
- 每步打印 `🤖 思考中... / 🔧 调用工具 xxx / ✅ 工具返回`，便于观察推理链路。

示例：
```
> 帮我算 3 * (4 + 5)，再告诉我现在几点
🔧 调用工具 calculator(expression='3 * (4 + 5)') → 27
🔧 调用工具 get_current_time() → 2026-05-09 15:20:11
🤖 3 × (4 + 5) = 27；当前时间是 2026-05-09 15:20:11。
```

### 6.2 Web 模式（推荐日常使用）

```bash
cd simple-agent
source .venv/bin/activate
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
```

浏览器打开 `http://localhost:8000`：
- 左侧会话列表，点击 `+ 新会话` 创建。
- 右侧聊天区，回车发送，Shift+Enter 换行。
- 工具调用以气泡形式内嵌在对话流中，流式渲染。
- 支持主题切换、响应式抽屉（< 768px 自动折叠）。

### 6.3 仅作为 Python 库调用

在其他项目里直接引入：
```python
from simple_agent.agent import run  # 假设把目录加入 PYTHONPATH
answer = run("帮我查一下纽约现在几点", session_id="user_42")
print(answer)
```

或调用 Web API（便于跨语言集成）：
```bash
# 非流式
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好","session_id":"demo"}'

# 流式（SSE）
curl -N http://localhost:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"算一下 1+1","session_id":"demo"}'
```

---

## 七、配置项总览

所有配置通过 `.env` / 环境变量注入，代码里均有默认值，零配置也能跑起来（除了 `LLM_API_KEY` 必填）。

### 7.1 LLM 配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | **必填** | DeepSeek / OpenAI / 其他兼容服务的 API Key |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容服务的 Base URL |
| `LLM_MODEL` | `deepseek-chat` | 模型名（如 `gpt-4o-mini`、`qwen-plus`） |
| `LLM_TIMEOUT` | `30` | 单次请求超时（秒） |
| `LLM_MAX_RETRIES` | `3` | 可重试错误的重试次数 |
| `PROMPT_VERSION` | `v2` | `v1` / `v2`，对应 [`prompts.py`](prompts.py) 里不同版本 |

### 7.2 短期记忆

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SHORT_TERM_TURNS` | `10` | 窗口保留轮数，超过触发摘要 |

### 7.3 会话存储

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ENABLE_SESSION_STORE` | `0` | `1` 开启 SQLite 持久化 |
| `SESSION_DB_PATH` | `sessions.db` | 数据库文件路径 |

### 7.4 长期记忆

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ENABLE_LONG_TERM` | `0` | `1` 开启 Chroma 向量检索 |
| `LONG_TERM_DIR` | `chroma_store` | 向量库持久化目录 |
| `EMBED_PROVIDER` | `hash` | `hash`（离线） / `openai`（联网） |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding 模型名（仅 `openai` 生效） |
| `EMBED_API_KEY` | 复用 `LLM_API_KEY` | 单独为 embedding 指定 key |
| `EMBED_BASE_URL` | 复用 `LLM_BASE_URL` | 单独为 embedding 指定 endpoint |

### 7.5 Agent 主循环

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAX_STEPS` | `8` | 单轮对话最多执行 LLM+工具的步数，防止死循环 |
| `TOOL_PARALLEL` | `1` | `1` 开启同轮多工具并发；`0` 串行 |

---

## 八、扩展指南

### 8.1 新增一个工具（3 步）

以「获取天气」为例：

**Step 1**：在 [`tools.py`](tools.py) 里实现函数
```python
def get_weather(city: str) -> str:
    # 真实场景调用气象 API，这里简化为 mock
    return f"{city} 今天多云 22℃"
```

**Step 2**：在同文件的 `TOOLS` 列表追加 schema
```python
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询指定城市的当日天气",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string", "description": "城市名"}},
      "required": ["city"],
    },
  },
}
```

**Step 3**：在 `TOOL_MAP` 注册
```python
TOOL_MAP = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "read_note": read_note,
    "get_weather": get_weather,   # 新增
}
```

重启服务即可，LLM 会自动感知新工具。

### 8.2 切换 LLM 服务商

只需改 `.env` 三项，无需改代码：
```bash
# 切到 OpenAI
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# 切到阿里云 DashScope（OpenAI 兼容模式）
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

### 8.3 替换记忆实现

所有记忆模块都是"接口 + 实现"分离：
- 短期记忆：替换 [`short_term.py`](memory/short_term.py) 的 `build()` 返回 messages 列表即可，例如改用 LLM 压缩+关键词提取。
- 会话存储：把 [`session_store.py`](memory/session_store.py) 的 SQLite 换成 Postgres/Redis，保持 `append/load/list/delete` 方法签名不变。
- 长期记忆：把 [`long_term.py`](memory/long_term.py) 的 Chroma 换成 Qdrant/Milvus/Pgvector，保持 `remember/recall/forget` 签名不变。

### 8.4 Docker 部署模板

`Dockerfile`：
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/
EXPOSE 8000
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`：
```yaml
services:
  agent:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./sessions.db:/app/sessions.db
      - ./chroma_store:/app/chroma_store
    restart: unless-stopped
```

### 8.5 接入 Prompt 版本管理

[`prompts.py`](prompts.py) 已支持多版本，新增 v3 只需：
```python
SYSTEM_V3 = """你是 ... (新 prompt)"""

def get_prompt(version: str = "v2") -> str:
    return {"v1": SYSTEM_V1, "v2": SYSTEM_V2, "v3": SYSTEM_V3}.get(version, SYSTEM_V2)
```

运行前 `export PROMPT_VERSION=v3` 即可灰度。

---

## 九、常见问题（FAQ）

### Q1：`ModuleNotFoundError: No module named 'openai'`
未激活虚拟环境或未装依赖。执行：
```bash
cd simple-agent && source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Q2：`openai.APIConnectionError` 反复重试仍失败
检查 `LLM_BASE_URL` 是否可达，国内机器访问 `api.openai.com` 需代理；建议切 DeepSeek 或通义千问兼容端点。

### Q3：Web 页面在桌面 Chrome 显示成移动端布局
旧版本 760px 断点过大，半屏窗口就触发。当前版本已改为 768px + 桌面端折叠按钮；若仍异常，强制刷新（Cmd+Shift+R）清缓存。

### Q4：SSE 流式返回 500 / 卡住
常见原因：LLM key 无效 / 超时。查看终端日志，或调用 `GET /api/health` 确认服务存活；`.env` 改完需重启 uvicorn。

### Q5：SQLite 报 `database is locked`
多进程写同一份 `sessions.db` 会冲突。本项目已用 `threading.Lock` 保证单进程内线程安全；跨进程请改用 Postgres。

### Q6：Chroma 报 `Expected metadata to be a non-empty dict`
老版本 chromadb 不允许空 metadata，[`long_term.py`](memory/long_term.py) 已自动补 `{"_": ""}` 占位；若仍报错，升级 `pip install -U chromadb`。

### Q7：测试跑不起来，提示 `attempted relative import`
请用 `python -m unittest discover -s tests -v`，而不是 `python tests/test_xxx.py`。前者会把项目根加入 sys.path。

### Q8：同时想跑 CLI 和 Web 会互相覆盖 session 吗？
不会。两者默认 session_id 不同：CLI 用 `default`，Web 用前端生成的 UUID。若要共享，CLI 启动前 `export AGENT_SESSION_ID=xxx`，并与 Web 传入一致的 session_id 即可。

### Q9：如何清空长期记忆？
```bash
rm -rf simple-agent/chroma_store
```
或在代码里调用 `from memory.long_term import clear_all; clear_all()`。

### Q10：想把本地知识库"喂"给 Agent？
调用 `remember(text, metadata)` 批量导入文档片段即可；复杂场景建议先做分段（chunk ~ 500 字），附上 `source` 元数据便于追溯。

---

## 十、从 0 到 1 运行（小白上手版）

假设你是一台**全新的 macOS / Linux 机器**，从克隆代码到看到 Agent 回复，一共 7 步、5 分钟搞定。

### Step 0 — 前置准备（一次性）

| 必需 | 检查命令 | 没有的话 |
|---|---|---|
| Python ≥ 3.10 | macOS/Linux：`python3 --version`<br>Windows：`python --version` | macOS：`brew install python@3.13`<br>Linux：`apt install python3 python3-venv`<br>Windows：[python.org 官网](https://www.python.org/downloads/windows/) 下载安装包，**勾选 ✅ Add Python to PATH** |
| pip | `python -m pip --version` | 一般随 Python 自带；缺则 `python -m ensurepip` |
| LLM API Key | —— | 推荐 [DeepSeek 控制台](https://platform.deepseek.com/) 注册后充 1 元，得 `sk-xxx` |

> **Windows 用户**：建议使用 **PowerShell**（Win+X 选 "终端" 或 "Windows PowerShell"），而不是老 cmd.exe。下文若 macOS/Linux 与 Windows 命令不同，会分别给出。

### Step 1 — 进入项目目录

```bash
cd /path/to/ViewCode/simple-agent
```

### Step 2 — 创建并激活虚拟环境

**macOS / Linux：**
```bash
python3 -m venv .venv          # 仅首次执行
source .venv/bin/activate      # 每次新开终端都要执行
```

**Windows（PowerShell）：**
```powershell
python -m venv .venv                    # 仅首次执行
.\.venv\Scripts\Activate.ps1            # 每次新开终端都要执行
```

> 如果 PowerShell 报 `无法加载文件 ... 因为在此系统上禁止运行脚本`，先以**管理员**身份打开 PowerShell 执行一次：
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

**Windows（cmd.exe）：**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

激活成功后命令行前会出现 `(.venv)` 前缀。

### Step 3 — 安装依赖（国内推荐加镜像）

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

约 30 秒装完 `openai / fastapi / uvicorn / python-dotenv / chromadb` 等。

### Step 4 — 配置 `.env`

```bash
cp .env.example .env
```

**Windows（PowerShell）**：
```powershell
Copy-Item .env.example .env
```

用任意编辑器（VSCode / Notepad / Notepad++）打开 `.env`，**至少填一项**：
```bash
LLM_API_KEY=sk-你刚才申请的key
# 其他保持默认即可（已默认 DeepSeek）
```

> 想用 OpenAI / 通义千问？参考 [§8.2](#82-切换-llm-服务商)。

### Step 5 — 跑一次单元测试，确认环境健康

```bash
python -m unittest discover -s tests -v
```

预期输出：
```
Ran 39 tests in 0.17s
OK
```

39 个用例全部离线 mock，**不需要真实 API Key 也能通过**。如果这步通过，说明依赖和代码都 OK，问题只可能在 LLM key。

### Step 6 — 选一种方式启动

#### A. CLI 模式（最快感受）
```bash
python agent.py
```
```
> 你好，介绍一下你自己
🤖 我是一个简单的 Agent，可以帮你计算、查时间、读笔记 ...
> 帮我算 (3+4)*5
🔧 调用工具 calculator(expression='(3+4)*5') → 35
🤖 (3+4)×5 = 35
> exit
```

#### B. Web 模式（推荐日常用）
```bash
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
```
看到 `Uvicorn running on http://0.0.0.0:8000` 后，浏览器打开：
```
http://localhost:8000
```
左侧 `+ 新会话` → 右侧输入框回车发送，工具调用会以气泡形式实时流式渲染。

### Step 7 — （可选）开启持久化 + 长期记忆

编辑 `.env` 追加：
```bash
ENABLE_SESSION_STORE=1   # 历史对话存到 sessions.db
ENABLE_LONG_TERM=1       # 关键信息向量化存到 chroma_store/
EMBED_PROVIDER=hash      # 离线 hash embedding，无需额外 key
```
重启服务后：
- 重启进程，CLI/Web 历史依然在；
- Agent 会自动 `recall` 相关历史片段拼到 prompt，实现"长期记忆"。

---

### 常用命令速查卡

```bash
# 激活环境（每次新终端）
source simple-agent/.venv/bin/activate

# 跑测试
python -m unittest discover -s simple-agent/tests -v

# 启动 Web
cd simple-agent && uvicorn web.server:app --reload

# 健康检查
curl http://localhost:8000/api/health

# 清空长期记忆
rm -rf simple-agent/chroma_store

# 清空会话历史
rm simple-agent/sessions.db

# 退出虚拟环境
deactivate
```

### 出问题怎么办？

1. **先看终端日志**：错误一般直接打印在 `uvicorn` 或 `python agent.py` 的输出里。
2. **再看 [§9 FAQ](#九常见问题faq)**：覆盖了 10 个最常见坑（key 失效、SSE 卡住、SQLite 锁、Chroma 校验等）。
3. **重置大法**：删掉 `.venv` / `sessions.db` / `chroma_store`，从 Step 2 重做一遍，95% 的问题都能解决。

---

## 十一、Windows 专项说明

Windows 用户常遇到的 4 个坑，一次性讲清楚。

### 11.1 命令速查表（macOS/Linux → Windows 对照）

| 操作 | macOS / Linux | Windows PowerShell |
|---|---|---|
| 查看 Python | `python3 --version` | `python --version` |
| 创建 venv | `python3 -m venv .venv` | `python -m venv .venv` |
| 激活 venv | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| 退出 venv | `deactivate` | `deactivate` |
| 复制文件 | `cp a b` | `Copy-Item a b` |
| 删除文件 | `rm file` | `Remove-Item file` |
| 删除目录 | `rm -rf dir` | `Remove-Item -Recurse -Force dir` |
| 查看文件尾 | `tail -f log` | `Get-Content log -Wait -Tail 20` |
| 设置环境变量（临时） | `export KEY=value` | `$env:KEY="value"` |

### 11.2 PowerShell 脚本策略报错

首次激活 venv 时如果报：
```
无法加载文件 ...Activate.ps1，因为在此系统上禁止运行脚本
```

**管理员权限**打开 PowerShell 执行一次（仅需一次）：
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新激活即可。

### 11.3 编码乱码（中文显示成 `??`）

Windows 默认终端编码是 GBK，遇到中文可能乱码。永久解决：
```powershell
# PowerShell 启动时强制 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
```
可把这两行写进 PowerShell Profile（`$PROFILE`）一次配置永久生效。

或临时在命令前加：
```powershell
chcp 65001       # 切换到 UTF-8 代码页
```

### 11.4 端口 8000 被占用

Windows 上 8000 端口偶尔被 Hyper-V / IIS / 其他服务占用：
```powershell
# 查看谁在占用 8000
netstat -ano | findstr :8000
# 根据 PID 结束进程
taskkill /PID <pid> /F

# 或直接换端口启动
uvicorn web.server:app --host 0.0.0.0 --port 8888 --reload
```

### 11.5 路径分隔符

代码里所有文件路径都用了 `os.path.join` 或 `pathlib`，Windows 上会自动变成 `\`，**无需改代码**。只有你写 `.env` 时需要注意：
```bash
# ✅ 正确（正斜杠，跨平台）
SESSION_DB_PATH=data/sessions.db

# ✅ 也正确（双反斜杠转义）
SESSION_DB_PATH=data\\sessions.db

# ❌ 错误（单反斜杠会被当转义符）
SESSION_DB_PATH=data\sessions.db
```

### 11.6 Windows 推荐的完整运行流程

从 0 到 1，PowerShell 版：
```powershell
# 1. 进入项目
cd D:\path\to\ViewCode\simple-agent

# 2. 建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. 配 .env
Copy-Item .env.example .env
notepad .env         # 填入 LLM_API_KEY

# 5. 跑测试
python -m unittest discover -s tests -v

# 6A. CLI 模式
python agent.py

# 6B. Web 模式
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
# 浏览器打开 http://localhost:8000
```

> **进阶建议**：Windows 用户如果有 WSL2（Ubuntu 子系统），直接在 WSL 里按 macOS/Linux 流程跑，体验比原生 Windows 更丝滑（尤其是 chromadb 等含 C 扩展的包）。

---

---

## 十二、部署到云服务器（生产可用）

本章覆盖**两种主流部署形态**：

- **方案 A：裸机 + systemd + Nginx + HTTPS**（最轻量、资源占用最低，适合 1~2 核 ECS）
- **方案 B：Docker / docker-compose**（最易迁移、版本回滚方便，适合多服务编排）

两种方案最终效果一致：浏览器访问 `https://yourdomain.com` 即可使用 Agent。

### 12.1 服务器选型与初始化

**最低配置**：1 核 1G、20G 系统盘、Linux（Ubuntu 22.04 / Debian 12 / CentOS Stream 9 均可）。
**推荐配置**：2 核 2G，便于并发 LLM 调用。

云厂商任选：阿里云 ECS、腾讯云 CVM、华为云 ECS、AWS Lightsail、DigitalOcean Droplet。

**安全组放行端口**：
| 端口 | 用途 | 必须？ |
|---|---|---|
| 22 | SSH | ✅ |
| 80 | HTTP（Let's Encrypt 验证） | ✅ |
| 443 | HTTPS | ✅ |
| 8000 | 应用直连（仅调试，生产可不开） | ❌ |

**首次登录后基础加固**（一次性）：
```bash
# 1. 更新系统
sudo apt update && sudo apt upgrade -y     # Ubuntu/Debian
# sudo dnf update -y                        # CentOS/RHEL

# 2. 创建非 root 部署用户
sudo adduser deploy
sudo usermod -aG sudo deploy
sudo su - deploy

# 3. 配置 SSH 公钥免密（在本地执行）
# ssh-copy-id deploy@your.server.ip

# 4. 防火墙
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 12.2 准备域名与 DNS

1. 在域名服务商（阿里云、Cloudflare 等）添加 A 记录：`agent.yourdomain.com → 你的服务器公网 IP`
2. 等待 DNS 生效（一般 1~10 分钟），用 `ping agent.yourdomain.com` 确认能解析到服务器 IP

> 没域名也能跑，只是无法签发 HTTPS，可先用 `http://公网IP:8000` 临时访问。

---

### 12.3 方案 A：裸机 + systemd + Nginx + HTTPS

#### Step A1：上传代码

```bash
# 服务器上
cd ~
git clone https://your-git-repo/simple-agent.git
# 或本地 scp 上传
# scp -r simple-agent deploy@your.ip:~/
cd simple-agent
```

#### Step A2：装 Python 3.10+ 与依赖

```bash
sudo apt install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
# 生产环境额外装 gunicorn 做进程管理
pip install "uvicorn[standard]" gunicorn
```

#### Step A3：写生产 `.env`

```bash
cp .env.example .env
nano .env
```

生产建议：
```bash
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TIMEOUT=60
LLM_MAX_RETRIES=3

ENABLE_SESSION_STORE=1
SESSION_DB_PATH=/home/deploy/simple-agent/data/sessions.db

ENABLE_LONG_TERM=1
LONG_TERM_DIR=/home/deploy/simple-agent/data/chroma_store
EMBED_PROVIDER=hash

MAX_STEPS=8
TOOL_PARALLEL=1
```

```bash
mkdir -p /home/deploy/simple-agent/data
```

#### Step A4：用 systemd 守护进程

新建 `/etc/systemd/system/simple-agent.service`：
```ini
[Unit]
Description=Simple Agent Web Service
After=network.target

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/home/deploy/simple-agent
EnvironmentFile=/home/deploy/simple-agent/.env
ExecStart=/home/deploy/simple-agent/.venv/bin/gunicorn \
    -k uvicorn.workers.UvicornWorker \
    -w 2 \
    -b 127.0.0.1:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    web.server:app
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/simple-agent.log
StandardError=append:/var/log/simple-agent.err.log

[Install]
WantedBy=multi-user.target
```

> `-w 2` = 2 个 worker 进程；CPU 越多可以增大，但注意 SQLite 在多进程下锁会更频繁，建议 ≤4 或换 Postgres。

启动：
```bash
sudo touch /var/log/simple-agent.log /var/log/simple-agent.err.log
sudo chown deploy:deploy /var/log/simple-agent.*

sudo systemctl daemon-reload
sudo systemctl enable --now simple-agent
sudo systemctl status simple-agent      # 应看到 active (running)
```

常用命令：
```bash
sudo systemctl restart simple-agent     # 重启
sudo systemctl stop simple-agent        # 停止
sudo journalctl -u simple-agent -f      # 实时日志
tail -f /var/log/simple-agent.log
```

#### Step A5：Nginx 反向代理 + HTTPS

装 Nginx 与 certbot：
```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

新建 `/etc/nginx/sites-available/simple-agent`：
```nginx
server {
    listen 80;
    server_name agent.yourdomain.com;

    # 客户端最大请求体（避免长 prompt 被拦截）
    client_max_body_size 10M;

    # SSE 流式必需配置：禁用缓冲、长连接
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
    }
}
```

启用并签发 HTTPS：
```bash
sudo ln -s /etc/nginx/sites-available/simple-agent /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 自动签发 + 配置 HTTPS（会修改上面的 nginx 配置文件）
sudo certbot --nginx -d agent.yourdomain.com \
    --agree-tos -m you@example.com --no-eff-email --redirect
```

certbot 会自动：
- 申请 Let's Encrypt 证书
- 改写 nginx 配置加入 `listen 443 ssl`
- 配置 80 → 443 强制跳转
- 安装 cron 自动续签（每 90 天）

完成后浏览器打开 `https://agent.yourdomain.com` 即可。

#### Step A6：发布更新流程

```bash
cd /home/deploy/simple-agent
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart simple-agent
```

---

### 12.4 方案 B：Docker / docker-compose

适合「我不想折腾 systemd 和 Python 环境」的同学。

#### Step B1：装 Docker

> Docker 在不同操作系统下的安装方式差异较大，按你的开发/部署机选对应方案即可。**生产服务器一般是 Linux**，本地开发可能是 macOS / Windows。

##### 🐧 Linux（Ubuntu / Debian / CentOS）

最常用：官方一键脚本（适配主流发行版）：
```bash
curl -fsSL https://get.docker.com | sudo sh

# 把当前用户加入 docker 组，免 sudo 跑 docker
sudo usermod -aG docker $USER
newgrp docker          # 让当前 shell 立刻生效（或重新登录）

# 开机自启
sudo systemctl enable --now docker

# 验证
docker --version
docker compose version
```

国内服务器拉镜像慢的话，配置加速器（阿里云/腾讯云控制台都送）：
```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": ["https://你的专属加速地址.mirror.aliyuncs.com"]
}
EOF
sudo systemctl restart docker
```

##### 🍎 macOS

推荐 **Docker Desktop**（图形界面，含 docker + compose + buildx）：

1. 访问 [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) 下载对应芯片的安装包：
   - Apple Silicon（M1/M2/M3/M4）→ 选 **Mac with Apple chip**
   - Intel → 选 **Mac with Intel chip**
2. 双击 `.dmg`，把 Docker.app 拖进 Applications
3. 启动 Docker Desktop，等顶部状态栏小鲸鱼图标变绿
4. 验证：
   ```bash
   docker --version
   docker compose version
   docker run --rm hello-world
   ```

如果不想装 Docker Desktop（License 限制公司用），可以用开源替代 **Colima**：
```bash
brew install colima docker docker-compose
colima start --cpu 2 --memory 4
docker --version
```

##### 🪟 Windows

推荐 **Docker Desktop for Windows**（基于 WSL2，性能好）：

1. **前置：启用 WSL2**（PowerShell 管理员权限运行一次）
   ```powershell
   wsl --install
   # 重启电脑
   wsl --set-default-version 2
   ```
2. 访问 [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) 下载 **Docker Desktop for Windows**，双击安装
3. 安装时勾选 ✅ **Use WSL 2 instead of Hyper-V**
4. 启动 Docker Desktop → Settings → Resources → WSL Integration，开启 Ubuntu 集成
5. 在 PowerShell 或 WSL 终端验证：
   ```powershell
   docker --version
   docker compose version
   docker run --rm hello-world
   ```

> **强烈建议** Windows 用户在 **WSL Ubuntu** 终端里跑后续命令（不要在 PowerShell 直接 build chromadb，C 扩展编译会出问题）。进入 WSL：
> ```powershell
> wsl
> cd /mnt/d/path/to/simple-agent     # Windows 盘符在 WSL 里映射到 /mnt/盘符/
> ```

##### ✅ 三平台通用验证

不论装在哪种系统，运行以下命令都应有输出：
```bash
docker --version              # Docker version 24.x 或更新
docker compose version        # Docker Compose version v2.x
docker run --rm hello-world   # 拉一次 hello-world 镜像验证全链路
```

#### Step B2：项目内新建 `Dockerfile`

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir "uvicorn[standard]" gunicorn

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", \
     "web.server:app"]
```

`.dockerignore`：
```
.venv
__pycache__
*.pyc
.env
sessions.db
chroma_store
.git
tests
```

#### Step B3：`docker-compose.yml`

```yaml
services:
  agent:
    build: .
    container_name: simple-agent
    restart: unless-stopped
    env_file: .env
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./data/sessions.db:/app/sessions.db
      - ./data/chroma_store:/app/chroma_store
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health').read()"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s

  nginx:
    image: nginx:alpine
    container_name: simple-agent-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deploy/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./deploy/certbot/conf:/etc/letsencrypt:ro
      - ./deploy/certbot/www:/var/www/certbot:ro
    depends_on:
      - agent
```

> `agent` 端口绑定 `127.0.0.1`，外部不能直连，必须经 Nginx，避免绕过 HTTPS。

#### Step B4：Nginx 配置（容器版）

`deploy/nginx.conf`：
```nginx
server {
    listen 80;
    server_name agent.yourdomain.com;

    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl http2;
    server_name agent.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/agent.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/agent.yourdomain.com/privkey.pem;

    client_max_body_size 10M;
    proxy_buffering off;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://agent:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
    }
}
```

#### Step B5：签发 HTTPS 证书

```bash
mkdir -p deploy/certbot/conf deploy/certbot/www data

# 临时用 standalone 签发
docker run --rm \
  -v $(pwd)/deploy/certbot/conf:/etc/letsencrypt \
  -v $(pwd)/deploy/certbot/www:/var/www/certbot \
  -p 80:80 \
  certbot/certbot certonly --standalone \
    -d agent.yourdomain.com \
    --agree-tos -m you@example.com --no-eff-email -n

# 启动完整服务
docker compose up -d --build
docker compose ps
docker compose logs -f
```

#### Step B6：证书自动续签

`deploy/renew.sh`：
```bash
#!/bin/bash
docker run --rm \
  -v $(pwd)/deploy/certbot/conf:/etc/letsencrypt \
  -v $(pwd)/deploy/certbot/www:/var/www/certbot \
  certbot/certbot renew --quiet
docker compose exec nginx nginx -s reload
```

```bash
chmod +x deploy/renew.sh
crontab -e
# 每月 1 号凌晨 3 点续签
0 3 1 * * cd /home/deploy/simple-agent && ./deploy/renew.sh >> /var/log/cert-renew.log 2>&1
```

#### Step B7：发布更新流程

```bash
cd /home/deploy/simple-agent
git pull
docker compose up -d --build
docker image prune -f
```

---

### 12.5 监控、备份、排错

#### 健康检查
```bash
curl https://agent.yourdomain.com/api/health
```
接入 UptimeRobot / 阿里云云监控，3 分钟探测一次，挂了自动告警。

#### 日志查看

| 部署方式 | 命令 |
|---|---|
| systemd | `sudo journalctl -u simple-agent -f` |
| systemd 文件 | `tail -f /var/log/simple-agent.log` |
| Docker | `docker compose logs -f agent` |
| Nginx 访问 | `tail -f /var/log/nginx/access.log` |
| Nginx 错误 | `tail -f /var/log/nginx/error.log` |

#### 数据备份

```bash
# 手动备份
tar -czf backup-$(date +%F).tar.gz data/

# cron 自动
crontab -e
0 2 * * * cd /home/deploy/simple-agent && tar -czf /backup/agent-$(date +\%F).tar.gz data/
```

#### 常见线上问题

| 现象 | 排查 |
|---|---|
| 502 Bad Gateway | 应用没起来：`systemctl status` 或 `docker compose ps` |
| SSE 流式断流 | Nginx 漏配 `proxy_buffering off`；或经过带缓冲 CDN |
| 502 + 长 prompt | `proxy_read_timeout` 调到 300s |
| HTTPS 证书过期 | `sudo certbot renew --dry-run` 检查续签 |
| `database is locked` | worker 数过多抢 SQLite，降到 1 或迁 Postgres |
| 内存 OOM | chromadb 向量过多，定期清理或换轻量 embedding |
| LLM 调用慢 | 服务器到 LLM 服务商延迟，优先同区域机房 |

#### 安全加固清单

- ✅ 关闭 8000 端口对公网暴露，只允许 Nginx 访问
- ✅ `chmod 600 .env`，仅 deploy 用户可读
- ✅ `.env` / `sessions.db` / `chroma_store` 不提交 git
- ✅ 内部系统加 IP 白名单或 Basic Auth
- ✅ 定期 `apt upgrade` 打安全补丁
- ✅ `fail2ban` 防 SSH 爆破
- ✅ LLM API Key 设配额告警

#### 性能调优速查

| 场景 | 优化 |
|---|---|
| 并发 > 10 QPS | gunicorn worker 数 = `2 * CPU + 1` |
| 向量库 > 10 万条 | `EMBED_PROVIDER=openai` 换高质量 embedding |
| 多机部署 | SQLite → Postgres；Chroma → Qdrant 集群 |
| 全球访问 | 套 Cloudflare CDN，注意关 SSE 缓冲 |

---

### 12.6 部署方案对比速查

| 维度 | 方案 A systemd | 方案 B Docker |
|---|---|---|
| 学习成本 | 需懂 systemd / Nginx | 需懂 Docker |
| 资源占用 | 最低（无容器开销） | 额外 100MB 左右 |
| 迁移难度 | 中（要重装依赖） | 低（镜像即走） |
| 多版本并存 | 难 | 容易（镜像 tag） |
| 回滚 | `git checkout` + 重启 | `docker compose` 切 tag |
| 适合场景 | 单机、长期运行 | 多环境、CI/CD |

没有强偏好就选 **方案 B Docker**，可移植性最好。

---

## 十三、Skill 系统（按需加载的能力包）

借鉴 Claude / JoyCode 的 Skill 设计，本项目支持把"领域知识 + 操作指南 + 辅助资源"打包成一个 skill 目录，Agent 在合适时机自动激活。

### 13.1 设计思路（两阶段激活）

| 阶段 | 加载内容 | 上下文成本 |
|---|---|---|
| **启动时** | 仅每个 skill 的 `name + description`（约 50 tokens/个） | 极低 |
| **运行时** | LLM 调 `activate_skill(name=...)` → 把完整 SKILL.md 正文回灌 | 按需展开 |

好处：**100 个 skill 注册进来，启动只多 5K tokens**，而不是全部塞进 system prompt。

### 13.2 目录结构

```
simple-agent/
├── skills/
│   ├── csv-analyzer/                     ← 一个 skill = 一个目录
│   │   ├── SKILL.md                      ← 必需：入口说明（含 frontmatter）
│   │   └── references/
│   │       └── pandas-cheatsheet.md      ← 可选：参考资料
│   └── sql-formatter/
│       └── SKILL.md
└── skill_loader.py                       ← 扫描器
```

### 13.3 SKILL.md 格式

frontmatter（YAML 风格）+ Markdown 正文：

```markdown
---
name: csv-analyzer
description: 当用户需要分析 CSV 文件、统计数据分布、查找异常值时使用。触发词：CSV、表格、统计、分组。
---

# CSV 分析技能

## 何时使用
- 用户明确提到 CSV
- 需要计算均值、分位数...

## 操作流程
1. 先读取前 100 行确认结构
2. ...

## 输出规范
- 先结论后数据
- 异常必须高亮
```

**字段说明**：
- `name`（必需）：skill 唯一标识，对应 `activate_skill(name=...)` 的参数
- `description`（必需）：**写清楚什么场景下用**，这是 LLM 决定要不要激活的唯一依据
- 正文：详细的 how-to，会在激活时整段塞进上下文，写得越具体效果越好

### 13.4 工作流程示例

用户问：「帮我分析一下 sales.csv 的数据分布」

```
轮 1：
  system: "...可用 skills: csv-analyzer / sql-formatter..."
  user:   "帮我分析一下 sales.csv 的数据分布"
  LLM 决策 → 调 activate_skill(name="csv-analyzer")
  工具返回 → CSV 分析技能的完整 SKILL.md 正文

轮 2：
  LLM 拿到详细指令 → 按 "Step 1: 探查数据结构" 行动
  调 read_file("sales.csv") 或反问用户路径
  ...
```

### 13.5 添加一个新 Skill（3 步）

**Step 1**：建目录与 SKILL.md
```bash
mkdir -p skills/my-skill
cat > skills/my-skill/SKILL.md <<'EOF'
---
name: my-skill
description: 当用户 XXX 时使用此技能...
---

# My Skill 详细说明
（操作步骤）
EOF
```

**Step 2**：重启服务（或调用 `prompts.refresh_skills()` 热刷新）
```python
from prompts import refresh_skills
refresh_skills()
```

**Step 3**：验证已被识别
```bash
python -c "from skill_loader import load_skills; print(list(load_skills().keys()))"
# 应输出：['csv-analyzer', 'my-skill', 'sql-formatter']
```

无需改任何 Python 代码——**纯文件驱动**。

### 13.6 与工具系统的关系

| 维度 | Tool（工具） | Skill（技能） |
|---|---|---|
| 形态 | Python 函数 + JSON Schema | Markdown 文件 |
| 注册方式 | 改 `tools.py` 的 `TOOLS` / `TOOL_MAP` | 加文件即可 |
| 作用 | 执行确定动作（计算、查时间、调 API） | 提供方法论、操作流程、约束 |
| 何时用 | 需要"做事" | 需要"按某种方式做事" |
| 是否消耗上下文 | Schema 常驻 | 仅激活时塞入 |

**两者协作**：skill 在正文里告诉 LLM「先调 tool A，再调 tool B」，把工具的串联策略文档化。

### 13.7 核心代码导览

| 文件 | 作用 |
|---|---|
| [`skill_loader.py`](skill_loader.py) | 扫描 `skills/`、解析 frontmatter、构建索引、提供 `activate_skill()` |
| [`prompts.py`](prompts.py) | `_build_v2_with_skills()` 自动把 skill 索引追加进 system prompt |
| [`tools.py`](tools.py) | 注册 `activate_skill` 工具，转发到 `skill_loader.activate_skill` |
| [`tests/test_skills.py`](tests/test_skills.py) | 10 个测试覆盖扫描、解析、激活、错误兜底 |

### 13.8 最佳实践

- **description 写"何时使用"，不要写"是什么"**：LLM 看的是触发条件，不是介绍。
  - ❌ `description: 一个 SQL 工具`
  - ✅ `description: 当用户需要格式化或美化 SQL 语句时使用`
- **正文要给可执行的步骤**：写"按步操作"而不是"理论介绍"，LLM 才知道下一步做什么。
- **明确"禁止事项"**：列 "不要把超过 20 行原始数据贴出来"、"不要修改用户文件" 等约束。
- **References 资料按需引**：体积大的速查表/规范放 `references/` 子目录，正文中只提"必要时查 X"，避免上下文爆炸。
- **粒度别太细**：一个 skill 解决一类问题就够，不要每个小操作都拆 skill，否则 LLM 选择困难。

### 13.9 排错

| 现象 | 原因 |
|---|---|
| Agent 完全不调用 activate_skill | description 写得太抽象，LLM 识别不到匹配场景 |
| 调用了但报"skill 未找到" | name 不一致，检查 SKILL.md 里 `name:` 字段和目录名 |
| 启动时 system prompt 没有 skill 列表 | `skills/` 目录路径不对，或 SKILL.md 缺 frontmatter |
| 修改了 SKILL.md 不生效 | 缓存：`prompts.CURRENT` 只在启动时构建一次，需调用 `refresh_skills()` 或重启 |

---

## 许可证

本项目为教学示例，可自由复制、修改、商用。如使用过程中遇到问题，欢迎在源码里直接加注释改造——这正是"Simple Agent"的本意：**少即是多，看得懂才改得动**。