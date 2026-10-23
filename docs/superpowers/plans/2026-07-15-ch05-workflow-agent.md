# Ch05 Workflow + Agent 混合架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 把客服后端重构成「确定性 Workflow 骨架 + 主力 Agent 核心节点」的混合架构，两个 API 入口都走图，前端加转人工/建工单两个按钮。

**Architecture:** `StateGraph` 编排骨架（指代消解→意图识别→分流→强制检索/置信度闸→主力 Agent→日志）；知识类强制先检索、证据弱生成前短路兜底；主力 Agent 是 `agent_llm ↔ agent_tools` 的 ReAct 环，复用 ch02 五工具；会话状态用 State 贯穿、AsyncSqliteSaver 持久化；转人工/建工单由后端产出可选项、前端按钮触发。

**Tech Stack:** LangGraph + langgraph-checkpoint-sqlite + aiosqlite；FastAPI（既有）；LangChain ChatOpenAI 直连上游（既有）；MySQL/SQLAlchemy（既有，存工单/低置信池/消息审计）；原生 HTML/CSS/JS 前端（既有，加按钮）。

**Spec:** `docs/superpowers/specs/2026-07-15-ch05-workflow-agent-design.md`

## Global Constraints

- 上游选型定死：LangGraph 图编排 + State + checkpointer；checkpointer = **AsyncSqliteSaver**（自带、持久）；前端 = 原生 HTML/CSS/JS，仅加按钮。发现矛盾停下问用户，不自行换方案。
- 复用 ch02 五工具（query_order/query_product/query_logistics/query_faq/create_ticket），**本章不新写业务工具**。D7 只修改现有 `query_logistics` 入参与 `query_order` 出参。
- 意图识别单标签七类（物流/订单/商品咨询/退款退货/售后/投诉/闲聊），指代消解原样透传——两者正式版留 ch06。
- 转人工、建工单是两件事，分开；后端**不自动执行**，只产出可选项；建工单写库只在按钮端点、点了才写；转人工纯前端模拟。**建工单不把会话状态改成“已转人工”**。
- 涉及 LangGraph/FastAPI/SQLAlchemy 具体 API，动手前用 Context7 核实当前接口。
- 不可单测的产出（纯 Prompt/数据）用标注样例/评估集验证代替 TDD；其余走 TDD。
- 端到端**真实**验收必须跑通（非纸面）：五条验收在浏览器/真实服务上复现。
- 每个任务末尾 commit；`dev-notes/ch05.md` 每完成一阶段追记。
- 模型非确定性：eval 抖动如实重跑记录。

## Repo Mapping（本仓库实际路径）

你贴的 plan 按另一套目录写；本仓库按下表映射，不新建平行 `app/core`。

| Plan 里的路径 | 本仓库实际 |
|---|---|
| `app/core/agent.py`（旧两次编排） | `app/services/chat.py::stream_chat`（改为图适配器） |
| `app/api/chat.py` | `app/api/routes.py`（保留 `/api/chat/stream`，新增 `/api/chat` 别名） |
| `app/api/agent.py` | `app/api/routes.py` 新增 `POST /api/agent` |
| `app/api/actions.py` | `app/api/actions.py`（新建，路由在 `app/main.py` 挂载） |
| `app/core/prompts.py` | `app/prompts/*.txt` + `app/graph/nodes.py` 固定话术常量 |
| `app/core/llm.py::get_chat_model` | `app/llm.py::build_chat_model` + `Runtime.chat_model` |
| `app/tools/registry.py::get_all_tools` | `app/tools/__init__.py::build_tools(repository, conversation_id, faq_search)` |
| `app/tools/infra.py::execute_tool_call` | `app/services/tool_exec.py::ToolRegistry.execute(name, args)` |
| `app/db/repository.py` | `app/db/repositories.py::ChatRepository` |
| `app/core/retrieval.py` | `app/knowledge/search.py::search_knowledge_result`、`app/knowledge/evidence.py` |
| `app/core/selfcheck.py` | `app/knowledge/self_check.py::EvidenceChecker` |
| `app/core/query_understanding.py` | `app/knowledge/query_understanding.py`（`search_knowledge_result` 内部已接改写） |
| `app/schemas/agent.py` | `app/schemas.py`（单文件，追加模型） |
| `app/static/index.html` | `app/static/chat.html` + `chat.js` + `chat.css` |
| `tests/test_agent_*.py` | `tests/test_chat_*.py`（旧图编排测试改造/退役） |

**依赖说明：** 本会话 `.venv` 没有 `langgraph-checkpoint-sqlite` / `aiosqlite`，沙箱不能拉 PyPI。Task 1 先加依赖与配置；真实 AsyncSqliteSaver 冒烟由用户在终端 `uv lock && uv sync` 后跑。本会话单测统一注入 `InMemorySaver`，不伪造 AsyncSqliteSaver 结果。

**流程说明：** 本计划用 `superpowers:executing-plans` 在当前会话逐任务实现；每个任务先写失败测试，再实现，再跑测试，再 commit。

## Review Focus

- **Agent 中间轮文本泄漏**：`stream_mode="messages"` 会流出 `agent_llm` 每一步的 token；工具调用轮的过渡文本可能先显示再被最终答案覆盖。期望：最终气泡只保留最终答复，不残留“我先查一下”之类的中间文本。
- **create_ticket 被 Agent 自动执行**：模型可能直接调 `create_ticket`。期望：`agent_tools` 拦截为 `suggested_actions`，绝不写 `tickets`；只有按钮端点写库。
- **建工单端点污染会话状态**：ch02 `create_ticket` 曾把会话改成“已转人工”。期望：本章建工单只写 `tickets`，会话状态不变；转人工纯前端。
- **闲聊/投诉节点花费模型调用**：期望闲聊固定话术零模型调用（纯问候走快路径）；投诉不调模型；只有意图分类/知识检索/Agent 花模型调用。
- **置信度闸放晚**：弱证据必须先于 Agent 流式生成短路。期望：`forced_rag → confidence_check → fallback_reply`，弱证据时不出现 Agent token。

---

## Task 1: 依赖 + 配置 + LangGraph/checkpointer/流式红线冒烟

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Create: `scripts/smoke_langgraph.py`
- Test: 无（冒烟脚本 + 本会话 InMemorySaver 探针）

**Interfaces:**
- Produces: `Settings.max_agent_steps: int = 6`、`Settings.checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"`、`Settings.intent_prompt_path: Path`、`Settings.agent_prompt_path: Path`

- [ ] **Step 1: Context7 核实 AsyncSqliteSaver / astream 接口**

查 `/websites/langchain_oss_python_langgraph`：`from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`、`AsyncSqliteSaver.from_conn_string(path)` 是 async context manager、`await checkpointer.setup()`；`graph.astream(inp, config, stream_mode=["messages","updates"])` 在默认 v1 下产出 `(mode, chunk)` 元组。把结论写进 `dev-notes/ch05.md`。

- [ ] **Step 2: 加依赖**

`pyproject.toml` dependencies 增加：

```toml
"langgraph>=1.2,<2",
"langgraph-checkpoint-sqlite>=2.0",
"aiosqlite>=0.20",
```

Run: `uv lock`（本会话沙箱会失败，记录错误；用户在终端执行后提交 `uv.lock`）
Expected: 用户终端安装成功，`uv pip show langgraph-checkpoint-sqlite aiosqlite` 有版本。

- [ ] **Step 3: 加配置**

`app/config.py` 的 `Settings` 增加：

```python
    # ch05 图编排
    max_agent_steps: int = 6
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"
    intent_prompt_path: Path = Path("app/prompts/intent_prompt.txt")
    agent_prompt_path: Path = Path("app/prompts/agent_prompt.txt")
    intent_temperature: float = 0.0
    intent_max_tokens: int = 128
```

- [ ] **Step 4: 写冒烟脚本**

`scripts/smoke_langgraph.py`：用 `AsyncSqliteSaver.from_conn_string(":memory:")`、`build_chat_model(settings)`（见 Task 11 的 streaming 说明）、一个单节点图，打印 `messages` 模式的 `(msg, meta)` 和 `updates` 模式的 `{node: update}`。导入失败时打印 `run uv lock && uv sync` 并以退出码 2 结束，不静默 fallback。

- [ ] **Step 5: 本会话用 InMemorySaver 做形状探针**

不提交临时脚本；用 `.venv/bin/python - <<'PY'` 跑一个 `InMemorySaver` + `FakeListChatModel(streaming=True)` 的单节点图，确认：

```text
('messages', (AIMessageChunk(...), {'langgraph_node': 'agent_llm', ...}))
('updates', {'agent_llm': {'messages': [AIMessage(...)]}})
```

把真实形状写进 `dev-notes/ch05.md`。AsyncSqliteSaver 的真实冒烟标记“待用户终端”。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml app/config.py scripts/smoke_langgraph.py dev-notes/ch05.md
git commit -m "feat(ch05): 加 LangGraph 依赖+配置,钉死流式形状"
```

## Task 2: 祛魅热身——手写裸 Agent 循环 + 工具数据依赖

**Files:**
- Create: `scripts/bare_agent_loop.py`
- Modify: `app/tools/order.py`
- Modify: `app/tools/logistics.py`
- Modify: `evals/run_tool_routing_eval.py`（假 args）
- Modify: `scripts/ch02_acceptance.py`（物流工具断言）
- Test: `tests/test_tools.py`、`tests/test_bare_agent_loop.py`（新增）

**Interfaces:**
- Produces: `query_order(order_id) -> dict` 含 `tracking_no: str`；`query_logistics(tracking_no) -> dict`；`run_bare_agent(model, tools, registry, query, max_turns=6) -> str`

- [ ] **Step 1: 先改工具测试（失败）**

`tests/test_tools.py`：
- `query_order` 返回含 `tracking_no`，形如 `SF` + 12 位数字。
- `query_logistics` 入参改为 `{"tracking_no": "SF123..."}`，返回 `tracking_no` 与 `status`。
- 旧的 `query_logistics({"order_id": ...})` 断言删除。

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: FAIL。

- [ ] **Step 2: 改工具实现**

`app/tools/order.py`：返回值增加 `"tracking_no": "SF" + 12 digits`（复用 `random.choices`）。
`app/tools/logistics.py`：参数改为 `tracking_no: str`；docstring 改为“查询物流轨迹，入参必须是订单查询返回的物流单号”；返回保留 `tracking_no`。

- [ ] **Step 3: 写裸循环**

`scripts/bare_agent_loop.py`：

```python
async def run_bare_agent(model, tools, registry, query: str, max_turns: int = 6) -> str:
    messages = [SystemMessage(AGENT_SYSTEM), HumanMessage(query)]
    for step in range(1, max_turns + 1):
        ai = await model.bind_tools(tools).ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print(f"[第{step}步] 无工具调用 -> 收敛")
            return chunk_text(ai.content)
        for tc in ai.tool_calls:
            result = await registry.execute(tc["name"], tc.get("args") or {})
            messages.append(ToolMessage(content=result.content,
                                        tool_call_id=tc["id"], name=tc["name"]))
    return "(未收敛)"
```

CLI 用 `Settings(_env_file=".env")`、`build_chat_model`、`DemoRepository`（内存版 `next_ticket_no`/`create_ticket`）、`build_tools`、`ToolRegistry`；默认问题 `订单1001的物流到哪了`。

- [ ] **Step 4: 加裸循环单测**

`tests/test_bare_agent_loop.py`：用 fake model（先返回一个 `query_order` tool_call，再返回最终答案）断言 registry 被调用一次、最终答案返回、循环步数为 2；max_turns 用尽返回 `(未收敛)`。

Run: `.venv/bin/pytest tests/test_bare_agent_loop.py tests/test_tools.py -v`
Expected: PASS。

- [ ] **Step 5: 运行裸循环（需上游）**

Run: `.venv/bin/python -m scripts.bare_agent_loop "订单1001的物流到哪了"`
Expected: 第 1 步 `query_order`，第 2 步 `query_logistics`（tracking_no 来自订单），最后收敛。上游不可用时标记“待用户终端”，不伪造输出。

- [ ] **Step 6: Commit**

```bash
git add scripts/bare_agent_loop.py app/tools/order.py app/tools/logistics.py evals/run_tool_routing_eval.py scripts/ch02_acceptance.py tests/test_tools.py tests/test_bare_agent_loop.py dev-notes/ch05.md
git commit -m "feat(ch05): 祛魅裸循环 + 物流单号数据依赖"
```

## Task 3: 会话 State 定义

**Files:**
- Create: `app/graph/__init__.py`（空）
- Create: `app/graph/state.py`
- Test: `tests/graph/__init__.py`（空）、`tests/graph/test_state.py`

**Interfaces:**
- Produces: `ConversationState`（TypedDict）、`merge_dict(a, b) -> dict`、`merge_trace(state, **updates) -> dict`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_state.py`：
- `ConversationState.__annotations__` 含 `messages/user_id/conversation_id/intent/route/evidence/citations/evidence_strong/answer/steps/tokens_used/suggested_actions/trace`。
- `merge_dict({"a":1},{"b":2}) == {"a":1,"b":2}`；`merge_dict(None,{"b":2}) == {"b":2}`；同键后写覆盖。
- `merge_trace({"trace":{"a":1}}, b=2) == {"a":1,"b":2}`。
- `add_messages` 追加行为。

- [ ] **Step 2: 跑失败**

Run: `.venv/bin/pytest tests/graph/test_state.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 State**

`app/graph/state.py`：

```python
def merge_dict(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}

def merge_trace(state, **updates) -> dict:
    return merge_dict(state.get("trace"), updates)

class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    conversation_id: int
    intent: str
    route: str
    evidence: str
    citations: list
    evidence_strong: bool
    answer: str
    steps: int
    tokens_used: int
    suggested_actions: list
    trace: dict          # 每轮输入显式重置；节点用 merge_trace 合并
```

> 与用户 plan 的差异：`trace` 不挂 `merge_dict` reducer。checkpointer 跨轮保留 state，若 trace 用累加 reducer，上一轮的 `suggested_actions`/`answer`/`route` 会污染下一轮。改为每轮输入重置 + 节点手动 `merge_trace`。

- [ ] **Step 4: 跑通过**

Run: `.venv/bin/pytest tests/graph/test_state.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/graph/__init__.py app/graph/state.py tests/graph/
git commit -m "feat(ch05): ConversationState + trace merge helper"
```

## Task 4: 意图分类器（Prompt/数据 → 评估集验证）

**Files:**
- Create: `app/prompts/intent_prompt.txt`
- Create: `app/graph/intent.py`
- Create: `scripts/eval_intent.py`
- Test: `tests/graph/test_intent.py`（fake 模型；Prompt 质量用评估集）

**Interfaces:**
- Produces: `INTENTS: tuple[str, ...]`、`IntentResult`、`IntentClassifier(model, prompt_path).classify(query) -> str`

- [ ] **Step 1: 写 Prompt**

`app/prompts/intent_prompt.txt`：七类定义 + 边界例子（“退款到哪了”=退款退货，“快递怎么还没到”=物流，“我要投诉”=投诉，“你好”=闲聊），要求只输出 `intent` 字段。纯问候快路径在代码里。

- [ ] **Step 2: 写失败测试**

`tests/graph/test_intent.py`：
- 纯问候 `"你好"`/`"谢谢"`/`"再见"` 直接返回 `闲聊`，且 fake model 的 `ainvoke` 调用次数为 0。
- 普通问题调用 fake structured model，返回其 `intent`。
- 模型抛异常 → `闲聊`。
- 返回越界值 → `闲聊`。

- [ ] **Step 3: 实现分类器**

`app/graph/intent.py`：`CHITCHAT_RE` 匹配纯问候/致谢/告别；`IntentClassifier` 持有 `with_structured_output(IntentResult)` 的 model 和 prompt 文本；`classify` 先快路径，再 `ainvoke([SystemMessage(prompt), HumanMessage(query)])`，异常/越界回退 `闲聊`。

- [ ] **Step 4: 写评估脚本**

`scripts/eval_intent.py`：14 条标注样例（七类各 2 条，见用户 plan），逐条打印 `✅/❌` 与判对率。

Run: `.venv/bin/python -m scripts.eval_intent`（需上游；本会话标记待终端）
Expected: 判对率高；明显偏差调 Prompt 重跑。结果记 dev-notes。

- [ ] **Step 5: 跑单测 + Commit**

Run: `.venv/bin/pytest tests/graph/test_intent.py -v`
Expected: PASS。

```bash
git add app/prompts/intent_prompt.txt app/graph/intent.py scripts/eval_intent.py tests/graph/test_intent.py dev-notes/ch05.md
git commit -m "feat(ch05): 单标签七类意图分类器 + 标注评估"
```

## Task 5: 路由与停止条件（确定性 → TDD）

**Files:**
- Create: `app/graph/routing.py`
- Test: `tests/graph/test_routing.py`

**Interfaces:**
- Produces: `INTENT_TO_ROUTE: dict[str,str]`、`route_by_intent(state)->str`、`confidence_gate(state)->str`、`should_continue(state)->Literal["continue","stop","cap"]`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_routing.py`：
- 七类映射：商品咨询/退款退货→knowledge；物流/订单/售后→business；投诉→complaint；闲聊→chitchat。
- 未知意图→business。
- `confidence_gate`：`evidence_strong=True`→strong；False/缺失→weak。
- `should_continue`：最后一条 AIMessage 无 tool_calls→stop；有 tool_calls 且 `steps < settings.max_agent_steps`→continue；有 tool_calls 且 `steps >= max`→cap；`tokens_used` 很大仍→continue（token 不参与停止）。

- [ ] **Step 2: 跑失败 / 实现 / 跑通过**

实现 `routing.py`。`should_continue` 读取 `settings.max_agent_steps`；`cap` 用于 Task 6 的 `budget_reply`。

Run: `.venv/bin/pytest tests/graph/test_routing.py -v`
Expected: PASS。

- [ ] **Step 3: Commit**

```bash
git add app/graph/routing.py tests/graph/test_routing.py
git commit -m "feat(ch05): 分流规则 + ReAct 停止条件"
```

## Task 6: 确定性出口节点（chitchat / complaint / fallback / budget）

**Files:**
- Create: `app/graph/nodes.py`（先放固定话术 + 这四个节点）
- Test: `tests/graph/test_reply_nodes.py`

**Interfaces:**
- Produces: `CHITCHAT_REPLY`、`COMPLAINT_REPLY`、`FALLBACK_REPLY`、`STEP_CAP_REPLY`；`chitchat_reply(state, *, deps)`、`complaint_reply(state, *, deps)`、`fallback_reply(state, *, deps)`、`budget_reply(state, *, deps)`、`_user_text(state)`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_reply_nodes.py`：
- `chitchat_reply` 返回 `answer == CHITCHAT_REPLY`，`trace.route=="chitchat"`，无 actions。
- `complaint_reply` 返回 `answer == COMPLAINT_REPLY`，actions 类型顺序 `["transfer_human","create_ticket"]`，ticket draft 的 `ticket_type=="投诉"`、`description` 等于用户原话。
- `fallback_reply` 用 fake deps.repository 捕获 `insert_low_confidence_question` 调用，断言 `source=="retrieval_low_conf"`、`conversation_id`、`reason` 非空；返回 `answer==FALLBACK_REPLY`。
- `budget_reply` 返回 `answer==STEP_CAP_REPLY`，actions 含 transfer_human 与 create_ticket。

- [ ] **Step 2: 实现四个节点**

固定话术按 spec 文案：
- `CHITCHAT_REPLY = "您好，我是客服小猫，很高兴为您服务～您可以问我订单、物流、商品或退换货政策方面的问题。"`
- `COMPLAINT_REPLY = "非常抱歉给您带来了不好的体验，您反映的问题我已经记录，会认真跟进。您可以点击下方按钮转接人工客服，或创建一个工单，我们会尽快处理。"`
- `FALLBACK_REPLY = "抱歉，知识库中没有找到能回答这个问题的依据，我不能凭空回答。建议您点击下方按钮转人工或建工单。"`
- `STEP_CAP_REPLY = "这个问题处理步骤较多，我先帮您转接人工客服，您也可以创建一个工单，我们会尽快跟进。"`

`fallback_reply` 调用 `deps.repository.insert_low_confidence_question(...)` + `commit()`；reason 取 `trace.evidence_top`/`trace.self_check`。

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/graph/test_reply_nodes.py -v`
Expected: PASS。

```bash
git add app/graph/nodes.py tests/graph/test_reply_nodes.py
git commit -m "feat(ch05): 闲聊/投诉/兜底/步数封顶四出口"
```

## Task 7: coref / classify_intent / forced_rag / confidence_check

**Files:**
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_forced_rag.py`
- Modify: `app/prompts/agent_prompt.txt`（新建，Task 8 使用）

**Interfaces:**
- Produces: `coref(state)->dict`、`classify_intent(state, *, deps)->dict`、`forced_rag(state, *, deps)->dict`、`confidence_check(state, *, deps)->dict`
- Consumes: `GraphDeps` 字段 `settings/knowledge_repository/evidence_checker/intent_classifier`（Task 11 定义；本任务测试用 SimpleNamespace）

- [ ] **Step 1: 写失败测试**

`tests/graph/test_forced_rag.py`：
- `coref` 返回 `trace.coref=="passthrough"`，不改 `messages`。
- `classify_intent` 用 fake classifier 返回 `订单`，state.intent/ trace.intent 正确。
- `forced_rag` 强证据：monkeypatch `nodes.search_knowledge_result` 返回 `RetrievalResult(strategy="hybrid_rerank", rewritten_query=..., reranked=True, hits=[...])`；fake evidence_checker `useful=True`；断言 `evidence_strong is True`、`"[1]" in evidence`、`citations[0]["n"]==1`、`trace.forced_rag is True`。
- `forced_rag` 无命中：`evidence_strong is False`，`trace.forced_rag is True`。
- `forced_rag` 有命中但 selfcheck `useful=False`：`evidence_strong is False`，`trace.self_check` 有 reason。
- `confidence_check` 强/弱写 trace。

- [ ] **Step 2: 实现**

- `coref`：`return {"trace": merge_trace(state, coref="passthrough")}`。
- `classify_intent`：`intent = await deps.intent_classifier.classify(_user_text(state))`；返回 `intent` + trace。
- `forced_rag`：
  - `result = await search_knowledge_result(query, repository=deps.knowledge_repository, settings=deps.settings, strategy="hybrid_rerank")`；
  - `ordered = prompt_order(result.hits)`；`full = citation_items_full(ordered)`；`light = citation_items_light(ordered)`；
  - `top = ordered[0][1].score if ordered else 0.0`；
  - 机械闸：`if not ordered or top < deps.settings.rerank_min_score` → weak；
  - 自评闸：`assessment = await deps.evidence_checker.assess(query, full)`；`if assessment is not None and not assessment.useful` → weak；
  - strong：`evidence=render_evidence_block(full)`、`citations=light`、`evidence_strong=True`；
  - trace：`forced_rag=True`、`evidence_top=top`、`self_check=...`、`retrieval_strategy=...`、`reranked=...`。
- `confidence_check`：`decision = "strong" if state.get("evidence_strong") else "weak"`；返回 trace。
- `app/prompts/agent_prompt.txt`：写 Agent 追加段（先调工具、不猜、证据 `[n]`、不重复 query_faq、缺信息追问、投诉/无法解决时调 create_ticket 表达提议、不要自动建单、保持 ch04 负面知识）。

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/graph/test_forced_rag.py -v`
Expected: PASS。

```bash
git add app/graph/nodes.py app/prompts/agent_prompt.txt tests/graph/test_forced_rag.py
git commit -m "feat(ch05): coref 透传 + 意图识别 + 强制检索 + 证据闸"
```

## Task 8: 主力 Agent 节点（agent_llm / agent_tools + create_ticket 拦截）

**Files:**
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_agent_nodes.py`

**Interfaces:**
- Produces: `_agent_messages(state, deps)->list`、`agent_llm(state, *, deps)->dict`、`agent_tools(state, *, deps)->dict`
- Consumes: `deps.chat_model`、`deps.tools`、`deps.tool_registry`、`deps.prompt_builder`、`deps.agent_prompt`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_agent_nodes.py`：
- `_agent_messages` 在 knowledge 路把证据拼进 SystemMessage，并含“不要再调用 query_faq”；business 路不含证据段。
- `agent_llm` 用 fake model（`bind_tools` 返回自身，`ainvoke` 返回带 usage 的 AIMessage）断言 `steps+1`、`tokens_used` 累加、messages 追加。
- `agent_tools` 普通工具：fake registry `execute` 返回 `ToolResult`；断言 ToolMessage 追加、`trace.tools` 记录 name/ok。
- `agent_tools` 拦截 `create_ticket`：fake registry 若被调用即失败；断言 `suggested_actions` 含 `create_ticket`、synthetic ToolMessage 的 `tool_call_id` 匹配、无写库调用。
- `agent_tools` 多个工具调用按顺序执行并返回多条 ToolMessage。

- [ ] **Step 2: 实现**

- `_agent_messages`：`system = deps.prompt_builder.render_system_text() + "\n\n" + deps.agent_prompt`；knowledge 且有 evidence 时追加证据段；返回 `[SystemMessage(system), *state["messages"]]`。
- `agent_llm`：`model = deps.chat_model.bind_tools(deps.tools)`；`ai = await model.ainvoke(_agent_messages(state, deps))`；`used = (ai.usage_metadata or {}).get("total_tokens", 0)`；返回 messages/steps/tokens_used。
- `agent_tools`：遍历最后一条 AIMessage 的 tool_calls；`create_ticket` 拦截；其他 `deps.tool_registry.execute`；ToolMessage 内容：`query_faq` 成功时把 JSON 解析后用 `render_evidence_block` 渲染，其他用 `result.content`；trace.tools 记录 `{"name","ok","error_code","duration_ms","args"}`；suggested_actions 合并去重。
- `create_ticket` 的 synthetic ToolMessage 文案：`"已把『建工单』选项交给用户自行确认。请用一句话简要说明并停止,不要再调用任何工具。"`

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/graph/test_agent_nodes.py -v`
Expected: PASS。

```bash
git add app/graph/nodes.py tests/graph/test_agent_nodes.py
git commit -m "feat(ch05): ReAct Agent 节点 + create_ticket 拦截"
```

## Task 9: 日志节点 log + resolve_answer

**Files:**
- Modify: `app/graph/nodes.py`
- Test: `tests/graph/test_log_node.py`

**Interfaces:**
- Produces: `resolve_answer(state)->str`、`log_node(state, *, deps)->dict`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_log_node.py`：
- `resolve_answer` 优先 `state["answer"]`；否则取最后 AIMessage 文本；block list 拼接。
- `log_node` 用 fake deps.repository 捕获 `add_message` + `commit`；断言 `role=="assistant"`、content 正确、日志含 `ch05 turn`（用 caplog）。
- 确定性节点（state 有 answer、最后消息不是同文 AIMessage）时，`log_node` 返回 `messages=[AIMessage(answer)]`；Agent 答复（最后 AIMessage 已是答案）不重复追加。

- [ ] **Step 2: 实现**

`log_node`：
```python
answer = resolve_answer(state)
updates = {}
if state.get("answer") and (not last_is_same_ai):
    updates["messages"] = [AIMessage(content=answer)]
if state.get("conversation_id"):
    await deps.repository.add_message(cid, "assistant", content=answer or None)
    await deps.repository.commit()
logger.info("ch05 turn conv=%s intent=%s route=%s trace=%s", ...)
return updates
```

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/graph/test_log_node.py -v`
Expected: PASS。

```bash
git add app/graph/nodes.py tests/graph/test_log_node.py
git commit -m "feat(ch05): 日志节点 + 消息审计"
```

## Task 10: 图组装 build_graph

**Files:**
- Create: `app/graph/build.py`
- Test: `tests/graph/test_build.py`

**Interfaces:**
- Produces: `build_graph(checkpointer, deps)`、`_builder(deps)`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_build.py`：
- `_builder(dummy_deps).compile()` 的节点集合包含 `coref/classify_intent/forced_rag/confidence_check/agent_llm/agent_tools/complaint_reply/chitchat_reply/fallback_reply/budget_reply/log`。
- 用 `InMemorySaver` 编译成功。
- 边映射：`confidence_check` 条件边 strong→agent_llm、weak→fallback_reply；`agent_llm` 条件边 continue→agent_tools、stop→log、cap→budget_reply。

- [ ] **Step 2: 实现**

`build.py` 用 `functools.partial` 把 `deps` 注入需要依赖的节点；纯节点直接注册。边按 spec §2/§3；`agent_tools → agent_llm`；所有出口 → `log → END`。

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/graph/test_build.py -v`
Expected: PASS。

```bash
git add app/graph/build.py tests/graph/test_build.py
git commit -m "feat(ch05): StateGraph 骨架组装"
```

## Task 11: 运行时——checkpointer 生命周期 + stream_turn/run_turn

**Files:**
- Create: `app/graph/runtime.py`
- Modify: `app/llm.py`（`build_chat_model` 加 `streaming=True`）
- Test: `tests/graph/test_runtime.py`

**Interfaces:**
- Produces:
  - `GraphDeps` dataclass
  - `async init_graph(settings)` / `async close_graph()` / `set_checkpointer(cp)`（测试）
  - `async stream_turn(deps, conversation_id, user_id, message) -> AsyncIterator[tuple[str, dict]]`
  - `async run_turn(deps, conversation_id, user_id, message) -> dict`
  - 事件：`session/token/tool/actions/citations/done/error`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_runtime.py` 注入 fake graph/InMemorySaver：
- `stream_turn` 过滤 `langgraph_node`：只把 `agent_llm` 的 messages token 变成 `token`；`classify_intent` 的 token 不出现。
- `updates` 中 `forced_rag.citations` → `citations` 事件；`agent_tools.trace.tools` → `tool` 事件；`suggested_actions` → 最后 `actions` 事件；末尾 `done` 带 conversation_id/trace。
- 确定性节点 `answer` → 一条 `token`。
- `run_turn` 返回 `{"conversation_id", "state"}`。
- 会话不存在 → `ConversationNotFound`。

- [ ] **Step 2: 实现**

- `GraphDeps` 字段：`settings/chat_model/intent_classifier/prompt_builder/repository/knowledge_repository/evidence_checker/faq_search/tools/tool_registry/agent_prompt`。
- `init_graph(settings)`：`AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)`（async context manager）→ `await setup()` → 存全局。
- `set_checkpointer(cp)` 供测试注入 InMemorySaver。
- `stream_turn`：
  - 写 user 消息 `add_message` + commit；
  - `graph = build_graph(get_checkpointer(), deps)`；
  - input 显式重置所有每轮字段（intent/route/evidence/citations/evidence_strong/answer/steps/tokens_used/suggested_actions/trace）；
  - `async for mode, chunk in graph.astream(input, config, stream_mode=["messages","updates"])`；
  - messages 过滤 `ANSWER_NODES={"agent_llm"}`；updates 处理确定性 answer/citations/tool/actions；
  - `final = await graph.aget_state(config)`；`done` 用 final.values 组装。
- `run_turn`：写 user 消息 → `graph.ainvoke` → 返回 final。
- 错误：`map_upstream_error` → `("error", {...})`；`ConversationNotFound` → 404 由 API 处理。

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/graph/test_runtime.py -v`
Expected: PASS。

```bash
git add app/graph/runtime.py app/llm.py tests/graph/test_runtime.py
git commit -m "feat(ch05): 图运行时 + checkpointer 生命周期 + 事件映射"
```

## Task 12: /api/agent 非流式 + schema

**Files:**
- Modify: `app/schemas.py`
- Modify: `app/api/routes.py`
- Test: `tests/graph/test_agent_view.py`、`tests/test_agent_api.py`（新增）

**Interfaces:**
- Produces: `AgentRequest`、`AgentResponse`、`ToolCallView`、`ToolResultView`；`_views_from_state(state)`；`POST /api/agent`

- [ ] **Step 1: 写失败测试**

- `_views_from_state` 从 `AIMessage.tool_calls` 与 `ToolMessage` 重建 calls/results。
- `POST /api/agent` 用 fake runtime 返回 answer/tool_calls/suggested_actions；会话不存在 404；上游异常 502。

- [ ] **Step 2: 实现**

`app/schemas.py` 加模型；`app/api/routes.py` 加路由：打开 repository scope，构建 deps（Task 11 的 helper），调用 `run_turn`，映射 `ConversationNotFound`/`SQLAlchemyError`/其他异常。

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/graph/test_agent_view.py tests/test_agent_api.py -v`
Expected: PASS。

```bash
git add app/schemas.py app/api/routes.py tests/graph/test_agent_view.py tests/test_agent_api.py
git commit -m "feat(ch05): /api/agent 走图 ainvoke"
```

## Task 13: /api/chat 流式接图 + actions SSE

**Files:**
- Modify: `app/services/chat.py`（旧两次编排 → 图适配器）
- Modify: `app/api/routes.py`（`/api/chat/stream` + `/api/chat` 别名）
- Test: `tests/test_chat_stream.py`（改造）、`tests/test_chat_service.py`（退役/改造）

**Interfaces:**
- Consumes: `runtime.stream_turn`
- Produces: SSE 帧 `session/token/tool/citations/actions/done/error`

- [ ] **Step 1: 写失败测试**

- `stream_chat` 用 fake runtime 产出 `token/tool/actions/citations/done`，断言事件顺序与 `done` 字段。
- SSE 路由：`/api/chat/stream` 与 `/api/chat` 都返回 `event: actions` 帧。
- 错误路径：上游异常 → `error` 帧。

- [ ] **Step 2: 实现**

`stream_chat` 保留签名，内部：加锁 → `repository.get_or_create_conversation` → 构建 `GraphDeps` → `async for event in stream_turn(...)` → yield。`/api/chat` 别名复用同一个 handler。

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/test_chat_stream.py tests/test_chat_service.py -v`
Expected: PASS。

```bash
git add app/services/chat.py app/api/routes.py tests/test_chat_stream.py tests/test_chat_service.py
git commit -m "feat(ch05): /api/chat 流式接图 + actions 帧"
```

## Task 14: 建工单端点 + 与转人工解耦

**Files:**
- Create: `app/services/tickets.py`
- Create: `app/api/actions.py`
- Modify: `app/tools/ticket.py`
- Modify: `app/schemas.py`
- Modify: `app/main.py`
- Test: `tests/test_actions_api.py`、`tests/test_tools.py`

**Interfaces:**
- Produces: `create_ticket_record(repository, conversation_id, description, ticket_type) -> Ticket`；`POST /api/actions/create-ticket` → `{ticket_no, status}`

- [ ] **Step 1: 写失败测试**

- `create_ticket_record` 写 `tickets`，不调用 `set_conversation_status`。
- `POST /api/actions/create-ticket` 合法类型写库并返回 ticket_no；非法类型 422；会话不存在 404；DB 错误 503。
- ch02 `build_ticket_tool` 仍可创建工单，但不再改会话状态。

- [ ] **Step 2: 实现**

抽取 `app/services/tickets.py`；`app/tools/ticket.py` 调它并删掉 `set_conversation_status`；`app/schemas.py` 加 `CreateTicketRequest/Response`；`app/api/actions.py` 打开 repository scope、校验会话存在、调 service；`app/main.py` include router。

- [ ] **Step 3: 跑通过 + Commit**

Run: `.venv/bin/pytest tests/test_actions_api.py tests/test_tools.py -v`
Expected: PASS。

```bash
git add app/services/tickets.py app/api/actions.py app/tools/ticket.py app/schemas.py app/main.py tests/test_actions_api.py tests/test_tools.py
git commit -m "feat(ch05): 建工单端点 + 与转人工解耦"
```

## Task 15: main.py lifespan 起图

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_static.py`（回归）+ 启动冒烟

- [ ] **Step 1: 改 lifespan**

lifespan 内 `await runtime.init_graph(settings)`；退出时 `await runtime.close_graph()`。若测试环境没有 AsyncSqliteSaver，允许注入 InMemorySaver（测试 fixture）。

- [ ] **Step 2: 启动冒烟**

Run: `.venv/bin/pytest tests/test_static.py -v`；用户终端 `make dev` 后 `curl -s localhost:8000/docs` 确认 `/api/actions/create-ticket` 与 `/api/agent` 存在。
Expected: PASS + 应用正常起。

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat(ch05): lifespan 起图 checkpointer"
```

## Task 16: 前端——actions 帧 + 转人工/建工单表单

**Files:**
- Modify: `app/static/chat.html`
- Modify: `app/static/chat.js`
- Modify: `app/static/chat.css`
- Test: 浏览器手动验收（无前端单测）

**Interfaces:**
- Consumes: SSE `actions` 帧；`POST /api/actions/create-ticket`

- [ ] **Step 1: 渲染 actions 按钮**

`chat.js` 在 SSE 分发中加 `actions` 分支；`renderActions(bubble, items)` 在 assistant 气泡下渲染独立按钮。`human_handoff` → 「转人工」；`create_ticket` → 「建工单」。

- [ ] **Step 2: 转人工**

点击后禁用按钮；追加两条 assistant 消息：`已转接人工客服`、`您好，我是客服小猫，请问有什么可以帮您的`；不调后端。

- [ ] **Step 3: 建工单表单**

`chat.html` 加 `<dialog id="ticketDialog">`：`<select id="ticketType">` 默认空（售后/投诉/咨询）、`<textarea id="ticketDescription">` 留空；提交按钮 + 取消。`chat.js` 点击「建工单」打开弹窗；提交前校验两项必填；提交 `POST /api/actions/create-ticket`，body `{conversation_id: Number(sessionId), ticket_type, description}`；成功显示 `工单已创建：{ticket_no}` 并禁用该气泡的建工单按钮；失败显示错误、按钮可重开；取消/遮罩/Esc 不建单。

- [ ] **Step 4: 样式**

`chat.css` 加 `.action-bar`、`.action-btn`、`.ticket-dialog`、表单校验错误态。

- [ ] **Step 5: 浏览器验收**

起全服务后：
- 问「我要投诉」→ 两个独立按钮；
- 点「转人工」→ 两条模拟消息；
- 点「建工单」→ 弹表单，空提交被拦；填写后提交 → 工单号出现；`SELECT * FROM tickets ORDER BY created_at DESC LIMIT 1` 有新行；
- 都不点继续发消息 → 正常对话。
截图存 `dev-notes/ch05-acceptance3-*.png`。

- [ ] **Step 6: Commit**

```bash
git add app/static/chat.html app/static/chat.js app/static/chat.css dev-notes/ch05-acceptance3-*.png
git commit -m "feat(ch05): 前端 actions 按钮 + 转人工模拟 + 工单表单"
```

## Task 17: 五验收端到端评估脚本 + 真实验收

**Files:**
- Create: `scripts/eval_ch05.py`
- Modify: `Makefile`
- Test: 真实服务

**Interfaces:**
- Consumes: `POST /api/agent`、`GET /health`

- [ ] **Step 1: 写 eval_ch05.py**

用 `/api/agent` 跑：
1. 政策问题 → `tool_calls` 不含 create_ticket；日志看 `trace.forced_rag`（脚本打印提示）。
2. 「订单1001的物流到哪了」→ `tool_calls` 含 `query_order` 与 `query_logistics`。
3. 「我要投诉」→ `suggested_actions` 含 `transfer_human` 与 `create_ticket`；再调 `/api/actions/create-ticket` 断言写库。
4. 「你好呀」→ `tool_calls==[]`、`answer` 非空。
5. 「订单1001我买了什么，帮我再查物流」→ `tool_calls` 至少两个且顺序含 query_order → query_logistics。

- [ ] **Step 2: Makefile 目标**

```makefile
eval-ch05:
	uv run python -m scripts.eval_ch05
```

- [ ] **Step 3: 真实跑五验收**

起全服务后 `make eval-ch05`；验收 1 看日志 `ch05 turn ... trace=...forced_rag: True`；验收 3 前端部分按 Task 16。结果与抖动记 dev-notes。

- [ ] **Step 4: 全量回归**

Run: `.venv/bin/pytest -q -m "not integration"`
Expected: 全绿；旧图编排测试已改造。

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_ch05.py Makefile dev-notes/ch05.md
git commit -m "test(ch05): 五验收端到端评估 + 真实留痕"
```

## Task 18: 退役旧编排 + README/dev-notes 收尾

**Files:**
- Modify: `app/services/chat.py`（删除旧两段式私有函数，保留图适配器）
- Modify: `README.md`
- Modify: `PROGRESS.md`
- Modify: `dev-notes/ch05.md`
- Test: 全量回归

- [ ] **Step 1: grep 旧编排引用**

Run: `grep -rn "chunk_text\|_openai_tool_calls\|ToolRegistry\|stream_chat" app tests scripts | head -50`

- 保留 `chunk_text`（裸循环/其他模块用）；删除旧 `stream_chat` 内的两段式逻辑与不再使用的 helper。
- 旧测试 `test_chat_service.py` 中依赖两次调用的用例改为图节点/runtime 用例；`test_chat_self_check.py` 若只测旧链路则删除，证据闸行为由 `tests/graph/test_forced_rag.py` 覆盖。

- [ ] **Step 2: README/PROGRESS**

补 CH05 能力、配置、演示命令、五验收、已知限制（AsyncSqliteSaver 单文件、InMemorySaver 仅测试、前端表单、建工单不改会话状态）。

- [ ] **Step 3: 全量回归 + Commit**

Run: `.venv/bin/pytest -q -m "not integration"`
Expected: 全绿。

```bash
git add app/services/chat.py tests README.md PROGRESS.md dev-notes/ch05.md
git commit -m "chore(ch05): 退役旧两次编排 + 文档收尾"
```

## Self-Review

**Spec coverage:**
- §2 架构 → Task 10；§3 十节点 → Task 6/7/8/9/10；§3.1 7→4 → Task 5；§4 ReAct/停止 → Task 5/8；§5 State+AsyncSqliteSaver → Task 3/11；§6 转人工/建工单 → Task 8/14/16；§7 两入口 → Task 12/13；§8 依赖/验证 → Task 1/4/17；§9 数据模型 → Task 9/14；§11 D1–D7 → Task 1/2/8/11/14；§12 R1/R2/R3 → Task 1/8/11/17；§13 五验收 → Task 16/17。
- 补充：`budget_reply` 覆盖 spec §4“步数封顶走兜底话术”，原用户 plan 的 `should_continue` 只返回 stop 会漏掉这一步。

**Step scan:** 每个任务有失败测试 → 实现 → 通过 → commit；Prompt 任务用 `eval_intent.py` 替代 TDD。无 TBD/占位。

**Type consistency:** `ConversationState` 字段（Task 3）↔ 节点读写（Task 6–9）一致；`should_continue` 返回 `continue/stop/cap`（Task 5）↔ `build.py` 映射（Task 10）；`suggested_actions` 结构 `{type, draft?}` 在 Task 6/8/12/13/16 一致；`stream_turn` 事件元组（Task 11）↔ SSE 映射（Task 13）一致。

**Review Focus tests:** 中间轮文本泄漏 → Task 11 的事件映射测试 + Task 16 浏览器验收；create_ticket 自动执行 → Task 8 拦截测试；建工单污染状态 → Task 14 测试；闲聊/投诉模型调用 → Task 6 测试（fake model 不被调用）；置信度闸放晚 → Task 7 强/弱测试 + Task 17 验收。
