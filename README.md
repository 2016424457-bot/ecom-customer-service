# 电商智能客服（CH01：纯对话）

基于 FastAPI + LangChain 的电商售后客服后端，本章跑通纯对话链路：

- `POST /api/chat/stream`：SSE 流式对话，同一 `session_id` 支持多轮上下文。
- `POST /api/chat/extract`：从售后描述中抽取 `order_id`、`request_type`、`desired_solution` 三个固定字段。
- 应用侧统一使用 OpenAI 兼容协议，可切换 DeepSeek / OpenAI / Ollama / Claude 兼容代理。
- 会话保存在进程内存，按 token 预算裁剪历史，不做持久化。

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

```bash
# macOS / Linux / Git Bash：已有 .env 时不覆盖
test -f .env || cp .env.example .env

# Windows PowerShell 等价写法：
# if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，至少把 `OPENAI_API_KEY` 换成真实密钥。默认上游是 DeepSeek，模型 `deepseek-flash`。

> `.env` 已被 `.gitignore` 忽略。不要提交真实密钥，也不要把密钥写进 README、日志或代码。

### 3. 启动服务

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","model":"deepseek-flash"}
```

本章只支持单 worker：会话在进程内存中，多 worker 会导致同一会话的上下文随机丢失。

## 切换上游模型

应用统一按 OpenAI 协议访问上游，切换时只需要改 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`。`LLM_THINKING` 非空时会作为 `extra_body.thinking.type` 发送；非 DeepSeek 上游通常留空。

### DeepSeek（默认）

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-flash
LLM_THINKING=disabled
```

### OpenAI

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
LLM_THINKING=
```

### Ollama（本地）

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b
LLM_THINKING=
```

### Claude 兼容代理

将 `OPENAI_BASE_URL` 指向提供 OpenAI 协议的服务地址（不是 Anthropic 原生协议地址）：

```dotenv
OPENAI_BASE_URL=https://your-proxy.example.com/v1
OPENAI_API_KEY=<代理密钥>
OPENAI_MODEL=claude-sonnet-4-5
LLM_THINKING=
```

结构化抽取默认使用 `STRUCTURED_OUTPUT_METHOD=function_calling`。如果上游不支持强制 function calling，可按上游能力改为 `json_mode` 或 `json_schema`。

## API 示例

### 流式对话

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-1","message":"你好，我叫小王，订单号 A12345，想退货"}'
```

响应是 SSE：先出现 `event: session`，随后是多个 `event: token`（`data` 中的 `delta` 为文本增量），最后是 `event: done`。复用同一个 `session_id` 再发一条消息即可验证多轮上下文（把 `message` 换成“我的订单号是多少？”）。

### 结构化抽取

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"我订单 A12345 买的外套不合身，想退货，希望退款"}'
```

响应示例：

```json
{
  "order_id": "订单 A12345",
  "request_type": "return",
  "desired_solution": "退货退款"
}
```

`order_id` 保留用户原文（例如“订单号：A12345”），未提及为 `null`；`request_type` 取值限定为 `refund`、`return`、`exchange`、`repair`、`logistics`、`complaint`、`other`。

## 测试、评估与验收

### 测试

```bash
uv run pytest
```

`uv run pytest` 会运行全部测试：单元测试，以及在配置了真实密钥时运行集成测试（集成测试只由 marker 标记，不会被默认排除）。没有 `.env`、密钥缺失或仍是 `sk-your-key` 占位符时，集成测试会自动跳过。只跑不访问网络的单元测试：

```bash
uv run pytest -m "not integration"
```

### 真实上游集成测试

```bash
uv run pytest -m integration -v
```

需要 `.env` 中存在以 `sk-` 开头的真实密钥；没有 `.env`、密钥缺失或仍是 `sk-your-key` 占位符时会自动跳过。

### 评估与验收脚本

先在另一个终端启动服务，然后运行：

```bash
uv run python evals/run_extraction_eval.py
uv run python evals/run_chat_eval.py
uv run python scripts/acceptance.py
```

- `run_extraction_eval.py`：逐条调用 `/api/chat/extract`，统计 JSON 合法率（100%）、`order_id` 原文精确匹配（≥80%）、`request_type` 匹配（≥90%）、`desired_solution` 有无判断（≥80%），不达标退出码为 1。
- `run_chat_eval.py`：逐条调用 `/api/chat/stream`，检查不编造物流、不擅自赔付、隐私提醒、高风险转人工、拒绝提示词注入，并打印完整回复供人工复核。
- `scripts/acceptance.py`：检查流式 token/done、多轮订单号记忆、结构化抽取三字段，全部通过时打印 `ACCEPTANCE PASS`。

脚本默认访问 `http://127.0.0.1:8000`，可用 `ECOM_BASE_URL` 环境变量覆盖。

## 已知限制

- 会话保存在进程内存中，服务重启或多 worker 都会丢失上下文；本章仅支持 `--workers 1`。
- 没有用户认证、限流、计费或多租户隔离，不要直接暴露到公网。
- 默认只监听 `127.0.0.1`，仅适合本地开发与验收。
- Prompt 中的业务信息（品牌名、服务时间）来自 `.env`，本章不接入订单/物流 API，无法核实真实订单状态。
- `LOG_CONTENT=false` 默认不记录用户消息与模型回复全文；改为 `true` 会把全文写入日志，存在隐私风险。
- 模型输出具有不确定性；评估脚本中的关键词检查只是底线，不能替代人工复核。
