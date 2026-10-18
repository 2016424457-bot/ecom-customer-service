"""Real-upstream smoke tests (DeepSeek via the OpenAI-compatible API).

These tests are opt-in via ``-m integration`` and skip when no usable real
key is configured. They never print or log the API key.
"""

import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from app.config import Settings
from app.runtime import Runtime, build_runtime
from app.schemas import RequestType
from app.services.chat import stream_chat
from app.services.extraction import extract_after_sales

# LangChain caches its default async httpx client globally, and that client
# binds connections to the first event loop that uses it. Keep every test in
# this module on one loop so the cached client is never reused across loops.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def runtime() -> Runtime:
    """Build the real runtime, or skip when no real key is configured."""
    if not (REPO_ROOT / ".env").is_file():
        pytest.skip("no .env file; integration tests need a real API key")

    try:
        settings = Settings()
    except Exception as exc:  # missing/invalid config -> skip, not fail
        pytest.skip(f"settings could not be loaded from .env: {exc}")

    key = settings.openai_api_key
    if not key.startswith("sk-") or key == "sk-your-key":
        pytest.skip("OPENAI_API_KEY is missing or still a placeholder")

    return build_runtime(settings)


async def collect_events(
    runtime: Runtime, session_id: str, message: str
) -> list[tuple[str, dict]]:
    """Run one real streaming turn and return its SSE events."""
    events: list[tuple[str, dict]] = []
    async for event, data in stream_chat(
        session_id=session_id,
        user_message=message,
        chat_model=runtime.chat_model,
        session_store=runtime.session_store,
        prompt_builder=runtime.prompt_builder,
        settings=runtime.settings,
    ):
        events.append((event, data))
    return events


def reply_text(events: list[tuple[str, dict]]) -> str:
    return "".join(
        str(data.get("delta", ""))
        for event, data in events
        if event == "token"
    )


async def test_real_stream_and_context(runtime: Runtime) -> None:
    """Real stream yields token + done, and the next turn sees history."""
    session_id = f"itest-{uuid.uuid4().hex[:12]}"

    first_events = await collect_events(
        runtime, session_id, "我叫小王，订单号 A12345，想退货"
    )
    assert any(event == "token" for event, _ in first_events), first_events
    assert any(event == "done" for event, _ in first_events), first_events
    assert all(event != "error" for event, _ in first_events), first_events

    second_events = await collect_events(
        runtime, session_id, "我的订单号是多少？"
    )
    assert any(event == "done" for event, _ in second_events), second_events
    assert all(event != "error" for event, _ in second_events), second_events
    assert "A12345" in reply_text(second_events), second_events


async def test_real_extraction(runtime: Runtime) -> None:
    """Real structured output stays inside the schema and finds the order id."""
    result = await extract_after_sales(
        text="我订单 A12345 买的外套不合身，想退货，希望退款",
        extraction_model=runtime.extraction_model,
        prompt_builder=runtime.prompt_builder,
    )

    assert isinstance(result.request_type, RequestType), result
    assert result.order_id, f"expected a non-empty order_id, got {result!r}"
