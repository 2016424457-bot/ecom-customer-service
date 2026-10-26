import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app import summarizer
from app.config import Settings


def _settings():
    return Settings(_env_file=None, openai_api_key="test")


def _rows(n, start=1):
    rows, uid = [], start
    for i in range(n):
        rows.append(HumanMessage(f"q{i}", id=f"db-{uid}"))
        rows.append(AIMessage(f"a{i}"))
        uid += 2
    return rows


class FakeConversation:
    def __init__(self, summary_upto, layer1_from):
        self.summary_upto_msg_id = summary_upto
        self.layer1_from_msg_id = layer1_from


class FakeRepo:
    def __init__(self, summary_upto=0, layer1_from=0):
        self.conversation = FakeConversation(
            summary_upto, layer1_from
        )
        self.inserted = []
        self.updated = None
        self.summaries = []
        self.seq = 0
        self.commits = 0

    async def get_conversation(self, conversation_id):
        return self.conversation

    async def next_summary_seq(self, conversation_id):
        self.seq += 1
        return self.seq

    async def insert_summary(
        self,
        conversation_id,
        seq,
        from_msg_id,
        upto_msg_id,
        content,
    ):
        self.inserted.append(
            (seq, from_msg_id, upto_msg_id, content)
        )
        row = SimpleNamespace(
            seq=seq,
            from_msg_id=from_msg_id,
            upto_msg_id=upto_msg_id,
            content=content,
        )
        self.summaries.append(row)
        return row

    async def list_recent_summaries(
        self, conversation_id, limit
    ):
        return self.summaries[-limit:]

    async def update_conversation_summary(
        self, conversation_id, projection, upto_msg_id
    ):
        self.updated = (projection, upto_msg_id)

    async def commit(self):
        self.commits += 1


class FakeModel:
    def __init__(self, summary):
        self.summary = summary
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(summary=self.summary)


def _scope(repo):
    @asynccontextmanager
    async def scope():
        yield repo

    return scope


def test_boundary_returns_layer2_user_range():
    msgs = _rows(5)  # user ids 1,3,5,7,9

    assert summarizer.compute_summary_boundary(msgs, 1, 9) == (3, 7)


def test_boundary_none_when_no_middle():
    assert summarizer.compute_summary_boundary(_rows(3), 0, 1) is None


def test_render_summary_dialog_compresses_assistant_and_skips_tool():
    msgs = [
        HumanMessage("q", id="db-1"),
        AIMessage("a" * 200),
        ToolMessage(
            "x" * 5000, tool_call_id="c1", name="query_order"
        ),
    ]

    text = summarizer.render_summary_dialog(msgs, _settings())

    assert "用户:q" in text
    assistant_line = [
        line for line in text.splitlines() if "客服:" in line
    ][0]
    assert len(assistant_line) < 120
    assert "x" * 100 not in text


def test_projection_keeps_recent_segments_and_caps_tokens():
    rows = [
        SimpleNamespace(seq=i, content=f"第{i}段" + "字" * 400)
        for i in range(1, 5)
    ]

    out = summarizer.build_projection(
        rows,
        max_segments=3,
        max_tokens=650,
        chars_per_token=1.5,
    )

    assert "第4段" in out
    assert "第1段" not in out


async def test_run_appends_segment_and_updates_anchors():
    repo = FakeRepo(summary_upto=1, layer1_from=9)
    service = summarizer.SummaryService(
        _settings(), FakeModel("订单1001 要退货"), _scope(repo)
    )

    await service.run(7, _rows(5), 1, 9)

    assert repo.inserted == [(1, 3, 7, "订单1001 要退货")]
    assert repo.updated == ("订单1001 要退货", 7)
    assert repo.commits == 1


async def test_run_skips_when_summary_upto_moved():
    repo = FakeRepo(summary_upto=3, layer1_from=9)
    service = summarizer.SummaryService(
        _settings(), FakeModel("x"), _scope(repo)
    )

    await service.run(7, _rows(5), 1, 9)

    assert repo.inserted == []
    assert repo.updated is None


async def test_maybe_schedule_below_budget_noop():
    repo = FakeRepo()
    service = summarizer.SummaryService(
        _settings(), FakeModel("x"), _scope(repo)
    )
    service._budget = SimpleNamespace(layer2=10**9)

    await service.maybe_schedule(7, _rows(5), 1, 9)

    assert service._running == {}


async def test_maybe_schedule_fires_and_debounces(monkeypatch):
    repo = FakeRepo()
    service = summarizer.SummaryService(
        _settings(), FakeModel("x"), _scope(repo)
    )
    service._budget = SimpleNamespace(layer2=1)
    gate = asyncio.Event()
    ran = []

    async def slow_run(cid, messages, upto, layer1_from):
        ran.append(cid)
        await gate.wait()

    monkeypatch.setattr(service, "run", slow_run)

    await service.maybe_schedule(7, _rows(5), 1, 9)
    await service.maybe_schedule(7, _rows(5), 1, 9)
    await asyncio.sleep(0)

    assert ran == [7]
    task = service._running[7]
    gate.set()
    await task
    await asyncio.sleep(0)
    assert 7 not in service._running