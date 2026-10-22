from dataclasses import dataclass

import pytest

from app.config import Settings
from app.knowledge.mining import (
    MiningError,
    extract_qas,
    finalize_staging,
    mine_once,
)
from app.knowledge.schemas import QaExtraction
from app.knowledge.types import ConversationWindow, StagingRow


@dataclass
class FakeMessage:
    role: str
    content: str


@dataclass
class FakeStaging:
    id: int
    batch_no: str
    source_ref: str
    category: str | None
    question: str
    answer: str
    status: str = "extracted"


@dataclass
class FakeChunk:
    id: int
    category: str
    questions: str
    answer: str
    content_type: str
    source_doc: str


class FakeMiningRepository:
    def __init__(self) -> None:
        self.windows: list[ConversationWindow] = []
        self.messages: dict[int, list[FakeMessage]] = {}
        self.staging: list[FakeStaging] = []
        self.chunks: list[FakeChunk] = []
        self.cursor = 0
        self.commits = 0
        self._next_staging_id = 1
        self._next_chunk_id = 1

    async def get_mining_cursor(self) -> int:
        return self.cursor

    async def set_mining_cursor(self, last_message_id: int) -> None:
        self.cursor = last_message_id

    async def list_conversations_to_mine(self, after_id, limit):
        return [
            window
            for window in self.windows
            if window.max_message_id > after_id
        ][:limit]

    async def list_conversation_messages(
        self, conversation_id, up_to_id, limit
    ):
        return self.messages.get(conversation_id, [])[-limit:]

    async def insert_staging_rows(self, rows: list[StagingRow]):
        ids = []
        for row in rows:
            staging = FakeStaging(
                self._next_staging_id,
                row.batch_no,
                row.source_ref or "",
                row.category,
                row.question,
                row.answer,
            )
            self._next_staging_id += 1
            self.staging.append(staging)
            ids.append(staging.id)
        return ids

    async def list_staging_by_status(self, status, batch_no=None):
        return [
            row
            for row in self.staging
            if row.status == status
            and (batch_no is None or row.batch_no == batch_no)
        ]

    async def set_staging_status(self, ids, status):
        for row in self.staging:
            if row.id in ids:
                row.status = status

    async def list_all_questions(self):
        return [chunk.questions for chunk in self.chunks]

    async def insert_knowledge_chunk(
        self,
        *,
        category,
        questions,
        answer,
        section_path,
        content_type,
        is_key_clause,
        source_doc,
    ):
        chunk = FakeChunk(
            self._next_chunk_id,
            category,
            questions,
            answer,
            content_type,
            source_doc,
        )
        self._next_chunk_id += 1
        self.chunks.append(chunk)
        return chunk.id

    async def commit(self):
        self.commits += 1


class FakeModel:
    def __init__(self, result, fail_on_call: int | None = None) -> None:
        self.result = result
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.messages = None

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        if self.fail_on_call == self.calls:
            raise RuntimeError("model failed")
        return self.result


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test",
        mining_batch_conversations=10,
    )


@pytest.fixture
def repository() -> FakeMiningRepository:
    return FakeMiningRepository()


def window(cid: int, max_id: int) -> ConversationWindow:
    return ConversationWindow(cid, max_id)


async def test_extract_qas_parses_parallel_arrays():
    model = FakeModel(
        QaExtraction(
            questions=["怎么退货？"],
            answers=["7 天内可退。"],
            categories=["退货"],
        )
    )

    pairs = await extract_qas("用户: 怎么退货？", model, "prompt")

    assert pairs[0].question == "怎么退货？"
    assert pairs[0].category == "退货"


async def test_mine_once_updates_cursor_and_ingests(repository, settings):
    repository.windows = [window(1, 20)]
    repository.messages[1] = [
        FakeMessage("user", "怎么退货？"),
        FakeMessage("assistant", "7 天内可退。"),
    ]
    model = FakeModel(
        QaExtraction(
            questions=["怎么退货？"],
            answers=["7 天内可退。"],
            categories=["退货"],
        )
    )

    stats = await mine_once(
        repository, settings, model, "prompt", sync=False
    )

    assert repository.cursor == 20
    assert stats.conversations == 1
    assert stats.extracted == 1
    assert stats.kept == 1
    assert repository.staging[0].status == "kept"
    assert repository.chunks[0].questions == "怎么退货？"
    assert repository.chunks[0].content_type == "qa"
    assert repository.chunks[0].source_doc == "conversation:1"


async def test_mine_stops_and_keeps_cursor_when_model_fails(
    repository, settings
):
    repository.windows = [window(1, 10), window(2, 20)]
    repository.messages[1] = [
        FakeMessage("user", "q1"),
        FakeMessage("assistant", "a1"),
    ]
    repository.messages[2] = [
        FakeMessage("user", "q2"),
        FakeMessage("assistant", "a2"),
    ]
    model = FakeModel(
        QaExtraction(
            questions=["q1"], answers=["a1"], categories=["其他"]
        ),
        fail_on_call=2,
    )

    with pytest.raises(MiningError):
        await mine_once(repository, settings, model, "prompt", sync=False)

    assert repository.cursor == 10


async def test_finalize_staging_recovers_extracted_rows(
    repository, settings
):
    repository.staging.append(
        FakeStaging(
            1,
            "b1",
            "conversation:1",
            "退货",
            "怎么退货？",
            "7 天内可退。",
        )
    )

    stats = await finalize_staging(repository, settings, sync=False)

    assert stats.kept == 1
    assert repository.staging[0].status == "kept"
    assert repository.chunks[0].source_doc == "conversation:1"


async def test_finalize_staging_discards_duplicate_question(
    repository, settings
):
    repository.chunks.append(
        FakeChunk(1, "退货", "怎么退货？", "旧答案。", "qa", "faq")
    )
    repository.staging.append(
        FakeStaging(
            1,
            "b1",
            "conversation:1",
            "退货",
            "怎么退货",
            "新答案。",
        )
    )

    stats = await finalize_staging(repository, settings, sync=False)

    assert stats.kept == 0
    assert stats.discarded == 1
    assert repository.staging[0].status == "discarded"
    assert len(repository.chunks) == 1


async def test_mine_once_with_no_conversations_is_noop(
    repository, settings
):
    stats = await mine_once(
        repository,
        settings,
        FakeModel(QaExtraction()),
        "prompt",
        sync=False,
    )

    assert stats.conversations == 0
    assert stats.kept == 0
    assert repository.cursor == 0


async def test_finalize_staging_syncs_pending_without_extracted_rows(
    repository, settings, monkeypatch
):
    calls: list[bool] = []

    async def fake_vectorize(repo, settings, **kwargs):
        calls.append(True)
        return 4

    monkeypatch.setattr(
        "app.knowledge.mining.vectorize_pending", fake_vectorize
    )

    stats = await finalize_staging(repository, settings, sync=True)

    assert stats.vectorized == 4
    assert calls == [True]


async def test_mine_once_retries_pending_without_new_conversations(
    repository, settings, monkeypatch
):
    async def fake_vectorize(repo, settings, **kwargs):
        return 4

    monkeypatch.setattr(
        "app.knowledge.mining.vectorize_pending", fake_vectorize
    )

    stats = await mine_once(
        repository,
        settings,
        FakeModel(QaExtraction()),
        "prompt",
        sync=True,
    )

    assert stats.vectorized == 4
