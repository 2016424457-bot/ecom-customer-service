"""Task 8: the user 👎 feedback endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.graph import runtime as graph_runtime
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeConversation, FakeFlywheel, FakeRepository


async def _matching_state(settings, conversation_id):
    return {
        "messages": [HumanMessage(content="能退吗")],
        "retrieval_chunks": [{"n": 1}],
    }


async def _text_block_state(settings, conversation_id):
    return {
        "messages": [
            HumanMessage(content=[{"type": "text", "text": "能退吗"}])
        ],
        "retrieval_chunks": [{"n": 2}],
    }


async def _stale_question_state(settings, conversation_id):
    return {
        "messages": [
            HumanMessage(content="能退吗"),
            AIMessage(content="七天无理由退货。"),
            HumanMessage(content="那运费谁出"),
        ],
        "retrieval_chunks": [{"n": 3}],
    }


async def _empty_snapshot_state(settings, conversation_id):
    return {
        "messages": [HumanMessage(content="能退吗")],
        "retrieval_chunks": [],
    }


async def _no_checkpoint_state(settings, conversation_id):
    return {}


async def _failing_state(settings, conversation_id):
    raise RuntimeError("graph checkpointer is not initialized")


@pytest.fixture
def fake_runtime(
    tmp_path: Path, clean_unit_env: pytest.MonkeyPatch
) -> Runtime:
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    repository = FakeRepository()
    repository.conversations[1] = FakeConversation(id=1, user_id="u1")
    return Runtime(
        settings=Settings(_env_file=None, openai_api_key="test"),
        prompt_builder=PromptBuilder(
            system_prompt_path=system,
            extract_prompt_path=extract,
            brand_name="示例商城",
            service_hours="9:00-21:00",
        ),
        chat_model=None,
        extraction_model=None,
        locks=ConversationLocks(),
        repository=repository,
    )


@pytest.fixture
def client(fake_runtime: Runtime) -> TestClient:
    return TestClient(create_app(runtime=fake_runtime))


def test_feedback_matching_checkpoint_saves_snapshot(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _matching_state
    )

    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )

    assert response.status_code == 200
    assert response.json() == {"pool_id": 1, "enqueued": False}
    row = fake_runtime.repository.low_confidence[-1]
    assert row["conversation_id"] == 1
    assert row["raw_question"] == "能退吗"
    assert row["source"] == "user_feedback"
    assert row["reason"] == "用户反馈未解决"
    assert row["retrieved_chunks"] == [{"n": 1}]


def test_feedback_matches_text_block_content(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _text_block_state
    )

    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )

    assert response.status_code == 200
    assert fake_runtime.repository.low_confidence[-1][
        "retrieved_chunks"
    ] == [{"n": 2}]


def test_feedback_mismatch_stores_null(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _stale_question_state
    )

    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )

    assert response.status_code == 200
    assert (
        fake_runtime.repository.low_confidence[-1]["retrieved_chunks"]
        is None
    )


def test_feedback_empty_snapshot_stores_null(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _empty_snapshot_state
    )

    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )

    assert response.status_code == 200
    assert (
        fake_runtime.repository.low_confidence[-1]["retrieved_chunks"]
        is None
    )


def test_feedback_without_checkpoint_stores_null(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _no_checkpoint_state
    )

    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )

    assert response.status_code == 200
    assert (
        fake_runtime.repository.low_confidence[-1]["retrieved_chunks"]
        is None
    )


def test_feedback_state_error_still_records(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _failing_state
    )

    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )

    assert response.status_code == 200
    assert (
        fake_runtime.repository.low_confidence[-1]["retrieved_chunks"]
        is None
    )


def test_feedback_unknown_conversation_returns_404(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_state(settings, conversation_id):
        raise AssertionError(
            "checkpoint state must not be read for unknown conversations"
        )

    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _unexpected_state
    )

    response = client.post(
        "/api/feedback", json={"conversation_id": 999, "question": "能退吗"}
    )

    assert response.status_code == 404
    assert fake_runtime.repository.low_confidence == []


def test_feedback_enqueues_after_commit(
    client: TestClient,
    fake_runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_runtime, "get_conversation_state", _matching_state
    )
    flywheel = FakeFlywheel(fake_runtime.repository)
    fake_runtime.flywheel = flywheel

    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )

    assert response.status_code == 200
    assert response.json() == {"pool_id": 1, "enqueued": True}
    assert flywheel.pool_ids == [1]
    assert flywheel.rows_at_enqueue == [1]
    assert flywheel.commits_at_enqueue == [1]


def test_feedback_rejects_invalid_payload(
    client: TestClient, fake_runtime: Runtime
) -> None:
    assert (
        client.post(
            "/api/feedback",
            json={"conversation_id": 0, "question": "能退吗"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/feedback", json={"conversation_id": 1, "question": ""}
        ).status_code
        == 422
    )