from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.api import routes
from app.schemas import AgentRequest


class FakeRepository:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists

    async def get_conversation(self, conversation_id):
        if not self.exists:
            return None
        return SimpleNamespace(id=conversation_id, user_id="u1")

    async def get_or_create_conversation(self, session_id, user_id):
        return SimpleNamespace(id=42, user_id=user_id)

    async def commit(self) -> None:
        return None


def _runtime(repository):
    @asynccontextmanager
    async def repository_scope():
        yield repository

    return SimpleNamespace(
        repository_scope=repository_scope,
        settings=SimpleNamespace(),
        chat_model=object(),
        intent_model=object(),
        prompt_builder=object(),
        evidence_checker=None,
        faq_search=None,
    )


@pytest.mark.asyncio
async def test_agent_endpoint_returns_tool_views(monkeypatch):
    async def fake_run(deps, *, conversation_id, user_id, message):
        return {
            "conversation_id": conversation_id,
            "state": {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[
                            {
                                "name": "query_logistics",
                                "args": {"tracking_no": "SF1"},
                                "id": "c1",
                            }
                        ],
                    ),
                    ToolMessage(
                        content='{"status":"派送中"}',
                        tool_call_id="c1",
                        name="query_logistics",
                    ),
                    AIMessage("正在派送中。"),
                ],
                "suggested_actions": [],
            },
        }

    monkeypatch.setattr(
        routes.graph_runtime, "build_deps", lambda **kwargs: object()
    )
    monkeypatch.setattr(routes.graph_runtime, "run_turn", fake_run)

    response = await routes.agent_endpoint(
        AgentRequest(
            user_id="u1", message="物流到哪了", conversation_id=5
        ),
        _runtime(FakeRepository()),
    )

    assert response.conversation_id == 5
    assert response.answer == "正在派送中。"
    assert response.tool_calls[0].name == "query_logistics"
    assert response.tool_results[0].tool_call_id == "c1"
    assert response.tool_results[0].ok is True


@pytest.mark.asyncio
async def test_agent_endpoint_missing_conversation_404():
    with pytest.raises(HTTPException) as exc:
        await routes.agent_endpoint(
            AgentRequest(
                user_id="u1", message="hi", conversation_id=999
            ),
            _runtime(FakeRepository(exists=False)),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_agent_endpoint_upstream_error_502(monkeypatch):
    async def boom(deps, *, conversation_id, user_id, message):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(
        routes.graph_runtime, "build_deps", lambda **kwargs: object()
    )
    monkeypatch.setattr(routes.graph_runtime, "run_turn", boom)

    with pytest.raises(HTTPException) as exc:
        await routes.agent_endpoint(
            AgentRequest(
                user_id="u1", message="hi", conversation_id=5
            ),
            _runtime(FakeRepository()),
        )

    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_agent_endpoint_database_error_503(monkeypatch):
    async def boom(deps, *, conversation_id, user_id, message):
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(
        routes.graph_runtime, "build_deps", lambda **kwargs: object()
    )
    monkeypatch.setattr(routes.graph_runtime, "run_turn", boom)

    with pytest.raises(HTTPException) as exc:
        await routes.agent_endpoint(
            AgentRequest(
                user_id="u1", message="hi", conversation_id=5
            ),
            _runtime(FakeRepository()),
        )

    assert exc.value.status_code == 503
