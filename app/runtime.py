from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from app.config import Settings
from app.db.repositories import (
    ChatRepository,
    FaithCaseRepository,
    KnowledgeRepository,
)
from app.db.session import create_engine_and_session
from app.knowledge.self_check import EvidenceChecker
from app.llm import (
    build_chat_model,
    build_evidence_check_model,
    build_extraction_model,
    build_intent_model,
)
from app.prompt_builder import PromptBuilder
from app.sessions import ConversationLocks


@dataclass
class Runtime:
    """Process-wide dependencies shared by the API routes."""

    settings: Settings
    prompt_builder: PromptBuilder
    chat_model: Any
    extraction_model: Any
    locks: ConversationLocks
    intent_model: Any = None
    session_factory: async_sessionmaker[AsyncSession] | None = None
    repository: Any = None
    engine: AsyncEngine | None = None
    faq_search: Any = None
    evidence_checker: Any = None
    knowledge_repository: Any = None
    faith_repository: Any = None

    @asynccontextmanager
    async def repository_scope(self) -> AsyncIterator[Any]:
        if self.session_factory is None:
            if self.repository is None:
                raise RuntimeError("runtime has no chat repository")
            yield self.repository
            return

        async with self.session_factory() as session:
            yield ChatRepository(session)

    @asynccontextmanager
    async def knowledge_repository_scope(self) -> AsyncIterator[Any]:
        if self.session_factory is None:
            if self.knowledge_repository is None:
                raise RuntimeError("runtime has no knowledge repository")
            yield self.knowledge_repository
            return

        async with self.session_factory() as session:
            yield KnowledgeRepository(session)

    @asynccontextmanager
    async def faith_repository_scope(self) -> AsyncIterator[Any]:
        if self.session_factory is None:
            if self.faith_repository is None:
                raise RuntimeError("runtime has no faith repository")
            yield self.faith_repository
            return

        async with self.session_factory() as session:
            yield FaithCaseRepository(session)

    async def aclose(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()


def build_runtime(settings: Settings) -> Runtime:
    """Assemble the runtime from settings; performs no network calls."""
    engine, session_factory = create_engine_and_session(settings)
    return Runtime(
        settings=settings,
        prompt_builder=PromptBuilder(
            system_prompt_path=settings.system_prompt_path,
            extract_prompt_path=settings.extract_prompt_path,
            brand_name=settings.brand_name,
            service_hours=settings.service_hours,
        ),
        chat_model=build_chat_model(settings),
        intent_model=build_intent_model(settings),
        extraction_model=build_extraction_model(settings),
        evidence_checker=EvidenceChecker(
            build_evidence_check_model(settings),
            settings.evidence_check_prompt_path,
        ),
        locks=ConversationLocks(settings.max_sessions),
        session_factory=session_factory,
        engine=engine,
    )


def get_runtime(request: Request) -> Runtime:
    """FastAPI dependency: return the runtime built during app startup."""
    return request.app.state.runtime
