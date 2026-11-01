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
from app.context import compute_context_budget
from app.db.repositories import (
    ChatRepository,
    EvalRunRepository,
    FaithCaseRepository,
    KnowledgeRepository,
    ReviewQueueRepository,
)
from app.db.session import create_engine_and_session
from app.flywheel.dedup import QuestionDeduplicator
from app.flywheel.normalize import QuestionNormalizer
from app.flywheel.service import FlywheelService
from app.knowledge.self_check import EvidenceChecker
from app.llm import (
    build_chat_model,
    build_coref_model,
    build_evidence_check_model,
    build_expansion_model,
    build_extraction_model,
    build_flywheel_model,
    build_intent_fast_model,
    build_intent_model,
    build_refund_judge_model,
    build_summary_model,
)
from app.observability import build_observability
from app.prompt_builder import PromptBuilder
from app.services.audit import ToolAuditWriter
from app.services.eval_scheduler import EvalScheduler
from app.services.mcp_tools import McpToolProvider
from app.sessions import ConversationLocks
from app.summarizer import SummaryService
from app.tools.registry import ensure_plugins_loaded


@dataclass
class Runtime:
    """Process-wide dependencies shared by the API routes."""

    settings: Settings
    prompt_builder: PromptBuilder
    chat_model: Any
    extraction_model: Any
    locks: ConversationLocks
    intent_model: Any = None
    coref_model: Any = None
    expansion_model: Any = None
    refund_judge_model: Any = None
    intent_fast_model: Any = None
    session_factory: async_sessionmaker[AsyncSession] | None = None
    repository: Any = None
    engine: AsyncEngine | None = None
    faq_search: Any = None
    evidence_checker: Any = None
    knowledge_repository: Any = None
    faith_repository: Any = None
    review_repository: Any = None
    eval_repository: Any = None
    context_budget: Any = None
    summary_service: Any = None
    mcp_provider: Any = None
    audit_writer: Any = None
    observability: Any | None = None
    flywheel: Any | None = None
    eval_scheduler: Any | None = None

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

    @asynccontextmanager
    async def eval_repository_scope(self) -> AsyncIterator[Any]:
        if self.session_factory is None:
            if self.eval_repository is None:
                raise RuntimeError("runtime has no eval repository")
            yield self.eval_repository
            return

        async with self.session_factory() as session:
            yield EvalRunRepository(session)

    @asynccontextmanager
    async def review_knowledge_scope(
        self,
    ) -> AsyncIterator[tuple[Any, Any]]:
        """Review queue and knowledge repositories sharing one session."""
        if self.session_factory is None:
            if (
                self.review_repository is None
                or self.knowledge_repository is None
            ):
                raise RuntimeError("runtime has no review repository")
            yield self.review_repository, self.knowledge_repository
            return

        async with self.session_factory() as session:
            yield (
                ReviewQueueRepository(session),
                KnowledgeRepository(session),
            )

    async def start(self) -> None:
        """Start background services owned by the runtime."""
        if self.flywheel is not None:
            await self.flywheel.start()
        if self.eval_scheduler is not None:
            await self.eval_scheduler.start()

    async def aclose(self) -> None:
        if self.eval_scheduler is not None:
            await self.eval_scheduler.aclose()
        if self.flywheel is not None:
            close = getattr(self.flywheel, "aclose", None)
            if close is not None:
                await close()
        if self.mcp_provider is not None:
            close = getattr(self.mcp_provider, "aclose", None)
            if close is not None:
                await close()
        if self.observability is not None:
            shutdown = getattr(self.observability, "shutdown", None)
            if shutdown is not None:
                shutdown()
        if self.engine is not None:
            await self.engine.dispose()


def _parse_csv(value: str) -> set[str]:
    return {
        item.strip()
        for item in (value or "").split(",")
        if item.strip()
    }


def build_mcp_connections(settings: Settings) -> dict[str, dict]:
    """Connections for the two business MCP servers.

    The HTTP timeout is intentionally higher than TOOL_TIMEOUT_SECONDS so the
    execution engine's timeout wins and the audit status stays deterministic.
    """
    return {
        "logistics": {
            "transport": "streamable_http",
            "url": settings.mcp_logistics_url,
            "timeout": settings.mcp_request_timeout_seconds,
        },
        "aftersales": {
            "transport": "streamable_http",
            "url": settings.mcp_aftersales_url,
            "timeout": settings.mcp_request_timeout_seconds,
        },
    }


def build_runtime(settings: Settings) -> Runtime:
    """Assemble the runtime from settings; performs no network calls."""
    ensure_plugins_loaded(settings.tool_plugin_dir)
    engine, session_factory = create_engine_and_session(settings)
    mcp_provider = McpToolProvider(
        build_mcp_connections(settings),
        read_only_servers=_parse_csv(settings.mcp_read_only_servers),
        write_tools=_parse_csv(settings.mcp_write_tools),
        enabled=settings.mcp_enabled,
    )
    observability = build_observability(settings)
    flywheel = None
    if settings.flywheel_enabled:
        flywheel = FlywheelService(
            session_factory,
            QuestionNormalizer(build_flywheel_model(settings)),
            QuestionDeduplicator(build_flywheel_model(settings)),
            candidate_limit=settings.flywheel_dedup_candidate_limit,
            enabled=settings.flywheel_enabled,
        )

    async def run_scheduled_eval() -> None:
        # Imported lazily so runtime assembly stays independent of the
        # evaluation pipeline's model/retrieval stack at import time.
        from app.services.eval_pipeline import run_eval_pipeline

        await run_eval_pipeline(
            settings=settings,
            session_factory=session_factory,
            trigger="定时",
        )

    eval_scheduler = EvalScheduler(settings, run_scheduled_eval)

    runtime = Runtime(
        settings=settings,
        prompt_builder=PromptBuilder(
            system_prompt_path=settings.system_prompt_path,
            extract_prompt_path=settings.extract_prompt_path,
            brand_name=settings.brand_name,
            service_hours=settings.service_hours,
        ),
        chat_model=build_chat_model(settings),
        intent_model=build_intent_model(settings),
        coref_model=build_coref_model(settings),
        expansion_model=build_expansion_model(settings),
        refund_judge_model=build_refund_judge_model(settings),
        intent_fast_model=build_intent_fast_model(settings),
        extraction_model=build_extraction_model(settings),
        evidence_checker=EvidenceChecker(
            build_evidence_check_model(settings),
            settings.evidence_check_prompt_path,
        ),
        locks=ConversationLocks(settings.max_sessions),
        session_factory=session_factory,
        engine=engine,
        mcp_provider=mcp_provider,
        audit_writer=ToolAuditWriter(
            session_factory,
            max_chars=settings.tool_audit_result_max_chars,
        ),
        observability=observability,
        flywheel=flywheel,
        eval_scheduler=eval_scheduler,
    )
    runtime.context_budget = compute_context_budget(settings)
    runtime.summary_service = SummaryService(
        settings,
        build_summary_model(settings),
        runtime.repository_scope,
    )
    return runtime


def get_runtime(request: Request) -> Runtime:
    """FastAPI dependency: return the runtime built during app startup."""
    return request.app.state.runtime
