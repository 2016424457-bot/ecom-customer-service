"""HTTP routes for the chat API."""

import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.errors import AppError, map_upstream_error
from app.graph import runtime as graph_runtime
from app.graph.nodes import resolve_answer
from app.runtime import Runtime, get_runtime
from app.schemas import (
    AfterSalesExtraction,
    AgentRequest,
    AgentResponse,
    ChatExtractRequest,
    ChatRequest,
    ChatResumeRequest,
    FaithCaseOut,
    FaithCaseUpdate,
    FeedbackRequest,
    KnowledgeChunkDetail,
    ToolCallView,
    ToolResultView,
)
from app.services.chat import resume_chat, stream_chat
from app.services.extraction import extract_after_sales

logger = logging.getLogger(__name__)

router = APIRouter()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _latest_user_text(values: dict[str, Any]) -> str:
    """Return the newest human turn's text from a checkpoint state."""
    for message in reversed(values.get("messages", []) or []):
        if isinstance(message, HumanMessage):
            return _message_text(getattr(message, "content", None))
    return ""


def _current_turn_messages(state: dict) -> list:
    messages = state.get("messages", [])
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _views_from_state(
    state: dict,
) -> tuple[list[ToolCallView], list[ToolResultView]]:
    calls: list[ToolCallView] = []
    results: list[ToolResultView] = []
    for message in _current_turn_messages(state):
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                calls.append(
                    ToolCallView(
                        id=call.get("id", ""),
                        name=call.get("name", ""),
                        args=call.get("args") or {},
                    )
                )
        elif isinstance(message, ToolMessage):
            results.append(
                ToolResultView(
                    tool_call_id=message.tool_call_id,
                    name=getattr(message, "name", "") or "",
                    ok=getattr(message, "status", "success") != "error",
                    content=_message_text(message.content),
                )
            )
    return calls, results


@router.post("/api/agent", response_model=AgentResponse)
async def agent_endpoint(
    payload: AgentRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AgentResponse:
    """Run one graph turn without streaming; used by evaluations."""
    async with runtime.repository_scope() as repository:
        try:
            if payload.conversation_id is not None:
                conversation = await repository.get_conversation(
                    payload.conversation_id
                )
                if conversation is None:
                    raise HTTPException(
                        status_code=404, detail="会话不存在"
                    )
            else:
                conversation = (
                    await repository.get_or_create_conversation(
                        None, payload.user_id
                    )
                )
                await repository.commit()

            deps = await graph_runtime.build_deps(
                settings=runtime.settings,
                chat_model=runtime.chat_model,
                intent_model=runtime.intent_model,
                prompt_builder=runtime.prompt_builder,
                repository=repository,
                conversation_id=conversation.id,
                evidence_checker=runtime.evidence_checker,
                faq_search=runtime.faq_search,
                coref_model=getattr(runtime, "coref_model", None),
                expansion_model=getattr(runtime, "expansion_model", None),
                refund_judge_model=getattr(
                    runtime, "refund_judge_model", None
                ),
                intent_fast_model=getattr(
                    runtime, "intent_fast_model", None
                ),
                context_budget=getattr(
                    runtime, "context_budget", None
                ),
                summary_service=getattr(
                    runtime, "summary_service", None
                ),
                mcp_provider=getattr(runtime, "mcp_provider", None),
                audit_writer=getattr(runtime, "audit_writer", None),
                observability=getattr(runtime, "observability", None),
                flywheel=getattr(runtime, "flywheel", None),
            )
            lock = runtime.locks.lock_for(str(conversation.id))
            async with lock:
                out = await graph_runtime.run_turn(
                    deps,
                    conversation_id=conversation.id,
                    user_id=payload.user_id,
                    message=payload.message,
                )
        except graph_runtime.ConversationNotFound:
            raise HTTPException(status_code=404, detail="会话不存在")
        except HTTPException:
            raise
        except SQLAlchemyError:
            raise HTTPException(
                status_code=503, detail="数据库暂时不可用,请稍后重试"
            )
        except AppError:
            raise
        except Exception as exc:
            error = map_upstream_error(exc)
            raise HTTPException(
                status_code=error.status_code, detail=error.message
            )

    state = out["state"]
    calls, results = _views_from_state(state)
    return AgentResponse(
        conversation_id=out["conversation_id"],
        answer=resolve_answer(state),
        tool_calls=calls,
        tool_results=results,
        suggested_actions=state.get("suggested_actions", []) or [],
        pending_confirm=out.get("pending_confirm"),
    )


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream_endpoint(
    payload: ChatRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AsyncIterator[ServerSentEvent]:
    """Stream one chat turn as Server-Sent Events."""
    async with runtime.repository_scope() as repository:
        async for event, data in stream_chat(
            session_id=payload.session_id,
            user_id=payload.user_id,
            user_message=payload.message,
            chat_model=runtime.chat_model,
            repository=repository,
            prompt_builder=runtime.prompt_builder,
            settings=runtime.settings,
            locks=runtime.locks,
            faq_search=runtime.faq_search,
            evidence_checker=runtime.evidence_checker,
            intent_model=runtime.intent_model,
            coref_model=getattr(runtime, "coref_model", None),
            expansion_model=getattr(runtime, "expansion_model", None),
            refund_judge_model=getattr(
                runtime, "refund_judge_model", None
            ),
            intent_fast_model=getattr(runtime, "intent_fast_model", None),
            context_budget=getattr(runtime, "context_budget", None),
            summary_service=getattr(
                runtime, "summary_service", None
            ),
            mcp_provider=getattr(runtime, "mcp_provider", None),
            audit_writer=getattr(runtime, "audit_writer", None),
            observability=getattr(runtime, "observability", None),
            flywheel=getattr(runtime, "flywheel", None),
        ):
            yield ServerSentEvent(event=event, data=data)


@router.post("/api/chat", response_class=EventSourceResponse)
async def chat_endpoint(
    payload: ChatRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AsyncIterator[ServerSentEvent]:
    """Alias for /api/chat/stream kept for the CH05 spec."""
    async for event in chat_stream_endpoint(payload, runtime):
        yield event


@router.post("/api/chat/resume", response_class=EventSourceResponse)
async def chat_resume_endpoint(
    payload: ChatResumeRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AsyncIterator[ServerSentEvent]:
    """Resume a paused ticket confirmation as Server-Sent Events."""
    async with runtime.repository_scope() as repository:
        async for event, data in resume_chat(
            conversation_id=payload.conversation_id,
            decision=payload.decision,
            interrupt_id=payload.interrupt_id,
            chat_model=runtime.chat_model,
            repository=repository,
            prompt_builder=runtime.prompt_builder,
            settings=runtime.settings,
            locks=runtime.locks,
            faq_search=runtime.faq_search,
            evidence_checker=runtime.evidence_checker,
            intent_model=runtime.intent_model,
            coref_model=getattr(runtime, "coref_model", None),
            expansion_model=getattr(runtime, "expansion_model", None),
            refund_judge_model=getattr(
                runtime, "refund_judge_model", None
            ),
            intent_fast_model=getattr(runtime, "intent_fast_model", None),
            context_budget=getattr(runtime, "context_budget", None),
            summary_service=getattr(
                runtime, "summary_service", None
            ),
            mcp_provider=getattr(runtime, "mcp_provider", None),
            audit_writer=getattr(runtime, "audit_writer", None),
            observability=getattr(runtime, "observability", None),
            flywheel=getattr(runtime, "flywheel", None),
        ):
            yield ServerSentEvent(event=event, data=data)


@router.get("/api/chat/pending")
async def chat_pending_endpoint(
    conversation_id: int = Query(gt=0),
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    """Return the pending ticket preview for a conversation, if any."""
    preview = await graph_runtime.get_pending(
        runtime.settings, conversation_id
    )
    return {"pending_confirm": preview}


@router.post("/api/feedback")
async def feedback_endpoint(
    payload: FeedbackRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    """Record a user 👎 in the question pool and enqueue the flywheel."""
    async with runtime.repository_scope() as repository:
        conversation = await repository.get_conversation(
            payload.conversation_id
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")

        try:
            values = await graph_runtime.get_conversation_state(
                runtime.settings, payload.conversation_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feedback_state_unavailable conversation_id=%s error=%r",
                payload.conversation_id,
                exc,
            )
            values = {}
        snapshot = (
            (values.get("retrieval_chunks") or None)
            if _latest_user_text(values) == payload.question
            else None
        )

        try:
            row = await repository.insert_low_confidence_question(
                payload.conversation_id,
                payload.question,
                "user_feedback",
                "用户反馈未解决",
                retrieved_chunks=snapshot,
            )
            await repository.commit()
        except SQLAlchemyError:
            logger.exception(
                "feedback_insert_failed conversation_id=%s",
                payload.conversation_id,
            )
            raise HTTPException(
                status_code=503,
                detail="反馈暂时无法保存,请稍后重试",
            )

    pool_id = int(row.id)
    enqueued = False
    flywheel = getattr(runtime, "flywheel", None)
    if flywheel is not None:
        enqueued = bool(await flywheel.enqueue(pool_id))
    return {"pool_id": pool_id, "enqueued": enqueued}


@router.post("/api/chat/extract", response_model=AfterSalesExtraction)
async def chat_extract_endpoint(
    payload: ChatExtractRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AfterSalesExtraction:
    """Extract the fixed after-sales fields from one message."""
    return await extract_after_sales(
        text=payload.text,
        extraction_model=runtime.extraction_model,
        prompt_builder=runtime.prompt_builder,
    )


@router.get(
    "/api/knowledge/chunks/{chunk_id}",
    response_model=KnowledgeChunkDetail,
)
async def knowledge_chunk_detail(
    chunk_id: int,
    runtime: Runtime = Depends(get_runtime),
) -> KnowledgeChunkDetail:
    """Return the source chunk behind a citation."""
    async with runtime.knowledge_repository_scope() as repository:
        row = await repository.get_chunk_by_id(chunk_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return KnowledgeChunkDetail.model_validate(row)


@router.get("/api/faith-cases")
async def list_faith_cases(
    status: str | None = Query(default=None),
    bucket: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    """List the faithfulness forensics ledger."""
    async with runtime.faith_repository_scope() as repository:
        items, total = await repository.list_cases(
            status=status,
            bucket=bucket,
            limit=limit,
            offset=offset,
        )
    return {
        "items": [FaithCaseOut.model_validate(item) for item in items],
        "total": total,
    }


@router.patch(
    "/api/faith-cases/{case_id}",
    response_model=FaithCaseOut,
)
async def update_faith_case(
    case_id: int,
    payload: FaithCaseUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> FaithCaseOut:
    """Change a case's status and resolution."""
    if payload.status != "未解决" and not (
        payload.resolution or ""
    ).strip():
        raise HTTPException(
            status_code=422, detail="resolution is required"
        )

    async with runtime.faith_repository_scope() as repository:
        case = await repository.update_status(
            case_id, payload.status, payload.resolution
        )
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        commit = getattr(repository, "commit", None)
        if commit is not None:
            await commit()
    return FaithCaseOut.model_validate(case)
