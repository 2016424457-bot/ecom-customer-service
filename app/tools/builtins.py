"""Core builtin tool catalog, built once at import and materialized per turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool

from app.tools.faq import build_faq_tool
from app.tools.formatters import CLASSIFIERS, FORMATTERS
from app.tools.order import query_order
from app.tools.product import query_product
from app.tools.registry import (
    ToolSpec,
    builtin_plugin_specs,
    spec_from_tool,
)
from app.tools.ticket import build_ticket_tool


async def _placeholder_search(*args, **kwargs):
    raise RuntimeError("placeholder search must not be called")


class _PlaceholderRepository:
    pass


@dataclass(frozen=True)
class CoreDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    build: Callable[..., BaseTool]
    format_result: Callable[[Any], Any] | None = None
    classify_result: Callable[[Any], str | None] | None = None


def _definition(
    tool: BaseTool,
    *,
    read_only: bool,
    build: Callable[..., BaseTool],
    format_result: Callable[[Any], Any] | None = None,
    classify_result: Callable[[Any], str | None] | None = None,
) -> CoreDefinition:
    spec = spec_from_tool(tool, source="builtin", read_only=read_only)
    return CoreDefinition(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
        read_only=read_only,
        build=build,
        format_result=format_result,
        classify_result=classify_result,
    )


_FAQ_PLACEHOLDER = build_faq_tool(_placeholder_search)
_TICKET_PLACEHOLDER = build_ticket_tool(_PlaceholderRepository(), 0)


CORE_DEFINITIONS: tuple[CoreDefinition, ...] = (
    _definition(
        query_order,
        read_only=True,
        build=lambda **ctx: query_order,
        format_result=FORMATTERS["query_order"],
        classify_result=CLASSIFIERS["query_order"],
    ),
    _definition(
        query_product,
        read_only=True,
        build=lambda **ctx: query_product,
        format_result=FORMATTERS["query_product"],
    ),
    _definition(
        _FAQ_PLACEHOLDER,
        read_only=True,
        build=lambda **ctx: build_faq_tool(ctx["faq_search"]),
        format_result=FORMATTERS["query_faq"],
        classify_result=CLASSIFIERS["query_faq"],
    ),
    _definition(
        _TICKET_PLACEHOLDER,
        read_only=False,
        build=lambda **ctx: build_ticket_tool(
            ctx["repository"],
            ctx["conversation_id"],
            delay_seconds=ctx.get(
                "ticket_create_delay_seconds", 0.0
            ),
        ),
        format_result=FORMATTERS["create_ticket"],
    ),
)

CORE_TOOL_NAMES = tuple(
    definition.name for definition in CORE_DEFINITIONS
)


def build_core_specs(
    *,
    repository: Any,
    conversation_id: int,
    faq_search: Any,
    ticket_create_delay_seconds: float = 0.0,
) -> list[ToolSpec]:
    context = {
        "repository": repository,
        "conversation_id": conversation_id,
        "faq_search": faq_search,
        "ticket_create_delay_seconds": ticket_create_delay_seconds,
    }
    return [
        spec_from_tool(
            definition.build(**context),
            source="builtin",
            read_only=definition.read_only,
            format_result=definition.format_result,
            classify_result=definition.classify_result,
        )
        for definition in CORE_DEFINITIONS
    ]


def build_builtin_specs(
    *,
    repository: Any,
    conversation_id: int,
    faq_search: Any,
    ticket_create_delay_seconds: float = 0.0,
) -> list[ToolSpec]:
    return build_core_specs(
        repository=repository,
        conversation_id=conversation_id,
        faq_search=faq_search,
        ticket_create_delay_seconds=ticket_create_delay_seconds,
    ) + builtin_plugin_specs()