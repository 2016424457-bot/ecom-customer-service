"""Structured after-sales extraction service."""

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.errors import ERROR_MESSAGES, AppError, map_upstream_error
from app.prompt_builder import PromptBuilder
from app.schemas import AfterSalesExtraction


async def extract_after_sales(
    *,
    text: str,
    extraction_model: Any,
    prompt_builder: PromptBuilder,
) -> AfterSalesExtraction:
    """Extract the fixed after-sales fields from one user message.

    Upstream failures are mapped through ``map_upstream_error``. A result
    that is not already an ``AfterSalesExtraction`` is validated against the
    schema; validation failure maps to ``parse_error`` (HTTP 502).
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=prompt_builder.extract_prompt),
        HumanMessage(content=text),
    ]
    try:
        result = await extraction_model.ainvoke(messages)
    except Exception as exc:
        raise map_upstream_error(exc) from exc

    if isinstance(result, AfterSalesExtraction):
        return result
    try:
        return AfterSalesExtraction.model_validate(result)
    except ValidationError as exc:
        raise AppError(
            code="parse_error",
            message=ERROR_MESSAGES["parse_error"],
            status_code=502,
        ) from exc
