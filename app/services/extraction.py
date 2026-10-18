"""Structured after-sales extraction service."""

import logging
import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.errors import ERROR_MESSAGES, AppError, map_upstream_error
from app.prompt_builder import PromptBuilder
from app.schemas import AfterSalesExtraction

logger = logging.getLogger(__name__)


def _log_outcome(outcome: str, started: float, error_code: str | None) -> None:
    """Log outcome metadata only; never log user text or model content."""
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    log = logger.info if outcome == "done" else logger.warning
    log(
        "extraction event=extract outcome=%s latency_ms=%s error_code=%s",
        outcome,
        latency_ms,
        error_code,
    )


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
    started = time.perf_counter()
    messages: list[BaseMessage] = [
        SystemMessage(content=prompt_builder.extract_prompt),
        HumanMessage(content=text),
    ]
    try:
        result = await extraction_model.ainvoke(messages)
    except ValidationError as exc:
        # with_structured_output parses tool-call arguments inside ainvoke,
        # so schema-invalid model output surfaces here as ValidationError.
        error = AppError(
            code="parse_error",
            message=ERROR_MESSAGES["parse_error"],
            status_code=502,
        )
        _log_outcome("error", started, error.code)
        raise error from exc
    except Exception as exc:
        error = map_upstream_error(exc)
        _log_outcome("error", started, error.code)
        raise error from exc

    if isinstance(result, AfterSalesExtraction):
        _log_outcome("done", started, None)
        return result
    try:
        extraction = AfterSalesExtraction.model_validate(result)
    except ValidationError as exc:
        error = AppError(
            code="parse_error",
            message=ERROR_MESSAGES["parse_error"],
            status_code=502,
        )
        _log_outcome("error", started, error.code)
        raise error from exc

    _log_outcome("done", started, None)
    return extraction
