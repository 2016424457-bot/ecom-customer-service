from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.schemas import AfterSalesExtraction


def build_model_kwargs(
    settings: Settings, *, temperature: float, max_tokens: int
) -> dict[str, Any]:
    """Build the keyword arguments shared by both model factories."""
    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "base_url": settings.openai_base_url,
        "api_key": settings.openai_api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "stream_usage": True,
    }
    if settings.llm_thinking:
        kwargs["extra_body"] = {"thinking": {"type": settings.llm_thinking}}
    return kwargs


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the streaming chat model."""
    return ChatOpenAI(
        **build_model_kwargs(
            settings,
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_tokens,
        )
    )


def build_extraction_model(settings: Settings) -> Runnable:
    """Create the structured-output extraction runnable."""
    chat_model = ChatOpenAI(
        **build_model_kwargs(
            settings,
            temperature=settings.extract_temperature,
            max_tokens=settings.extract_max_tokens,
        )
    )
    return chat_model.with_structured_output(
        AfterSalesExtraction,
        method=settings.structured_output_method,
    )
