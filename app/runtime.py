from dataclasses import dataclass

from fastapi import Request
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.llm import build_chat_model, build_extraction_model
from app.prompt_builder import PromptBuilder
from app.sessions import SessionStore


@dataclass
class Runtime:
    """Process-wide dependencies shared by the API routes."""

    settings: Settings
    session_store: SessionStore
    prompt_builder: PromptBuilder
    chat_model: ChatOpenAI
    extraction_model: Runnable


def build_runtime(settings: Settings) -> Runtime:
    """Assemble the runtime from settings; performs no network calls."""
    return Runtime(
        settings=settings,
        session_store=SessionStore(
            settings.max_sessions, settings.max_stored_messages
        ),
        prompt_builder=PromptBuilder(
            system_prompt_path=settings.system_prompt_path,
            extract_prompt_path=settings.extract_prompt_path,
            brand_name=settings.brand_name,
            service_hours=settings.service_hours,
        ),
        chat_model=build_chat_model(settings),
        extraction_model=build_extraction_model(settings),
    )


def get_runtime(request: Request) -> Runtime:
    """FastAPI dependency: return the runtime built during app startup."""
    return request.app.state.runtime
