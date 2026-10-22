from datetime import date
from pathlib import Path

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


class PromptBuilder:
    """Load prompt files and render chat messages.

    The system prompt is rendered separately and is never part of the
    history budget; callers trim ``history`` before passing it in.
    """

    def __init__(
        self,
        *,
        system_prompt_path: Path,
        extract_prompt_path: Path,
        brand_name: str,
        service_hours: str,
    ) -> None:
        self._system_template = PromptTemplate.from_template(
            system_prompt_path.read_text(encoding="utf-8")
        )
        self._extract_prompt = extract_prompt_path.read_text(encoding="utf-8")
        self._brand_name = brand_name
        self._service_hours = service_hours
        self._chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_text}"),
                MessagesPlaceholder("history"),
                ("human", "{message}"),
            ]
        )

    @property
    def extract_prompt(self) -> str:
        """Return the raw extraction prompt text."""
        return self._extract_prompt

    def render_system_text(self, today: date | None = None) -> str:
        """Render the system prompt with brand, hours, and date."""
        return self._system_template.format(
            brand_name=self._brand_name,
            service_hours=self._service_hours,
            today=(today or date.today()).isoformat(),
        )

    def build_messages(
        self,
        history: list[BaseMessage],
        user_message: str,
        *,
        today: date | None = None,
    ) -> list[BaseMessage]:
        """Render the system message, trimmed history, and user message."""
        messages = self._chat_prompt.format_messages(
            system_text=self.render_system_text(today),
            history=history,
            message=user_message,
        )
        return list(messages)
