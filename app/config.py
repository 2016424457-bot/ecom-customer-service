from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_base_url: str = "https://api.deepseek.com"
    openai_api_key: str = Field(min_length=1)
    openai_model: str = "deepseek-flash"

    chat_temperature: float = 0.3
    extract_temperature: float = 0.0
    chat_max_tokens: int = 1024
    extract_max_tokens: int = 512

    llm_timeout_seconds: float = 60
    llm_max_retries: int = 2
    llm_thinking: str = "disabled"

    structured_output_method: Literal[
        "function_calling", "json_mode", "json_schema"
    ] = "function_calling"

    history_token_budget: int = Field(default=2000, gt=0)
    max_stored_messages: int = Field(default=50, gt=0)
    max_sessions: int = Field(default=1000, gt=0)

    database_url: str = (
        "mysql+asyncmy://root:123456@127.0.0.1:3306/ecom_customer_service"
        "?charset=utf8mb4"
    )
    tool_timeout_seconds: float = Field(default=5, gt=0)
    tool_max_retries: int = Field(default=2, ge=0)

    system_prompt_path: Path = Path("app/prompts/system_prompt.txt")
    extract_prompt_path: Path = Path("app/prompts/extract_prompt.txt")

    brand_name: str = "示例商城"
    service_hours: str = "9:00-21:00"

    log_level: str = "INFO"
    log_content: bool = False

    @field_validator("openai_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Normalize the upstream base URL and require absolute http(s)."""
        normalized = value.rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "openai_base_url must be an absolute http:// or https:// URL"
            )
        return normalized
