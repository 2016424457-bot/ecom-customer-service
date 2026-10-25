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

    # ch05 graph orchestration
    max_agent_steps: int = Field(default=6, gt=0)
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"
    intent_prompt_path: Path = Path("app/prompts/intent_prompt.txt")
    agent_prompt_path: Path = Path("app/prompts/agent_prompt.txt")
    intent_temperature: float = 0.0
    intent_max_tokens: int = 128

    # ch06 router: intent / coref / expansion / refund judge
    intent_model: str = ""
    intent_fast_model: str = ""
    intent_confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    coref_prompt_path: Path = Path("app/prompts/coref_prompt.txt")
    coref_temperature: float = 0.0
    coref_max_tokens: int = 256
    coref_timeout_seconds: float = 8
    coref_max_retries: int = 0
    coref_history_messages: int = Field(default=10, gt=0)
    coref_max_history_chars: int = Field(default=2000, gt=0)
    expansion_prompt_path: Path = Path("app/prompts/query_expansion_prompt.txt")
    expansion_temperature: float = 0.0
    expansion_max_tokens: int = 256
    expansion_timeout_seconds: float = 8
    expansion_max_retries: int = 0
    refund_judge_prompt_path: Path = Path("app/prompts/refund_judge_prompt.txt")
    refund_judge_temperature: float = 0.0
    refund_judge_max_tokens: int = 512

    database_url: str = (
        "mysql+asyncmy://root:123456@127.0.0.1:3306/ecom_customer_service"
        "?charset=utf8mb4"
    )
    tool_timeout_seconds: float = Field(default=5, gt=0)
    tool_max_retries: int = Field(default=2, ge=0)

    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = Field(default=1024, gt=0)
    embed_timeout_seconds: float = Field(default=10, gt=0)
    embed_max_retries: int = Field(default=1, ge=0)

    milvus_uri: str = "./data/milvus_knowledge.db"
    milvus_collection: str = "knowledge"
    milvus_analyzer_type: str = "jieba"

    chunk_size: int = Field(default=700, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    table_max_rows: int = Field(default=12, gt=0)
    faq_top_k: int = Field(default=5, gt=0)
    faq_min_score: float = 0.35
    faq_tool_timeout_seconds: float = Field(default=30, gt=0)
    knowledge_docs_dir: Path = Path("knowledge_docs")

    retrieval_candidates: int = Field(default=50, gt=0)
    rag_top_k: int = Field(default=10, gt=0)

    query_rewrite_enabled: bool = True
    query_rewrite_prompt_path: Path = Path(
        "app/prompts/query_rewrite_prompt.txt"
    )
    query_rewrite_temperature: float = 0.0
    query_rewrite_max_tokens: int = 256
    query_rewrite_timeout_seconds: float = Field(default=8, gt=0)
    query_rewrite_max_retries: int = Field(default=0, ge=0)

    rerank_enabled: bool = True
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = Field(default=10, gt=0)
    rerank_min_score: float = 0.0
    rerank_timeout_seconds: float = Field(default=10, gt=0)
    rerank_max_retries: int = Field(default=0, ge=0)

    evidence_check_prompt_path: Path = Path(
        "app/prompts/evidence_check_prompt.txt"
    )
    evidence_check_temperature: float = 0.0
    evidence_check_max_tokens: int = 256
    refusal_message: str = (
        "抱歉，知识库中没有找到能回答这个问题的依据，我不能凭空回答。"
        "建议您转人工客服进一步确认。"
    )

    judge_model: str = ""
    faithfulness_prompt_path: Path = Path(
        "app/prompts/faithfulness_judge_prompt.txt"
    )
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024

    mining_batch_conversations: int = Field(default=10, gt=0)
    mining_max_messages: int = Field(default=60, gt=0)
    mining_temperature: float = 0.0
    mining_max_tokens: int = 1024
    mining_prompt_path: Path = Path("app/prompts/qa_mining_prompt.txt")

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
