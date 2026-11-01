from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator
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

    # ch07 context management: budget derived from the model window
    model_context_window: int = Field(default=128000, gt=0)
    max_output_tokens: int = Field(default=2000, gt=0)
    max_user_input_tokens: int = Field(default=2000, gt=0)
    tool_result_max_tokens: int = Field(default=1200, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
    system_overhead_tokens: int = Field(default=2000, gt=0)
    evidence_tokens_per_item: int = Field(default=300, gt=0)
    summary_projection_tokens: int = Field(default=650, gt=0)
    context_safety_ratio: float = Field(default=0.10, ge=0, le=1)
    context_desired_turns: int = Field(default=40, gt=0)
    context_turn_steady_tokens: int = Field(default=600, gt=0)
    token_chars_per_token: float = Field(default=1.5, gt=0)
    layer2_assistant_chars: int = Field(default=60, gt=0)
    layer2_tool_result_keep_tokens: int = Field(default=120, gt=0)
    summary_model: str = ""
    summary_prompt_path: Path = Path("app/prompts/summary_prompt.txt")
    summary_temperature: float = 0.0
    summary_max_tokens: int = Field(default=512, gt=0)
    summary_projection_segments: int = Field(default=3, gt=0)
    summary_timeout_seconds: float = Field(default=30, gt=0)
    summary_max_retries: int = Field(default=0, ge=0)
    log_file: Path = Path("log/app.log")

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

    # ch08 tool system
    mcp_enabled: bool = True
    mcp_logistics_url: str = "http://127.0.0.1:8765/mcp"
    mcp_aftersales_url: str = "http://127.0.0.1:8766/mcp"
    mcp_request_timeout_seconds: float = Field(default=10, gt=0)
    mcp_read_only_servers: str = "logistics,aftersales"
    mcp_write_tools: str = ""
    tool_plugin_dir: Path = Path("app/tools/plugins")
    tool_audit_result_max_chars: int = Field(default=1000, gt=0)
    ticket_create_delay_seconds: float = Field(default=0, ge=0)

    # ch09 observability + data flywheel
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        # Accept the legacy Langfuse SDK name too; the first set wins.
        validation_alias=AliasChoices(
            "LANGFUSE_BASE_URL", "LANGFUSE_HOST"
        ),
    )
    langfuse_trace_name: str = "chat"
    evidence_confidence_threshold: float = Field(
        default=0.50, ge=0, le=1
    )  # calibrated 2026-11-01 on evals/ch04_eval_cases.jsonl
    evidence_effective_floor: float = Field(default=0.2, ge=0, le=1)
    flywheel_enabled: bool = True
    flywheel_dedup_candidate_limit: int = Field(default=100, gt=0)
    flywheel_model: str = ""
    eval_schedule_enabled: bool = False
    eval_interval_hours: float = Field(default=24, gt=0)
    eval_dataset_path: Path = Path("evals/ch04_eval_cases.jsonl")
    review_queue_page_size: int = Field(default=20, gt=0)

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
