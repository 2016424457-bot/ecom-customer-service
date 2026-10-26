from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.prompt_builder import PromptBuilder


@dataclass
class GraphDeps:
    """Per-turn dependencies injected into graph nodes.

    Nodes are bound with ``functools.partial`` so no repository session or
    model is stored in module globals; concurrent turns stay isolated.
    """

    settings: Settings
    chat_model: Any
    intent_classifier: Any
    prompt_builder: PromptBuilder
    repository: Any
    knowledge_repository: Any
    evidence_checker: Any | None
    faq_search: Any
    tools: list
    tool_registry: Any
    agent_prompt: str
    coref_rewriter: Any
    query_expander: Any
    refund_judge_model: Any
    refund_judge_prompt: str
    order_provider: Any
    context_budget: Any
    summary_service: Any | None = None
