from dataclasses import dataclass


@dataclass(frozen=True)
class StagingRow:
    batch_no: str
    source_ref: str | None
    category: str | None
    question: str
    answer: str


@dataclass(frozen=True)
class ConversationWindow:
    conversation_id: int
    max_message_id: int
