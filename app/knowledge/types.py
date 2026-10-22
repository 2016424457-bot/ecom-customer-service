from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str
    is_key_clause: bool
    source_doc: str


@dataclass(frozen=True)
class VectorRecord:
    id: int
    text: str
    vector: list[float]
    category: str
    content_type: str | None
    section_path: str | None
    source_doc: str


@dataclass(frozen=True)
class KnowledgeHit:
    id: int
    category: str
    questions: str
    answer: str
    score: float
    section_path: str | None
    content_type: str | None
    source_doc: str | None = None


@dataclass(frozen=True)
class QaPair:
    question: str
    answer: str
    category: str


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
