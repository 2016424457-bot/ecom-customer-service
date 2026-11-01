from dataclasses import dataclass

from pydantic import BaseModel


class NormalizedQuestion(BaseModel):
    """A colloquial question rewritten for the review queue."""

    question: str
    suggested_answer: str
    grounded: bool


class DedupVerdict(BaseModel):
    """Result of matching one normalized question against pending candidates."""

    matched_id: int | None
    reason: str


@dataclass(frozen=True)
class ReviewCandidate:
    """One pending review-queue row offered to the deduplicator."""

    id: int
    normalized_question: str
    occurrence_count: int