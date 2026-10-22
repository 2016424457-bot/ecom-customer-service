import logging

from pydantic import BaseModel, Field

from app.knowledge.types import QaPair

logger = logging.getLogger(__name__)


class QaExtraction(BaseModel):
    """Use parallel scalar arrays: nested object arrays break some upstreams."""

    questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


def parse_extraction(
    result: QaExtraction | dict,
) -> list[QaPair]:
    if not isinstance(result, QaExtraction):
        result = QaExtraction.model_validate(result)

    questions = result.questions
    answers = result.answers
    categories = result.categories
    if len(questions) != len(answers):
        logger.warning(
            "挖知识并列数组长度不一致 questions=%d answers=%d，按较短对齐",
            len(questions),
            len(answers),
        )

    pairs: list[QaPair] = []
    for index in range(min(len(questions), len(answers))):
        question = questions[index].strip()
        answer = answers[index].strip()
        if not question or not answer:
            continue
        category = (
            categories[index].strip()
            if index < len(categories) and categories[index].strip()
            else "历史对话"
        )
        pairs.append(
            QaPair(question=question, answer=answer, category=category)
        )
    return pairs
