from pydantic import BaseModel


class RefundVerdict(BaseModel):
    eligible: bool | None
    answer: str
