from typing import Literal

from pydantic import BaseModel


class RefundVerdict(BaseModel):
    eligible: bool | None
    answer: str


def request_type_for(query: str) -> Literal["退款", "退货", "换货"]:
    """Map the standard question's keywords to a request type.

    换货/换新 is a service request, not a refund form; the judge node turns it
    into an after-sales ticket instead of ``refund_form``.
    """
    if "换货" in query or "换新" in query:
        return "换货"
    if "退货" in query or "退回去" in query:
        return "退货"
    return "退款"
