from pydantic import BaseModel


class CorefResult(BaseModel):
    standalone_query: str
    order_id: str | None = None
