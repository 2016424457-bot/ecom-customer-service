from pydantic import BaseModel


class QueryExpansion(BaseModel):
    queries: list[str]
